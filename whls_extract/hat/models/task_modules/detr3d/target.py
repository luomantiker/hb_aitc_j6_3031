# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, Tuple

import torch
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import multi_apply
from hat.utils.distributed import get_dist_info
from .post_process import decode_preds

__all__ = ["Detr3dTarget"]


class HungarianAssigner3D(nn.Module):
    """Basic Assigner class.

    Args:
        cls_cost: classification cost module.
        reg_cost: regression cost module.
    """

    def __init__(self, cls_cost: nn.Module, reg_cost: nn.Module):
        super(HungarianAssigner3D, self).__init__()
        self.cls_cost = cls_cost
        self.reg_cost = reg_cost

    def _match(self, cost: Tensor) -> Tuple[Tensor, Tensor]:
        """Match bbox.

        Match the predicted bounding box
        with ground truth boxes based on
        the cost matrix.

        Args:
            cost : The cost matrix.

        Returns:
            The matched row indices and column indices.
        """
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        return matched_row_inds, matched_col_inds

    def forward(
        self,
        cls_gts: Tensor,
        cls_preds: Tensor,
        reg_gts: Tensor,
        reg_preds: Tensor,
        box_weight: Tensor,
    ) -> Tensor:
        """Forward pass of the module.

        Args:
            cls_gts : The ground truth classification tensor.
            cls_preds : The predicted classification tensor.
            reg_gts : The ground truth regression tensor.
            reg_preds : The predicted regression tensor.
            box_weight : The tensor containing weights
                         for regression bounding boxes.

        Returns:
            The matched row indices and column indices.
        """
        cls_cost = self.cls_cost(cls_preds, cls_gts)

        reg_cost = self.reg_cost(reg_preds[..., :8], reg_gts[..., :8])

        cost = reg_cost + cls_cost

        cost = cost.cpu().detach()
        return self._match(cost)


@OBJECT_REGISTRY.register
class Detr3dTarget(nn.Module):
    """Generate detr3d targets.

    Args:
        cls_cost: classification cost module.
        reg_cost: regression cost module.
        num_classes: Number of calassification.
        bbox_weight: Weight for bbox meta.
    """

    def __init__(
        self,
        cls_cost: nn.Module,
        reg_cost: nn.Module,
        bev_range,
        num_classes: int = 10,
        bbox_weight: float = None,
    ):
        super(Detr3dTarget, self).__init__()
        self.assigner = HungarianAssigner3D(cls_cost, reg_cost)
        self.num_classes = num_classes
        self.bev_range = bev_range
        if bbox_weight is not None:
            self.bbox_weight = bbox_weight
        else:
            self.bbox_weight = [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.2,
                0.2,
            ]

        self.bbox_weight = torch.tensor(self.bbox_weight)

    def _normalize(self, label: Tensor) -> Tuple[Tensor, Tensor]:
        """Normalize the label tensor.

        Args:
            label : The label tensor.

        Returns:
            The normalized classification tensor and regression tensor.
        """
        center = label[..., 0:3]
        dim = label[..., 3:6].log()
        rot = label[..., 6]
        rot_sine = torch.sin(rot).view(-1, 1)
        rot_cos = torch.cos(rot).view(-1, 1)
        if label.shape[1] == 10:
            vel = label[..., 7:9]
            cls = label[..., 9].long()
            reg = torch.cat(
                [center, dim, rot_sine, rot_cos, vel], dim=-1
            ).float()
        else:
            cls = label[..., 7].long()
            reg = torch.cat([center, dim, rot_sine, rot_cos], dim=-1).float()
        return cls, reg

    def _get_target_single(
        self, label: Tensor, cls_preds: Tensor, reg_preds: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Get the target values for a single example.

        Args:
            label: The label tensor.
            cls_preds: The predicted classification tensor.
            reg_preds: The predicted regression tensor.

        Returns:
            The target values for the classification and regression branches.
        """
        label = torch.tensor(label).to(device=cls_preds.device)
        cls_label = (
            (torch.ones((cls_preds.shape[0],)) * self.num_classes)
            .long()
            .to(device=cls_preds.device)
        )
        reg_label = (
            torch.zeros_like(reg_preds).to(device=reg_preds.device).float()
        )

        bbox_weight = torch.zeros_like(reg_preds).to(device=reg_preds.device)

        if len(label) == 0:
            return cls_preds, cls_label, reg_preds, reg_label, bbox_weight
        cls_gts, reg_gts = self._normalize(label)
        pred_idx, gts_idx = self.assigner(
            cls_gts, cls_preds, reg_gts, reg_preds, self.bbox_weight
        )
        cls_label[pred_idx] = cls_gts[gts_idx]

        reg_label[pred_idx] = reg_gts[gts_idx]
        bbox_weight[pred_idx] = 1.0

        return cls_preds, cls_label, reg_preds, reg_label, bbox_weight

    def forward(
        self,
        label: Tensor,
        cls_preds: Tensor,
        reg_preds: Tensor,
        reference_points: Tensor,
    ) -> Tuple[Dict, Dict]:
        """Forward pass of the module.

        Args:
            label : The label tensor.
            cls_preds : The predicted classification tensor.
            reg_preds : The predicted regression tensor.

        Returns:
            Dictionaries containing the target values
            for the classification and regression branches.
        """
        cls_preds, reg_preds = decode_preds(
            self.bev_range, cls_preds, reg_preds, reference_points
        )
        (
            cls_preds_list,
            cls_gts_list,
            reg_preds_list,
            reg_gts_list,
            bbox_weight_list,
        ) = multi_apply(
            self._get_target_single,
            label,
            cls_preds,
            reg_preds,
        )
        cls_preds = torch.cat(cls_preds_list)
        cls_gts = torch.cat(cls_gts_list)

        pos_idx = cls_gts < self.num_classes
        pos_num = torch.count_nonzero(pos_idx)

        pos_avg = torch.clamp(pos_num, min=1.0).float()

        if dist.is_initialized is True:
            dist.all_reduce(pos_avg, op=dist.ReduceOp.SUM)
            _, world_size = get_dist_info()
            pos_avg /= world_size
        cls_target = {
            "pred": cls_preds,
            "target": cls_gts,
            "avg_factor": pos_avg,
        }

        reg_preds = torch.cat(reg_preds_list)
        reg_gts = torch.cat(reg_gts_list)
        bbox_weight = torch.cat(bbox_weight_list)
        bbox_weight = bbox_weight * self.bbox_weight.to(
            device=reg_preds.device
        )
        reg_target = {
            "pred": reg_preds,
            "target": reg_gts,
            "weight": bbox_weight,
            "avg_factor": pos_avg,
        }
        return cls_target, reg_target
