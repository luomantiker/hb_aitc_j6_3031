# Copyright (c) Horizon Robotics. All rights reserved.
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = ["SegLoss", "MultiStrideLosses", "SegEdgeLoss"]


@OBJECT_REGISTRY.register
class SegLoss(torch.nn.Module):
    """
    Segmentation loss wrapper.

    Args:
        loss (dict): loss config.

    Note:
        This class is not universe. Make sure you know this class limit before
        using it.

    """

    def __init__(
        self,
        loss: List[torch.nn.Module],
    ):
        super(SegLoss, self).__init__()
        self.loss = loss

    def forward(self, pred: Any, target: List[Dict]) -> Dict:
        # Since pred is resized in SegTarget, use target directly.
        assert len(target) == len(self.loss)
        res = [
            single_loss(**single_target)
            for single_target, single_loss in zip(target, self.loss)
        ]

        return res


@OBJECT_REGISTRY.register
class MixSegLoss(nn.Module):
    """Calculate multi-losses with same prediction and target.

    Args:
        losses: List of losses with the same input pred and target.
        losses_weight: List of weights used for loss calculation.
            Default: None

    """

    def __init__(
        self,
        losses: List[nn.Module],
        losses_weight: List[float] = None,
        loss_name="mixsegloss",
    ):
        super(MixSegLoss, self).__init__()
        assert losses is not None
        self.losses_name = []
        self.losses = []
        self.loss_name = loss_name
        for loss in losses:
            self.losses.append(loss)
            self.losses_name.append(loss.loss_name)

        if losses_weight is None:
            losses_weight = [1.0 for _ in range(len(losses))]
        self.losses_weight = losses_weight

    def forward(self, pred, target):
        losses_res = {}
        for idx, loss in enumerate(self.losses):
            # for class SegTarget
            if isinstance(target, Dict):
                loss_val = loss(**target)
            else:
                loss_val = loss(pred, target)
            for key, item in loss_val.items():
                loss_val[key] = item * self.losses_weight[idx]
            losses_res.update(loss_val)

        return losses_res


@OBJECT_REGISTRY.register
class MixSegLossMultipreds(MixSegLoss):
    """Calculate multi-losses with multi-preds and correspondence targets.

    Args:
        losses: List of losses with different prediction and target.
        losses_weight: List of weights used for loss calculation.
            Default: None
        loss_name: Name of output loss
    """

    def __init__(
        self,
        losses: List[nn.Module],
        losses_weight: List[float] = None,
        loss_name: str = "multipredsloss",
    ):
        super(MixSegLossMultipreds, self).__init__(losses, losses_weight)
        self.loss_name = loss_name

    def forward(self, pred, target):
        pred = _as_list(pred)
        target = _as_list(target)
        if len(target) == 1:
            target = target * len(pred)
        losses_res = {}
        for idx, loss in enumerate(self.losses):
            loss_val = loss(pred[idx], target[idx])
            for key, item in loss_val.items():
                loss_val[key] = item * self.losses_weight[idx]
            losses_res.update(loss_val)

        return {self.loss_name: losses_res}


@OBJECT_REGISTRY.register
class MultiStrideLosses(nn.Module):
    """Multiple Stride Losses.

    Apply the same loss function with different loss weights
    to multiple outputs.

    Args:
        num_classes: Number of classes.
        out_strides: strides of output feature maps
        loss: Loss module.
        loss_weights: Loss weight.
    """

    def __init__(
        self,
        num_classes: int,
        out_strides: List[int],
        loss: nn.Module,
        loss_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        if loss_weights is not None:
            assert len(loss_weights) == len(out_strides)
        else:
            loss_weights = [1.0] * len(out_strides)
        self.out_strides = out_strides
        self.loss = loss
        self.loss_weights = loss_weights

    @autocast(enabled=False)
    def forward(
        self,
        preds: List[torch.Tensor],
        targets: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # convert to float32 while using amp
        preds = [pred.float() for pred in preds]
        assert (
            len(preds) == len(targets) == len(self.loss_weights)
        ), "%d vs. %d vs. %d" % (
            len(preds),
            len(targets),
            len(self.loss_weights),
        )

        targets, weights = self.slice_vanilla_labels(targets)

        losses = OrderedDict()

        for pred, target, weight, stride, loss_weight in zip(
            preds, targets, weights, self.out_strides, self.loss_weights
        ):
            losses[f"stride_{stride}_loss"] = (
                self.loss(pred, target, weight=weight) * loss_weight
            )

        return losses

    def slice_vanilla_labels(self, target):
        labels, weights = [], []

        # (N, 1, H, W) --> (N, 1, H, W, C) --> (N, C, H, W)
        for target_i in target:
            assert torch.all(target_i.abs() < self.num_classes)

            label_neg_mask = target_i < 0
            all_pos_label = target_i.detach().clone()

            # set neg label to positive to work around torch one_hot
            all_pos_label[label_neg_mask] *= -1

            target_i = F.one_hot(
                all_pos_label.type(torch.long), num_classes=self.num_classes
            )
            target_i[label_neg_mask] = -1

            target_i.squeeze_(axis=1)
            target_i = target_i.permute(0, 3, 1, 2)

            label_weight = target_i != -1

            labels.append(target_i)
            weights.append(label_weight)

        return labels, weights


@OBJECT_REGISTRY.register
class SegEdgeLoss(torch.nn.Module):
    def __init__(
        self,
        edge_graph: List[List[int]],
        kernel_half_size: int = 2,
        ignore_index: int = 255,
        loss_name: Optional[str] = None,
        loss_weight: float = 1e-5,
    ):
        """Edge loss for segmetaion task.

        Args:
            edge_graph (List[List[int]]): edges of graph like [
                [0, 1],
                [1, 2],
                [1, 3]
            ] means using the edge bewteen label idx 0 and 1, 1 and 2, 1 and 3.
            kernel_half_size (int): kernel_half_size. Defaults to 2.
            ignore_index (int): ignore_index. Defaults to 255.
            loss_name (str, optional): loss name.Defaults
                to None and return scalar.
            loss_weight (float): loss weight. Defaults to 1e-5.
        """
        super(SegEdgeLoss, self).__init__()

        self.edge_graph = edge_graph
        self.kernel_size = 2 * kernel_half_size + 1
        self.kernel = torch.ones(
            size=[1, 1, self.kernel_size, self.kernel_size]
        )
        self.ignore_index = ignore_index
        self.loss_name = loss_name
        self.loss_weight = loss_weight

        self.padding = torch.nn.MaxPool2d(
            kernel_size=self.kernel_size,
            padding=kernel_half_size,
            stride=1,
        )
        self.ce_loss = torch.nn.CrossEntropyLoss(
            reduction="none", ignore_index=ignore_index
        )

    def forward(self, pred, target, weight=None, avg_factor=None):
        pred_seg = torch.argmax(pred, dim=1, keepdim=True).float()
        gt_seg = target.view_as(pred_seg).float()

        edge_map_pred = self.get_edge_map(pred_seg)
        edge_map_gt = self.get_edge_map(gt_seg)

        loss = self.ce_loss(pred.float(), target.long()).unsqueeze(dim=1)
        loss = loss * edge_map_pred + loss * edge_map_gt
        loss = loss.mean(dim=0).sum()
        if self.loss_name is not None:
            return {self.loss_name: loss * self.loss_weight}
        else:
            return loss * self.loss_weight

    def get_edge_map(self, seg_map):
        edge_map = torch.zeros_like(seg_map)
        for edge in self.edge_graph:
            label_a = edge[0]
            label_b = edge[1]

            mask_a = torch.where(seg_map == label_a, 1.0, 0.0)
            mask_b = torch.where(seg_map == label_b, 1.0, 0.0)

            dilated_a = self.padding(mask_a)
            dilated_b = self.padding(mask_b)

            intersection_ab = dilated_b * mask_a
            intersection_ba = dilated_a * mask_b
            intersection = torch.logical_or(intersection_ab, intersection_ba)

            edge_map = torch.logical_or(edge_map, intersection)

        return edge_map
