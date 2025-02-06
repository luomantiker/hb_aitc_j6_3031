# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast

from hat.models.task_modules.bevformer.utils import normalize_bbox
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import multi_apply
from hat.utils.distributed import reduce_mean

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class BevFormerCriterion(nn.Module):
    """The basic structure of BevFormerCriterion.

    Args:
        assigner: Assigner module.
        num_classes: The num of class.
        sync_cls_avg_factor: Whether to sync cls avg factor.
        loss_cls: Cls loss module.
        loss_bbox: Bbox loss module.
        code_weights: Weights for the regression loss.
        pc_range: VCS range or point cloud range.
        num_group_detr: The group num of detr decoder.
        bg_cls_weight: The cls weight for background.
        bbox_key: The key name of bbox in data, available values are
                    'ego_bboxes_labels' and 'lidar_bboxes_labels'.
    """

    def __init__(
        self,
        assigner: nn.Module = None,
        num_classes: int = 10,
        sync_cls_avg_factor: bool = True,
        loss_cls: nn.Module = None,
        loss_bbox: nn.Module = None,
        code_weights: List[float] = None,
        pc_range: List[float] = None,
        num_group_detr: int = 1,
        bg_cls_weight: int = 0,
        bbox_key: str = "ego_bboxes_labels",
    ):
        super(BevFormerCriterion, self).__init__()
        self.assigner = assigner
        self.num_classes = num_classes
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.pc_range = pc_range
        self.bg_cls_weight = bg_cls_weight
        self.bbox_key = bbox_key
        assert bbox_key in ["ego_bboxes_labels", "lidar_bboxes_labels"]
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [
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
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False),
            requires_grad=False,
        )

        self.num_group_detr = num_group_detr

    def _get_target_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        gt_labels: Union[Tensor, None],
        gt_bboxes: Union[Tensor, None],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Get single target."""
        num_bboxes = bbox_pred.size(0)
        if gt_bboxes is None:
            labels = (
                torch.ones(
                    (num_bboxes,), dtype=torch.int64, device=bbox_pred.device
                )
                * self.num_classes
            )
            label_weights = torch.ones(
                (num_bboxes,), dtype=torch.int64, device=bbox_pred.device
            )
            bbox_targets = torch.zeros_like(bbox_pred)[..., :9].to(
                torch.float32
            )
            bbox_weights = torch.zeros_like(bbox_pred).to(torch.float32)
            pos_inds = torch.tensor([])
            neg_inds = torch.tensor(
                list(range(num_bboxes)),
                dtype=torch.int64,
                device=bbox_pred.device,
            )

            return (
                labels,
                label_weights,
                bbox_targets,
                bbox_weights,
                pos_inds,
                neg_inds,
            )

        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assigned_gt_inds = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels
        )
        pos_inds = (
            torch.nonzero(assigned_gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )

        neg_inds = (
            torch.nonzero(assigned_gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )

        # label targets
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]

        labels = gt_bboxes.new_full(
            (num_bboxes,), self.num_classes, dtype=torch.long
        )

        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c].to(
            torch.float32
        )
        bbox_weights = torch.zeros_like(bbox_pred).to(torch.float32)
        bbox_weights[pos_inds] = 1.0

        bbox_targets[pos_inds] = pos_gt_bboxes
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_inds,
            neg_inds,
        )

    def get_targets(
        self,
        cls_scores_list: List[Tensor],
        bbox_preds_list: List[Tensor],
        gt_bboxes_list: Union[List[Tensor], List[None]],
        gt_labels_list: Union[List[Tensor], List[None]],
    ) -> Tuple[
        List[Tensor], List[Tensor], List[Tensor], List[Tensor], int, int
    ]:
        """Get all targets."""
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_labels_list,
            gt_bboxes_list,
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def loss_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        gt_bboxes_list: Union[List[Tensor], List[None]],
        gt_labels_list: Union[List[Tensor], List[None]],
    ) -> Tuple[Tensor, Tensor]:
        """Get single loss."""
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.num_classes)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = (
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        )
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor])
            )

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )
        loss_cls = loss_cls["cls"]
        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos,
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @autocast(enabled=False)
    def forward(
        self,
        preds_dicts: Dict,
        data: Dict,
    ) -> Dict:
        """Forward bevformer criterion to get loss."""
        gt_bboxes_list = []
        gt_labels_list = []

        for gt_l in data["seq_meta"][0][self.bbox_key]:
            if len(gt_l):
                gt_bboxes_list.append(
                    torch.tensor(
                        gt_l[:, :9],
                        dtype=torch.float32,
                        device=preds_dicts["all_cls_scores"].device,
                    )
                )
                gt_labels_list.append(
                    torch.tensor(
                        gt_l[:, 9],
                        dtype=torch.int64,
                        device=preds_dicts["all_cls_scores"].device,
                    )
                )
            else:
                gt_bboxes_list.append(None)
                gt_labels_list.append(None)
        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]

        num_dec_layers = len(all_cls_scores)
        num_query = all_cls_scores.shape[-2]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        loss_dict = {}
        loss_dict["loss_cls"] = 0
        loss_dict["loss_bbox"] = 0
        for num_dec_layer in range(all_cls_scores.shape[0] - 1):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = 0
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = 0

        num_query_per_group = num_query // self.num_group_detr
        for group_index in range(self.num_group_detr):
            group_query_start = group_index * num_query_per_group
            group_query_end = (group_index + 1) * num_query_per_group
            group_cls_scores = all_cls_scores[
                :, :, group_query_start:group_query_end, :
            ]
            group_bbox_preds = all_bbox_preds[
                :, :, group_query_start:group_query_end, :
            ]
            losses_cls, losses_bbox = multi_apply(
                self.loss_single,
                group_cls_scores,
                group_bbox_preds,
                all_gt_bboxes_list,
                all_gt_labels_list,
            )
            loss_dict["loss_cls"] += losses_cls[-1] / self.num_group_detr
            loss_dict["loss_bbox"] += losses_bbox[-1] / self.num_group_detr
            # loss from other decoder layers
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(
                losses_cls[:-1], losses_bbox[:-1]
            ):
                loss_dict[f"d{num_dec_layer}.loss_cls"] += (
                    loss_cls_i / self.num_group_detr
                )
                loss_dict[f"d{num_dec_layer}.loss_bbox"] += (
                    loss_bbox_i / self.num_group_detr
                )
                num_dec_layer += 1
        return loss_dict
