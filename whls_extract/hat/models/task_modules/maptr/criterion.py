# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp import autocast

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import multi_apply
from hat.utils.distributed import reduce_mean

logger = logging.getLogger(__name__)

__all__ = ["MapTRCriterion"]


def bbox_cxcywh_to_xyxy(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox: Bounding boxes tensor of shape (n, 4).

    Returns:
        Converted bboxes tensor.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


def bbox_xyxy_to_cxcywh(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox: Shape (n, 4) for bboxes.

    Returns:
        Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)


def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (
        bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0]
    )
    bboxes[..., 1::2] = (
        bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    )

    return bboxes


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = (
        pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    )
    new_pts[..., 1:2] = (
        pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    )
    return new_pts


@OBJECT_REGISTRY.register
class MapTRCriterion(nn.Module):
    """MapTRCriterion.

    Args:
        assigner: The object responsible for assigning targets.
        num_classes: Number of classes for classification.
        sync_cls_avg_factor: Whether to sync classification average factor.
        loss_cls: Classification loss function.
        loss_pts: Points regression loss function.
        loss_dir: Direction loss function.
        loss_seg: Segmentation loss function.
        loss_pv_seg: Point cloud segmentation loss function.
        code_weights: Weights for bounding box regression.
        pc_range: Point cloud range.
        dir_interval: Interval for direction prediction.
        num_pts_per_vec: Number of points per predicted vector.
        num_pts_per_gt_vec: Number of points per ground truth vector.
        gt_shift_pts_pattern: Pattern for ground truth shift points.
        aux_seg: Auxiliary segmentation parameters.
        pred_absolute_points: Whether to predict absolute points.
            Default: False.
    """

    def __init__(
        self,
        assigner: nn.Module = None,
        num_classes: int = 10,
        sync_cls_avg_factor: bool = True,
        loss_cls: nn.Module = None,
        loss_pts: nn.Module = None,
        loss_dir: nn.Module = None,
        loss_seg: nn.Module = None,
        loss_pv_seg: nn.Module = None,
        code_weights: Optional[List[float]] = None,
        pc_range: Optional[List[float]] = None,
        dir_interval: int = 1,
        num_pts_per_vec: int = 2,
        num_pts_per_gt_vec: int = 2,
        gt_shift_pts_pattern: str = "v0",
        aux_seg: Optional[dict] = None,
        pred_absolute_points: bool = False,
    ):
        super(MapTRCriterion, self).__init__()
        self.assigner = assigner
        self.num_classes = num_classes
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.bg_cls_weight = 0
        self.cls_out_channels = num_classes
        self.pc_range = pc_range

        self.loss_cls = loss_cls
        self.loss_pts = loss_pts
        self.loss_dir = loss_dir
        self.loss_seg = loss_seg
        self.loss_pv_seg = loss_pv_seg

        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        self.dir_interval = dir_interval

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

        self.aux_seg = aux_seg
        self.pred_absolute_points = pred_absolute_points

    def _get_target_single(
        self,
        cls_score,
        bbox_pred,
        pts_pred,
        gt_labels,
        gt_bboxes,
        gt_shifts_pts,
        gt_bboxes_ignore=None,
    ):

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        (
            num_gts,
            assigned_gt_inds,
            _,
            assigned_labels,
            order_index,
        ) = self.assigner.assign(
            bbox_pred,
            cls_score,
            pts_pred,
            gt_bboxes,
            gt_labels,
            gt_shifts_pts,
            gt_bboxes_ignore,
        )

        # sampling_result = self.sampler.sample(
        #     assign_result, bbox_pred, gt_bboxes
        # )

        # pos_inds = sampling_result.pos_inds
        # neg_inds = sampling_result.neg_inds
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

        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]

        # label targets
        labels = gt_bboxes.new_full(
            (num_bboxes,), self.num_classes, dtype=torch.long
        )
        # labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c].to(
            torch.float32
        )
        bbox_weights = torch.zeros_like(bbox_pred).to(torch.float32)
        bbox_weights[pos_inds] = 1.0

        # pts targets
        # import pdb;pdb.set_trace()
        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[pos_inds, pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros(
            (pts_pred.size(0), pts_pred.size(1), pts_pred.size(2))
        )
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[
            pos_assigned_gt_inds, assigned_shift, :, :
        ]
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pts_targets,
            pts_weights,
            pos_inds,
            neg_inds,
        )

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        pts_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_shifts_pts_list,
        gt_bboxes_ignore_list=None,
    ):
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            pts_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        pts_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_shifts_pts_list,
        gt_bboxes_ignore_list=None,
    ):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        # import pdb;pdb.set_trace()
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            pts_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = (
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        )
        if self.sync_cls_avg_factor:
            cls_avg_factor = cls_scores.new_tensor([cls_avg_factor])
            if cls_avg_factor.device.type != "cpu":
                cls_avg_factor = reduce_mean(cls_avg_factor)

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )
        loss_cls = loss_cls["cls"]

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        if num_total_pos.device.type != "cpu":
            num_total_pos = torch.clamp(
                reduce_mean(num_total_pos), min=1
            ).item()
        else:
            num_total_pos = torch.clamp(num_total_pos, min=1).item()
        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(
            bbox_targets, self.pc_range
        )
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        # regression pts CD loss
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(
            -1, pts_preds.size(-2), pts_preds.size(-1)
        )
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(
                pts_preds,
                size=(self.num_pts_per_gt_vec),
                mode="linear",
                align_corners=True,
            )
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()
        if self.pred_absolute_points:
            pts_preds = normalize_2d_pts(pts_preds, self.pc_range)

        loss_pts = self.loss_pts(
            pts_preds[isnotnan, :, :],
            normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos,
        )
        dir_weights = pts_weights[:, : -self.dir_interval, 0]
        if self.pred_absolute_points:
            denormed_pts_preds = pts_preds
        else:
            denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = (
            denormed_pts_preds[:, self.dir_interval :, :]
            - denormed_pts_preds[:, : -self.dir_interval, :]
        )
        pts_targets_dir = (
            pts_targets[:, self.dir_interval :, :]
            - pts_targets[:, : -self.dir_interval, :]
        )
        # dir_weights = pts_weights[:, indice,:-1,0]
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :],
            pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos,
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_pts = torch.nan_to_num(loss_pts)
        loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_pts, loss_dir

    @autocast(enabled=False)
    def forward(
        self,
        preds_dicts,
        data,
        gt_bboxes_ignore=None,
    ):

        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )

        if "seq_meta" in data:
            gt_bboxes_list = data["seq_meta"][0]["gt_instances"]
            gt_labels_list = data["seq_meta"][0]["gt_labels_map"]
        else:
            gt_bboxes_list = data["gt_instances"]
            gt_labels_list = data["gt_labels_map"]
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)

        all_cls_scores = preds_dicts["all_cls_scores"].float()
        all_bbox_preds = preds_dicts["all_bbox_preds"].float()
        all_pts_preds = preds_dicts["all_pts_preds"].float()
        enc_cls_scores = preds_dicts["enc_cls_scores"]
        enc_bbox_preds = preds_dicts["enc_bbox_preds"]
        enc_pts_preds = preds_dicts["enc_pts_preds"]

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [
            gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list
        ]
        gt_pts_list = [
            gt_bboxes.fixed_num_sampled_points.to(device)
            for gt_bboxes in gt_vecs_list
        ]
        if self.gt_shift_pts_pattern == "v0":
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device)
                for gt_bboxes in gt_vecs_list
            ]
        elif self.gt_shift_pts_pattern == "v1":
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device)
                for gt_bboxes in gt_vecs_list
            ]
        elif self.gt_shift_pts_pattern == "v2":
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device)
                for gt_bboxes in gt_vecs_list
            ]
        elif self.gt_shift_pts_pattern == "v3":
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device)
                for gt_bboxes in gt_vecs_list
            ]
        elif self.gt_shift_pts_pattern == "v4":
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device)
                for gt_bboxes in gt_vecs_list
            ]
        elif self.gt_shift_pts_pattern == "sparsedrive":
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_sparsedrive.to(device)
                for gt_bboxes in gt_vecs_list
            ]
        else:
            raise NotImplementedError
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [
            gt_shifts_pts_list for _ in range(num_dec_layers)
        ]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        (losses_cls, losses_pts, losses_dir,) = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_pts_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list,
        )

        loss_dict = {}

        if self.aux_seg and self.aux_seg["use_aux_seg"]:
            if self.aux_seg["bev_seg"]:
                if preds_dicts["seg"] is not None:
                    gt_seg_mask = data["seq_meta"][0]["gt_seg_mask"]
                    seg_output = preds_dicts["seg"].float()
                    num_imgs = seg_output.size(0)
                    seg_gt = torch.stack(
                        [gt_seg_mask[i] for i in range(num_imgs)], dim=0
                    )
                    loss_seg = self.loss_seg(seg_output, seg_gt.float())
                    loss_dict["loss_seg"] = loss_seg
            if self.aux_seg["pv_seg"]:
                if preds_dicts["pv_seg"] is not None:
                    if "seq_meta" in data:
                        gt_pv_seg_masks = data["seq_meta"][0]["gt_pv_seg_mask"]
                    else:
                        gt_pv_seg_masks = data["gt_pv_seg_mask"]
                    pv_seg_output = preds_dicts["pv_seg"]
                    num_levels = len(pv_seg_output)
                    pv_seg_gt = [
                        torch.stack(
                            [
                                seg_mask[i].float()
                                for seg_mask in gt_pv_seg_masks
                            ],
                            dim=0,
                        )
                        for i in range(num_levels)
                    ]
                    loss_pv_seg = self.loss_pv_seg(pv_seg_output, pv_seg_gt)
                    loss_dict["loss_pv_seg"] = loss_pv_seg
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            # TODO bug here
            (enc_loss_cls, enc_losses_pts, enc_losses_dir,) = self.loss_single(
                enc_cls_scores.float(),
                enc_bbox_preds.float(),
                enc_pts_preds.float(),
                gt_bboxes_list,
                binary_labels_list,
                gt_pts_list,
                gt_bboxes_ignore,
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_losses_pts"] = enc_losses_pts
            loss_dict["enc_losses_dir"] = enc_losses_dir

        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_pts"] = losses_pts[-1]
        loss_dict["loss_dir"] = losses_dir[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i, loss_dir_i in zip(
            losses_cls[:-1],
            losses_pts[:-1],
            losses_dir[:-1],
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_pts"] = loss_pts_i
            loss_dict[f"d{num_dec_layer}.loss_dir"] = loss_dir_i
            num_dec_layer += 1
        return loss_dict
