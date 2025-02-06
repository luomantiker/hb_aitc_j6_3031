# Copyright (c) Horizon Robotics. All rights reserved.
# Source code reference to mmdetection
import itertools
import math
from collections import OrderedDict
from math import pi as PI
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from hat.core.box_utils import bbox_overlaps
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import multi_apply

__all__ = [
    "FCOSTarget",
    "DynamicFcosTarget",
    "get_points",
    "distance2bbox",
    "FCOSTarget4RPNHead",
    "VehicleSideFCOSTarget",
    "DynamicVehicleSideFcosTarget",
]

INF = 1e8


def distance2bbox(
    points: Tensor, distance: Tensor, max_shape: Optional[Tuple] = None
) -> Tensor:
    """Decode distance prediction to bounding box.

    Args:
        points: Shape (n, 2), [x, y].
        distance: Distance from the given point to 4 boundaries
            (left, top, right, bottom).
        max_shape: Shape of the image, used to clamp decoded
            bbox in max_shape range.

    Returns:
        Decoded bbox, with shape (n, 4).

    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def frcnn_regression2bbox(
    rel_codes: Tensor,
    points: Tensor,
    widths: int = 3,
    heights: int = 3,
    bbox_xform_clip: float = math.log(1000.0 / 16),  # noqa: B008
) -> Tensor:
    """Get the decoded boxes.

    Args:
        rel_codes (Tensor): encoded boxes.
        points (Tensor): reference points (N, 2).
        widths (int): the width of the corresponding anchor.
        heights (int): the height of the corresponding anchor.
        bbox_xform_clip (float): prevent sending too large values into exp().

    Returns:
        Decoded bbox, with shape (n, 4).
    """
    points = points.to(rel_codes.dtype)

    ctr_x = points[:, 0] - 0.5
    ctr_y = points[:, 1] - 0.5

    dx = rel_codes[:, 0::4]
    dy = rel_codes[:, 1::4]
    dw = rel_codes[:, 2::4]
    dh = rel_codes[:, 3::4]

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)

    pred_ctr_x = dx * widths + ctr_x[:, None]
    pred_ctr_y = dy * heights + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    # Distance from center to box's corner.
    c_to_c_h = (
        torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device)
        * pred_h
    )
    c_to_c_w = (
        torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device)
        * pred_w
    )

    pred_boxes1 = pred_ctr_x - c_to_c_w
    pred_boxes2 = pred_ctr_y - c_to_c_h
    pred_boxes3 = pred_ctr_x + c_to_c_w
    pred_boxes4 = pred_ctr_y + c_to_c_h
    pred_boxes = torch.stack(
        (pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2
    ).flatten(1)
    return pred_boxes


def distancealpha2polygon(points, distance, tanalpha, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (torch.Tensor): Shape (n, 2), [x, y].
        distance (torch.Tensor): Distance from the given point to 4 boundaries
            (left, top, right, bottom).
        max_shape (tuple, optional): Shape of the image(height, width),
            used to clamp decoded
            bbox in max_shape range.

    Returns:
        torch.Tensor: Decoded bbox, with shape (n, 4).

    """
    left = points[..., 0] - distance[..., 0]
    top = points[..., 1] - distance[..., 1]
    right = points[..., 0] + distance[..., 2]
    bottom = points[..., 1] + distance[..., 3]
    bottom_right = bottom + distance[..., 2] * tanalpha
    bottom_left = bottom - distance[..., 0] * tanalpha
    if max_shape is not None:
        left = left.clamp(min=0, max=max_shape[1])
        right = right.clamp(min=0, max=max_shape[1])
        bottom_left = bottom_left.clamp(min=0, max=max_shape[0])
        bottom_right = bottom_right.clamp(min=0, max=max_shape[0])
    return torch.stack(
        [left, top, right, top, right, bottom_right, left, bottom_left], -1
    )


def get_points(
    feat_sizes: List[Tuple],
    strides: List[int],
    dtype: torch.dtype,
    device: torch.device,
    flatten: bool = False,
) -> List[Tensor]:
    """Generate points according to feat_sizes.

    Args:
        feat_sizes: Multi-level feature map sizes, the value is the HW of
            a certain layer.
        strides: Multi-level feature map strides.
        dtype: Type of points should be.
        device: Device of points should be.
        flatten: Whether to flatten 2D coordinates into 1D dimension.

    Returns:
        Points of multiple levels belong to each image,
            the value in mlvl_points is [Tensor(H1W1, 2), Tensor(H2W2, 2), ...]
    """

    def _get_points_single(feat_size, stride):
        h, w = feat_size
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        points = (
            torch.stack(
                (x.reshape(-1) * stride, y.reshape(-1) * stride), dim=-1
            )
            + stride // 2
        )
        return points

    mlvl_points = []
    for i in range(len(feat_sizes)):
        mlvl_points.append(_get_points_single(feat_sizes[i], strides[i]))
    return mlvl_points


@OBJECT_REGISTRY.register
class FCOSTarget(nn.Module):
    """Generate cls and reg targets for FCOS in training stage.

    Args:
        strides: Strides of points in multiple feature levels.
        regress_ranges: Regress range of multiple level points.
        cls_out_channels: Out_channels of cls_score.
        background_label: Label ID of background, set as num_classes.
        center_sampling: If true, use center sampling.
        center_sample_radius: Radius of center sampling. Default: 1.5.
        norm_on_bbox: If true, normalize the regression targets with
            FPN strides.
        use_iou_replace_ctrness: If true, use iou as box quality
            assessment method, else use ctrness. Default: false.
        task_batch_list: Mask for different label source dataset.
    """

    def __init__(
        self,
        strides: Tuple[int, ...],
        regress_ranges: Tuple[Tuple[int, int], ...],
        cls_out_channels: int,
        background_label: int,
        norm_on_bbox: bool = True,
        center_sampling: bool = True,
        center_sample_radius: float = 1.5,
        use_iou_replace_ctrness: bool = False,
        task_batch_list: Optional[List[int]] = None,
    ):
        super(FCOSTarget, self).__init__()
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.cls_out_channels = cls_out_channels
        self.background_label = background_label
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.use_iou_replace_ctrness = use_iou_replace_ctrness
        self.norm_on_bbox = norm_on_bbox
        self.task_batch_list = task_batch_list

    @staticmethod
    def _get_ignore_bboxes(gt_bboxes_list, gt_labels_list):
        # Currently, the box corresponding to label <0 indicates that it needs
        # to be ignored
        gt_bboxes_ignore_list = [None] * len(gt_bboxes_list)
        for ii, (gt_bboxes, gt_labels) in enumerate(
            zip(gt_bboxes_list, gt_labels_list)
        ):
            if gt_bboxes.shape[0] > 0:
                gt_bboxes_ignore = gt_bboxes[gt_labels < 0]
                if len(gt_bboxes_ignore.shape) == 1:
                    gt_bboxes_ignore = gt_bboxes_ignore.unsqueeze(0)
                gt_bboxes_ignore_list[ii] = gt_bboxes_ignore
        return gt_bboxes_ignore_list

    def _get_target_single(
        self,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore,
        points,
        regress_ranges,
        num_points_per_lvl,
        background_label,
    ):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        # If the gt label is full of -1, do not calculate the image loss
        num_pos_gts = sum(gt_labels != -1)
        num_ignore = 0
        if gt_bboxes_ignore is not None:
            num_ignore = gt_bboxes_ignore.size(0)
        if num_pos_gts == 0:
            return (
                gt_labels.new_full((num_points,), background_label),
                gt_bboxes.new_zeros((num_points, 4)),
                gt_bboxes.new_zeros((num_points,)),
            )

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1]
        )
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2
        )
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)

        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        if num_ignore != 0:
            gt_bboxes_ignore = gt_bboxes_ignore[None].expand(
                num_points, num_ignore, 4
            )
            xs_ignore = points[:, 0][:, None].expand(num_points, num_ignore)
            ys_ignore = points[:, 1][:, None].expand(num_points, num_ignore)

            left = xs_ignore - gt_bboxes_ignore[..., 0]
            right = gt_bboxes_ignore[..., 2] - xs_ignore
            top = ys_ignore - gt_bboxes_ignore[..., 1]
            bottom = gt_bboxes_ignore[..., 3] - ys_ignore
            ignore_bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(
                x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0]
            )
            center_gts[..., 1] = torch.where(
                y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1]
            )
            center_gts[..., 2] = torch.where(
                x_maxs > gt_bboxes[..., 2], gt_bboxes[..., 2], x_maxs
            )
            center_gts[..., 3] = torch.where(
                y_maxs > gt_bboxes[..., 3], gt_bboxes[..., 3], y_maxs
            )

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1
            )
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]
        ) & (max_regress_distance <= regress_ranges[..., 1])

        # If gt_bboxes_ignore is not none, condition: limit the regression
        # range out of gt_bboxes_ignore
        if num_ignore != 0:
            inside_ignore_bbox_mask = ignore_bbox_targets.min(-1)[0] > 0

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        label_weights = labels.new_ones(labels.shape[0], dtype=torch.float)
        if num_ignore != 0:
            inside_ignore, _ = inside_ignore_bbox_mask.max(dim=1)
            label_weights[inside_ignore == 1] = 0.0

        labels[min_area == INF] = background_label  # set as BG
        labels = labels.long()
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets, label_weights

    @staticmethod
    def _centerness_target(pos_bbox_targets: Tensor) -> Tensor:
        """Compute centerness targets.

        Args:
            pos_bbox_targets: BBox targets of positive bboxes,
                with shape (num_pos, 4).

        """
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]
        ) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def forward(self, label, pred, *args):
        assert len(pred) == 3
        if isinstance(pred, dict):
            # assert order is cls_scores, bbox_preds, centernesses
            assert isinstance(pred, OrderedDict)
            cls_scores, bbox_preds, centernesses = pred.values()
        else:
            assert isinstance(pred, (list, tuple))
            cls_scores, bbox_preds, centernesses = pred

        # flatten and concat head output
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)

        gt_bboxes_list = label["gt_bboxes"]
        gt_labels_list = label["gt_classes"]
        feat_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = bbox_preds[0].device
        dtype = bbox_preds[0].dtype

        all_level_points = get_points(feat_sizes, self.strides, dtype, device)
        assert len(all_level_points) == len(self.regress_ranges)
        num_levels = len(all_level_points)
        num_imgs = len(gt_bboxes_list)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            all_level_points[i]
            .new_tensor(self.regress_ranges[i])[None]
            .expand_as(all_level_points[i])
            for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(all_level_points, dim=0)

        # the number of points per lvl, all imgs are equal
        num_points = [center.size(0) for center in all_level_points]
        gt_bboxes_ignore_list = self._get_ignore_bboxes(
            gt_bboxes_list, gt_labels_list
        )

        with torch.no_grad():
            labels_list, bbox_targets_list, label_weights_list = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                gt_bboxes_ignore_list,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points,
                background_label=self.background_label,
            )
        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        label_weights_list = [
            label_weights.split(num_points, 0)
            for label_weights in label_weights_list
        ]  # noqa
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_label_weights = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list])
            )
            concat_lvl_label_weights.append(
                torch.cat(
                    [label_weights[i] for label_weights in label_weights_list]
                )
            )  # noqa
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list]
            )
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)

        # generate bbox targets and centerness targets of positive points
        flatten_labels = torch.cat(concat_lvl_labels)
        flatten_label_weights = torch.cat(concat_lvl_label_weights)
        flatten_bbox_targets = torch.cat(concat_lvl_bbox_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points]
        )

        pos_inds = (
            (
                (flatten_labels >= 0)
                & (flatten_labels != self.background_label)
                & (flatten_label_weights > 0)
            )
            .nonzero()
            .reshape(-1)
        )
        num_pos = len(pos_inds)
        cls_avg_factor = max(num_pos, 1.0)

        if num_pos == 0:
            # generate fake pos_inds
            pos_inds = flatten_labels.new_zeros((1,))

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_points = flatten_points[pos_inds]
        # decode pos_bbox_targets to bbox for calculating IOU-like loss
        pos_decoded_targets = distance2bbox(pos_points, pos_bbox_targets)

        # fix loss conflict
        points_per_strides = []
        for feat_size in feat_sizes:
            points_per_strides.append(feat_size[0] * feat_size[1])
        valid_classes_list = None
        if self.task_batch_list is not None:
            valid_classes_list = []
            accumulate_list = list(itertools.accumulate(self.task_batch_list))
            task_id = 0
            for ii in range(num_imgs):
                if ii >= accumulate_list[task_id]:
                    task_id += 1
                valid_classes = []
                for cls in label["gt_classes"][ii].unique():
                    if cls >= 0:
                        valid_classes.append(cls.item())
                if len(valid_classes) == 0:
                    valid_classes.append(task_id)
                valid_classes_list.append(valid_classes)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        eps = 1e-5  # to avoid iou loss not converge bug
        pos_decoded_bbox_preds = distance2bbox(
            pos_points, pos_bbox_preds.relu() + eps
        )
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos == 0:
            pos_centerness_targets = torch.tensor([0.0], device=device)
            centerness_weight = torch.tensor([0.0], device=device)  # noqa
            bbox_weight = torch.tensor([0.0], device=device)
            bbox_avg_factor = None
        else:
            centerness_weight = None
            if not self.use_iou_replace_ctrness:
                pos_centerness_targets = self._centerness_target(
                    pos_bbox_targets
                )  # noqa
                bbox_weight = pos_centerness_targets
                bbox_avg_factor = pos_centerness_targets.sum()
            else:
                pos_centerness_targets = bbox_overlaps(
                    pos_decoded_bbox_preds.detach(),  # noqa
                    pos_decoded_targets,
                    is_aligned=True,
                )
                bbox_weight = None
                bbox_avg_factor = None

        cls_target = {
            "pred": flatten_cls_scores,
            "target": flatten_labels,
            "weight": flatten_label_weights,
            "avg_factor": cls_avg_factor,
            "points_per_strides": points_per_strides,
            "valid_classes_list": valid_classes_list,
        }
        giou_target = {
            "pred": pos_decoded_bbox_preds,
            "target": pos_decoded_targets,
            "weight": bbox_weight,
            "avg_factor": bbox_avg_factor,
        }
        centerness_target = {
            "pred": pos_centerness,
            "target": pos_centerness_targets,
            "weight": centerness_weight,
        }
        return cls_target, giou_target, centerness_target


@OBJECT_REGISTRY.register
class DynamicFcosTarget(nn.Module):
    """Generate cls and box training targets for FCOS based on simOTA label \
    assignment strategy used in YOLO-X.

    Args:
        strides: Strides of points in multiple feature levels.
        topK: Number of positive sample for each ground truth to keep.
        cls_out_channels: Out_channels of cls_score.
        background_label: Label ID of background, set as num_classes.
        loss_cls: Loss for cls to choose positive target.
        loss_reg: Loss for reg to choose positive target.
        center_sampling: Whether to perform center sampling.
        center_sampling_radius: The radius of the center sampling area.
        bbox_relu: Whether apply relu to bbox preds.
    """

    def __init__(
        self,
        strides: Sequence[int],
        topK: int,
        loss_cls: nn.Module,
        loss_reg: nn.Module,
        cls_out_channels: int,
        background_label: int,
        center_sampling: bool = False,
        center_sampling_radius: float = 2.5,
        bbox_relu: bool = False,
    ):
        super(DynamicFcosTarget, self).__init__()
        self.strides = strides
        self.topK = topK
        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.cls_out_channels = cls_out_channels
        self.background_label = background_label
        self.center_sampling = center_sampling
        self.center_sampling_radius = center_sampling_radius
        self.bbox_relu = bbox_relu

    def _get_iou(self, flatten_bbox_preds, bbox_targets, points):
        decoded_targets = distance2bbox(points, bbox_targets)
        decoded_preds = distance2bbox(points, flatten_bbox_preds)
        return bbox_overlaps(decoded_preds, decoded_targets, is_aligned=True)

    def _get_cost(
        self, cls_preds, cls_targets, bbox_preds, bbox_targets, points
    ):

        cls_loss = list(self.loss_cls(cls_preds, cls_targets.long()).values())[
            0
        ]
        cls_loss = cls_loss.sum(2)
        bbox_preds_decoded = distance2bbox(points, bbox_preds)
        bbox_targets_decoded = distance2bbox(points, bbox_targets)
        reg_loss = list(
            self.loss_reg(bbox_preds_decoded, bbox_targets_decoded).values()
        )[0]
        return cls_loss + reg_loss

    def _get_target_single(
        self,
        gt_bboxes,
        gt_labels,
        flatten_cls_scores,
        flatten_bbox_preds,
        strides,
        points,
        device,
        num_points_per_lvl,
    ):
        num_gts = gt_bboxes.shape[0]
        if num_gts == 0:
            bbox_targets = torch.zeros_like(flatten_bbox_preds, device=device)
            cls_targets = (
                torch.ones((flatten_cls_scores.shape[0]), device=device)
                * self.background_label
            )
            centerness_targets = torch.zeros(
                (flatten_bbox_preds.shape[0]), device=device
            )
            label_weights = torch.ones_like(
                cls_targets, dtype=torch.float, device=device
            )
            return (
                cls_targets.long(),
                bbox_targets,
                centerness_targets,
                label_weights,
            )

        gt_labels = gt_labels.view(1, -1).repeat(
            flatten_cls_scores.shape[0], 1
        )
        bbox_targets = torch.zeros(
            (flatten_bbox_preds.shape[0], num_gts, 4), device=device
        )

        bbox_targets[..., 0:2] = points[:, None, :] - gt_bboxes[None, :, 0:2]
        bbox_targets[..., 2:4] = gt_bboxes[None, :, 2:4] - points[:, None, :]

        # do center sampling
        if self.center_sampling:
            # condition1: inside a `center bbox`
            num_points = points.size(0)
            xs, ys = points[:, 0], points[:, 1]
            xs = xs[:, None].expand(num_points, num_gts)
            ys = ys[:, None].expand(num_points, num_gts)
            gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)

            radius = self.center_sampling_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(
                x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0]
            )
            center_gts[..., 1] = torch.where(
                y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1]
            )
            center_gts[..., 2] = torch.where(
                x_maxs > gt_bboxes[..., 2], gt_bboxes[..., 2], x_maxs
            )
            center_gts[..., 3] = torch.where(
                y_maxs > gt_bboxes[..., 3], gt_bboxes[..., 3], y_maxs
            )

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1
            )
            inside_gt_bbox_mask = (center_bbox.min(-1)[0] > 0).int()
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = (bbox_targets.min(-1)[0] > 0).int()

        ignore_labels = gt_labels < 0
        ignore_labels[inside_gt_bbox_mask != 1] = 0

        inside_gt_bbox_mask[ignore_labels] = -1
        bbox_targets /= strides[:, None, :]
        points = points / strides[..., 0:2]
        points = points.view(-1, 1, 2).repeat(1, num_gts, 1)
        flatten_bbox_preds = flatten_bbox_preds.view(-1, 1, 4).repeat(
            1, num_gts, 1
        )
        flatten_cls_scores = flatten_cls_scores.view(
            -1, 1, self.cls_out_channels
        ).repeat(1, num_gts, 1)

        ious = self._get_iou(flatten_bbox_preds, bbox_targets, points)
        ious[inside_gt_bbox_mask != 1] = 0.0

        cost = self._get_cost(
            flatten_cls_scores,
            gt_labels,
            flatten_bbox_preds,
            bbox_targets,
            points,
        )
        cost[inside_gt_bbox_mask != 1] = INF
        cost = cost.permute(1, 0).contiguous()
        n_candidate_k = min(self.topK, ious.size(0))
        topk_ious, _ = torch.topk(ious, n_candidate_k, dim=0)

        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        matching_matrix = torch.zeros_like(
            cost, device=flatten_bbox_preds.device
        )
        for gt_idx in range(num_gts):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0
        matching_gt = matching_matrix.sum(0)

        if (matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, matching_gt > 1], dim=0)
            matching_matrix[:, matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, matching_gt > 1] = 1.0

        matching_matrix = matching_matrix.permute(1, 0).contiguous()
        matching_matrix[inside_gt_bbox_mask != 1] = 0.0

        gt_labels[matching_matrix == 0] = self.background_label
        argmin = gt_labels.argmin(1).view(-1)

        cls_targets = gt_labels[torch.arange(gt_labels.shape[0]), argmin]

        bbox_targets = bbox_targets[
            torch.arange(bbox_targets.shape[0]), argmin
        ]
        centerness_targets = ious[torch.arange(bbox_targets.shape[0]), argmin]
        # if (inside_gt_bbox_mask == -1).sum().cpu().numpy() != 0:
        #     print((inside_gt_bbox_mask == -1).sum())
        label_weights = torch.ones_like(gt_labels, dtype=torch.float)
        label_weights[inside_gt_bbox_mask == -1] = 0
        label_weights, _ = label_weights.min(-1)
        return cls_targets, bbox_targets, centerness_targets, label_weights

    def forward(self, label, pred, *args):
        cls_scores, bbox_preds, centernesses = pred
        if self.bbox_relu:
            bbox_preds = [nn.functional.relu(i) for i in bbox_preds]

        gt_bboxes_list = label["gt_bboxes"]
        gt_labels_list = label["gt_classes"]
        num_imgs = len(gt_bboxes_list)
        # flatten and concat head output
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.cls_out_channels
            )
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for centerness in centernesses
        ]

        device = bbox_preds[0].device
        dtype = bbox_preds[0].dtype

        flatten_strides = [
            torch.tensor([s, s, s, s], device=device)
            .view(-1, 4)
            .repeat(flatten_bbox_pred.shape[1], 1)
            for flatten_bbox_pred, s in zip(flatten_bbox_preds, self.strides)
        ]
        flatten_strides = torch.cat(flatten_strides)
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_centerness = torch.cat(flatten_centerness, dim=1)

        feat_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        all_level_points = get_points(feat_sizes, self.strides, dtype, device)
        # expand regress ranges to align with points
        concat_points = torch.cat(all_level_points, dim=0)
        # the number of points per lvl, all imgs are equal
        num_points = [center.size(0) for center in all_level_points]

        with torch.no_grad():
            (
                cls_targets_list,
                bbox_targets_list,
                centerness_targets_list,
                label_weights_list,
            ) = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                flatten_cls_scores,
                flatten_bbox_preds,
                strides=flatten_strides,
                points=concat_points,
                device=device,
                num_points_per_lvl=num_points,
            )
        cls_targets = torch.cat(cls_targets_list)
        bbox_targets = torch.cat(bbox_targets_list)
        centerness_targets = torch.cat(centerness_targets_list)

        label_weights = torch.cat(label_weights_list)

        pos_inds = (
            (
                (cls_targets >= 0)
                & (cls_targets != self.background_label)
                & (label_weights > 0)
            )
            .nonzero()
            .reshape(-1)
        )

        num_pos = len(pos_inds)
        cls_avg_factor = max(num_pos, 1.0)
        if num_pos == 0:
            pos_inds = flatten_cls_scores.new_zeros((1,)).long()
        pos_bbox_targets = bbox_targets[pos_inds]
        if num_pos == 0:
            centerness_weight = torch.tensor([0.0], device=device)  # noqa
            bbox_weight = torch.tensor([0.0], device=device)
            bbox_avg_factor = None
        else:
            centerness_weight = None
            bbox_weight = FCOSTarget._centerness_target(
                pos_bbox_targets
            )  # noqa
            bbox_avg_factor = bbox_weight.sum()

        flatten_cls_scores = flatten_cls_scores.view(-1, self.cls_out_channels)
        flatten_bbox_preds = flatten_bbox_preds.view(-1, 4)
        flatten_centerness = flatten_centerness.view(-1)

        cls_target = {
            "pred": flatten_cls_scores,
            "target": cls_targets.long(),
            "weight": label_weights,
            "avg_factor": cls_avg_factor,
        }

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_points = torch.zeros((2), device=device)
        pos_decoded_targets = distance2bbox(pos_points, pos_bbox_targets)
        pos_decoded_preds = distance2bbox(pos_points, pos_bbox_preds)

        giou_target = {
            "pred": pos_decoded_preds,
            "target": pos_decoded_targets,
            "weight": bbox_weight,
            "avg_factor": bbox_avg_factor,
        }

        pos_centerness = flatten_centerness[pos_inds]
        pos_centerness_targets = centerness_targets[pos_inds]
        centerness_target = {
            "pred": pos_centerness,
            "target": pos_centerness_targets,
            "weight": centerness_weight,
        }
        return cls_target, giou_target, centerness_target


@OBJECT_REGISTRY.register
class FCOSTarget4RPNHead(FCOSTarget):
    """Generate fcos-style cls and reg targets for RPNHead and HingeLoss.

    Args:
        strides: Strides of points in multiple feature levels.
        regress_ranges: Regress range of multiple level points.
        cls_out_channels: Out_channels of cls_score.
        background_label: Label ID of background, set as num_classes.
        center_sampling: If true, use center sampling.
        center_sample_radius: Radius of center sampling. Default: 1.5.
        norm_on_bbox: If true, normalize the regression targets with
            FPN strides.
        use_iou_replace_ctrness: If true, use iou as box quality
            assessment method, else use ctrness. Default: false.
        soft_label: If true, Use iou as class ground truth.
        task_batch_list: Mask for different label source dataset.
        reference_anchor_width: the width of the corresponding anchor.
        reference_anchor_height: the height of the corresponding anchor.
    """

    def __init__(
        self,
        strides: Tuple[int, ...],
        regress_ranges: Tuple[Tuple[int, int], ...],
        cls_out_channels: int,
        background_label: int,
        norm_on_bbox: bool = True,
        center_sampling: bool = True,
        center_sample_radius: float = 1.5,
        use_iou_replace_ctrness: bool = False,
        soft_label: bool = False,
        task_batch_list: Optional[List[int]] = None,
        reference_anchor_width: int = 3,
        reference_anchor_height: int = 3,
    ):
        super(FCOSTarget4RPNHead, self).__init__(
            strides=strides,
            regress_ranges=regress_ranges,
            cls_out_channels=cls_out_channels,
            background_label=background_label,
            center_sampling=center_sampling,
            center_sample_radius=center_sample_radius,
            use_iou_replace_ctrness=use_iou_replace_ctrness,
            norm_on_bbox=norm_on_bbox,
            task_batch_list=task_batch_list,
        )
        self.soft_label = soft_label
        self.reference_anchor_width = reference_anchor_width
        self.reference_anchor_height = reference_anchor_height

    def __call__(self, label, pred, *args):
        assert "rpn_head_out" in pred
        cls_scores, bbox_preds = pred["rpn_cls_pred"], pred["rpn_reg_pred"]

        # flatten and concat head output
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)

        gt_bboxes_list = label["gt_bboxes"]
        gt_labels_list = label["gt_classes"]
        feat_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = bbox_preds[0].device
        dtype = bbox_preds[0].dtype

        all_level_points = get_points(feat_sizes, self.strides, dtype, device)
        assert len(all_level_points) == len(self.regress_ranges)
        num_levels = len(all_level_points)
        num_imgs = len(gt_bboxes_list)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            all_level_points[i]
            .new_tensor(self.regress_ranges[i])[None]
            .expand_as(all_level_points[i])
            for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(all_level_points, dim=0)

        # the number of points per lvl, all imgs are equal
        num_points = [center.size(0) for center in all_level_points]
        gt_bboxes_ignore_list = self._get_ignore_bboxes(
            gt_bboxes_list, gt_labels_list
        )

        with torch.no_grad():
            labels_list, bbox_targets_list, label_weights_list = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                gt_bboxes_ignore_list,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points,
                background_label=self.background_label,
            )
        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        label_weights_list = [
            label_weights.split(num_points, 0)
            for label_weights in label_weights_list
        ]  # noqa
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_label_weights = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list])
            )
            concat_lvl_label_weights.append(
                torch.cat(
                    [label_weights[i] for label_weights in label_weights_list]
                )
            )  # noqa
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list]
            )
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)

        # generate bbox targets and centerness targets of positive points
        flatten_labels = torch.cat(concat_lvl_labels)
        flatten_label_weights = torch.cat(concat_lvl_label_weights)
        flatten_bbox_targets = torch.cat(concat_lvl_bbox_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points]
        )

        pos_inds = (
            (
                (flatten_labels >= 0)
                & (flatten_labels != self.background_label)
                & (flatten_label_weights > 0)
            )
            .nonzero()
            .reshape(-1)
        )
        num_pos = len(pos_inds)
        cls_avg_factor = max(num_pos, 1.0)

        if num_pos == 0:
            # generate fake pos_inds
            pos_inds = flatten_labels.new_zeros((1,))

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_points = flatten_points[pos_inds]
        # decode pos_bbox_targets to bbox for calculating IOU-like loss
        pos_decoded_targets = distance2bbox(pos_points, pos_bbox_targets)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_decoded_bbox_preds = frcnn_regression2bbox(
            pos_bbox_preds,
            pos_points,
            widths=self.reference_anchor_width,
            heights=self.reference_anchor_height,
        )

        if num_pos == 0:
            pos_centerness_targets = torch.tensor([0.0], device=device)
            bbox_weight = torch.tensor([0.0], device=device)
            bbox_avg_factor = None
        else:
            if not self.use_iou_replace_ctrness:
                pos_centerness_targets = self._centerness_target(
                    pos_bbox_targets
                )  # noqa
                bbox_weight = pos_centerness_targets
                bbox_avg_factor = pos_centerness_targets.sum()
            else:
                pos_centerness_targets = bbox_overlaps(
                    pos_decoded_bbox_preds.detach(),
                    pos_decoded_targets,
                    is_aligned=True,
                )
                bbox_weight = None
                bbox_avg_factor = None

        flatten_labels[flatten_labels < 0] = self.cls_out_channels
        flatten_labels = nn.functional.one_hot(
            flatten_labels, self.cls_out_channels + 1
        )  # N x C+1
        N, C, H, W = cls_scores[0].shape
        flatten_labels = flatten_labels[..., : self.cls_out_channels]  # N x C
        if self.soft_label:
            flatten_labels = flatten_labels.to(pos_centerness_targets.dtype)
            flatten_labels[pos_inds] *= pos_centerness_targets[:, None]
        flatten_labels = flatten_labels.reshape([N, C, H, W])
        flatten_label_weights = flatten_label_weights.reshape(
            N, -1, H, W
        ).repeat(1, self.cls_out_channels, 1, 1)

        cls_target = {
            "pred": cls_scores[0],
            "target": flatten_labels,
            "weight": flatten_label_weights,
            "avg_factor": cls_avg_factor,
        }
        giou_target = {
            "pred": pos_decoded_bbox_preds,
            "target": pos_decoded_targets,
            "weight": bbox_weight,
            "avg_factor": bbox_avg_factor,
        }
        return cls_target, giou_target


@OBJECT_REGISTRY.register
class VehicleSideFCOSTarget(FCOSTarget):
    def __init__(
        self,
        strides: Tuple[int, ...],
        regress_ranges: Tuple[Tuple[int, int], ...],
        cls_out_channels: int,
        background_label: int,
        norm_on_bbox: bool = True,
        center_sampling: bool = True,
        center_sample_radius: float = 1.5,
        use_iou_replace_ctrness: bool = False,
        task_batch_list: Optional[List[int]] = None,
        decouple_h: bool = False,
    ):
        super(VehicleSideFCOSTarget, self).__init__(
            strides,
            regress_ranges,
            cls_out_channels,
            background_label,
            norm_on_bbox,
            center_sampling,
            center_sample_radius,
            use_iou_replace_ctrness,
            task_batch_list,
        )
        self.decouple_h = decouple_h

    @staticmethod
    def _get_ignore_bboxes_and_tanalphas(
        gt_bboxes_list, gt_tanalphas_list, gt_labels_list
    ):
        # Currently, the box corresponding to label <0 indicates that it needs
        # to be ignored
        gt_bboxes_ignore_list = [None] * len(gt_bboxes_list)
        gt_tanalphas_ignore_list = [None] * len(gt_tanalphas_list)
        for ii, (gt_bboxes, gt_tanalphas, gt_labels) in enumerate(
            zip(gt_bboxes_list, gt_tanalphas_list, gt_labels_list)
        ):
            if gt_bboxes.shape[0] > 0:
                gt_bboxes_ignore = gt_bboxes[gt_labels < 0]
                if len(gt_bboxes_ignore.shape) == 1:
                    gt_bboxes_ignore = gt_bboxes_ignore.unsqueeze(0)
                gt_bboxes_ignore_list[ii] = gt_bboxes_ignore
                gt_tanalphas_ignore = gt_tanalphas[gt_labels < 0]
                if len(gt_tanalphas_ignore.shape) == 0:
                    gt_tanalphas_ignore = gt_tanalphas_ignore.unsqueeze(0)
                gt_tanalphas_ignore_list[ii] = gt_tanalphas_ignore
        return gt_bboxes_ignore_list, gt_tanalphas_ignore_list

    def _get_target_single(
        self,
        gt_bboxes,
        gt_tanalphas,
        gt_labels,
        gt_bboxes_ignore,
        gt_tanalphas_ignore,
        points,
        regress_ranges,
        num_points_per_lvl,
        background_label,
    ):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        # If the gt label is full of -1, do not calculate the image loss
        num_pos_gts = sum(gt_labels != -1)
        num_ignore = 0
        if gt_bboxes_ignore is not None:
            num_ignore = gt_bboxes_ignore.size(0)
        if num_pos_gts == 0:
            return (
                gt_labels.new_full((num_points,), background_label),
                gt_bboxes.new_zeros((num_points, 4)),
                gt_bboxes.new_zeros((num_points,)),
                gt_bboxes.new_zeros((num_points,)),
            )

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1]
        )
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2
        )
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        gt_tanalphas = gt_tanalphas[None].expand(num_points, num_gts)
        gt_ybs = (
            gt_bboxes[..., 3]
            if self.decouple_h
            else (
                gt_bboxes[..., 3]
                - (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2.0 * gt_tanalphas
            )
        )

        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        ybs = ys if self.decouple_h else (ys - xs * gt_tanalphas)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_ybs - ybs
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        tanalpha_targets = gt_tanalphas
        if num_ignore != 0:
            gt_bboxes_ignore = gt_bboxes_ignore[None].expand(
                num_points, num_ignore, 4
            )
            gt_tanalphas_ignore = gt_tanalphas_ignore[None].expand(
                num_points, num_ignore
            )
            gt_ybs_ignore = (
                gt_bboxes_ignore[..., 3]
                if self.decouple_h
                else (
                    gt_bboxes_ignore[..., 3]
                    - (gt_bboxes_ignore[..., 0] + gt_bboxes_ignore[..., 2])
                    / 2.0
                    * gt_tanalphas_ignore
                )
            )
            xs_ignore = points[:, 0][:, None].expand(num_points, num_ignore)
            ys_ignore = points[:, 1][:, None].expand(num_points, num_ignore)
            ybs_ignore = (
                ys_ignore
                if self.decouple_h
                else (ys_ignore - xs_ignore * gt_tanalphas_ignore)
            )
            left = xs_ignore - gt_bboxes_ignore[..., 0]
            right = gt_bboxes_ignore[..., 2] - xs_ignore
            top = ys_ignore - gt_bboxes_ignore[..., 1]
            bottom = gt_ybs_ignore - ybs_ignore
            ignore_bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            yb_maxs = (
                y_maxs
                if self.decouple_h
                else (y_maxs - center_xs * gt_tanalphas)
            )
            center_gts[..., 0] = torch.where(
                x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0]
            )
            center_gts[..., 1] = torch.where(
                y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1]
            )
            center_gts[..., 2] = torch.where(
                x_maxs > gt_bboxes[..., 2], gt_bboxes[..., 2], x_maxs
            )
            center_gts[..., 3] = torch.where(yb_maxs > gt_ybs, gt_ybs, yb_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ybs
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1
            )
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]
        ) & (max_regress_distance <= regress_ranges[..., 1])

        # If gt_bboxes_ignore is not none, condition: limit the regression
        # range out of gt_bboxes_ignore
        if num_ignore != 0:
            inside_ignore_bbox_mask = ignore_bbox_targets.min(-1)[0] > 0

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        label_weights = labels.new_ones(labels.shape[0], dtype=torch.float)
        if num_ignore != 0:
            inside_ignore, _ = inside_ignore_bbox_mask.max(dim=1)
            label_weights[inside_ignore == 1] = 0.0

        labels[min_area == INF] = background_label  # set as BG
        labels = labels.long()
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        tanalpha_targets = tanalpha_targets[range(num_points), min_area_inds]

        return labels, bbox_targets, tanalpha_targets, label_weights

    def forward(self, label, pred, *args):
        assert len(pred) == 4
        if isinstance(pred, dict):
            # assert order is cls_scores, bbox_preds, centernesses
            assert isinstance(pred, OrderedDict)
            cls_scores, bbox_preds, alpha_preds, centernesses = pred.values()
        else:
            assert isinstance(pred, (list, tuple))
            cls_scores, bbox_preds, alpha_preds, centernesses = pred
        # flatten and concat head output
        # N C H W --> N H W C
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_alpha_preds = [
            alpha_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for alpha_pred in alpha_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_alpha_preds = torch.cat(flatten_alpha_preds)
        flatten_centerness = torch.cat(flatten_centerness)

        gt_bboxes_list = label["gt_bboxes"]
        gt_tanalphas_list = label[
            "gt_tanalphas"
        ]  # gt_tanalphas_list.shape = (n,)
        gt_labels_list = label["gt_classes"]  # gt_labels_list.shape = (n,)
        feat_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = bbox_preds[0].device
        dtype = bbox_preds[0].dtype

        all_level_points = get_points(feat_sizes, self.strides, dtype, device)
        assert len(all_level_points) == len(self.regress_ranges)
        num_levels = len(all_level_points)
        num_imgs = len(gt_bboxes_list)
        assert num_imgs == len(gt_tanalphas_list)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            all_level_points[i]
            .new_tensor(self.regress_ranges[i])[None]
            .expand_as(all_level_points[i])
            for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(all_level_points, dim=0)

        # the number of points per lvl, all imgs are equal
        num_points = [center.size(0) for center in all_level_points]
        (
            gt_bboxes_ignore_list,
            gt_tanalphas_ignore_list,
        ) = self._get_ignore_bboxes_and_tanalphas(
            gt_bboxes_list, gt_tanalphas_list, gt_labels_list
        )
        (
            labels_list,
            bbox_targets_list,
            tanalpha_targets_list,
            label_weights_list,
        ) = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_tanalphas_list,
            gt_labels_list,
            gt_bboxes_ignore_list,
            gt_tanalphas_ignore_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points,
            background_label=self.background_label,
        )

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        label_weights_list = [
            label_weights.split(num_points, 0)
            for label_weights in label_weights_list
        ]  # noqa
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        tanalpha_targets_list = [
            tanalpha_targets.split(num_points, 0)
            for tanalpha_targets in tanalpha_targets_list
        ]
        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_tanalpha_targets = []
        concat_lvl_label_weights = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list])
            )
            concat_lvl_label_weights.append(
                torch.cat(
                    [label_weights[i] for label_weights in label_weights_list]
                )
            )  # noqa
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list]
            )
            tanalpha_targets = torch.cat(
                [
                    tanalpha_targets[i]
                    for tanalpha_targets in tanalpha_targets_list
                ]
            )
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_tanalpha_targets.append(tanalpha_targets)

        # generate bbox targets and centerness targets of positive points
        flatten_labels = torch.cat(concat_lvl_labels)
        flatten_label_weights = torch.cat(concat_lvl_label_weights)
        flatten_bbox_targets = torch.cat(concat_lvl_bbox_targets)
        flatten_tanalpha_targets = torch.cat(concat_lvl_tanalpha_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points]
        )

        pos_inds = (
            (
                (flatten_labels >= 0)
                & (flatten_labels != self.background_label)
                & (flatten_label_weights > 0)
            )
            .nonzero()
            .reshape(-1)
        )
        num_pos = len(pos_inds)
        cls_avg_factor = max(num_pos, 1.0)

        if num_pos == 0:
            # generate fake pos_inds
            pos_inds = flatten_labels.new_zeros((1,))

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_alpha_targets = flatten_tanalpha_targets[pos_inds]
        pos_alpha_targets.arctan_()
        pos_alpha_targets.divide_(PI / 2)
        pos_points = flatten_points[pos_inds]
        # decode pos_bbox_targets to bbox for calculating IOU-like loss
        pos_decoded_targets = distance2bbox(pos_points, pos_bbox_targets)

        # fix loss conflict
        points_per_strides = []
        for feat_size in feat_sizes:
            points_per_strides.append(feat_size[0] * feat_size[1])
        valid_classes_list = None
        if self.task_batch_list is not None:
            valid_classes_list = []
            accumulate_list = list(itertools.accumulate(self.task_batch_list))
            task_id = 0
            for ii in range(num_imgs):
                if ii >= accumulate_list[task_id]:
                    task_id += 1
                valid_classes = []
                for cls in label["gt_classes"][ii].unique():
                    if cls >= 0:
                        valid_classes.append(cls.item())
                if len(valid_classes) == 0:
                    valid_classes.append(task_id)
                valid_classes_list.append(valid_classes)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
        pos_alpha_preds = flatten_alpha_preds[pos_inds]
        pos_alpha_preds.squeeze_()
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos == 0:
            pos_centerness_targets = torch.tensor([0.0], device=device)
            centerness_weight = torch.tensor([0.0], device=device)  # noqa
            bbox_weight = torch.tensor([0.0], device=device)
            bbox_avg_factor = None
            tanalpha_weight = torch.tensor([0.0], device=device)
            tanalpha_avg_factor = None
        else:
            centerness_weight = None
            if not self.use_iou_replace_ctrness:
                pos_centerness_targets = self._centerness_target(
                    pos_bbox_targets
                )  # noqa
                bbox_weight = pos_centerness_targets
                bbox_avg_factor = pos_centerness_targets.sum()
                tanalpha_weight = pos_centerness_targets
                tanalpha_avg_factor = pos_centerness_targets.sum()
            else:
                raise NotImplementedError()

        cls_target = {
            "pred": flatten_cls_scores,
            "target": flatten_labels,
            "weight": flatten_label_weights,
            "avg_factor": cls_avg_factor,
            "points_per_strides": points_per_strides,
            "valid_classes_list": valid_classes_list,
        }
        giou_target = {
            "pred": pos_decoded_bbox_preds,
            "target": pos_decoded_targets,
            "weight": bbox_weight,
            "avg_factor": bbox_avg_factor,
        }
        l1_target = {
            "pred": pos_alpha_preds,
            "target": pos_alpha_targets,
            "weight": tanalpha_weight,
            "avg_factor": tanalpha_avg_factor,
        }
        centerness_target = {
            "pred": pos_centerness,
            "target": pos_centerness_targets,
            "weight": centerness_weight,
        }

        return cls_target, giou_target, l1_target, centerness_target


@OBJECT_REGISTRY.register
class DynamicVehicleSideFcosTarget(nn.Module):
    """Generate cls and box training targets for FCOS based on simOTA label \
    assignment strategy used in YOLO-X.

    Args:
        strides: Strides of points in multiple feature levels.
        topK: Number of positive sample for each ground truth to keep.
        cls_out_channels: Out_channels of cls_score.
        background_label: Label ID of background, set as num_classes.
        loss_cls: Loss for cls to choose positive target.
        loss_reg: Loss for reg to choose positive target.
        center_sampling: Whether to perform center sampling.
        center_sampling_radius: The radius of the center sampling area.
        bbox_relu: Whether apply relu to bbox preds.
        decouple_h: Whether decouple height when calculating targets.
    """

    def __init__(
        self,
        strides: Sequence[int],
        topK: int,
        loss_cls: nn.Module,
        loss_reg: nn.Module,
        cls_out_channels: int,
        background_label: int,
        center_sampling: bool = False,
        center_sampling_radius: float = 2.5,
        bbox_relu: bool = False,
        decouple_h: bool = False,
    ):
        super(DynamicVehicleSideFcosTarget, self).__init__()
        self.strides = strides
        self.topK = topK
        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.cls_out_channels = cls_out_channels
        self.background_label = background_label
        self.center_sampling = center_sampling
        self.center_sampling_radius = center_sampling_radius
        self.bbox_relu = bbox_relu
        self.decouple_h = decouple_h

    def _get_iou(self, flatten_bbox_preds, bbox_targets, points):
        decoded_targets = distance2bbox(points, bbox_targets)
        decoded_preds = distance2bbox(points, flatten_bbox_preds)
        return bbox_overlaps(decoded_preds, decoded_targets, is_aligned=True)

    def _get_cost(
        self, cls_preds, cls_targets, bbox_preds, bbox_targets, points
    ):

        cls_loss = list(self.loss_cls(cls_preds, cls_targets.long()).values())[
            0
        ]
        cls_loss = cls_loss.sum(2)
        bbox_preds_decoded = distance2bbox(points, bbox_preds)
        bbox_targets_decoded = distance2bbox(points, bbox_targets)
        reg_loss = list(
            self.loss_reg(bbox_preds_decoded, bbox_targets_decoded).values()
        )[0]
        return cls_loss + reg_loss

    def _get_target_single(
        self,
        gt_bboxes,
        gt_tanalphas,
        gt_labels,
        flatten_cls_scores,
        flatten_bbox_preds,
        flatten_alpha_preds,
        strides,
        points,
        device,
        num_points_per_lvl,
    ):
        num_gts = gt_bboxes.shape[0]
        if num_gts == 0:
            bbox_targets = torch.zeros_like(flatten_bbox_preds, device=device)
            alpha_targets = torch.zeros(
                (flatten_alpha_preds.shape[0]), device=device
            )
            cls_targets = (
                torch.ones((flatten_cls_scores.shape[0]), device=device)
                * self.background_label
            )
            centerness_targets = torch.zeros(
                (flatten_bbox_preds.shape[0]), device=device
            )
            label_weights = torch.ones_like(
                cls_targets, dtype=torch.float, device=device
            )
            return (
                cls_targets.long(),
                bbox_targets,
                alpha_targets,
                centerness_targets,
                label_weights,
            )

        gt_labels = gt_labels.view(1, -1).repeat(
            flatten_cls_scores.shape[0], 1
        )
        gt_bboxes = gt_bboxes.view(1, -1, 4).repeat(
            flatten_bbox_preds.shape[0], 1, 1
        )
        gt_tanalphas = gt_tanalphas.view(1, -1).repeat(
            flatten_alpha_preds.shape[0], 1
        )
        gt_ybs = (
            gt_bboxes[..., 3]
            if self.decouple_h
            else (
                gt_bboxes[..., 3]
                - (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2.0 * gt_tanalphas
            )
        )

        num_points = points.size(0)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        ybs = ys if self.decouple_h else (ys - xs * gt_tanalphas)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_ybs - ybs
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        tanalpha_targets = gt_tanalphas

        # do center sampling
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sampling_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            yb_maxs = (
                y_maxs
                if self.decouple_h
                else (y_maxs - center_xs * gt_tanalphas)
            )
            center_gts[..., 0] = torch.where(
                x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0]
            )
            center_gts[..., 1] = torch.where(
                y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1]
            )
            center_gts[..., 2] = torch.where(
                x_maxs > gt_bboxes[..., 2], gt_bboxes[..., 2], x_maxs
            )
            center_gts[..., 3] = torch.where(yb_maxs > gt_ybs, gt_ybs, yb_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ybs
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1
            )
            inside_gt_bbox_mask = (center_bbox.min(-1)[0] > 0).int()
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = (bbox_targets.min(-1)[0] > 0).int()

        ignore_labels = gt_labels < 0
        ignore_labels[inside_gt_bbox_mask != 1] = 0

        inside_gt_bbox_mask[ignore_labels] = -1
        bbox_targets /= strides[:, None, :]
        points = points / strides[..., 0:2]
        points = points.view(-1, 1, 2).repeat(1, num_gts, 1)
        flatten_bbox_preds = flatten_bbox_preds.view(-1, 1, 4).repeat(
            1, num_gts, 1
        )
        flatten_cls_scores = flatten_cls_scores.view(
            -1, 1, self.cls_out_channels
        ).repeat(1, num_gts, 1)

        ious = self._get_iou(flatten_bbox_preds, bbox_targets, points)
        ious[inside_gt_bbox_mask != 1] = 0.0

        cost = self._get_cost(
            flatten_cls_scores,
            gt_labels,
            flatten_bbox_preds,
            bbox_targets,
            points,
        )
        cost[inside_gt_bbox_mask != 1] = INF
        cost = cost.permute(1, 0).contiguous()
        n_candidate_k = min(self.topK, ious.size(0))
        topk_ious, _ = torch.topk(ious, n_candidate_k, dim=0)

        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        matching_matrix = torch.zeros_like(
            cost, device=flatten_bbox_preds.device
        )
        for gt_idx in range(num_gts):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0
        matching_gt = matching_matrix.sum(0)

        if (matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, matching_gt > 1], dim=0)
            matching_matrix[:, matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, matching_gt > 1] = 1.0

        matching_matrix = matching_matrix.permute(1, 0).contiguous()
        matching_matrix[inside_gt_bbox_mask != 1] = 0.0

        gt_labels[matching_matrix == 0] = self.background_label
        argmin = gt_labels.argmin(1).view(-1)

        cls_targets = gt_labels[torch.arange(gt_labels.shape[0]), argmin]

        bbox_targets = bbox_targets[
            torch.arange(bbox_targets.shape[0]), argmin
        ]
        tanalpha_targets = tanalpha_targets[
            torch.arange(tanalpha_targets.shape[0]), argmin
        ]
        centerness_targets = ious[torch.arange(bbox_targets.shape[0]), argmin]

        label_weights = torch.ones_like(gt_labels, dtype=torch.float)
        label_weights[inside_gt_bbox_mask == -1] = 0
        label_weights, _ = label_weights.min(-1)
        return (
            cls_targets,
            bbox_targets,
            tanalpha_targets,
            centerness_targets,
            label_weights,
        )

    def forward(self, label, pred, *args):
        assert len(pred) == 4
        # cls_scores, bbox_preds, centernesses = pred
        if isinstance(pred, dict):
            # assert order is cls_scores, bbox_preds, centernesses
            assert isinstance(pred, OrderedDict)
            cls_scores, bbox_preds, alpha_preds, centernesses = pred.values()
        else:
            assert isinstance(pred, (list, tuple))
            cls_scores, bbox_preds, alpha_preds, centernesses = pred

        if self.bbox_relu:
            bbox_preds = [nn.functional.relu(i) for i in bbox_preds]

        gt_bboxes_list = label["gt_bboxes"]
        gt_tanalphas_list = label[
            "gt_tanalphas"
        ]  # gt_tanalphas_list.shape = (n,)
        gt_labels_list = label["gt_classes"]

        num_imgs = len(gt_bboxes_list)
        # flatten and concat head output
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.cls_out_channels
            )
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_alpha_preds = [
            alpha_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            for alpha_pred in alpha_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for centerness in centernesses
        ]

        device = bbox_preds[0].device
        dtype = bbox_preds[0].dtype

        flatten_strides = [
            torch.tensor([s, s, s, s], device=device)
            .view(-1, 4)
            .repeat(flatten_bbox_pred.shape[1], 1)
            for flatten_bbox_pred, s in zip(flatten_bbox_preds, self.strides)
        ]
        flatten_strides = torch.cat(flatten_strides)
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_alpha_preds = torch.cat(flatten_alpha_preds, dim=1)
        flatten_centerness = torch.cat(flatten_centerness, dim=1)

        feat_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        all_level_points = get_points(feat_sizes, self.strides, dtype, device)
        # expand regress ranges to align with points
        concat_points = torch.cat(all_level_points, dim=0)
        # the number of points per lvl, all imgs are equal
        num_points = [center.size(0) for center in all_level_points]

        with torch.no_grad():
            (
                cls_targets_list,
                bbox_targets_list,
                tanalpha_targets_list,
                centerness_targets_list,
                label_weights_list,
            ) = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_tanalphas_list,
                gt_labels_list,
                flatten_cls_scores,
                flatten_bbox_preds,
                flatten_alpha_preds,
                strides=flatten_strides,
                points=concat_points,
                device=device,
                num_points_per_lvl=num_points,
            )
        cls_targets = torch.cat(cls_targets_list)
        bbox_targets = torch.cat(bbox_targets_list)
        tanalpha_targets = torch.cat(tanalpha_targets_list)
        centerness_targets = torch.cat(centerness_targets_list)

        label_weights = torch.cat(label_weights_list)

        pos_inds = (
            (
                (cls_targets >= 0)
                & (cls_targets != self.background_label)
                & (label_weights > 0)
            )
            .nonzero()
            .reshape(-1)
        )

        num_pos = len(pos_inds)
        cls_avg_factor = max(num_pos, 1.0)
        if num_pos == 0:
            pos_inds = flatten_cls_scores.new_zeros((1,)).long()
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_alpha_targets = tanalpha_targets[pos_inds]
        pos_alpha_targets.arctan_()
        pos_alpha_targets.divide_(PI / 2)

        if num_pos == 0:
            centerness_weight = torch.tensor([0.0], device=device)  # noqa
            bbox_weight = torch.tensor([0.0], device=device)
            bbox_avg_factor = None
            tanalpha_weight = torch.tensor([0.0], device=device)
            tanalpha_avg_factor = None
        else:
            centerness_weight = None
            bbox_weight = FCOSTarget._centerness_target(
                pos_bbox_targets
            )  # noqa
            bbox_avg_factor = bbox_weight.sum()
            tanalpha_weight = bbox_weight
            tanalpha_avg_factor = bbox_weight.sum()

        flatten_cls_scores = flatten_cls_scores.view(-1, self.cls_out_channels)
        flatten_bbox_preds = flatten_bbox_preds.view(-1, 4)
        flatten_centerness = flatten_centerness.view(-1)
        flatten_alpha_preds = flatten_alpha_preds.view(-1, 1)

        cls_target = {
            "pred": flatten_cls_scores,
            "target": cls_targets.long(),
            "weight": label_weights,
            "avg_factor": cls_avg_factor,
        }

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_points = torch.zeros((2), device=device)
        pos_decoded_targets = distance2bbox(pos_points, pos_bbox_targets)
        pos_decoded_preds = distance2bbox(pos_points, pos_bbox_preds)
        pos_alpha_preds = flatten_alpha_preds[pos_inds]
        pos_alpha_preds.squeeze_()

        giou_target = {
            "pred": pos_decoded_preds,
            "target": pos_decoded_targets,
            "weight": bbox_weight,
            "avg_factor": bbox_avg_factor,
        }
        l1_target = {
            "pred": pos_alpha_preds,
            "target": pos_alpha_targets,
            "weight": tanalpha_weight,
            "avg_factor": tanalpha_avg_factor,
        }

        pos_centerness = flatten_centerness[pos_inds]
        pos_centerness_targets = centerness_targets[pos_inds]
        centerness_target = {
            "pred": pos_centerness,
            "target": pos_centerness_targets,
            "weight": centerness_weight,
        }
        return cls_target, giou_target, l1_target, centerness_target
