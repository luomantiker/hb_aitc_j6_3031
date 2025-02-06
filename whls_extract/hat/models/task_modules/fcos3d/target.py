# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Tuple

import torch
from torch import nn

from hat.models.task_modules.fcos.target import get_points
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import multi_apply

INF = 1e8


@OBJECT_REGISTRY.register
class FCOS3DTarget(nn.Module):
    """Generate cls/reg targets for FCOS3D in training stage.

    Args:
        num_classes: Number of categories excluding the background category.
        background_label: Label ID of background.
        bbox_code_size: Dimensions of predicted bounding boxes.
        regress_ranges: Regress range of multiple level points.
        strides: Downsample factor of each feature map.
        pred_attrs: Whether to predict attributes.
        num_attrs: The number of attributes to be predicted.
        center_sampling: If true, use center sampling. Default: True.
        center_sample_radius: Radius of center sampling. Default: 1.5.
        centerness_alpha: Parameter used to adjust the intensity
            attenuation from the center to the periphery. Default: 2.5.
        norm_on_bbox: If true, normalize the regression targets
            with FPN strides. Default: True.
    """  # noqa

    def __init__(
        self,
        num_classes: int,
        background_label: int,
        bbox_code_size: int,
        regress_ranges: Tuple[Tuple[int, int]],
        strides: Tuple[int],
        pred_attrs: bool,
        num_attrs: int,
        center_sampling: bool,
        center_sample_radius: float = 1.5,
        centerness_alpha: float = 2.5,
        norm_on_bbox: bool = True,
    ):
        super(FCOS3DTarget, self).__init__()
        self.background_label = (
            num_classes if background_label is None else background_label
        )
        self.bbox_code_size = bbox_code_size
        self.regress_ranges = regress_ranges
        self.strides = strides
        self.attr_background_label = -1
        if pred_attrs:
            self.attr_background_label = num_attrs
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.centerness_alpha = centerness_alpha
        self.norm_on_bbox = norm_on_bbox

    def _get_target_single(
        self,
        gt_bboxes,
        gt_labels,
        gt_bboxes_3d,
        gt_labels_3d,
        centers2d,
        depths,
        attr_labels,
        points,
        regress_ranges,
        num_points_per_lvl,
    ):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if not isinstance(gt_bboxes_3d, torch.Tensor):
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(gt_bboxes.device)
        if num_gts == 0:
            return (
                gt_labels.new_full((num_points,), self.background_label),
                gt_bboxes.new_zeros((num_points, 4)),
                gt_labels_3d.new_full((num_points,), self.background_label),
                gt_bboxes_3d.new_zeros((num_points, self.bbox_code_size)),
                gt_bboxes_3d.new_zeros((num_points,)),
                attr_labels.new_full(
                    (num_points,), self.attr_background_label
                ),
            )

        # change orientation to local yaw
        gt_bboxes_3d[..., 6] = (
            -torch.atan2(gt_bboxes_3d[..., 0], gt_bboxes_3d[..., 2])
            + gt_bboxes_3d[..., 6]
        )

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1]
        )
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2
        )
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        centers2d = centers2d[None].expand(num_points, num_gts, 2)
        gt_bboxes_3d = gt_bboxes_3d[None].expand(
            num_points, num_gts, self.bbox_code_size
        )
        depths = depths[None, :, None].expand(num_points, num_gts, 1)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        delta_xs = (xs - centers2d[..., 0])[..., None]
        delta_ys = (ys - centers2d[..., 1])[..., None]
        bbox_targets_3d = torch.cat(
            (delta_xs, delta_ys, depths, gt_bboxes_3d[..., 3:]), dim=-1
        )

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        assert self.center_sampling is True, (
            "Setting center_sampling to False "
            "has not been implemented for FCOS3D."
        )

        # condition1: inside a `center bbox`
        radius = self.center_sample_radius
        center_xs = centers2d[..., 0]
        center_ys = centers2d[..., 1]
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_zeros(center_xs.shape)

        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        center_gts[..., 0] = center_xs - stride
        center_gts[..., 1] = center_ys - stride
        center_gts[..., 2] = center_xs + stride
        center_gts[..., 3] = center_ys + stride

        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1
        )
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]
        ) & (max_regress_distance <= regress_ranges[..., 1])

        # center-based criterion to deal with ambiguity
        dists = torch.sqrt(torch.sum(bbox_targets_3d[..., :2] ** 2, dim=-1))
        dists[inside_gt_bbox_mask == 0] = INF
        dists[inside_regress_range == 0] = INF
        min_dist, min_dist_inds = dists.min(dim=1)

        labels = gt_labels[min_dist_inds]
        labels_3d = gt_labels_3d[min_dist_inds]
        attr_labels = attr_labels[min_dist_inds]
        labels[min_dist == INF] = self.background_label  # set as BG
        labels_3d[min_dist == INF] = self.background_label  # set as BG
        attr_labels[min_dist == INF] = self.attr_background_label

        bbox_targets = bbox_targets[range(num_points), min_dist_inds]
        bbox_targets_3d = bbox_targets_3d[range(num_points), min_dist_inds]
        relative_dists = torch.sqrt(
            torch.sum(bbox_targets_3d[..., :2] ** 2, dim=-1)
        ) / (1.414 * stride[:, 0])
        # [N, 1] / [N, 1]
        centerness_targets = torch.exp(-self.centerness_alpha * relative_dists)

        return (
            labels,
            bbox_targets,
            labels_3d,
            bbox_targets_3d,
            centerness_targets,
            attr_labels,
        )

    def forward(
        self,
        cls_scores,
        bbox_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_bboxes_3d_list,
        gt_labels_3d_list,
        centers2d_list,
        depths_list,
        attr_labels_list,
    ):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points = get_points(
            featmap_sizes,
            self.strides,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
        )
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i]
            .new_tensor(self.regress_ranges[i])[None]
            .expand_as(points[i])
            for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        if attr_labels_list is None:
            attr_labels_list = [
                gt_labels.new_full(gt_labels.shape, self.attr_background_label)
                for gt_labels in gt_labels_list
            ]

        # get labels and bbox_targets of each image
        (
            _,
            _,
            labels_3d_list,
            bbox_targets_3d_list,
            centerness_targets_list,
            attr_targets_list,
        ) = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            gt_bboxes_3d_list,
            gt_labels_3d_list,
            centers2d_list,
            depths_list,
            attr_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points,
        )

        # split to per img, per level
        labels_3d_list = [
            labels_3d.split(num_points, 0) for labels_3d in labels_3d_list
        ]
        bbox_targets_3d_list = [
            bbox_targets_3d.split(num_points, 0)
            for bbox_targets_3d in bbox_targets_3d_list
        ]
        centerness_targets_list = [
            centerness_targets.split(num_points, 0)
            for centerness_targets in centerness_targets_list
        ]
        attr_targets_list = [
            attr_targets.split(num_points, 0)
            for attr_targets in attr_targets_list
        ]

        # concat per level image
        concat_lvl_labels_3d = []
        concat_lvl_bbox_targets_3d = []
        concat_lvl_centerness_targets = []
        concat_lvl_attr_targets = []

        for i in range(num_levels):
            concat_lvl_labels_3d.append(
                torch.cat([labels[i] for labels in labels_3d_list])
            )
            concat_lvl_centerness_targets.append(
                torch.cat(
                    [
                        centerness_targets[i]
                        for centerness_targets in centerness_targets_list
                    ]
                )
            )
            bbox_targets_3d = torch.cat(
                [
                    bbox_targets_3d[i]
                    for bbox_targets_3d in bbox_targets_3d_list
                ]
            )
            concat_lvl_attr_targets.append(
                torch.cat(
                    [attr_targets[i] for attr_targets in attr_targets_list]
                )
            )
            if self.norm_on_bbox:
                bbox_targets_3d[:, :2] = (
                    bbox_targets_3d[:, :2] / self.strides[i]
                )
            concat_lvl_bbox_targets_3d.append(bbox_targets_3d)
        return (
            concat_lvl_labels_3d,
            concat_lvl_bbox_targets_3d,
            concat_lvl_centerness_targets,
            concat_lvl_attr_targets,
        )
