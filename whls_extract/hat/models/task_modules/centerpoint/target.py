# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List, Sequence

import numpy as np
import torch
from torch import nn

from hat.core.center_utils import (
    draw_umich_gaussian,
    draw_umich_gaussian_torch,
    gaussian_radius,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import multi_apply

__all__ = ["CenterPointTarget"]


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


@OBJECT_REGISTRY.register
class CenterPointTarget(nn.Module):
    """Generate centerpoint targets for bev task.

    Args:
        class_names: List of class names for bev detection.
        tasks: List of tasks
        gaussian_overlap: Gaussian overlap for genenrate heatmap target.
        min_radius: Min values for radius.
        out_size_factor: Output size for factor.
        norm_bbox: Whether using norm bbox.
        max_num: Max number for bbox.
        bbox_weight: Weight for bbox meta.
    """

    def __init__(
        self,
        class_names: Sequence[str],
        tasks: Sequence[dict],
        gaussian_overlap: float = 0.1,
        min_radius: int = 2,
        out_size_factor: int = 4,
        norm_bbox: bool = True,
        max_num: int = 500,
        bbox_weight: float = None,
        use_heatmap: bool = True,
    ):
        super(CenterPointTarget, self).__init__()

        self.class_names = class_names
        self.out_size_factor = out_size_factor
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.norm_bbox = norm_bbox
        self.max_num = max_num
        self.tasks = tasks
        self.use_heatmap = use_heatmap

        if bbox_weight is None:
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

    def _get_task_targets(self, gt_bboxes_3d, preds, task):
        heatmap_pred = preds["heatmap"]
        reg_pred = preds["reg"]
        height_pred = preds["height"]
        dim_pred = preds["dim"]
        rot_pred = preds["rot"]
        if "vel" in preds:
            vel_pred = preds["vel"]
        else:
            vel_pred = None
        heatmaps, indices, bbox_targets_list = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            heatmap_pred,
            reg_pred,
            dim_pred,
            rot_pred,
            vel_pred,
            task=task,
            max_obj=self.max_num,
        )

        heatmaps = torch.stack(heatmaps)
        heatmaps_target = {
            "logits": clip_sigmoid(heatmap_pred),
            "labels": heatmaps,
        }

        if vel_pred is None:
            bbox_pred = torch.cat(
                [reg_pred, height_pred, dim_pred, rot_pred], dim=1
            )
        else:
            bbox_pred = torch.cat(
                [reg_pred, height_pred, dim_pred, rot_pred, vel_pred], dim=1
            )

        bbox_pred = torch.permute(bbox_pred, (0, 2, 3, 1)).contiguous()
        if self.use_heatmap is True:
            pos_bbox_targets = torch.stack(bbox_targets_list)
            if self.bbox_weight:
                bbox_weight = torch.tensor(self.bbox_weight).to(
                    device=heatmap_pred.device
                )
            else:
                bbox_weight = torch.ones(bbox_pred.shape[3]).to(
                    device=heatmap_pred.device
                )
            bbox_weight_heatmap = torch.zeros_like(pos_bbox_targets[..., 0])
            num_cls = heatmaps.shape[1]
            for i in range(num_cls):
                heatmaps_weight = heatmaps[:, i]
                torch.max(
                    heatmaps_weight,
                    bbox_weight_heatmap,
                    out=bbox_weight_heatmap,
                )

            avg_factor = max(bbox_weight_heatmap.sum(), 1)

            bbox_weight_heatmap = (
                bbox_weight_heatmap.unsqueeze(-1) * bbox_weight
            )
            bbox_weight = bbox_weight_heatmap
            pos_bbox_pred = bbox_pred
            refact_indices = []
        else:
            refact_indices = []
            pos_bbox_targets = []
            for i in range(len(indices)):
                if len(indices[i]) != 0:
                    pos_bbox_targets.append(bbox_targets_list[i])
                for index in indices[i]:
                    ref_index = torch.tensor([i, *index])
                    refact_indices.append(ref_index)

            if len(refact_indices) == 0:
                bbox_weight = torch.zeros(bbox_pred.shape[3]).to(
                    device=bbox_pred.device
                )
                refact_indices = torch.zeros((1, 3)).long()
                pos_bbox_targets = torch.zeros((1, bbox_pred.shape[3])).to(
                    device=bbox_pred.device
                )
            else:
                refact_indices = torch.stack(refact_indices)
                pos_bbox_targets = torch.cat(pos_bbox_targets)
                if self.bbox_weight:
                    bbox_weight = torch.tensor(self.bbox_weight).to(
                        device=heatmap_pred.device
                    )
                else:
                    bbox_weight = torch.ones(bbox_pred.shape[3]).to(
                        device=heatmap_pred.device
                    )
            pos_bbox_pred = bbox_pred[
                refact_indices[:, 0],
                refact_indices[:, 1],
                refact_indices[:, 2],
            ]
            avg_factor = pos_bbox_pred.shape[0]

        bbox_targets = {
            "pred": pos_bbox_pred,
            "target": pos_bbox_targets,
            "weight": bbox_weight,
            "avg_factor": avg_factor,
        }
        return {
            "task_name": task["name"],
            "cls_target": heatmaps_target,
            "reg_target": bbox_targets,
            "pos_indices": refact_indices,
        }

    def _gen_offset_map(self, bbox_target, center, radius):
        center = center  # .cpu().numpy()
        center_int = center.int()  # (int(center[0]), int(center[1]))
        center_offset = (
            center - center_int
        )  # np.array(center) - np.array(center_int)
        x, y = center_int

        y_grid = torch.arange(y - radius, y + radius + 1).to(
            device=center.device
        )
        x_grid = torch.arange(x - radius, x + radius + 1).to(
            device=center.device
        )
        y_reg = center[1] - y_grid
        x_reg = center[0] - x_grid

        y_reg[radius] = center_offset[1]
        x_reg[radius] = center_offset[0]
        xv, yv = torch.meshgrid(x_reg, y_reg)
        ct_off_reg_map = torch.stack([xv, yv], dim=-1).to(device=center.device)

        self._gen_heatmap(ct_off_reg_map, bbox_target, center, radius, 0, 2)

    def _gen_heatmap(self, src_map, heatmap, center, radius, start, end):
        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        rh, rw = src_map.shape[:2]
        ry, rx = (rh - 1) // 2, (rw - 1) // 2

        heatmap[
            y - top : y + bottom, x - left : x + right, start:end
        ] = src_map[ry - top : ry + bottom, rx - left : rx + right]

    def _gen_reg_map(self, value, bbox_target, center, radius, start, end):
        h = w = radius * 2 + 1
        src_map = torch.tile(value, (h, w, 1))
        self._gen_heatmap(src_map, bbox_target, center, radius, start, end)

    def get_targets_single(
        self,
        gt_bboxes_3d,
        heatmap_pred,
        reg_pred,
        dim_pred,
        rot_pred,
        vel_pred,
        task,
        max_obj,
    ):
        feat_size = heatmap_pred.shape[1:]
        # reorganize the gt_dict by tasks
        gt_bboxes_task = []
        if len(gt_bboxes_3d) != 0:
            for cls in task["class_names"]:
                cat_id = self.class_names.index(cls)
                task_indices = gt_bboxes_3d[:, 9] == cat_id
                gt_bbox = gt_bboxes_3d[task_indices]
                gt_bboxes_task.append(gt_bbox)

        indices = []
        if self.use_heatmap is True:
            bbox_dim = 10 if vel_pred is not None else 9
            bbox_targets = torch.zeros(
                (feat_size[0], feat_size[1], bbox_dim)
            ).to(device=heatmap_pred.device)
        else:
            bbox_targets = []
        heatmaps = []

        for _, gt_bbox in enumerate(gt_bboxes_task):
            heatmap = np.zeros((feat_size[0], feat_size[1]))
            for bbox in gt_bbox:
                width, length = bbox[3:5] / self.out_size_factor
                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width), min_overlap=self.gaussian_overlap
                    )
                    radius = max(self.min_radius, int(radius))
                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y = bbox[:2] / self.out_size_factor
                    z = bbox[2]
                    hi = torch.tensor([z]).to(device=heatmap_pred.device)
                    center = torch.tensor(
                        [x, y], dtype=torch.float32, device=heatmap_pred.device
                    )
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (
                        0 <= center_int[0] < feat_size[1]
                        and 0 <= center_int[1] < feat_size[0]
                    ):
                        continue
                    heatmap = draw_umich_gaussian(
                        heatmap, center_int.cpu().numpy(), radius
                    )

                    x, y = center_int[0], center_int[1]

                    assert y * feat_size[1] + x < feat_size[0] * feat_size[1]
                    indices.append([y, x])
                    reg = center - center_int
                    box_dim = torch.tensor(bbox[3:6]).to(
                        device=heatmap_pred.device
                    )
                    if self.norm_bbox:
                        box_dim = torch.log(box_dim)

                    rot = torch.tensor(bbox[6])
                    rot_sine = (
                        torch.sin(rot).view(-1).to(device=heatmap_pred.device)
                    )
                    rot_cos = (
                        torch.cos(rot).view(-1).to(device=heatmap_pred.device)
                    )

                    if self.use_heatmap is True:
                        self._gen_offset_map(bbox_targets, center, radius)
                        self._gen_reg_map(
                            hi, bbox_targets, center, radius, 2, 3
                        )
                        self._gen_reg_map(
                            box_dim, bbox_targets, center, radius, 3, 6
                        )
                        self._gen_reg_map(
                            rot_sine, bbox_targets, center, radius, 6, 7
                        )
                        self._gen_reg_map(
                            rot_cos, bbox_targets, center, radius, 7, 8
                        )
                        if vel_pred is not None:
                            vel = torch.tensor(bbox[7:9]).to(
                                device=heatmap_pred.device
                            )
                            self._gen_reg_map(
                                vel, bbox_targets, center, radius, 8, 10
                            )
                    else:
                        if vel_pred is None:
                            bbox_target = torch.stack(
                                [reg, hi, box_dim, rot_sine, rot_cos]
                            )
                        else:
                            vel = torch.tensor(bbox[7:9]).to(
                                device=heatmap_pred.device
                            )
                            if torch.isnan(vel).any():
                                vel = torch.zeros(2).to(
                                    device=heatmap_pred.device
                                )
                            bbox_target = torch.cat(
                                [reg, hi, box_dim, rot_sine, rot_cos, vel]
                            )
                            bbox_targets.append(bbox_target)
                            if len(bbox_targets) >= self.max_num:
                                break

            heatmaps.append(heatmap)
            if self.use_heatmap is False:
                if len(bbox_targets) >= self.max_num:
                    break
        if len(heatmaps) == 0:
            heatmaps = torch.zeros(
                (len(task["class_names"]), feat_size[0], feat_size[1])
            ).to(device=heatmap_pred.device)
        else:
            heatmaps = np.stack(heatmaps)
            heatmaps = torch.tensor(heatmaps).to(device=heatmap_pred.device)

        if self.use_heatmap is False:
            if len(indices) == 0:
                bbox_targets = torch.zeros((1,))
            else:
                bbox_targets = torch.stack(bbox_targets)
        return heatmaps, indices, bbox_targets

    def forward(self, label, preds, *args):
        task_targets = []
        for task_preds, task in zip(preds, self.tasks):
            task_targets.append(
                self._get_task_targets(label, task_preds, task)
            )
        return task_targets


@OBJECT_REGISTRY.register
class CenterPointLidarTarget(nn.Module):
    """Generate CenterPoint targets.

    Args:
        grid_size: List of grid sizes (W, H, D).
        voxel_size: List of voxel sizes (dx, dy, dz).
        point_cloud_range: List specifying the point cloud range
            (x_min, y_min, z_min, x_max, y_max, z_max).
        tasks: List of task dictionaries.
        dense_reg: Density of regression targets.
        max_objs: Maximum number of objects.
        gaussian_overlap: Gaussian overlap for generating
            heatmap targets.
        min_radius: Minimum radius for generating heatmap targets.
        out_size_factor: Output size factor.
        norm_bbox: Whether to use normalized bounding boxes.
        with_velocity: Whether to include velocity information in targets.
    """

    def __init__(
        self,
        grid_size: List[int],
        voxel_size: List[float],
        point_cloud_range: List[float],
        tasks: List[dict],
        dense_reg: int = 1,
        max_objs: int = 500,
        gaussian_overlap: float = 0.1,
        min_radius: int = 2,
        out_size_factor: int = 4,
        norm_bbox: bool = True,
        with_velocity: bool = False,
    ):
        super(CenterPointLidarTarget, self).__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.num_classes = num_classes

        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.dense_reg = dense_reg
        self.max_objs = max_objs
        self.out_size_factor = out_size_factor
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.norm_bbox = norm_bbox
        self.with_velocity = with_velocity

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d: Ground truth 3D bounding boxes.
            gt_labels_3d: Labels of the boxes.

        Returns:
            Tuple of target lists containing:
                - Heatmap scores.
                - Ground truth boxes.
                - Indexes indicating the position of the valid boxes.
                - Masks indicating which boxes are valid.
        """
        device = gt_labels_3d.device

        max_objs = self.max_objs * self.dense_reg
        grid_size = torch.tensor(self.grid_size)
        pc_range = torch.tensor(self.point_cloud_range)
        voxel_size = torch.tensor(self.voxel_size)

        feature_map_size = grid_size[:2] // self.out_size_factor

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append(
                [
                    torch.where(gt_labels_3d == class_name.index(i) + 1 + flag)
                    for i in class_name
                ]
            )
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for _, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, labels added 1 earlier.
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_umich_gaussian_torch
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, _ in enumerate(self.num_classes):
            heatmap = gt_bboxes_3d.new_zeros(
                (
                    len(self.class_names[idx]),
                    feature_map_size[1],
                    feature_map_size[0],
                )
            )

            if self.with_velocity:
                anno_box = gt_bboxes_3d.new_zeros(
                    (max_objs, 10), dtype=torch.float32
                )
            else:
                anno_box = gt_bboxes_3d.new_zeros(
                    (max_objs, 8), dtype=torch.float32
                )

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.out_size_factor
                length = length / voxel_size[1] / self.out_size_factor

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length.item(), width.item()),
                        min_overlap=self.gaussian_overlap,
                    )
                    radius = max(self.min_radius, int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = (
                        task_boxes[idx][k][0],
                        task_boxes[idx][k][1],
                        task_boxes[idx][k][2],
                    )

                    coor_x = (
                        (x - pc_range[0])
                        / voxel_size[0]
                        / self.out_size_factor
                    )
                    coor_y = (
                        (y - pc_range[1])
                        / voxel_size[1]
                        / self.out_size_factor
                    )

                    center = torch.tensor(
                        [coor_x, coor_y], dtype=torch.float32, device=device
                    )
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (
                        0 <= center_int[0] < feature_map_size[0]
                        and 0 <= center_int[1] < feature_map_size[1]
                    ):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (
                        y * feature_map_size[0] + x
                        < feature_map_size[0] * feature_map_size[1]
                    )

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # index based on dataset
                    rot = task_boxes[idx][k][8]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    if self.with_velocity:
                        vx, vy = task_boxes[idx][k][6:8]
                        anno_box[new_idx] = torch.cat(
                            [
                                center - torch.tensor([x, y], device=device),
                                z.unsqueeze(0),
                                box_dim,
                                torch.sin(rot).unsqueeze(0),
                                torch.cos(rot).unsqueeze(0),
                                vx.unsqueeze(0),
                                vy.unsqueeze(0),
                            ]
                        )
                    else:
                        anno_box[new_idx] = torch.cat(
                            [
                                center - torch.tensor([x, y], device=device),
                                z.unsqueeze(0),
                                box_dim,
                                torch.sin(rot).unsqueeze(0),
                                torch.cos(rot).unsqueeze(0),
                            ]
                        )

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    def forward(self, gt_bboxes_3d, gt_labels_3d):
        """Generate CenterPoint training targets for a batch of samples.

        Args:
            gt_bboxes_3d: Ground truth 3D bounding boxes.
            gt_labels_3d: Labels of the boxes.

        Returns:
            Tuple of target lists containing:
                - Heatmap scores.
                - Ground truth boxes.
                - Indexes indicating the position of the valid boxes.
                - Masks indicating which boxes are valid.
        """
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d
        )
        # Transpose heatmaps
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]

        return heatmaps, anno_boxes, inds, masks
