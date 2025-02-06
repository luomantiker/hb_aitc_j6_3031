# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import geopandas as gpd
    from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
    from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
except ImportError:
    Quaternion = None
    quaternion_yaw = None
    NuScenesMap = None
    NuScenesMapExplorer = None
    gpd = None

from shapely import affinity, ops
from shapely.geometry import LineString, MultiLineString, MultiPolygon, box

from hat.data.datasets.map_utils.geo_opensfm import TopocentricConverter
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import convert_tensor_v2 as to_tensor
from hat.utils.package_helper import require_packages
from .nuscenes_dataset import NuscenesBevDataset, NuscenesFromImage

logger = logging.getLogger(__name__)

__all__ = [
    "LiDARInstanceLines",
    "NuscenesMapDataset",
]


def add_rotation_noise(extrinsics, std=0.01, mean=0.0):
    # n = extrinsics.shape[0]
    noise_angle = torch.normal(mean, std=std, size=(3,))
    # extrinsics[:, 0:3, 0:3] *= (1 + noise)
    sin_noise = torch.sin(noise_angle)
    cos_noise = torch.cos(noise_angle)
    rotation_matrix = torch.eye(4).view(4, 4)
    #  rotation_matrix[]
    rotation_matrix_x = rotation_matrix.clone()
    rotation_matrix_x[1, 1] = cos_noise[0]
    rotation_matrix_x[1, 2] = sin_noise[0]
    rotation_matrix_x[2, 1] = -sin_noise[0]
    rotation_matrix_x[2, 2] = cos_noise[0]

    rotation_matrix_y = rotation_matrix.clone()
    rotation_matrix_y[0, 0] = cos_noise[1]
    rotation_matrix_y[0, 2] = -sin_noise[1]
    rotation_matrix_y[2, 0] = sin_noise[1]
    rotation_matrix_y[2, 2] = cos_noise[1]

    rotation_matrix_z = rotation_matrix.clone()
    rotation_matrix_z[0, 0] = cos_noise[2]
    rotation_matrix_z[0, 1] = sin_noise[2]
    rotation_matrix_z[1, 0] = -sin_noise[2]
    rotation_matrix_z[1, 1] = cos_noise[2]

    rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z

    rotation = torch.from_numpy(extrinsics.astype(np.float32))
    rotation[:3, -1] = 0.0
    # import pdb;pdb.set_trace()
    rotation = rotation_matrix @ rotation
    extrinsics[:3, :3] = rotation[:3, :3].numpy()
    return extrinsics


def add_translation_noise(extrinsics, std=0.01, mean=0.0):
    # n = extrinsics.shape[0]
    noise = torch.normal(mean, std=std, size=(3,))
    extrinsics[0:3, -1] += noise.numpy()
    return extrinsics


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


class LiDARInstanceLines(object):
    """LiDARInstanceLines.

    Args:
        instance_line_list: List of instance lines.
        sample_dist: Distance between samples. Default is 1.
        num_samples: Number of samples. Default is 250.
        padding: Whether to pad the samples. Default is False.
        fixed_num: Fixed number of samples. Default is -1.
        padding_value: Value to use for padding. Default is -10000.
        patch_size: Size of the patch. Default is None.
    """

    def __init__(
        self,
        instance_line_list: list,
        sample_dist: int = 1,
        num_samples: int = 250,
        padding: bool = False,
        fixed_num: int = -1,
        padding_value: int = -10000,
        patch_size: int = None,
    ):
        assert isinstance(instance_line_list, list)
        assert patch_size is not None
        if len(instance_line_list) != 0:
            assert isinstance(instance_line_list[0], LineString)
        self.patch_size = patch_size
        self.max_x = self.patch_size[1] / 2
        self.max_y = self.patch_size[0] / 2
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_num
        self.padding_value = padding_value

        self.instance_list = instance_line_list

    @property
    def start_end_points(self):
        """Return Tensor([N,4]), in xstart, ystart, xend, yend form."""
        assert len(self.instance_list) != 0
        instance_se_points_list = []
        for instance in self.instance_list:
            se_points = []
            se_points.extend(instance.coords[0])
            se_points.extend(instance.coords[-1])
            instance_se_points_list.append(se_points)
        instance_se_points_array = np.array(instance_se_points_list)
        instance_se_points_tensor = to_tensor(instance_se_points_array)
        instance_se_points_tensor = instance_se_points_tensor.to(
            dtype=torch.float32
        )
        instance_se_points_tensor[:, 0] = torch.clamp(
            instance_se_points_tensor[:, 0], min=-self.max_x, max=self.max_x
        )
        instance_se_points_tensor[:, 1] = torch.clamp(
            instance_se_points_tensor[:, 1], min=-self.max_y, max=self.max_y
        )
        instance_se_points_tensor[:, 2] = torch.clamp(
            instance_se_points_tensor[:, 2], min=-self.max_x, max=self.max_x
        )
        instance_se_points_tensor[:, 3] = torch.clamp(
            instance_se_points_tensor[:, 3], min=-self.max_y, max=self.max_y
        )
        return instance_se_points_tensor

    @property
    def bbox(self):
        """Return Tensor([N,4]), in xmin, ymin, xmax, ymax form."""
        assert len(self.instance_list) != 0
        instance_bbox_list = []
        for instance in self.instance_list:
            # bounds is bbox: [xmin, ymin, xmax, ymax]
            instance_bbox_list.append(instance.bounds)
        instance_bbox_array = np.array(instance_bbox_list)
        instance_bbox_tensor = to_tensor(instance_bbox_array)
        instance_bbox_tensor = instance_bbox_tensor.to(dtype=torch.float32)
        instance_bbox_tensor[:, 0] = torch.clamp(
            instance_bbox_tensor[:, 0], min=-self.max_x, max=self.max_x
        )
        instance_bbox_tensor[:, 1] = torch.clamp(
            instance_bbox_tensor[:, 1], min=-self.max_y, max=self.max_y
        )
        instance_bbox_tensor[:, 2] = torch.clamp(
            instance_bbox_tensor[:, 2], min=-self.max_x, max=self.max_x
        )
        instance_bbox_tensor[:, 3] = torch.clamp(
            instance_bbox_tensor[:, 3], min=-self.max_y, max=self.max_y
        )
        return instance_bbox_tensor

    @property
    def fixed_num_sampled_points(self):
        """Return Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form."""
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array(
                [
                    list(instance.interpolate(distance).coords)
                    for distance in distances
                ]
            ).reshape(-1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(
            instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x
        )
        instance_points_tensor[:, :, 1] = torch.clamp(
            instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y
        )
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_ambiguity(self):
        """Return Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form."""
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array(
                [
                    list(instance.interpolate(distance).coords)
                    for distance in distances
                ]
            ).reshape(-1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(
            instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x
        )
        instance_points_tensor[:, :, 1] = torch.clamp(
            instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y
        )
        instance_points_tensor = instance_points_tensor.unsqueeze(1)
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_torch(self):
        """Return Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form."""
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            poly_pts = to_tensor(np.array(list(instance.coords)))
            poly_pts = poly_pts.unsqueeze(0).permute(0, 2, 1)
            sampled_pts = torch.nn.functional.interpolate(
                poly_pts,
                size=(self.fixed_num),
                mode="linear",
                align_corners=True,
            )
            sampled_pts = sampled_pts.permute(0, 2, 1).squeeze(0)
            instance_points_list.append(sampled_pts)
        # instance_points_array = np.array(instance_points_list)
        # instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = torch.stack(instance_points_list, dim=0)
        instance_points_tensor = instance_points_tensor.to(dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(
            instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x
        )
        instance_points_tensor[:, :, 1] = torch.clamp(
            instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y
        )
        return instance_points_tensor

    @property
    def shift_fixed_num_sampled_points(self):
        """Return  [instances_num, num_shifts, fixed_num, 2]."""
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                # import pdb;pdb.set_trace()
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            shift_pts[:, :, 0] = torch.clamp(
                shift_pts[:, :, 0], min=-self.max_x, max=self.max_x
            )
            shift_pts[:, :, 1] = torch.clamp(
                shift_pts[:, :, 1], min=-self.max_y, max=self.max_y
            )

            if not is_poly:
                padding = torch.full(
                    [fixed_num - shift_pts.shape[0], fixed_num, 2],
                    self.padding_value,
                )
                shift_pts = torch.cat([shift_pts, padding], dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v1(self):
        """Return  [instances_num, num_shifts, fixed_num, 2]."""
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1, :]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros(
                    (shift_num, pts_num, num_coords)
                )
                tmp_shift_pts[:, :-1, :] = shift_pts
                tmp_shift_pts[:, -1, :] = shift_pts[:, 0, :]
                shift_pts = tmp_shift_pts

            shift_pts[:, :, 0] = torch.clamp(
                shift_pts[:, :, 0], min=-self.max_x, max=self.max_x
            )
            shift_pts[:, :, 1] = torch.clamp(
                shift_pts[:, :, 1], min=-self.max_y, max=self.max_y
            )

            if not is_poly:
                padding = torch.full(
                    [shift_num - shift_pts.shape[0], pts_num, 2],
                    self.padding_value,
                )
                shift_pts = torch.cat([shift_pts, padding], dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v2(self):
        """Return  [instances_num, num_shifts, fixed_num, 2]."""
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate(
                        (shift_pts, pts_to_concat), axis=0
                    )
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [
                            list(shift_instance.interpolate(distance).coords)
                            for distance in distances
                        ]
                    ).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                # import pdb;pdb.set_trace()
            else:
                sampled_points = np.array(
                    [
                        list(instance.interpolate(distance).coords)
                        for distance in distances
                    ]
                ).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)

            multi_shifts_pts = np.stack(shift_pts_list, axis=0)
            shifts_num, _, _ = multi_shifts_pts.shape

            if shifts_num > final_shift_num:
                index = np.random.choice(
                    multi_shifts_pts.shape[0], final_shift_num, replace=False
                )
                multi_shifts_pts = multi_shifts_pts[index]

            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                dtype=torch.float32
            )

            multi_shifts_pts_tensor[:, :, 0] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 0],
                min=-self.max_x,
                max=self.max_x,
            )
            multi_shifts_pts_tensor[:, :, 1] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 1],
                min=-self.max_y,
                max=self.max_y,
            )
            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < final_shift_num:
                padding = torch.full(
                    [
                        final_shift_num - multi_shifts_pts_tensor.shape[0],
                        self.fixed_num,
                        2,
                    ],
                    self.padding_value,
                )
                multi_shifts_pts_tensor = torch.cat(
                    [multi_shifts_pts_tensor, padding], dim=0
                )
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v3(self):
        """Return  [instances_num, num_shifts, fixed_num, 2]."""
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate(
                        (shift_pts, pts_to_concat), axis=0
                    )
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [
                            list(shift_instance.interpolate(distance).coords)
                            for distance in distances
                        ]
                    ).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                flip_pts_to_shift = np.flip(pts_to_shift, axis=0)
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(
                        flip_pts_to_shift, shift_right_i, axis=0
                    )
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate(
                        (shift_pts, pts_to_concat), axis=0
                    )
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [
                            list(shift_instance.interpolate(distance).coords)
                            for distance in distances
                        ]
                    ).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
            else:
                sampled_points = np.array(
                    [
                        list(instance.interpolate(distance).coords)
                        for distance in distances
                    ]
                ).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)

            multi_shifts_pts = np.stack(shift_pts_list, axis=0)
            shifts_num, _, _ = multi_shifts_pts.shape
            if shifts_num > 2 * final_shift_num:
                index = np.random.choice(
                    shift_num, final_shift_num, replace=False
                )
                flip0_shifts_pts = multi_shifts_pts[index]
                flip1_shifts_pts = multi_shifts_pts[index + shift_num]
                multi_shifts_pts = np.concatenate(
                    (flip0_shifts_pts, flip1_shifts_pts), axis=0
                )

            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                dtype=torch.float32
            )

            multi_shifts_pts_tensor[:, :, 0] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 0],
                min=-self.max_x,
                max=self.max_x,
            )
            multi_shifts_pts_tensor[:, :, 1] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 1],
                min=-self.max_y,
                max=self.max_y,
            )
            if multi_shifts_pts_tensor.shape[0] < 2 * final_shift_num:
                padding = torch.full(
                    [
                        final_shift_num * 2 - multi_shifts_pts_tensor.shape[0],
                        self.fixed_num,
                        2,
                    ],
                    self.padding_value,
                )
                multi_shifts_pts_tensor = torch.cat(
                    [multi_shifts_pts_tensor, padding], dim=0
                )
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v4(self):
        """Return  [instances_num, num_shifts, fixed_num, 2]."""
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        for fixed_num_pts in fixed_num_sampled_points:
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            shift_pts_list = []
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i, 0))
                flip_pts_to_shift = pts_to_shift.flip(0)
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(
                        flip_pts_to_shift.roll(shift_right_i, 0)
                    )
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros(
                    (shift_num * 2, pts_num, num_coords)
                )
                tmp_shift_pts[:, :-1, :] = shift_pts
                tmp_shift_pts[:, -1, :] = shift_pts[:, 0, :]
                shift_pts = tmp_shift_pts

            shift_pts[:, :, 0] = torch.clamp(
                shift_pts[:, :, 0], min=-self.max_x, max=self.max_x
            )
            shift_pts[:, :, 1] = torch.clamp(
                shift_pts[:, :, 1], min=-self.max_y, max=self.max_y
            )

            if not is_poly:
                padding = torch.full(
                    [shift_num * 2 - shift_pts.shape[0], pts_num, 2],
                    self.padding_value,
                )
                shift_pts = torch.cat([shift_pts, padding], dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_torch(self):
        """Return  [instances_num, num_shifts, fixed_num, 2]."""
        fixed_num_sampled_points = self.fixed_num_sampled_points_torch
        instances_list = []
        is_poly = False

        for fixed_num_pts in fixed_num_sampled_points:
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            shift_pts[:, :, 0] = torch.clamp(
                shift_pts[:, :, 0], min=-self.max_x, max=self.max_x
            )
            shift_pts[:, :, 1] = torch.clamp(
                shift_pts[:, :, 1], min=-self.max_y, max=self.max_y
            )

            if not is_poly:
                padding = torch.full(
                    [fixed_num - shift_pts.shape[0], fixed_num, 2],
                    self.padding_value,
                )
                shift_pts = torch.cat([shift_pts, padding], dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_sparsedrive(self, padding=1e5):
        """Return  [instances_num, 2*(fixed_num-1), fixed_num, 2]."""
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            line = np.array(
                [
                    list(instance.interpolate(distance).coords)
                    for distance in distances
                ]
            ).squeeze()
            is_closed = np.allclose(line[0], line[-1], atol=1e-3)
            # num_points = len(line)
            num_points, coords_dim = line.shape
            permute_num = num_points - 1
            permute_lines_list = []
            if is_closed:
                pts_to_permute = line[
                    :-1, :
                ]  # throw away replicate start end pts
                for shift_i in range(permute_num):
                    permute_lines_list.append(
                        np.roll(pts_to_permute, shift_i, axis=0)
                    )
                flip_pts_to_permute = np.flip(pts_to_permute, axis=0)
                for shift_i in range(permute_num):
                    permute_lines_list.append(
                        np.roll(flip_pts_to_permute, shift_i, axis=0)
                    )
            else:
                permute_lines_list.append(line)
                permute_lines_list.append(np.flip(line, axis=0))

            permute_lines_array = np.stack(permute_lines_list, axis=0)

            if is_closed:
                tmp = np.zeros((permute_num * 2, num_points, coords_dim))
                tmp[:, :-1, :] = permute_lines_array
                tmp[:, -1, :] = permute_lines_array[
                    :, 0, :
                ]  # add replicate start end pts
                permute_lines_array = tmp

            else:
                # padding
                padding = np.full(
                    [permute_num * 2 - 2, num_points, coords_dim], padding
                )
                permute_lines_array = np.concatenate(
                    (permute_lines_array, padding), axis=0
                )

            instances_list.append(to_tensor(permute_lines_array))
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor


class VectorizedLocalMap(object):
    """VectorizedLocalMap.

    Args:
        dataroot: Root directory of the dataset.
        patch_size: Size of the patch.
        map_classes: Tuple of map classes. Default is ("divider",
            "ped_crossing", "boundary").
        line_classes: Tuple of line classes. Default is ("road_divider",
            "lane_divider").
        ped_crossing_classes: Tuple of pedestrian crossing classes. Default is
            ("ped_crossing").
        contour_classes: Tuple of contour classes. Default is ("road_segment",
            "lane").
        sample_dist: Distance between samples. Default is 1.
        num_samples: Number of samples. Default is 250.
        padding: Whether to pad the samples. Default is False.
        fixed_ptsnum_per_line: Fixed number of points per line. Default is -1.
        padding_value: Value to use for padding. Default is -10000.
        thickness: Thickness of the lines. Default is 3.
        canvas_size: Size of the canvas. Default is (200, 200).
        aux_seg: Auxiliary segmentation information. Default is None.
        sd_map_path: Path to the SD map. Default is None.
        osm_thickness: Thickness of the lines for OSM (OpenStreetMap) features.
            Default is 5.
    """

    CLASS2LABEL = {
        "road_divider": 0,
        "lane_divider": 0,
        "ped_crossing": 1,
        "contours": 2,
        "others": -1,
    }

    def __init__(
        self,
        dataroot: str,
        patch_size: int,
        map_classes: tuple = ("divider", "ped_crossing", "boundary"),
        line_classes: tuple = ("road_divider", "lane_divider"),
        ped_crossing_classes: tuple = ("ped_crossing",),
        contour_classes: tuple = ("road_segment", "lane"),
        sample_dist: int = 1,
        num_samples: int = 250,
        padding: bool = False,
        fixed_ptsnum_per_line: int = -1,
        padding_value: int = -10000,
        thickness: int = 3,
        canvas_size: tuple = (200, 200),
        aux_seg: any = None,
        sd_map_path: Optional[str] = None,
        osm_thickness: int = 5,
    ):
        super().__init__()
        self.data_root = dataroot
        self.MAPS = [
            "boston-seaport",
            "singapore-hollandvillage",
            "singapore-onenorth",
            "singapore-queenstown",
        ]
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=self.data_root, map_name=loc
            )
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value

        # for semantic mask
        self.canvas_size = canvas_size
        self.thickness = thickness
        self.scale_x = self.canvas_size[1] / self.patch_size[1]
        self.scale_y = self.canvas_size[0] / self.patch_size[0]
        self.aux_seg = aux_seg

        # sdmap
        self.osm_thickness = osm_thickness
        self.sd_maps = None
        if sd_map_path is not None:
            self.sd_maps = {}
            options = [
                "trunk",
                "primary",
                "secondary",
                "tertiary",
                "unclassified",
                "residential",
                "trunk_link",
                "primary_link",
                "secondary_link",
                "tertiary_link" "living_street",
                "road",
            ]  # osm(OpenStreetMap) road categories
            map_origin = {
                "boston-seaport": (
                    42.336849169438615,
                    -71.05785369873047,
                    0.0,
                ),
                "singapore-onenorth": (
                    1.2882100868743724,
                    103.78475189208984,
                    0.0,
                ),
                "singapore-hollandvillage": (
                    1.2993652317780957,
                    103.78217697143555,
                    0.0,
                ),
                "singapore-queenstown": (
                    1.2782562240223188,
                    103.76741409301758,
                    0.0,
                ),
            }  # coordinates of osm areas in nuscenes dataset

            for loc in self.MAPS:
                lat, lon, alt = map_origin[loc]
                sd_map = gpd.read_file(
                    os.path.join(sd_map_path, "{}.shp".format(loc))
                )
                converter = TopocentricConverter(lat, lon, alt)
                sd_map = sd_map[sd_map["type"].isin(options)]
                sd_map_topo_list = []
                for _, row in sd_map.iterrows():
                    tmp_sd_data = list(row.geometry.coords)
                    tmp_sd_data_topo = [
                        converter.to_topocentric(lonlat[1], lonlat[0], 0.0)[:2]
                        for lonlat in tmp_sd_data
                    ]
                    sd_map_topo_list.append(tmp_sd_data_topo)
                self.sd_maps[loc] = MultiLineString(sd_map_topo_list)

    def gen_vectorized_samples(
        self,
        location,
        homo_translation,
        homo_rotation,
        img_shape,
        homo2img,
        num_cam=6,
    ):
        """Use coords2global to get gt map layers."""

        map_pose = homo_translation[:2]
        rotation = Quaternion(homo_rotation)

        patch_box = (
            map_pose[0],
            map_pose[1],
            self.patch_size[0],
            self.patch_size[1],
        )
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == "divider":
                line_geom = self.get_map_geom(
                    patch_box, patch_angle, self.line_classes, location
                )
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        vectors.append(
                            (
                                LineString(np.array(instance.coords)),
                                self.CLASS2LABEL.get(line_type, -1),
                            )
                        )
            elif vec_class == "ped_crossing":
                ped_geom = self.get_map_geom(
                    patch_box, patch_angle, self.ped_crossing_classes, location
                )
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    vectors.append(
                        (
                            LineString(np.array(instance.coords)),
                            self.CLASS2LABEL.get("ped_crossing", -1),
                        )
                    )
            elif vec_class == "boundary":
                polygon_geom = self.get_map_geom(
                    patch_box, patch_angle, self.polygon_classes, location
                )
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for instance in poly_bound_list:
                    vectors.append(
                        (
                            LineString(np.array(instance.coords)),
                            self.CLASS2LABEL.get("contours", -1),
                        )
                    )
            else:
                raise ValueError(f"WRONG vec_class: {vec_class}")

        gt_labels = []
        gt_instance = []
        if self.aux_seg is not None:
            if self.aux_seg["bev_seg"]:
                gt_semantic_mask = np.zeros(
                    (1, self.canvas_size[0], self.canvas_size[1]),
                    dtype=np.uint8,
                )
            else:
                gt_semantic_mask = None
            # import ipdb;ipdb.set_trace()
            if self.aux_seg["pv_seg"]:
                num_cam = num_cam
                img_shape = img_shape
                gt_pv_semantic_masks = []
                lidar2feats = []
                for _, stride in self.aux_seg["feat_down_sample"]:
                    feat_down_sample = stride
                    gt_pv_semantic_mask = np.zeros(
                        (
                            num_cam,
                            1,
                            img_shape[0] // feat_down_sample,
                            img_shape[1] // feat_down_sample,
                        ),
                        dtype=np.uint8,
                    )
                    homo2img = homo2img
                    scale_factor = np.eye(4)
                    scale_factor[0, 0] *= 1 / feat_down_sample
                    scale_factor[1, 1] *= 1 / feat_down_sample
                    lidar2feat = [scale_factor @ l2i for l2i in homo2img]
                    gt_pv_semantic_masks.append(gt_pv_semantic_mask)
                    lidar2feats.append(lidar2feat)
            else:
                gt_pv_semantic_masks = None

            for instance, type in vectors:
                if type != -1:
                    gt_instance.append(instance)
                    gt_labels.append(type)
                    if instance.geom_type == "LineString":
                        if self.aux_seg["bev_seg"]:
                            self.line_ego_to_mask(
                                instance,
                                gt_semantic_mask[0],
                                color=1,
                                thickness=self.thickness,
                            )
                        if self.aux_seg["pv_seg"]:
                            for i, (
                                gt_pv_semantic_mask,
                                lidar2feat,
                            ) in enumerate(
                                zip(gt_pv_semantic_masks, lidar2feats)
                            ):
                                for cam_index in range(num_cam):
                                    self.line_ego_to_pvmask(
                                        instance,
                                        gt_pv_semantic_mask[cam_index][0],
                                        lidar2feat[cam_index],
                                        color=1,
                                        thickness=self.aux_seg["pv_thickness"],
                                    )
                                gt_pv_semantic_masks[i] = gt_pv_semantic_mask
                    else:
                        print(instance.geom_type)

        else:
            for instance, type in vectors:
                if type != -1:
                    gt_instance.append(instance)
                    gt_labels.append(type)
            gt_semantic_mask = None
            gt_pv_semantic_masks = None

        gt_instance = LiDARInstanceLines(
            gt_instance,
            self.sample_dist,
            self.num_samples,
            self.padding,
            self.fixed_num,
            self.padding_value,
            patch_size=self.patch_size,
        )

        # sdmap
        if self.sd_maps is not None:
            osm_geom = self.get_osm_geom(patch_box, patch_angle, location)
            osm_vector_list = []

            for line in osm_geom:
                if not line.is_empty:
                    if line.geom_type == "MultiLineString":
                        for single_line in line.geoms:
                            pts, pts_num = self.sample_fixed_pts_from_line(
                                single_line, padding=False, fixed_num=50
                            )
                            if pts_num >= 2:
                                osm_vector_list.append(LineString(pts))
                    elif line.geom_type == "LineString":
                        pts, pts_num = self.sample_fixed_pts_from_line(
                            line, padding=False, fixed_num=50
                        )
                        if pts_num >= 2:
                            osm_vector_list.append(LineString(pts))
                    else:
                        raise NotImplementedError

            local_box = (0.0, 0.0, self.patch_size[0], self.patch_size[1])
            osm_mask = self.line_osms_to_mask(
                osm_vector_list,
                local_box,
                self.canvas_size,
                thickness=self.osm_thickness,
            )
            osm_vectors = LiDARInstanceLines(
                osm_vector_list,
                self.sample_dist,
                self.num_samples,
                self.padding,
                self.fixed_num,
                self.padding_value,
                patch_size=self.patch_size,
            )

        anns_results = {
            "gt_vecs_pts_loc": gt_instance,
            "gt_vecs_label": gt_labels,
            "gt_semantic_mask": gt_semantic_mask,
            "gt_pv_semantic_mask": gt_pv_semantic_masks,
            "osm_vectors": osm_vectors if self.sd_maps is not None else None,
            "osm_mask": osm_mask if self.sd_maps is not None else None,
        }
        return anns_results

    def line_ego_to_pvmask(
        self, line_ego, mask, lidar2feat, color=1, thickness=1, z=-1.6
    ):
        distances = np.linspace(0, line_ego.length, 200)
        coords = np.array(
            [
                list(line_ego.interpolate(distance).coords)
                for distance in distances
            ]
        ).reshape(-1, 2)
        pts_num = coords.shape[0]
        zeros = np.zeros((pts_num, 1))
        zeros[:] = z
        ones = np.ones((pts_num, 1))
        lidar_coords = np.concatenate([coords, zeros, ones], axis=1).transpose(
            1, 0
        )
        pix_coords = perspective(lidar_coords, lidar2feat)
        cv2.polylines(
            mask,
            np.int32([pix_coords]),
            False,
            color=color,
            thickness=thickness,
        )

    def line_ego_to_mask(
        self,
        line_ego: LineString,
        mask: np.ndarray,
        color: int = 1,
        thickness: int = 3,
    ):
        """Rasterize a single line to mask.

        Args:
            line_ego: LineString representing the line.
            mask: Semantic mask to paint on.
            color: Positive label. Default is 1.
            thickness: Thickness of rasterized lines. Default is 3.
        """

        trans_x = self.canvas_size[1] / 2
        trans_y = self.canvas_size[0] / 2
        line_ego = affinity.scale(
            line_ego, self.scale_x, self.scale_y, origin=(0, 0)
        )
        line_ego = affinity.affine_transform(
            line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y]
        )
        # print(np.array(list(line_ego.coords), dtype=np.int32).shape)
        coords = np.array(list(line_ego.coords), dtype=np.int32)[:, :2]
        coords = coords.reshape((-1, 2))
        assert len(coords) >= 2

        cv2.polylines(
            mask, np.int32([coords]), False, color=color, thickness=thickness
        )

    def line_osms_to_mask(
        self,
        lines: List[LineString],
        local_box: Sequence[float],
        canvas_size: Sequence[int],
        color: int = 1,
        thickness: int = 5,
    ):
        """Convert OSM lines to a mask image with specified thickness and color.

        Args:
            lines: List of OSM line geometries to be converted to a mask.
            local_box: Coordinates of the local box.
            canvas_size: Size of the canvas for the output mask.
            color: Color of the lines in the mask. Default is 1.
            thickness: Thickness of the lines in pixels. Default is 5.

        Return:
            A mask array with the OSM lines drawn.
        """

        patch_x, patch_y, patch_h, patch_w = local_box
        patch = NuScenesMapExplorer.get_patch_coord(local_box)
        canvas_h = canvas_size[0]
        canvas_w = canvas_size[1]
        scale_height = canvas_h / patch_h
        scale_width = canvas_w / patch_w
        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        osm_mask = np.zeros(canvas_size, np.uint8)
        for line in lines:
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.affine_transform(
                    new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y]
                )
                new_line = affinity.scale(
                    new_line,
                    xfact=scale_width,
                    yfact=scale_height,
                    origin=(0, 0),
                )

                coords = np.array(list(new_line.coords), dtype=np.int32)
                coords = coords.reshape((-1, 2))
                assert len(coords) >= 2

                cv2.polylines(
                    osm_mask,
                    np.int32([coords]),
                    False,
                    color=color,
                    thickness=thickness,
                )
        osm_mask = osm_mask[np.newaxis, :, :]
        osm_mask = osm_mask.astype(np.int32)
        osm_mask[np.where(osm_mask == 0)] = -1
        osm_mask = osm_mask.astype(np.float32)
        return osm_mask

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(
                    patch_box, patch_angle, layer_name, location
                )
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(
                    patch_box, patch_angle, layer_name, location
                )
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(
                    patch_box, patch_angle, location
                )
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == "MultiLineString":
                    for single_line in line.geoms:
                        line_vectors.append(
                            self.sample_pts_from_line(single_line)
                        )
                elif line.geom_type == "LineString":
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == "MultiLineString":
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == "LineString":
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != "MultiPolygon":
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def ped_poly_geoms_to_instances(self, ped_geom):
        ped = ped_geom[0][1]
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != "MultiPolygon":
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != "MultiPolygon":
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = {}
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(
                a_type_of_lines
            )
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = {}
        for line_type, a_type_of_lines in line_geom:
            one_type_instances = self._one_type_line_geom_to_instances(
                a_type_of_lines
            )
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != "MultiPolygon":
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_contour_line(self, patch_box, patch_angle, layer_name, location):
        if (
            layer_name
            not in self.map_explorer[
                location
            ].map_api.non_geometric_polygon_layers
        ):
            raise ValueError("{} is not a polygonal layer".format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(
            patch_box, patch_angle
        )

        records = getattr(self.map_explorer[location].map_api, layer_name)

        polygon_list = []
        if layer_name == "drivable_area":
            for record in records:
                polygons = [
                    self.map_explorer[location].map_api.extract_polygon(
                        polygon_token
                    )
                    for polygon_token in record["polygon_tokens"]
                ]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(
                            new_polygon,
                            -patch_angle,
                            origin=(patch_x, patch_y),
                            use_radians=False,
                        )
                        new_polygon = affinity.affine_transform(
                            new_polygon,
                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y],
                        )
                        if new_polygon.geom_type == "Polygon":
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.map_explorer[location].map_api.extract_polygon(
                    record["polygon_token"]
                )

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(
                            new_polygon,
                            -patch_angle,
                            origin=(patch_x, patch_y),
                            use_radians=False,
                        )
                        new_polygon = affinity.affine_transform(
                            new_polygon,
                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y],
                        )
                        if new_polygon.geom_type == "Polygon":
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list

    def get_divider_line(self, patch_box, patch_angle, layer_name, location):
        if (
            layer_name
            not in self.map_explorer[
                location
            ].map_api.non_geometric_line_layers
        ):
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == "traffic_light":
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(
            patch_box, patch_angle
        )

        line_list = []
        records = getattr(self.map_explorer[location].map_api, layer_name)
        for record in records:
            line = self.map_explorer[location].map_api.extract_line(
                record["line_token"]
            )
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(
                    new_line,
                    -patch_angle,
                    origin=(patch_x, patch_y),
                    use_radians=False,
                )
                new_line = affinity.affine_transform(
                    new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y]
                )
                line_list.append(new_line)

        return line_list

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(
            patch_box, patch_angle
        )
        polygon_list = []
        records = self.map_explorer[location].map_api.ped_crossing
        # records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].map_api.extract_polygon(
                record["polygon_token"]
            )
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(
                        new_polygon,
                        -patch_angle,
                        origin=(patch_x, patch_y),
                        use_radians=False,
                    )
                    new_polygon = affinity.affine_transform(
                        new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y]
                    )
                    if new_polygon.geom_type == "Polygon":
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        return polygon_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array(
                [
                    list(line.interpolate(distance).coords)
                    for distance in distances
                ]
            ).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length/self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array(
                [
                    list(line.interpolate(distance).coords)
                    for distance in distances
                ]
            ).reshape(-1, 2)

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate(
                    [sampled_points, padding], axis=0
                )
            else:
                sampled_points = sampled_points[: self.num_samples, :]
                num_valid = self.num_samples

        return sampled_points, num_valid

    def sample_fixed_pts_from_line(self, line, padding=False, fixed_num=100):
        if padding:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array(
                [
                    list(line.interpolate(distance).coords)
                    for distance in distances
                ]
            ).reshape(-1, 2)
        else:
            distances = np.linspace(0, line.length, fixed_num)
            sampled_points = np.array(
                [
                    list(line.interpolate(distance).coords)
                    for distance in distances
                ]
            ).reshape(-1, 2)

        num_valid = len(sampled_points)

        if num_valid < fixed_num:
            padding = np.zeros((fixed_num - len(sampled_points), 2))
            sampled_points = np.concatenate([sampled_points, padding], axis=0)
        elif num_valid > fixed_num:
            sampled_points = sampled_points[:fixed_num, :]
            num_valid = fixed_num

        num_valid = len(sampled_points)
        return sampled_points, num_valid

    def get_osm_geom(self, patch_box, patch_angle, location):
        osm_map = self.sd_maps[location]
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        for geom_line in osm_map.geoms:
            if geom_line.is_empty:
                continue
            new_line = geom_line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(
                    new_line,
                    -patch_angle,
                    origin=(patch_x, patch_y),
                    use_radians=False,
                )
                new_line = affinity.affine_transform(
                    new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y]
                )
                line_list.append(new_line)
        return line_list


@OBJECT_REGISTRY.register
class NuscenesMapDataset(NuscenesBevDataset):
    """Dataset object for packed NuScenes.

    This dataset adds static map elements.

    Args:
        pc_range: Range of the point cloud.
        map_ann_file: Path to the map annotation file.
        queue_length: Length of the queue.
        bev_size: Size of the BEV image. Default is (200, 200).
        fixed_ptsnum_per_line: Fixed number of points per line. Default is -1.
        padding_value: Value to use for padding. Default is -10000.
        map_classes: Tuple of map classes. Default is None.
        map_path: Path to the map. Default is None.
        aux_seg: Auxiliary segmentation information. Default is None.
        test_mode: Whether in test mode. Default is False.
        filter_empty_gt: Whether to filter empty ground truth. Default is True.
        use_lidar_gt: Whether to use LiDAR ground truth. Default is True.
        sd_map_path: Path to the SD map. Default is None.
        **kwargs: Additional keyword arguments.
    """

    MAPCLASSES: Tuple[str] = ("divider",)

    @require_packages("nuscenes")
    def __init__(
        self,
        pc_range: List[int],
        map_ann_file: Optional[str] = None,
        queue_length: int = 1,
        bev_size: Tuple[int, int] = (200, 200),
        fixed_ptsnum_per_line: int = -1,
        padding_value: int = -10000,
        map_classes: Optional[Tuple[str]] = None,
        map_path: Optional[str] = None,
        aux_seg: Optional[any] = None,
        test_mode: Optional[bool] = False,
        filter_empty_gt: Optional[bool] = True,
        use_lidar_gt: bool = True,
        sd_map_path: Optional[str] = None,
        **kwargs,
    ):
        super(NuscenesMapDataset, self).__init__(**kwargs)
        self.map_ann_file = map_ann_file

        self.queue_length = queue_length
        self.bev_size = bev_size

        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.aux_seg = aux_seg
        self.vector_map = VectorizedLocalMap(
            map_path,
            patch_size=self.patch_size,
            map_classes=self.MAPCLASSES,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=self.padding_value,
            canvas_size=bev_size,
            aux_seg=aux_seg,
            sd_map_path=sd_map_path,
        )

        self.filter_empty_gt = filter_empty_gt
        self.test_mode = test_mode
        self.flag = np.zeros(len(self), dtype=np.uint8)
        self.use_lidar_gt = use_lidar_gt

    def get_sample(self, idx: int):
        return self._decode(self.pack_file, self.samples[idx])

    @classmethod
    def get_map_classes(
        cls, map_classes: Optional[Sequence[str]] = None
    ) -> List[str]:
        """Get class names of current dataset.

        Args:
            map_classes : If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            A list of class names.
        """
        if map_classes is None:
            return cls.MAPCLASSES

        if isinstance(map_classes, (tuple, list)):
            class_names = map_classes
        else:
            raise ValueError(
                f"Unsupported type {type(map_classes)} of map classes."
            )

        return class_names

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def vectormap_pipeline(self, example):
        # import pdb;pdb.set_trace()
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(
            example["meta"]["sensor2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = example["meta"]["sensor2ego_translation"]
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(
            example["meta"]["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = example["meta"]["ego2global_translation"]

        lidar2global = ego2global @ lidar2ego
        example["lidar2global"] = lidar2global

        translation = list(lidar2global[:3, 3])
        rotation = list(Quaternion(matrix=lidar2global).q)
        homo2img = example["lidar2img"]

        if not self.use_lidar_gt:
            translation = example["meta"]["ego2global_translation"]
            rotation = example["meta"]["ego2global_rotation"]
            homo2img = example["ego2img"]

        location = example["meta"]["location"]
        num_cam = len(example["img"])
        if self.aux_seg:
            if isinstance(example["img"][0], Image.Image):
                img_shape = example["img"][0].size[::-1]
            else:
                img_shape = example["img"][0].shape[1:]
        else:
            img_shape = None

        anns_results = self.vector_map.gen_vectorized_samples(
            location,
            translation,
            rotation,
            img_shape=img_shape,
            homo2img=homo2img,
            num_cam=num_cam,
        )

        gt_vecs_label = to_tensor(anns_results["gt_vecs_label"])
        if isinstance(anns_results["gt_vecs_pts_loc"], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]
        else:
            gt_vecs_pts_loc = to_tensor(anns_results["gt_vecs_pts_loc"])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(
                    dtype=torch.float32
                )
            except BaseException:
                # empty tensor, will be passed in train,
                # but we preserve it for test
                gt_vecs_pts_loc = gt_vecs_pts_loc
        example["gt_labels_map"] = gt_vecs_label
        example["gt_instances"] = gt_vecs_pts_loc
        if anns_results["osm_vectors"] is not None:
            example["osm_vectors"] = anns_results["osm_vectors"]
        if anns_results["osm_mask"] is not None:
            example["osm_mask"] = to_tensor(anns_results["osm_mask"])
        if anns_results["gt_semantic_mask"] is not None:
            example["gt_seg_mask"] = to_tensor(
                anns_results["gt_semantic_mask"]
            )
        if anns_results["gt_pv_semantic_mask"] is not None:
            example["gt_pv_seg_mask"] = [
                to_tensor(mask) for mask in anns_results["gt_pv_semantic_mask"]
            ]
        return example

    def prepare_train_data(self, item):

        start = item
        end = item - self.queue_length
        queue_list = list(range(start, end, -1))

        # queue_list = [item]
        # index_list = list(range(item-self.queue_length, item))
        # random.shuffle(index_list)
        # index_list = sorted(index_list[1:], reverse=True)
        # queue_list.extend(index_list)

        seq_data = []
        for i in queue_list:
            idx = max(i, 0)
            example = super().__getitem__(idx)
            example = self.vectormap_pipeline(example)
            if self.filter_empty_gt and (
                example is None or ~(example["gt_labels_map"] != -1).any()
            ):
                return None

            seq_data.append(example)

        return seq_data

    def prepare_test_data(self, item):
        # example = super().__getitem__(item)
        # example = self.vectormap_pipeline(example)
        start = item
        end = item - self.queue_length
        seq_data = []

        for i in range(start, end, -1):
            idx = max(i, 0)
            example = super().__getitem__(idx)
            example = self.vectormap_pipeline(example)

            seq_data.append(example)
        return seq_data

        # return [example]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data


@OBJECT_REGISTRY.register
class NuscenesMapFromImageSequence(NuscenesFromImage):
    """Dataset type for processing NuScenes map data from image sequences.

    Args:
        num_seq: Num of sequential images. Default is 1.
        map_path: Path to the map. Default is None.
        map_classes: Tuple of map classes. Default is 3.
        pc_range: Range of the point cloud. Default is None.
        bev_size: Size of the BEV image. Default is (200, 200).
        fixed_ptsnum_per_line: Fixed number of points per line. Default is -1.
        padding_value: Value to use for padding. Default is -10000.
        use_lidar_gt: Whether to use LiDAR ground truth. Default is True.
        sd_map_path: Path to the SD map. Default is None.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        num_seq: int = 1,
        map_path: str = None,
        map_classes: int = 3,
        pc_range: List[int] = None,
        bev_size: Tuple[int, int] = (200, 200),
        fixed_ptsnum_per_line: int = -1,
        padding_value: int = -10000,
        use_lidar_gt: bool = True,
        sd_map_path: Optional[str] = None,
        **kwargs,
    ):
        super(NuscenesMapFromImageSequence, self).__init__(**kwargs)
        self.num_seq = num_seq
        self.use_lidar_gt = use_lidar_gt

        self.pc_range = pc_range
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.vector_map = VectorizedLocalMap(
            map_path,
            patch_size=self.patch_size,
            map_classes=map_classes,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=padding_value,
            canvas_size=bev_size,
            aux_seg=None,
            sd_map_path=sd_map_path,
        )

    def vectormap_pipeline(self, example):
        # import pdb;pdb.set_trace()
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(
            example["meta"]["sensor2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = example["meta"]["sensor2ego_translation"]
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(
            example["meta"]["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = example["meta"]["ego2global_translation"]

        lidar2global = ego2global @ lidar2ego
        example["lidar2global"] = lidar2global

        translation = list(lidar2global[:3, 3])
        rotation = list(Quaternion(matrix=lidar2global).q)
        homo2img = example["lidar2img"]

        if not self.use_lidar_gt:
            translation = example["meta"]["ego2global_translation"]
            rotation = example["meta"]["ego2global_rotation"]
            homo2img = example["ego2img"]

        location = example["meta"]["location"]
        num_cam = len(example["img"])

        anns_results = self.vector_map.gen_vectorized_samples(
            location,
            translation,
            rotation,
            homo2img=homo2img,
            num_cam=num_cam,
            img_shape=None,
        )

        gt_vecs_label = to_tensor(anns_results["gt_vecs_label"])
        if isinstance(anns_results["gt_vecs_pts_loc"], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]
        else:
            gt_vecs_pts_loc = to_tensor(anns_results["gt_vecs_pts_loc"])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(
                    dtype=torch.float32
                )
            except BaseException:
                # empty tensor, will be passed in train,
                # but we preserve it for test
                gt_vecs_pts_loc = gt_vecs_pts_loc
        example["gt_labels_map"] = gt_vecs_label
        example["gt_instances"] = gt_vecs_pts_loc
        if anns_results["osm_vectors"] is not None:
            example["osm_vectors"] = anns_results["osm_vectors"]
        if anns_results["osm_mask"] is not None:
            example["osm_mask"] = to_tensor(anns_results["osm_mask"])
        return example

    def __getitem__(self, item):
        start = item
        end = item - self.num_seq
        seq_data = []

        for i in range(start, end, -1):
            idx = max(i, 0)
            data = super().__getitem__(idx)
            data = self.vectormap_pipeline(data)
            seq_data.append(data)
        return seq_data


@OBJECT_REGISTRY.register
class NuscenesSparseMapDataset(NuscenesMapDataset):
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)[0]
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data[0]
