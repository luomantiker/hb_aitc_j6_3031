# Copyright (c) Horizon Robotics. All rights reserved.

# Functional transforms used during the data pipeline for lidar data.
import copy
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from hat.core import box_np_ops
from hat.core.center_utils import draw_umich_gaussian, gaussian_radius
from hat.data.transforms.lidar_utils import VoxelGenerator, preprocess
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "ParsePointCloud",
    "Voxelization",
    "DetectionTargetGenerator",
    "BBoxSelector",
    "Point2VCS",
    "DetectionAnnoToBEVFormat",
]


def cat_sampled_points(
    points: np.ndarray,
    sampled_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cat sampled points into origin points.

    Args:
        points (np.ndarray): point clouds. The first 3 dimensions specify
            xyz.
        sampled_points (np.ndarray): point clouds.
            The first 3 dimensions specify xyz.
    Returns:
        Cated points and original points shape
    """
    original_shape = points.shape
    pad_len = points.shape[1] - sampled_points.shape[1]
    sampled_points = np.pad(
        sampled_points,
        ((0, 0), (0, pad_len)),
        "constant",
        constant_values=((0.0, 0.0), (0.0, 0.0)),
    )
    points = np.concatenate([points, sampled_points], axis=0)
    return points, original_shape


def decat_sampled_points(
    points: np.ndarray,
    original_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Decat merge points to origin points and sample points.

    Args:
        points (np.ndarray): point clouds. The first 3 dimensions specify
            xyz.
        original_shape: Tuple[int, int, int]: points array shape

    Returns:
        Origin points and sample points
    """
    sampled_points = points[original_shape[0] :, :]
    points = points[: original_shape[0], :]
    return points, sampled_points


def _dict_select(dict_: Dict[str, Any], inds: np.ndarray) -> None:
    """Filter each value in the dict, choose those that are selected by inds.

    Args:
        dict_ (Dict[str, Any]): the dictionary to be filtered.
        inds (np.ndarray): selected values' indices.
    """
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            if k not in {"gt_seg"}:
                dict_[k] = v[inds]


@OBJECT_REGISTRY.register_module
class BBoxSelector:
    """Filter out GT BBoxes.

    Support multiframe and multimodal data.
    """

    def __init__(
        self,
        category2id_map: Dict,
        vcs_range: Tuple[float, float, float, float],
        min_points_in_gt: int = 0,
    ):
        """Init BBoxSelector.

        Args:
            category2id_map: category to id map, which includes category user
                wants to keep. Defaults to None.
            vcs_range: vcs range. Defaults to None.
            min_points_in_gt: min points number inside each gt bbox as a
                threshold. Defaults to 0.
        """
        self.category2id_map = category2id_map
        self.vcs_range = vcs_range
        self.min_points_in_gt = min_points_in_gt

    def __call__(self, sample):
        anno_dict = sample["lidar_gt"]
        gt_dict = {
            "gt_boxes": anno_dict["gt_boxes"],
            "gt_names": np.array(anno_dict["gt_names"]).reshape(-1),
        }

        point_counts = box_np_ops.points_count_rbbox(
            sample["point_clouds"][0], gt_dict["gt_boxes"]
        )
        min_point_mask = point_counts >= self.min_points_in_gt

        gt_boxes_mask = np.array(
            [n in self.category2id_map.keys() for n in gt_dict["gt_names"]],
            dtype=np.bool_,
        )

        gt_names_mask = np.array(
            [int(self.category2id_map[n]) >= 0 for n in gt_dict["gt_names"]],
            dtype=np.bool_,
        )

        vcs_mask = preprocess.filter_gt_box_outside_range(
            gt_dict["gt_boxes"], limit_range=self.vcs_range
        )

        mask = np.logical_and(
            np.logical_and(gt_boxes_mask, gt_names_mask),
            np.logical_and(min_point_mask, vcs_mask),
        )
        _dict_select(gt_dict, mask)

        gt_classes = np.array(
            [
                int(self.category2id_map[n]) + 1
                for n in gt_dict["gt_names"]
                if self.category2id_map[n] >= 0
            ],
            dtype=np.int32,
        )
        gt_dict["gt_classes"] = gt_classes

        sample["lidar_gt"] = gt_dict

        return sample


@OBJECT_REGISTRY.register_module
class Voxelization:
    """Perform voxelization for points in multiple frames."""

    def __init__(
        self,
        range: Tuple[float, ...],
        voxel_size: Tuple[float, float, float],
        max_points_in_voxel: int,
        max_voxel_num: int = 20000,
        voxel_key: str = "voxel",
        nframe: int = 1,
    ):
        """Transform point cloud into voxels.

        Each voxel is either empty or filled with some points.

        Args:
            range: point cloud range [xmin, ymin, zmin, xmax, ymax, zmax].
            voxel_size: unit voxel size [dx, dy, dz].
            max_points_in_voxel: max point number in each voxel.
            max_voxel_num: max voxel number. Defaults to 20000.
            voxel_key: voxel name. Defaults to "voxel".
            nframe: number of frames. Defaults to 1.
        """
        self.range = range
        self.voxel_size = voxel_size
        self.max_points_in_voxel = max_points_in_voxel
        self.max_voxel_num = max_voxel_num
        assert voxel_key in ["voxel", "pillar"]  # related to collate_fn
        self.voxel_key = voxel_key

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num,
        )

        self.nframe = nframe

    def __call__(self, sample):
        grid_size = self.voxel_generator.grid_size

        voxels_list = []
        coordinates_list = []
        num_points_list = []
        num_voxels_list = []

        for idx in range(self.nframe):
            voxels, coordinates, num_points = self.voxel_generator.generate(
                sample["point_clouds"][idx]
            )
            num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

            voxels_list.append(voxels)
            coordinates_list.append(coordinates)
            num_points_list.append(num_points)
            num_voxels_list.append(num_voxels)

        sample.update(
            {
                f"{self.voxel_key}_data": voxels_list,
                f"{self.voxel_key}_coordinates": coordinates_list,
                f"{self.voxel_key}_num_points": num_points_list,
                f"num_{self.voxel_key}s": num_voxels_list,
                f"{self.voxel_key}_shape": grid_size,
            }
        )

        return sample


def merge_multi_group_label(gt_classes, num_classes_by_task) -> np.ndarray:
    num_task = len(gt_classes)
    flag = 0

    for i in range(num_task):
        gt_classes[i] += flag
        flag += num_classes_by_task[i]

    return np.concatenate(gt_classes, axis=0)


def update_category(tasks):
    """
    Add default attribute index_info' and 'target_class_names' for tasks.

    Args:
        tasks(dict):
            'index_info' will be used in assign pipeline and
            'target_class_names' will be used
            in training.
            For 8 class tasks config:
                num_class: 8
                class_names: ['Car', 'Cyclist', 'Tricycle',
                    'Pedestrian', 'Construction',
                    'Truck', 'Bus', 'Blur']
                target_class_names: ['Car', 'Cyclist', 'Tricycle',
                    'Pedestrian', 'Construction',
                    'Truck', 'Bus', 'Blur'] (same as class_names)
                index_info: [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)]
            For 3 class tasks config:
                num_class: 3
                class_names: ['Car', 'Cyclist', 'Tricycle',
                    'Pedestrian', 'Construction',
                    'Truck', 'Bus', 'Blur']
                target_class_names: ['Vehicle',
                'Cyclist', 'Pedestrian']
                index_info: [(0, 4, 5, 6, 7), (1, 2), (3,)]
    """
    total_num_class = 0
    for t in tasks:
        if "index_info" not in t:
            t["index_info"] = [
                (i + total_num_class,) for i in range(t["num_class"])
            ]
        if "target_class_names" not in t:
            t["target_class_names"] = t["class_names"]
        total_num_class += t["num_class"]
    return tasks


def get_ctoff_map(wh, center):
    """Calculate the regression map of bev center offset.

    Args:
        wh (int): the inserted kernel size.(order (w,h))
        center (float): object center in bev.(order u, v)

    Returns:
        (ndarray): bev center offset regression map

    """

    w, h = wh
    radius = (w // 2, h // 2)
    n, m = radius
    center_int = (int(center[0]), int(center[1]))
    center_offset = np.array(center) - np.array(center_int)
    x, y = center_int

    y_grid = np.arange(y - m, y + m + 1)
    x_grid = np.arange(x - n, x + n + 1)

    y_reg = center[1] - y_grid
    x_reg = center[0] - x_grid

    y_reg[m] = center_offset[1]
    x_reg[n] = center_offset[0]
    xv, yv = np.meshgrid(x_reg, y_reg)
    ct_off_reg_map = np.concatenate(
        [xv[:, :, np.newaxis], yv[:, :, np.newaxis]], axis=-1
    )

    xv_grid, yv_grid = np.meshgrid(x_grid, y_grid)
    ct_off_reg_grid = np.concatenate(
        [xv_grid[:, :, np.newaxis], yv_grid[:, :, np.newaxis]], axis=-1
    )

    return ct_off_reg_map, ct_off_reg_grid


class BaseTargetGenerator:
    def __init__(
        self,
        feature_stride: int,
        id2label: Optional[Dict],
        to_bev: bool = True,
        pc_range: Tuple[float, ...] = (-50, -50, -10, 50, 50, 10),
        feat_shape: Tuple[int, int] = (100, 100),
        voxel_size: Tuple[float, float, float] = (0.1, 0.1, 0.1),
    ):
        """Generate CenterNet training targets.

        Args:
            feature_stride: feature map stride.
            id2label: class id to label name map.
            to_bev: whether to convert gt to bev feature coordinate.
            pc_range: point cloud range. (xmin, ymin, zmin, xmax, ymax, zmax)
            voxel_size: voxel size. (sx, sy, sz)
        """
        self.feature_stride = feature_stride
        self.id2label = id2label
        self.to_bev = to_bev
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.feat_shape = feat_shape

    def __call__(self, data):
        return self.generate(data)

    def generate(self, data):
        raise NotImplementedError


@OBJECT_REGISTRY.register_module
class DetectionTargetGenerator(BaseTargetGenerator):
    """Create detection training targets."""

    def __init__(
        self,
        feature_stride: int,
        id2label: Dict,
        pc_range: Tuple[float, ...],
        feat_shape: Tuple[int, int],
        voxel_size: Tuple[float, float, float],
        to_bev: bool = True,
        max_objs: int = 500,
        min_gaussian_overlap: float = 0.1,
        min_gaussian_radius: float = 2.0,
        use_gaussian_reg_loss: bool = False,
    ):
        """Init function.

        Args:
            feature_stride: feature map stride.
            id2label: class id to label name map.
            to_bev: whether to convert gt to bev feature coordinate.
            pc_range: point cloud range. (xmin, ymin, zmin, xmax, ymax, zmax)
            voxel_size: voxel size. (sx, sy, sz)
            max_objs: maximum number of objects in one sample.
            min_gaussian_overlap: minimum overlap threshold between kernels.
            min_gaussian_radius: minimum radius of gaussian kernel.
            use_gaussian_reg_loss: whether to use gaussian kernel in regression
        """
        super().__init__(
            feature_stride=feature_stride,
            id2label=id2label,
            to_bev=to_bev,
            pc_range=pc_range,
            voxel_size=voxel_size,
            feat_shape=feat_shape,
        )
        self.min_gaussian_overlap = min_gaussian_overlap
        self.max_objs = max_objs
        self.min_gaussian_radius = min_gaussian_radius
        self.use_gaussian_reg_loss = use_gaussian_reg_loss

    def generate(self, data: Tuple[Dict]):
        # Calculate output featuremap size
        gt_dict = data.pop("lidar_gt")  # TODO: use general gt

        feature_map_resolution = (
            (self.pc_range[3] - self.pc_range[0]) / self.feat_shape[0],
            (self.pc_range[4] - self.pc_range[1]) / self.feat_shape[1],
        )

        # reorganize the gt_dict by category
        cls_mask = [
            np.where(gt_dict["gt_classes"] == i + 1)
            for i in self.id2label.keys()
        ]

        det_box = []
        det_class = []
        det_name = []
        for m in cls_mask:
            det_box.append(gt_dict["gt_boxes"][m])
            det_class.append(gt_dict["gt_classes"][m])
            det_name.append(gt_dict["gt_names"][m])

        det_box = np.concatenate(det_box, axis=0)
        det_class = np.concatenate(det_class)
        det_name = np.concatenate(det_name)

        # limit rad to [-pi, pi]
        det_box[:, -1] = box_np_ops.limit_period(
            det_box[:, -1], offset=0.5, period=np.pi * 2
        )

        gt_dict["gt_classes"] = det_class
        gt_dict["gt_names"] = det_name
        gt_dict["gt_boxes"] = det_box

        data["annos_dict"] = gt_dict

        hm = np.zeros(
            (
                len(self.id2label),
                self.feat_shape[0],
                self.feat_shape[1],
            ),
            dtype=np.float32,
        )
        anno_box = np.zeros((self.max_objs, 8), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat = np.zeros((self.max_objs), dtype=np.int64)
        gt_boxes_task = np.zeros((self.max_objs, 7), dtype=np.float32)

        num_objs = min(gt_dict["gt_boxes"].shape[0], self.max_objs)

        if self.use_gaussian_reg_loss:
            valid_idx = 0
            ind_reg = np.zeros((self.max_objs * 25), dtype=np.int64)
            mask_reg = np.zeros((self.max_objs * 25), dtype=np.float32)
            anno_box_reg = np.zeros((self.max_objs * 25, 8), dtype=np.float32)

        for k in range(num_objs):
            cls_id = gt_dict["gt_classes"][k] - 1

            width, length = (
                gt_dict["gt_boxes"][k][3] / feature_map_resolution[0],
                gt_dict["gt_boxes"][k][4] / feature_map_resolution[1],
            )
            if not all([width > 0, length > 0]):
                continue

            radius = gaussian_radius(
                (length, width), min_overlap=self.min_gaussian_overlap
            )
            radius = max(self.min_gaussian_radius, int(radius))

            x, y, z = (
                gt_dict["gt_boxes"][k][0],
                gt_dict["gt_boxes"][k][1],
                gt_dict["gt_boxes"][k][2],
            )
            coor_x = (x - self.pc_range[0]) / feature_map_resolution[0]
            coor_y = (y - self.pc_range[1]) / feature_map_resolution[1]

            xi_limit, yi_limit = self.feat_shape[:2]

            if self.to_bev:
                coor_x, coor_y = (
                    self.feat_shape[1] - coor_y - 1,
                    self.feat_shape[0] - coor_x - 1,
                )
                xi_limit, yi_limit = yi_limit, xi_limit

            ct = np.array([coor_x, coor_y], dtype=np.float32)
            xi, yi = ct.astype(np.int32)

            if not (0 <= xi < xi_limit and 0 <= yi < yi_limit):
                continue

            if self.use_gaussian_reg_loss:
                _, masked_gaussian, valid_mask = draw_umich_gaussian(
                    hm[cls_id], ct, radius, return_gaussian=True
                )
                # add more positive regression samples
                diameter = 2 * radius + 1
                ct_off_reg, ct_off_reg_grid = get_ctoff_map(
                    (diameter, diameter), ct
                )  # [diameter, diameter, 2]

                # get gaussian mask index
                ct_off_reg_reshape = ct_off_reg[
                    valid_mask.astype(np.bool_)
                ].reshape(
                    (-1, 2)
                )  # [diameter*diameter, 2]
                ct_off_reg_grid_reshape = ct_off_reg_grid[
                    valid_mask.astype(np.bool_)
                ].reshape(
                    (-1, 2)
                )  # [diameter*diameter, 2]
                masked_gaussian_reshape = masked_gaussian.reshape(
                    -1
                )  # [diameter*diameter]

                num_reg = masked_gaussian_reshape.shape[0]

                for grid_idx in range(num_reg):
                    cur_x, cur_y = (
                        ct_off_reg_grid_reshape[grid_idx, 0],
                        ct_off_reg_grid_reshape[grid_idx, 1],
                    )
                    ind_reg[valid_idx] = cur_y * self.feat_shape[1] + cur_x
                    mask_reg[valid_idx] = masked_gaussian_reshape[grid_idx]

                    rot = gt_dict["gt_boxes"][k][-1]
                    anno_box_reg[valid_idx] = np.concatenate(
                        (
                            ct_off_reg_reshape[grid_idx],
                            z,
                            np.log(gt_dict["gt_boxes"][k][3:6]),
                            np.sin(rot),
                            np.cos(rot),
                        ),
                        axis=None,
                    )

                    valid_idx += 1
            else:
                draw_umich_gaussian(hm[cls_id], ct, radius)

            cat[k] = cls_id
            ind[k] = yi * self.feat_shape[1] + xi
            mask[k] = 1
            gt_boxes_task[k, :] = gt_dict["gt_boxes"][
                k, [0, 1, 2, 3, 4, 5, -1]
            ]
            rot = gt_dict["gt_boxes"][k][-1]
            anno_box[k] = np.concatenate(
                (
                    ct - (xi, yi),
                    z,
                    np.log(gt_dict["gt_boxes"][k][3:6]),
                    np.sin(rot),
                    np.cos(rot),
                ),
                axis=None,
            )

        targets = {
            "hm": hm,
            "anno_box": anno_box,
            "ind": ind,
            "mask": mask,
            "cat": cat,
            "gt_boxes_tasks": gt_boxes_task,
        }

        if self.use_gaussian_reg_loss:
            targets.update(
                {
                    "anno_box_reg": anno_box_reg,
                    "ind_reg": ind_reg,
                    "mask_reg": mask_reg,
                }
            )

        data.update(targets)
        return data


@OBJECT_REGISTRY.register_module
class DetectionAnnoToBEVFormat:
    def __init__(
        self,
        bev_size: Sequence[int] = None,
        vcs_range: Sequence[float] = None,
        max_objs: int = 100,
        category2id_map: Mapping = None,
        enable_ignore: bool = False,
    ):
        """Init function.

        Args:
            bev_size: bev feature shape.
            vcs_range: vcs range.
            max_objs: maximum number of objects in one sample.
            category2id_map: category to id map, which includes category user
                wants to keep. Defaults to None.
            enable_ignore: whether to enable ignore flag.

        """
        self.bev_size = bev_size
        self.vcs_range = vcs_range
        self.max_objs = max_objs
        self.category2id_map = category2id_map
        self.enable_ignore = enable_ignore

        self.m_perpixel = (
            abs(vcs_range[2] - vcs_range[0]) / bev_size[0],
            abs(vcs_range[3] - vcs_range[1]) / bev_size[1],
        )  # bev coord y, x

    def __call__(self, data: Dict) -> Dict:
        bev_gts = []
        boxes = data["annotations"]["gt_boxes"]
        names = data["annotations"]["gt_names"]
        for i, box in enumerate(boxes):
            box = box.tolist()
            bev_gt = {}
            bev_gt["dimension"] = [box[5], box[3], box[4]]
            bev_gt["yaw"] = -box[-1] - math.pi / 2  # TODO check
            # bev_gt["yaw"] =box[-1]#TODO check
            bev_gt["location"] = box[0:3]
            bev_gt["score"] = 1.0
            bev_gt["label"] = names[i]
            bev_gt["ignore"] = False
            bev_gts.append(bev_gt)

        annotations = copy.deepcopy(bev_gts)

        # used for eval
        vcs_loc_ = torch.zeros((self.max_objs, 3), dtype=torch.float32)
        vcs_dim_ = torch.zeros((self.max_objs, 3), dtype=torch.float32)
        vcs_rot_z_ = torch.zeros((self.max_objs), dtype=torch.float32)
        vcs_cls_ = torch.zeros((self.max_objs), dtype=torch.float32) - 99
        vcs_ignore_ = torch.zeros((self.max_objs), dtype=torch.bool)
        vcs_visible_ = torch.zeros((self.max_objs), dtype=torch.float32)

        count_id = 0
        for anno in annotations:
            if count_id > (self.max_objs):
                break
            cls_id = int(self.category2id_map[anno["label"]])
            if cls_id <= -99 or (
                not self.enable_ignore and anno.get("ignore", False)
            ):
                continue
            vcs_loc = anno["location"]
            bev_ct = (
                ((self.vcs_range[3] - vcs_loc[1]) / self.m_perpixel[1]),
                ((self.vcs_range[2] - vcs_loc[0]) / self.m_perpixel[0]),
            )  # (u, v)
            bev_ct_int = (int(bev_ct[0]), int(bev_ct[1]))

            if (
                0 <= bev_ct_int[0] < self.bev_size[1]
                and 0 <= bev_ct_int[1] < self.bev_size[0]
            ):

                vcs_loc_[count_id] = torch.tensor(vcs_loc)
                vcs_cls_[count_id] = cls_id
                vcs_dim_[count_id] = torch.tensor(anno["dimension"])
                # transfer the yaw to value [-pi,pi]
                vcs_yaw = np.arctan2(np.sin(anno["yaw"]), np.cos(anno["yaw"]))
                vcs_rot_z_[count_id] = vcs_yaw
                if anno.get("ignore", False):
                    vcs_ignore_[count_id] = 1
                # Visibility reflects the occlusion degree of the target
                if anno.get("visibility", False):
                    vcs_visible_[count_id] = anno["visibility"]
                count_id += 1

        annotation_bev_3d = {
            "vcs_loc_": vcs_loc_,
            "vcs_cls_": vcs_cls_,
            "vcs_rot_z_": vcs_rot_z_,
            "vcs_dim_": vcs_dim_,
            "vcs_ignore_": vcs_ignore_,
            "vcs_visible_": vcs_visible_,
        }
        data["annotation_bev_3d"] = annotation_bev_3d
        return data


@OBJECT_REGISTRY.register_module
class ParsePointCloud:
    """Parse point cloud from bytes to numpy array."""

    def __init__(
        self,
        dtype: np.dtype = np.float32,
        load_dim: int = 4,
        keep_dim: int = 4,
    ):
        """Init function.

        Args:
            dtype: data type stored in bin buffer.
            load_dim: dimension of each point.
            keep_dim: dimension of each point to keep.
        """
        self.dtype = dtype
        self.load_dim = load_dim
        self.keep_dim = keep_dim

    def __call__(self, data):
        raw_point_list = data["point_clouds"]
        point_list = []
        for raw_point in raw_point_list:
            encode_point = np.frombuffer(raw_point, self.dtype)
            point = encode_point.reshape((-1, self.load_dim))
            point = point[:, : self.keep_dim]
            point_list.append(point.astype(np.float32))
        data["point_clouds"] = point_list
        return data


@OBJECT_REGISTRY.register_module
class Point2VCS:
    """Transform pointclouds from lidar CS to VCS."""

    def __init__(self, shuffle_points: bool = False):
        """Init function.

        Args:
            shuffle_points: whether to randomly set the order of points.
        """
        self.shuffle_points = shuffle_points

    def transform_points(self, points, lidar2vcs):
        points_h = np.insert(points, 3, values=1, axis=1).T
        points_update = np.dot(lidar2vcs, points_h).T
        return points_update[:, :3]

    def __call__(self, data: Dict):
        pointclouds_list = data.pop("point_clouds")
        rt_lidar2vcs = data["meta_info"]["T_lidar2vcs"]

        new_pointclouds_list = []
        for pointcloud in pointclouds_list:
            points_feature = pointcloud[:, 3:]
            points_transform = self.transform_points(
                pointcloud[:, :3], rt_lidar2vcs
            )
            points = np.concatenate(
                [points_transform.astype(pointcloud.dtype), points_feature],
                axis=1,
            )
            if self.shuffle_points:
                np.random.shuffle(points)
            new_pointclouds_list.append(points)

        data["point_clouds"] = new_pointclouds_list
        return data
