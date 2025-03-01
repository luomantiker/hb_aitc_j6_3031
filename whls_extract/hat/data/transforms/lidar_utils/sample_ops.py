# Copyright https://github.com/V2AI/Det3D.
# @article{zhu2019class,
#   title={Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection}, # noqa E501
#   author={Zhu, Benjin and Jiang, Zhengkai and Zhou, Xiangxin and Li, Zeming and Yu, Gang}, # noqa E501
#   journal={arXiv preprint arXiv:1908.09492},
#   year={2019}
# }
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Added line34-line47, line62-line87, line248-line277, line280-line282.


import copy
import pathlib
import pickle
from functools import reduce
from typing import Dict, List

import numpy as np

from hat.core import box_np_ops, box_utils
from hat.data.transforms.lidar_utils import preprocess
from hat.data.transforms.lidar_utils.preprocess import DataBasePreprocessor
from hat.registry import OBJECT_REGISTRY, build_from_registry

__all__ = ["DataBaseSampler"]


def is_array_like(x):
    return isinstance(x, (list, tuple, np.ndarray))


def shape_mergeable(x, expected_shape):
    mergeable = True
    if is_array_like(x) and is_array_like(expected_shape):
        x = np.array(x)
        if len(x.shape) == len(expected_shape):
            for s, s_ex in zip(x.shape, expected_shape):
                if s_ex is not None and s != s_ex:
                    mergeable = False
                    break
    return mergeable


@OBJECT_REGISTRY.register_module
class DataBaseSampler:
    def __init__(
        self,
        enable,
        db_info_path: str,
        sample_groups: Dict,
        db_prep_steps: Dict,
        global_random_rotation_range_per_object: List[float],
        rate: float = 1.0,
        root_path: str = None,
    ):
        """DB_sampler for lidar dataset.

        Sample some object's points to put into current points.

        Args:
            db_info_path : thd db_sample data path.
            sample_groups : the sample class group.
            db_prep_steps :pre-process for filter sample data.
            global_random_rotation_range_per_object.
        """
        prepors = [build_from_registry(c) for c in db_prep_steps]
        db_prepor = DataBasePreprocessor(prepors)
        self.enable = enable
        rate = rate
        groups = sample_groups
        # groups = [dict(g.name_to_max_num) for g in groups]
        info_path = db_info_path
        self.root_path = root_path
        with open(info_path, "rb") as f:
            db_infos = pickle.load(f)
        global_random_rotation_range_per_object = list(
            global_random_rotation_range_per_object
        )
        if len(global_random_rotation_range_per_object) == 0:
            global_random_rotation_range_per_object = None
        global_rot_range = global_random_rotation_range_per_object
        for k, v in db_infos.items():
            print(f"load {len(v)} {k} database infos")

        if db_prepor is not None:
            db_infos = db_prepor(db_infos)
            print("After filter database:")
            for k, v in db_infos.items():
                print(f"load {len(v)} {k} database infos")

        self.db_infos = db_infos
        self._rate = rate
        self._groups = groups
        self._group_db_infos = {}
        self._group_name_to_names = []
        self._sample_classes = []
        self._sample_max_nums = []
        self._use_group_sampling = False  # slower
        if any([len(g) > 1 for g in groups]):
            self._use_group_sampling = True
        if not self._use_group_sampling:
            self._group_db_infos = self.db_infos  # just use db_infos
            for group_info in groups:
                group_names = list(group_info.keys())
                self._sample_classes += group_names
                self._sample_max_nums += list(group_info.values())
        else:
            for group_info in groups:
                group_dict = {}
                group_names = list(group_info.keys())
                group_name = ", ".join(group_names)
                self._sample_classes += group_names
                self._sample_max_nums += list(group_info.values())
                self._group_name_to_names.append((group_name, group_names))
                # self._group_name_to_names[group_name] = group_names
                for name in group_names:
                    for item in db_infos[name]:
                        gid = item["group_id"]
                        if gid not in group_dict:
                            group_dict[gid] = [item]
                        else:
                            group_dict[gid] += [item]
                if group_name in self._group_db_infos:
                    raise ValueError("group must be unique")
                group_data = list(group_dict.values())
                self._group_db_infos[group_name] = group_data
                info_dict = {}
                if len(group_info) > 1:
                    for group in group_data:
                        names = [item["name"] for item in group]
                        names = sorted(names)
                        group_name = ", ".join(names)
                        if group_name in info_dict:
                            info_dict[group_name] += 1
                        else:
                            info_dict[group_name] = 1
                print(info_dict)

        self._sampler_dict = {}
        for k, v in self._group_db_infos.items():
            self._sampler_dict[k] = preprocess.BatchSampler(v, k)
        self._enable_global_rot = False
        if global_rot_range is not None:
            if not isinstance(global_rot_range, (list, tuple, np.ndarray)):
                global_rot_range = [-global_rot_range, global_rot_range]
            else:
                assert shape_mergeable(global_rot_range, [2])
            if np.abs(global_rot_range[0] - global_rot_range[1]) >= 1e-3:
                self._enable_global_rot = True
        self._global_rot_range = global_rot_range

    @property
    def use_group_sampling(self):
        return self._use_group_sampling

    def sample_all(
        self,
        root_path: str,
        gt_boxes: np.ndarray,
        gt_names: List,
        num_point_features: int,
        random_crop: bool = False,
        gt_group_ids: List = None,
        calib: np.ndarray = None,
        road_planes: List[int] = None,
    ):
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(
            self._sample_classes, self._sample_max_nums
        ):
            sampled_num = int(
                max_sample_num - np.sum([n == class_name for n in gt_names])
            )

            sampled_num = np.round(self._rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled_groups = self._sample_classes
        if self._use_group_sampling:
            assert gt_group_ids is not None
            sampled_groups = []
            sample_num_per_class = []
            for group_name, class_names in self._group_name_to_names:
                sampled_nums_group = [sampled_num_dict[n] for n in class_names]
                sampled_num = np.max(sampled_nums_group)
                sample_num_per_class.append(sampled_num)
                sampled_groups.append(group_name)
            total_group_ids = gt_group_ids
        sampled = []
        sampled_gt_boxes = []
        avoid_coll_boxes = gt_boxes

        for class_name, sampled_num in zip(
            sampled_groups, sample_num_per_class
        ):
            if sampled_num > 0:
                if self._use_group_sampling:
                    sampled_cls = self.sample_group(
                        class_name,
                        sampled_num,
                        avoid_coll_boxes,
                        total_group_ids,
                    )
                else:
                    sampled_cls = self.sample_class_v2(
                        class_name, sampled_num, avoid_coll_boxes
                    )

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][
                            np.newaxis, ...
                        ]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0
                        )

                    sampled_gt_boxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0
                    )
                    if self._use_group_sampling:
                        if len(sampled_cls) == 1:
                            sampled_group_ids = np.array(
                                sampled_cls[0]["group_id"]
                            )[np.newaxis, ...]
                        else:
                            sampled_group_ids = np.stack(
                                [s["group_id"] for s in sampled_cls], axis=0
                            )
                        total_group_ids = np.concatenate(
                            [total_group_ids, sampled_group_ids], axis=0
                        )

        if len(sampled) > 0:
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)

            if road_planes is not None:
                # Only support KITTI
                # image plane
                # assert False, "Not correct yet!"
                a, b, c, d = road_planes

                center = sampled_gt_boxes[:, :3]
                center[:, 2] -= sampled_gt_boxes[:, 5] / 2
                center_cam = box_np_ops.lidar_to_camera(
                    center, calib["rect"], calib["Trv2c"]
                )

                cur_height_cam = (
                    -d - a * center_cam[:, 0] - c * center_cam[:, 2]
                ) / b
                center_cam[:, 1] = cur_height_cam
                lidar_tmp_point = box_np_ops.camera_to_lidar(
                    center_cam, calib["rect"], calib["Trv2c"]
                )
                cur_lidar_height = lidar_tmp_point[:, 2]

                # botom to middle center
                # kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
                sampled_gt_boxes[:, 2] = (
                    cur_lidar_height + sampled_gt_boxes[:, 5] / 2
                )

                # mv_height = sampled_gt_boxes[:, 2] - cur_lidar_height
                # sampled_gt_boxes[:, 2] -= mv_height

            num_sampled = len(sampled)
            s_points_list = []
            if self.root_path and not root_path:
                root_path = self.root_path

            for info in sampled:
                try:
                    # TODO fix point read error by shijie.sun
                    s_points = np.fromfile(
                        str(pathlib.Path(root_path) / info["path"]),
                        dtype=np.float32,
                    ).reshape(-1, num_point_features)
                    # if not add_rgb_to_points:
                    #     s_points = s_points[:, :4]
                    if "rot_transform" in info:
                        rot = info["rot_transform"]
                        s_points[
                            :, :3
                        ] = box_np_ops.rotation_points_single_angle(
                            s_points[:, :4], rot, axis=2
                        )
                    s_points[:, :3] += info["box3d_lidar"][:3]
                    s_points_list.append(s_points)
                except FileNotFoundError:
                    continue
            # gt_bboxes = np.stack([s["bbox"] for s in sampled], axis=0)
            # if np.random.choice([False, True], replace=False, p=[0.3, 0.7]):
            # do random crop.
            if random_crop:
                s_points_list_new = []
                assert calib is not None
                rect = calib["rect"]
                Trv2c = calib["Trv2c"]
                P2 = calib["P2"]
                gt_bboxes = box_np_ops.box3d_to_bbox(
                    sampled_gt_boxes, rect, Trv2c, P2
                )
                crop_frustums = preprocess.random_crop_frustum(
                    gt_bboxes, rect, Trv2c, P2
                )
                for i in range(crop_frustums.shape[0]):
                    s_points = s_points_list[i]
                    mask = preprocess.mask_points_in_corners(
                        s_points, crop_frustums[i : i + 1]
                    ).reshape(-1)
                    num_remove = np.sum(mask)
                    if (
                        num_remove > 0
                        and (s_points.shape[0] - num_remove) > 15
                    ):
                        s_points = s_points[np.logical_not(mask)]
                    s_points_list_new.append(s_points)
                s_points_list = s_points_list_new
            ret = {
                "gt_names": np.array([s["name"] for s in sampled]),
                "difficulty": np.array([s["difficulty"] for s in sampled]),
                "gt_boxes": sampled_gt_boxes,
                "points": np.concatenate(s_points_list, axis=0),
                "gt_masks": np.ones((num_sampled,), dtype=np.bool_),
            }
            if self._use_group_sampling:
                ret["group_ids"] = np.array([s["group_id"] for s in sampled])
            else:
                ret["group_ids"] = np.arange(
                    gt_boxes.shape[0], gt_boxes.shape[0] + len(sampled)
                )
        else:
            ret = None
        return ret

    def sample(self, name, num):
        if self._use_group_sampling:
            group_name = name
            ret = self._sampler_dict[group_name].sample(num)
            groups_num = [len(ll) for ll in ret]
            return reduce(lambda x, y: x + y, ret), groups_num
        else:
            ret = self._sampler_dict[name].sample(num)
            return ret, np.ones((len(ret),), dtype=np.int64)

    def sample_v1(self, name, num):
        if isinstance(name, (list, tuple)):
            group_name = ", ".join(name)
            ret = self._sampler_dict[group_name].sample(num)
            groups_num = [len(ll) for ll in ret]
            return reduce(lambda x, y: x + y, ret), groups_num
        else:
            ret = self._sampler_dict[name].sample(num)
            return ret, np.ones((len(ret),), dtype=np.int64)

    def sample_class_v2(self, name, num, gt_boxes):
        sampled = self._sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled)
        gt_boxes_bv = box_utils.center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, -1]
        )

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)

        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask, np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0
        )
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        if self._enable_global_rot:
            # place samples to any place in a circle.
            preprocess.noise_per_object_v3_(
                boxes,
                None,
                valid_mask,
                0,
                0,
                self._global_rot_range,
                num_try=100,
            )

        sp_boxes_new = boxes[gt_boxes.shape[0] :]
        sp_boxes_bv = box_utils.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, -1]
        )

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        # coll_mat = collision_test_allbox(total_bv)
        coll_mat = preprocess.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                if self._enable_global_rot:
                    sampled[i - num_gt]["box3d_lidar"][:2] = boxes[i, :2]
                    sampled[i - num_gt]["box3d_lidar"][-1] = boxes[i, -1]
                    sampled[i - num_gt]["rot_transform"] = (
                        boxes[i, -1] - sp_boxes[i - num_gt, -1]
                    )
                valid_samples.append(sampled[i - num_gt])
        return valid_samples

    def sample_group(self, name, num, gt_boxes, gt_group_ids):
        sampled, group_num = self.sample(name, num)
        sampled = copy.deepcopy(sampled)
        # rewrite sampled group id to avoid duplicated with gt group ids
        gid_map = {}
        max_gt_gid = np.max(gt_group_ids)
        sampled_gid = max_gt_gid + 1
        for s in sampled:
            gid = s["group_id"]
            if gid in gid_map:
                s["group_id"] = gid_map[gid]
            else:
                gid_map[gid] = sampled_gid
                s["group_id"] = sampled_gid
                sampled_gid += 1

        num_gt = gt_boxes.shape[0]
        gt_boxes_bv = box_utils.center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, -1]
        )

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        sp_group_ids = np.stack([i["group_id"] for i in sampled], axis=0)
        valid_mask = np.zeros([gt_boxes.shape[0]], dtype=np.bool_)
        valid_mask = np.concatenate(
            [valid_mask, np.ones([sp_boxes.shape[0]], dtype=np.bool_)], axis=0
        )
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        group_ids = np.concatenate([gt_group_ids, sp_group_ids], axis=0)
        if self._enable_global_rot:
            # place samples to any place in a circle.
            preprocess.noise_per_object_v3_(
                boxes,
                None,
                valid_mask,
                0,
                0,
                self._global_rot_range,
                group_ids=group_ids,
                num_try=100,
            )
        sp_boxes_new = boxes[gt_boxes.shape[0] :]
        sp_boxes_bv = box_utils.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, -1]
        )
        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        # coll_mat = collision_test_allbox(total_bv)
        coll_mat = preprocess.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False
        valid_samples = []
        idx = num_gt
        for num in group_num:
            if coll_mat[idx : idx + num].any():
                coll_mat[idx : idx + num] = False
                coll_mat[:, idx : idx + num] = False
            else:
                for i in range(num):
                    if self._enable_global_rot:
                        sampled[idx - num_gt + i]["box3d_lidar"][:2] = boxes[
                            idx + i, :2
                        ]
                        sampled[idx - num_gt + i]["box3d_lidar"][-1] = boxes[
                            idx + i, -1
                        ]
                        sampled[idx - num_gt + i]["rot_transform"] = (
                            boxes[idx + i, -1] - sp_boxes[idx + i - num_gt, -1]
                        )

                    valid_samples.append(sampled[idx - num_gt + i])
            idx += num
        return valid_samples
