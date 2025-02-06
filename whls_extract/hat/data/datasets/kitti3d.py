# Copyright (c) Horizon Robotics. All rights reserved.

import os
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import msgpack
import msgpack_numpy
import numpy as np
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm

from hat.core.box_np_ops import (
    box_camera_to_lidar,
    change_box3d_center_,
    points_in_rbbox,
    remove_outside_points,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type import PackTypeMapper
from hat.utils.pack_type.utils import get_packtype_from_path
from .data_packer import Packer

__all__ = [
    "Kitti3DReader",
    "Kitti3DDetection",
    "Kitti3D",
    "Kitti3DDetectionPacker",
]

KITTI_DICT = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


def remove_dontcare(annos):
    filtered_annotations = {}
    valid_indices = [i for i, x in enumerate(annos["name"]) if x != "DontCare"]
    for key in annos.keys():
        filtered_annotations[key] = annos[key][valid_indices]
    return filtered_annotations


class Kitti3DReader(object):
    """Kitti3D dataset processor.

    Args:
        data_dir: Root directory path of Kitti3D dataset.
          And the directory structure of `data_dir` should be like this:
          ```
          |--- data_dir
          |   |--- ImageSets
          |   |   |--- train.txt
          |   |   |--- val.txt
          |   |   |--- ...
          |   |--- training
          |   |   |--- calib
          |   |   |--- image_2
          |   |   |--- label_2
          |   |   |--- velodyne
          |   |--- testing
          |   |   |--- ...
          ```
        split: Dataset split, in ["train", "val", "test"].
        num_point_feature: Number of feature in points, default 4 (x, y, z, r).
    """

    def __init__(
        self,
        data_dir: str,
        split_name: str = "train",
        num_point_feature: int = 4,
    ):

        assert split_name in [
            "train",
            "val",
            "test",
        ], f"`split_name` should be one of ['train', 'val', 'test'], but get {split_name}"  # noqa E501

        self.data_dir = data_dir
        self.split = split_name

        self.sample_ids = self.get_split_img_ids()
        self.num_point_feature = num_point_feature
        self.test_mode = True if split_name == "val" else False
        self.split_dir = "training" if self.split != "test" else "testing"

    def get_split_img_ids(self) -> List[int]:
        """Get all index of split dataset.

        Returns:
            List[int]: All index of split dataset.
        """
        split_txt = os.path.join(
            self.data_dir, "ImageSets", f"{self.split}.txt"
        )
        assert os.path.exists(split_txt), f"{split_txt} does not exist."
        with open(split_txt, "r") as f:
            lines = f.readlines()
        return [int(line) for line in lines]

    def get_calib(self, index: int, extend_matrix: bool = True) -> Dict:
        """Get the calibration information of one sample.

        Args:
            index: Int value in sample name. For example,
                the `index` value of sample '000026.bin' will be `int(26)`.
            extend_matrix: Whether to pad calibration matrix from
                shape (3, 4) to (4,4).

        Returns:
            Dict: Calibration info.
        """
        dir_calib = os.path.join(self.data_dir, self.split_dir, "calib")
        calib_path = os.path.join(dir_calib, "{:06d}.txt".format(index))
        assert os.path.exists(calib_path), f"{calib_path} does not exists."

        with open(calib_path, "r") as f:
            lines = f.readlines()

        # Calibration info
        p0 = np.array(
            [item for item in lines[0].split(" ")[1:13]],  # noqa C416
            dtype=np.float64,
        ).reshape(3, 4)
        p1 = np.array(
            [item for item in lines[1].split(" ")[1:13]],  # noqa C416
            dtype=np.float64,
        ).reshape(3, 4)
        p2 = np.array(
            [item for item in lines[2].split(" ")[1:13]],  # noqa C416
            dtype=np.float64,
        ).reshape(3, 4)
        p3 = np.array(
            [item for item in lines[3].split(" ")[1:13]],  # noqa C416
            dtype=np.float64,
        ).reshape(3, 4)

        r0_rect = np.array(
            [item for item in lines[4].split(" ")[1:10]],  # noqa C416
            dtype=np.float64,
        ).reshape(3, 3)

        tr_velo_to_cam = np.array(
            [item for item in lines[5].split(" ")[1:13]],  # noqa C416
            dtype=np.float64,
        ).reshape(3, 4)
        tr_imu_to_velo = np.array(
            [item for item in lines[6].split(" ")[1:13]],  # noqa C416
            dtype=np.float64,
        ).reshape(3, 4)

        if extend_matrix:
            p0 = self._extend_matrix(p0)
            p1 = self._extend_matrix(p1)
            p2 = self._extend_matrix(p2)
            p3 = self._extend_matrix(p3)

            tr_velo_to_cam = self._extend_matrix(tr_velo_to_cam)
            tr_imu_to_velo = self._extend_matrix(tr_imu_to_velo)

            r0_rect_extend = np.eye(4, dtype=r0_rect.dtype)
            r0_rect_extend[:3, :3] = r0_rect
            r0_rect = r0_rect_extend

        return {
            "sample_idx": index,
            "P0": p0,
            "P1": p1,
            "P2": p2,
            "P3": p3,
            "R0_rect": r0_rect,
            "Tr_velo_to_cam": tr_velo_to_cam,
            "Tr_imu_to_velo": tr_imu_to_velo,
        }

    def _extend_matrix(self, mat: np.ndarray) -> np.ndarray:
        mat = np.concatenate([mat, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
        return mat

    def get_img(self, index: int) -> Dict:
        """Get the image information of one sample.

        Args:
            index: Int value in sample name.

        Returns:
            Dict: Image info.
        """
        dir_imgs = os.path.join(self.data_dir, self.split_dir, "image_2")
        img_path = os.path.join(dir_imgs, "{:06d}.png".format(index))
        assert os.path.exists(img_path), f"{img_path} does not exists."

        image = np.asarray(Image.open(img_path).convert("RGB"))
        image_shape = image.shape  # (H, W, C)

        img_info = {
            "image_idx": index,
            "image_shape": image_shape,
            "image": image,
            "image_path": os.path.relpath(img_path, self.data_dir),
        }

        return img_info

    def get_ponitcloud_from_bin(
        self,
        index: int,
        remove_outside: bool = False,
    ) -> np.ndarray:
        """Get the points cloud data of one sample.

        Args:
            index: Int value in sample name.

        Returns:
            Points cloud data.
        """
        dir_velodyne = os.path.join(self.data_dir, self.split_dir, "velodyne")
        velodyne_path = os.path.join(dir_velodyne, "{:06d}.bin".format(index))
        assert os.path.exists(
            velodyne_path
        ), f"{velodyne_path} does not exists."
        points = np.fromfile(
            velodyne_path, dtype=np.float32, count=-1
        ).reshape((-1, 4))

        point_cloud = {
            "num_features": self.num_point_feature,
            "velodyne_path": os.path.relpath(velodyne_path, self.data_dir),
            "points": points,
        }

        if remove_outside:
            image_info = self.get_img(index)
            calib_info = self.get_calib(index)
            reduced_points = self.generate_reduced_pointcloud(
                points=points,
                rect=calib_info["R0_rect"],
                Trv2c=calib_info["Tr_velo_to_cam"],
                P2=calib_info["P2"],
                image_shape=image_info["image_shape"],
            )
            point_cloud["points"] = reduced_points

        return point_cloud

    def get_label_annotation(
        self,
        index: int,
        add_difficulty: bool = True,
        add_num_points_in_gt: bool = True,
    ) -> Dict:
        """Get the annotation of one sample.

        Args:
            index: Int value in sample name.

        Returns:
            Dict: annotations.
        """
        dir_labels = os.path.join(self.data_dir, self.split_dir, "label_2")
        label_path = os.path.join(dir_labels, "{:06d}.txt".format(index))
        assert os.path.exists(label_path), f"{label_path} does not exists."

        with open(label_path, "r") as f:
            lines = f.readlines()

        content = [line.strip().split(" ") for line in lines]
        num_objects = len([x[0] for x in content if x[0] != "DontCare"])

        annotations = {}
        annotations["name"] = np.array([x[0] for x in content])
        num_gt = len(annotations["name"])
        annotations["truncated"] = np.array([float(x[1]) for x in content])
        annotations["occluded"] = np.array([int(x[2]) for x in content])
        annotations["alpha"] = np.array([float(x[3]) for x in content])
        annotations["bbox"] = np.array(
            [[float(info) for info in x[4:8]] for x in content]
        ).reshape(-1, 4)
        # dimensions will convert hwl format to standard lhw(camera) format.
        annotations["dimensions"] = np.array(
            [[float(info) for info in x[8:11]] for x in content]
        ).reshape(-1, 3)[:, [2, 0, 1]]
        annotations["location"] = np.array(
            [[float(info) for info in x[11:14]] for x in content]
        ).reshape(-1, 3)
        annotations["rotation_y"] = np.array([float(x[14]) for x in content])

        if content and len(content[0]) == 16:  # have score
            annotations["score"] = np.array([float(x[15]) for x in content])
        else:
            annotations["score"] = np.zeros((annotations["bbox"].shape[0],))

        idx = list(range(num_objects)) + [-1] * (num_gt - num_objects)
        annotations["index"] = np.array(idx, dtype=np.int32)
        annotations["group_ids"] = np.arange(num_gt, dtype=np.int32)

        # add difficulty
        if add_difficulty:
            difficulty = self.judge_difficulty(
                truncated=annotations["truncated"],
                occluded=annotations["occluded"],
                bbox=annotations["bbox"],
            )
            annotations["difficulty"] = difficulty

        if add_num_points_in_gt:
            point_cloud_info = self.get_ponitcloud_from_bin(
                index, remove_outside=True
            )
            reduced_points = point_cloud_info["points"]
            calib_info = self.get_calib(index)
            num_points_in_gt = self.calculate_num_points_in_gt(
                points=reduced_points,
                r0_rect=calib_info["R0_rect"],
                tr_velo_to_cam=calib_info["Tr_velo_to_cam"],
                dimensions=annotations["dimensions"],
                location=annotations["location"],
                rotation_y=annotations["rotation_y"],
                name=annotations["name"],
            )
            annotations["num_points_in_gt"] = num_points_in_gt.astype(np.int32)

        return annotations

    def generate_reduced_pointcloud(
        self,
        points: np.ndarray,
        rect: np.ndarray,
        Trv2c: np.ndarray,
        P2: np.ndarray,
        image_shape: np.ndarray,
    ) -> np.ndarray:
        """Generate reduced pointcloud.

        Args:
            points: Point cloud, shape=[N, 3] or shape=[N, 4].
            rect: matrix rect, shape=[4, 4].
            Trv2c: Translate matrix vel2cam, shape=[4, 4].
            P2: Project matrix, shape=[4, 4].
            image_shape: Image shape, (H, W, ...) format.

        Returns:
            Point clouds are distributed inside the image range.
        """

        return remove_outside_points(
            points,
            rect,
            Trv2c,
            P2,
            image_shape,
        )

    def judge_difficulty(
        self,
        truncated: np.ndarray,
        occluded: np.ndarray,
        bbox: np.ndarray,
    ) -> np.ndarray:
        """Judge whether bbox is difficult to detect.

        Args:
            truncated: Truncated info from kitti3d label.
            occluded: occluded info from kitti3d label.
            bbox: 2D bounding boxes from kitti3d label.

        Returns:
            Difficult level to detection.
        """

        height = bbox[:, 3] - bbox[:, 1]

        min_heights = [40, 25, 25]
        max_occlusion = [0, 1, 2]
        max_truncation = [0.15, 0.30, 0.50]
        difficultys = []
        for h, o, t in zip(height, occluded, truncated):
            difficulty = -1
            for i in range(2, -1, -1):
                if (
                    h > min_heights[i]
                    and o <= max_occlusion[i]
                    and t <= max_truncation[i]
                ):
                    difficulty = i
            difficultys.append(difficulty)
        return np.array(difficultys, dtype=np.int64)

    def calculate_num_points_in_gt(
        self,
        points: np.ndarray,
        r0_rect: np.ndarray,
        tr_velo_to_cam: np.ndarray,
        dimensions: np.ndarray,
        location: np.ndarray,
        rotation_y: np.ndarray,
        name: Union[List[str], np.ndarray],
    ) -> np.ndarray:
        """Calculate point cloud number in ground truth.

        Args:
            points: Point cloud, shape=[N, 3] or shape=[N, 4].
            r0_rect: matrix rect, shape=[4, 4].
            tr_velo_to_cam: Translate matrix vel2cam, shape=[4, 4].
            dimensions: 3D object dimensions (height, width, length).
            location: 3D objection location (x, y, z) in camera coordinates.
            rotation_y: Rotation ry around Y-axis in camera coordinates.
            name: Object class name.

        Returns:
            Point cloud number in ground truth.
        """
        valid_name = [item for item in name if item != "DontCare"]
        num_valid_obj = len(valid_name)

        loc = location[:num_valid_obj]
        dims = dimensions[:num_valid_obj]
        rot_y = rotation_y[:num_valid_obj]

        gt_boxes_camera = np.concatenate(
            [loc, dims, rot_y[..., np.newaxis]], axis=1
        )
        gt_boxes_lidar = box_camera_to_lidar(
            gt_boxes_camera,
            r0_rect,
            tr_velo_to_cam,
        )
        indices = points_in_rbbox(points[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(dimensions) - num_valid_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])]
        )
        return num_points_in_gt

    def get_infos(self) -> Dict:
        """Read data and label in from origin kitti3d dataset."""
        data_infos = []
        for sample_id in tqdm(self.sample_ids):
            info = {}

            point_info = self.get_ponitcloud_from_bin(sample_id)
            image_info = self.get_img(sample_id)
            calib_info = self.get_calib(sample_id)

            # pop
            image_info.pop("image")
            point_info.pop("points")

            info["image"] = image_info
            info["point_cloud"] = point_info
            info["calib"] = calib_info

            if self.split in ["train", "val"]:
                anno_info = self.get_label_annotation(sample_id)
                info["annos"] = anno_info
            data_infos.append(info)
        return data_infos

    def get_groundruth_database(
        self,
        db_path=None,
        dbinfo_path=None,
        used_classes=None,
    ) -> None:
        """Generate ground truth info and dump to pkl.

        Args:
            db_path: Dir to save ground truth bin file.
            dbinfo_path: Path of pkl where to dump ground truth info.
            used_classes: Class names to filter.
        """

        assert self.split == "train"
        if db_path is None:
            db_path = os.path.join(self.data_dir, "kitti3d_gt_database")
            if not os.path.exists(db_path):
                os.makedirs(db_path)
        if dbinfo_path is None:
            dbinfo_path = os.path.join(
                self.data_dir, "kitti3d_dbinfos_train.pkl"
            )

        all_db_infos = {}
        group_counter = 0
        for sample_id in tqdm(self.sample_ids):

            point_info = self.get_ponitcloud_from_bin(
                sample_id, remove_outside=True
            )
            image_info = self.get_img(sample_id)
            calib_info = self.get_calib(sample_id)
            anno_info = self.get_label_annotation(sample_id)
            anno_info = remove_dontcare(anno_info)

            # camera: [x, y, z, l, h, w, rot_y]
            gt_boxes = np.concatenate(
                [
                    anno_info["location"],
                    anno_info["dimensions"],
                    anno_info["rotation_y"][..., np.newaxis],
                ],
                axis=1,
            ).astype(np.float32)

            # lidar: [x_liar, y_lidar, z_lidar, w, l, h, rot_y]
            gt_boxes = box_camera_to_lidar(
                gt_boxes, calib_info["R0_rect"], calib_info["Tr_velo_to_cam"]
            )
            # only center format is allowed. so we need to convert
            # kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
            change_box3d_center_(gt_boxes, [0.5, 0.5, 0], [0.5, 0.5, 0.5])

            num_obj = gt_boxes.shape[0]
            names = anno_info["name"]
            image_idx = image_info["image_idx"]
            points = point_info["points"]
            point_indices = points_in_rbbox(points, gt_boxes)

            if "difficulty" in anno_info:
                difficulty = anno_info["difficulty"]
            else:
                difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)

            group_dict = {}
            group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
            if "group_ids" in anno_info:
                group_ids = anno_info["group_ids"]
            else:
                group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)

            for i in range(num_obj):
                filename = f"{image_idx}_{names[i]}_{i}.bin"
                filepath = os.path.join(db_path, filename)
                gt_points = points[point_indices[:, i]]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, "w") as f:
                    gt_points[:, : self.num_point_feature].tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_dump_path = str(
                        f"{os.path.basename(db_path)}/{filename}"
                    )

                    db_info = {
                        "name": names[i],
                        "path": db_dump_path,
                        "image_idx": image_idx,
                        "gt_idx": i,
                        "box3d_lidar": gt_boxes[i],
                        "num_points_in_gt": gt_points.shape[0],
                        "difficulty": difficulty[i],
                    }
                    local_group_id = group_ids[i]
                    if local_group_id not in group_dict:
                        group_dict[local_group_id] = group_counter
                        group_counter += 1
                    db_info["group_id"] = group_dict[local_group_id]
                    if "score" in anno_info:
                        db_info["score"] = anno_info["score"][i]
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        for k, v in all_db_infos.items():
            print(f"load {len(v)} {k} database infos")

        with open(dbinfo_path, "wb") as f:
            pickle.dump(all_db_infos, f)


@OBJECT_REGISTRY.register
class Kitti3DDetection(data.Dataset):
    """Kitti 3D Detection Dataset.

    Args:
        source_path: Root directory where images are downloaded to.
        split_name: Dataset split, 'train' or 'val'.
        transforms: A function transform that takes input
            sample and its target as entry and returns a transformed version.
        num_point_feature: Number of feature in points, default 4 (x, y, z, r).
    """

    def __init__(
        self,
        source_path: str,
        split_name: str,
        transforms: Optional[Callable] = None,
        num_point_feature: int = 4,
    ) -> None:
        super(Kitti3DDetection, self).__init__()

        assert split_name in [
            "train",
            "val",
        ], f"`split_name` should one of ['train', 'val'], but get {split_name}"
        self.source_path = source_path
        self.split_name = split_name
        self.transforms = transforms
        self.num_point_feature = num_point_feature
        self.test_mode = True if split_name == "val" else False

        self.kitti_pkl_infos, self.indexes = self._gen_indexes()

    def _gen_indexes(self):
        kitti_pkl_file = os.path.join(
            self.source_path, f"kitti3d_infos_{self.split_name}.pkl"
        )

        with open(kitti_pkl_file, "rb") as f:
            kitti_pkl_infos = pickle.load(f)
        indexes = list(range(len(kitti_pkl_infos)))
        return kitti_pkl_infos, indexes

    def __len__(self):
        return len(self.kitti_pkl_infos)

    def __getitem__(self, idx):
        index = self.indexes[idx]
        kitti_info = self.kitti_pkl_infos[index]

        image_info = kitti_info["image"]
        point_cloud_info = kitti_info["point_cloud"]
        calib_info = kitti_info["calib"]
        annos_info = kitti_info["annos"]

        # read image
        img_path = image_info["image_path"]
        img_path = os.path.join(self.source_path, img_path)
        image = np.asarray(Image.open(img_path).convert("RGB"))
        image_info["image"] = image

        # read points
        point_cloud_info["pointclould_num_features"] = np.asarray(
            point_cloud_info["num_features"], np.int64
        )
        # get velodyne points, the points outside of image are removed
        velodyne_path = os.path.join(
            self.source_path,
            os.path.dirname(point_cloud_info["velodyne_path"]) + "_reduced",
            os.path.basename(point_cloud_info["velodyne_path"]),
        )
        points = np.fromfile(str(velodyne_path), dtype=np.float32, count=-1)
        point_cloud_info["points"] = points.reshape(-1, self.num_point_feature)

        # remove 'DoneCare'
        annos_info = remove_dontcare(annos_info)

        locs = annos_info["location"]
        dims = annos_info["dimensions"]  # lhw
        rots = annos_info["rotation_y"]
        gt_bboxes = np.concatenate(
            [locs, dims, rots[..., np.newaxis]], axis=1
        ).astype(
            np.float32
        )  # [x, y, z, l, h, w, rot_y]

        # [x, y, z, l, h, w, rot_y] -> [x_liar, y_lidar, z_lidar, w, l, h, rot_y]  # noqa E501
        gt_bboxes = box_camera_to_lidar(
            gt_bboxes, calib_info["R0_rect"], calib_info["Tr_velo_to_cam"]
        )

        # only center format is allowed. so we need to convert
        # kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
        # (x, y, z+0.5h, w, l, h, rot)
        change_box3d_center_(gt_bboxes, [0.5, 0.5, 0], [0.5, 0.5, 0.5])

        sample_info = {}

        sample_info["lidar"] = {
            "points": point_cloud_info["points"],
            "annotations": {
                "boxes": gt_bboxes,
                "names": annos_info["name"],
            },
        }
        sample_info["cam"] = {
            "annotations": {
                "boxes": annos_info["bbox"],
                "names": annos_info["name"],
            }
        }

        sample_info["calib"] = calib_info
        sample_info["metadata"] = {
            "img": image_info["image"],
            "num_point_features": self.num_point_feature,
            "image_idx": image_info["image_idx"],
            "image_shape": image_info["image"].shape,
            "token": str(image_info["image_idx"]),
        }
        sample_info["metadata"].update(annos_info)
        sample_info["mode"] = "val" if self.test_mode else "train"

        if self.transforms:
            sample_info = self.transforms(sample_info)

        return sample_info


@OBJECT_REGISTRY.register
class Kitti3D(data.Dataset):  # noqa: D205,D400
    """Kitti3D provides the method of reading kitti3d data
    from target pack type.

    Args:
        data_path: The path of LMDB file.
        transforms: Transforms of voc before using.
        pack_type: The pack type.
        pack_kwargs: Kwargs for pack type.
    """

    def __init__(
        self,
        data_path: str,
        num_point_feature: int = 4,
        transforms: Optional[List] = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
    ):
        self.root = data_path
        self.transforms = transforms
        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        self.num_point_feature = num_point_feature
        self.pack_type = pack_type

        if self.pack_type is not None:
            self.pack_type = PackTypeMapper(pack_type.lower())
        else:
            self.pack_type = get_packtype_from_path(data_path)

        self.pack_file = self.pack_type(
            self.root, writable=False, **self.kwargs
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        raw_data = self.pack_file.read(self.samples[item])
        raw_data = msgpack.unpackb(
            raw_data,
            object_hook=msgpack_numpy.decode,
            raw=True,
        )

        sample_info = self._parse_data(raw_data)

        if self.transforms is not None:
            sample_info = self.transforms(sample_info)
        return sample_info

    def _parse_data(self, raw_data):
        """Read encoded data from lmdb and parse to formatted data.

        Args:
            raw_data: Encoded data read from lmdb.

        Returns:
            Formatted data.
        """

        lidar_info = raw_data[b"lidar"]
        metadata_info = raw_data[b"metadata"]
        calib_info = raw_data[b"calib"]
        camera_info = raw_data[b"cam"]

        res = {
            "lidar": {
                "type": "lidar",
                "points": np.array(
                    lidar_info[b"points"], dtype=np.float32
                ).reshape((-1, self.num_point_feature)),
                "annotations": {
                    "boxes": np.array(
                        lidar_info[b"annotations"][b"boxes"], dtype=np.float32
                    ),
                    "names": np.array(lidar_info[b"annotations"][b"names"]),
                },
            },
            "metadata": {
                # "image_prefix": self.root_dir,
                "num_point_features": self.num_point_feature,
                # annotation info
                "image_idx": metadata_info[b"image_idx"],
                "image_shape": metadata_info[b"image_shape"],
                "token": metadata_info[b"token"].decode("utf-8"),
                "name": np.array(metadata_info[b"name"]),  # noqa
                "truncated": metadata_info[b"truncated"],
                "occluded": metadata_info[b"occluded"],
                "alpha": metadata_info[b"alpha"],
                "bbox": metadata_info[b"bbox"],
                "dimensions": metadata_info[b"dimensions"],
                "location": metadata_info[b"location"],
                "rotation_y": metadata_info[b"rotation_y"],
                "category_id": self._category_to_id(
                    np.array(metadata_info[b"name"])
                ),
            },
            "calib": {
                "P0": np.array(calib_info[b"P0"], dtype=np.float32).reshape(
                    (4, 4)
                ),
                "P1": np.array(calib_info[b"P1"], dtype=np.float32).reshape(
                    (4, 4)
                ),
                "P2": np.array(calib_info[b"P2"], dtype=np.float32).reshape(
                    (4, 4)
                ),
                "P3": np.array(calib_info[b"P3"], dtype=np.float32).reshape(
                    (4, 4)
                ),
                "R0_rect": np.array(
                    calib_info[b"R0_rect"], dtype=np.float32
                ).reshape((4, 4)),
                "Tr_velo_to_cam": np.array(
                    calib_info[b"Tr_velo_to_cam"], dtype=np.float32
                ).reshape((4, 4)),
                "Tr_imu_to_velo": np.array(
                    calib_info[b"Tr_imu_to_velo"], dtype=np.float32
                ).reshape((4, 4)),
            },
            "cam": {
                "annotations": {
                    "bbox": np.array(
                        camera_info[b"annotations"][b"boxes"], dtype=np.float32
                    ),
                    "names": np.array(camera_info[b"annotations"][b"names"]),
                }
            },
            "mode": raw_data[b"mode"].decode("utf-8"),
        }
        return res

    def _category_to_id(self, names):
        ids = [
            KITTI_DICT.get(name) for name in names if name in KITTI_DICT.keys()
        ]
        return np.array(ids)


class Kitti3DDetectionPacker(Packer):  # noqa: D205,D400
    """Kitti3DDetectionPacker is used for converting
    kitti3D dataset to target DataType format.

    Args:
        src_data_dir: The dir of original kitti2D data.
        target_data_dir: Path for LMDB file.
        split_name: Dataset split, 'train' or 'val'.
        num_workers: The num workers for reading data
            using multiprocessing.
        pack_type: The file type for packing.
        num_samples: the number of samples you want to pack. You
            will pack all the samples if num_samples is None.
    """

    def __init__(
        self,
        src_data_dir: str,
        target_data_dir: str,
        split_name: str,
        num_workers: int,
        pack_type: str,
        num_samples: Optional[int] = None,
        **kwargs,
    ):
        self.dataset = Kitti3DDetection(
            source_path=src_data_dir,
            split_name=split_name,
        )

        self.source_path = src_data_dir
        if num_samples is None:
            num_samples = len(self.dataset)
        super(Kitti3DDetectionPacker, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def pack_data(self, idx):
        info = self.dataset[idx]
        return msgpack.packb(info, default=msgpack_numpy.encode)


def create_reduced_point_cloud(
    root_path: str, save_path: Optional[str] = None
):
    """Generate reduced point cloud data.

    Args:
        root_path: Root path of dataset.
        save_path: Path to save reduced point cloud data.
    """
    print("Generate reduced point cloud. this may take several minutes.")

    for split in ["train", "val", "test"]:
        reader = Kitti3DReader(data_dir=root_path, split_name=split)
        kitti3d_infos = reader.get_infos()

        for info in tqdm(kitti3d_infos):
            pc_info = info["point_cloud"]
            image_info = info["image"]
            calib_info = info["calib"]
            velodyne_path = pc_info["velodyne_path"]
            points_v = np.fromfile(
                os.path.join(root_path, velodyne_path),
                dtype=np.float32,
                count=-1,
            ).reshape([-1, 4])
            reduced_points = reader.generate_reduced_pointcloud(
                points=points_v,
                rect=calib_info["R0_rect"],
                Trv2c=calib_info["Tr_velo_to_cam"],
                P2=calib_info["P2"],
                image_shape=image_info["image_shape"],
            )
            if save_path is None:
                velodyne_path = Path(velodyne_path)
                Path.mkdir(
                    (
                        root_path
                        / velodyne_path.parent.parent
                        / f"{velodyne_path.parent.stem}_reduced"
                    ),
                    exist_ok=True,
                )
                save_filename = (
                    root_path
                    / velodyne_path.parent.parent
                    / f"{velodyne_path.parent.stem}_reduced"
                ) / velodyne_path.name
            else:
                save_filename = str(Path(save_path) / velodyne_path.name)
            with open(save_filename, "w") as f:
                reduced_points.tofile(f)


def create_kitti_info_file(root_path: str, save_path: Optional[str] = None):
    """Read origin kitti3d data and dump useful info to pkl.

    Args:
        root_path: Root path of dataset.
        save_path: Path to save pkl.
    """

    print("Generate info. this may take several minutes.")
    train_val_info = []
    if save_path is None:
        save_path = root_path
    else:
        if not os.path.exists(save_path):
            os.makedirs()
    for split in ["train", "val", "test"]:
        # create pkl
        print(f"Start generate kitti_info_{split}")
        reader = Kitti3DReader(data_dir=root_path, split_name=split)
        kitti3d_infos = reader.get_infos()
        filename = os.path.join(save_path, f"kitti3d_infos_{split}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(kitti3d_infos, f)
        print(f"Kitti info {split} file is saved to {filename}")
        if split in ["train", "val"]:
            train_val_info.append(kitti3d_infos)

        #

    filename = os.path.join(save_path, "kitti3d_infos_trainval.pkl")

    with open(filename, "wb") as f:
        pickle.dump(train_val_info[0] + train_val_info[1], f)
    print(f"Kitti info trainval file is saved to {filename}")


def create_groundtruth_database(
    root_path: str,
    db_path: Optional[str] = None,
    dbinfo_path: Optional[str] = None,
):
    """Generate ground truth info and dump to pkl.

    Args:
        root_path: Root dir of dataset.
        db_path: Dir to save ground truth bin file.
        dbinfo_path: Path of pkl where to dump ground truth info.
    """
    reader = Kitti3DReader(data_dir=root_path, split_name="train")
    reader.get_groundruth_database()
