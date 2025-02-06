# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import os
from typing import List, Optional

import cv2
import msgpack
import numpy as np
import torch.utils.data as data

from hat.data.utils import decode_img
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type import PackTypeMapper
from hat.utils.pack_type.utils import get_packtype_from_path
from .data_packer import Packer

__all__ = ["CuLaneDataset", "CuLanePacker", "CuLaneFromImage"]


def encode_hook(obj):
    if isinstance(obj, np.ndarray):
        if len(obj.shape) == 3:
            img_bytes = cv2.imencode(".png", obj)[1]
            obj = np.asarray(img_bytes).astype(np.uint8).tobytes()
        elif len(obj.shape) == 2:
            obj = obj.tobytes()
    return obj


def decode_hook(obj):
    def _decode_bytes(obj):
        if isinstance(obj, bytes):
            obj = obj.decode("utf-8")
        return obj

    new_obj = {}
    for k, v in obj.items():
        k = _decode_bytes(k)

        if k == "img":
            v = decode_img(v, cv2.IMREAD_COLOR)
        elif k == "gt_lines" and v:
            gt_num = len(v)
            for i in range(gt_num):
                v[i] = np.array(
                    np.frombuffer(v[i], dtype=np.float32).reshape(-1, 2)
                ).astype(np.float32)

        else:
            v = _decode_bytes(v)
        new_obj[k] = v
    return new_obj


@OBJECT_REGISTRY.register
class CuLaneDataset(data.Dataset):  # noqa: D205,D400
    """
    CuLaneDataset provides the method of reading CuLaneDataset data
    from target pack type.

    Args:
        data_path: The path of packed file.
        transforms: Transfroms of data before using.
        pack_type: The pack type.
        pack_kwargs: Kwargs for pack type.
        to_rgb: Whether to convert to `rgb` color_space.
    """

    def __init__(
        self,
        data_path: str,
        transforms: Optional[List] = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
        to_rgb: bool = True,
    ):

        self.data_path = data_path
        self.transforms = transforms
        self.to_rgb = to_rgb

        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        if pack_type is not None:
            self.pack_type = PackTypeMapper[pack_type.lower()]
        else:
            self.pack_type = get_packtype_from_path(data_path)

        self.pack_file = self.pack_type(
            self.data_path, writable=False, **self.kwargs
        )
        self.samples = self.pack_file.get_keys()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_data = self.pack_file.read(self.samples[idx])
        data = msgpack.unpackb(raw_data, object_hook=decode_hook, raw=True)
        data["img_shape"] = data["img"].shape
        data["layout"] = "hwc"

        data["ori_gt_lines"] = copy.deepcopy(data["gt_lines"])
        if self.to_rgb and data["color_space"] != "rgb":
            cv2.cvtColor(data["img"], cv2.COLOR_BGR2RGB, data["img"])
            data["color_space"] = "rgb"
        data["ori_img"] = copy.deepcopy(data["img"])

        if self.transforms is not None:
            data = self.transforms(data)
        return data


@OBJECT_REGISTRY.register
class CuLanePacker(Packer):  # noqa: D205,D400
    """
    CuLanePacker is used for converting Culane dataset
    to target DataType format.

    Args:
        src_data_dir: The dir of original culane data.
        target_data_dir: Path for packed file.
        split_name: Split name of data, must be train or test.
        num_workers: Num workers for reading data using multiprocessing.
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
        assert split_name in [
            "train",
            "test",
        ], "split_name must be one of train and test."
        if split_name == "train":
            train_flag = True
        if split_name == "test":
            train_flag = False
        self.dataset = CuLaneFromImage(src_data_dir, train_flag=train_flag)

        if num_samples is None:
            num_samples = len(self.dataset)
        super(CuLanePacker, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )
        self.pop_keys = ["img_shape", "layout", "ori_gt_lines", "ori_img"]

    def pack_data(self, idx):
        data = self.dataset[idx]
        for pop_key in self.pop_keys:
            data.pop(pop_key)
        return msgpack.packb(data, default=encode_hook)


@OBJECT_REGISTRY.register
class CuLaneFromImage(data.Dataset):
    """CuLane dataset which gets img data and gt lines from the data_path.

    Args:
        data_path: The path where the image and gt lines is stored.
        transforms: List of transform.
        to_rgb: Whether to convert to `rgb` color_space.
        train_flag: Whether the data use to train or test.
    """

    def __init__(
        self,
        data_path: str,
        transforms: Optional[List] = None,
        to_rgb: bool = False,
        train_flag: bool = False,
    ):
        self.data_path = data_path

        data_list = self._path_join(self.data_path, "list/test.txt")
        if train_flag:
            data_list = self._path_join(self.data_path, "list/train.txt")
        self.img_infos, self.annotations = self.parser_datalist(data_list)
        self.transforms = transforms
        self.to_rgb = to_rgb

    def _path_join(self, root, name):
        if root == "":
            return name
        if name[0] == "/":
            return os.path.join(root, name[1:])
        else:
            return os.path.join(root, name)

    def parser_datalist(self, data_list):
        img_infos, annotations = [], []
        with open(data_list) as f:
            lines = f.readlines()
            for line in lines:
                img_dir = line.strip()
                img_infos.append(img_dir)
                anno_dir = img_dir.replace(".jpg", ".lines.txt")
                annotations.append(anno_dir)
        return img_infos, annotations

    def load_labels(self, idx):
        anno_dir = self._path_join(self.data_path, self.annotations[idx])
        annos = []
        with open(anno_dir, "r") as anno_f:
            lines = anno_f.readlines()
            for line in lines:
                coords_str = line.strip().split(" ")
                num_point = len(coords_str) // 2
                if num_point < 2:
                    continue
                points = np.zeros([num_point, 2], dtype=np.float32)
                for i in range(num_point):
                    points[i][0] = float(coords_str[2 * i])
                    points[i][1] = float(coords_str[2 * i + 1])
                annos.append(points)
        return annos

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        data = {}
        img_name = self.img_infos[idx]
        img_path = self._path_join(self.data_path, img_name)

        img = cv2.imread(img_path)

        color_space = "bgr"
        if self.to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            color_space = "rgb"

        ori_shape = img.shape
        gt_lines = self.load_labels(idx)

        data["image_name"] = img_name
        data["img"] = img
        data["gt_lines"] = gt_lines
        data["img_shape"] = ori_shape
        data["color_space"] = color_space
        data["layout"] = "hwc"

        data["ori_gt_lines"] = copy.deepcopy(data["gt_lines"])
        data["ori_img"] = copy.deepcopy(data["img"])

        if self.transforms is not None:
            data = self.transforms(data)

        return data
