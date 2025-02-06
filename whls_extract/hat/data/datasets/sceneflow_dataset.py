# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import os
import re
from typing import List, Optional

import cv2
import msgpack
import numpy as np
import torch.utils.data as data
from PIL import Image

from hat.data.utils import decode_img
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type import PackTypeMapper
from hat.utils.pack_type.utils import get_packtype_from_path
from .data_packer import Packer

__all__ = ["SceneFlow", "SceneFlowPacker", "SceneFlowFromImage"]


def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


def encode_hook(obj):
    if isinstance(obj, np.ndarray):
        if len(obj.shape) == 3:
            if obj.dtype == np.uint8:
                img_bytes = cv2.imencode(".png", obj)[1]
                obj = np.asarray(img_bytes).astype(np.uint8).tobytes()
            else:
                obj = obj.tobytes()
        elif len(obj.shape) == 2:
            obj = obj.tobytes()
        elif len(obj.shape) == 1:
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

        if k == "img_left" or k == "img_right":
            v = decode_img(v, iscolor=cv2.IMREAD_COLOR)
        elif k == "gt_disp":
            v = np.array(np.frombuffer(v, dtype=np.float32)).astype(np.float32)
        else:
            v = _decode_bytes(v)
        new_obj[k] = v
    return new_obj


@OBJECT_REGISTRY.register
class SceneFlow(data.Dataset):  # noqa: D205,D400
    """
    SceneFlow provides the method of reading SceneFlow data
    from target pack type.

    Args:
        data_path: The path of packed file.
        transforms: Transfroms of data before using.
        pack_type: The pack type.
        pack_kwargs: Kwargs for pack type.
    """

    def __init__(
        self,
        data_path: str,
        transforms: Optional[List] = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
    ):

        self.data_path = data_path
        self.transforms = transforms

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
        data["gt_disp"] = data["gt_disp"].reshape(data["img_left"].shape[:2])
        data["img_shape"] = data["img_left"].shape
        data["layout"] = "hwc"
        data["color_space"] = "rgb"
        img_l = data.pop("img_left")
        img_r = data.pop("img_right")
        img = np.concatenate((img_l, img_r), axis=2)
        data["img"] = img
        data["ori_img"] = copy.deepcopy(img)

        if self.transforms is not None:
            data = self.transforms(data)
        return data


@OBJECT_REGISTRY.register
class SceneFlowPacker(Packer):  # noqa: D205,D400
    """
    SceneFlowPacker is used for converting sceneflow dataset
    to target DataType format.

    Args:
        src_data_dir: The dir of original sceneflow data.
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
            data_list = os.path.join(
                src_data_dir, "SceneFlow_finalpass_train.txt"
            )
        if split_name == "test":
            data_list = os.path.join(
                src_data_dir, "SceneFlow_finalpass_test.txt"
            )

        self.dataset = SceneFlowFromImage(
            data_path=src_data_dir,
            data_list=data_list,
        )
        if num_samples is None:
            num_samples = len(self.dataset)
        super(SceneFlowPacker, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )
        self.pop_keys = ["img_shape", "layout", "color_space", "ori_img"]

    def pack_data(self, idx):
        data = self.dataset[idx]
        for pop_key in self.pop_keys:
            data.pop(pop_key)
        imgs = data.pop("img")

        data["img_left"] = imgs[..., :3]
        data["img_right"] = imgs[..., 3:]
        return msgpack.packb(data, default=encode_hook)


@OBJECT_REGISTRY.register
class SceneFlowFromImage(data.Dataset):
    """SceneFlowFromImage which gets img data and gt from the data_path.

    Args:
        data_path: The dir of sceneflow data.
        data_list: The filelist of data.
        transforms: List of transform.
    """

    def __init__(
        self,
        data_path: str,
        data_list: str,
        transforms: Optional[List] = None,
    ):

        self.data_path = data_path
        self.data_list = data_list
        self.transforms = transforms
        (
            self.left_filenames,
            self.right_filenames,
            self.disp_filenames,
        ) = self.load_path(data_list)

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        data = Image.open(filename).convert("RGB")
        return data

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, item):
        left_img = self.load_image(
            os.path.join(self.data_path, self.left_filenames[item])
        )
        right_img = self.load_image(
            os.path.join(self.data_path, self.right_filenames[item])
        )
        disparity = self.load_disp(
            os.path.join(self.data_path, self.disp_filenames[item])
        )
        w, h = left_img.size

        img_l = np.array(left_img)
        img_r = np.array(right_img)
        img = np.concatenate((img_l, img_r), axis=2)

        data = {}
        data["img_name"] = self.left_filenames[item]
        data["layout"] = "hwc"
        data["img_shape"] = img_l.shape
        data["gt_disp"] = disparity
        data["color_space"] = "rgb"
        data["img"] = img
        data["ori_img"] = copy.deepcopy(img)
        if self.transforms is not None:
            data = self.transforms(data)

        return data


def pfm_imread(filename):
    file = open(filename, "rb")
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
