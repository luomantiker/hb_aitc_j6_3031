import json
import os
from typing import List, Optional

import msgpack
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type import PackTypeMapper
from hat.utils.pack_type.utils import get_packtype_from_path
from .data_packer import Packer

__all__ = ["CarfusionPacker", "CarfusionPackData", "CarfusionCroppedData"]

KEYPOINT_NAMES = [
    "Right_Front_wheel",
    "Left_Front_wheel",
    "Right_Back_wheel",
    "Left_Back_wheel",
    "Right_Front_HeadLight",
    "Left_Front_HeadLight",
    "Right_Back_HeadLight",
    "Left_Back_HeadLight",
    # "Exhaust",
    "Right_Front_Top",
    "Left_Front_Top",
    "Right_Back_Top",
    "Left_Back_Top",
    # "Center",
]

FLIP_MAP = {
    "Right_Front_wheel": "Left_Front_wheel",
    "Right_Back_wheel": "Left_Back_wheel",
    "Right_Front_HeadLight": "Left_Front_HeadLight",
    "Right_Back_HeadLight": "Left_Back_HeadLight",
    "Right_Front_Top": "Left_Front_Top",
    "Right_Back_Top": "Left_Back_Top",
}


def _create_ldmk_pairs(names, flip_map):
    name_idx = {names[i]: i for i in range(len(names))}
    ldmk_pairs = [[name_idx[k], name_idx[flip_map[k]]] for k in flip_map]
    return ldmk_pairs


class CarfusionPacker(Packer):
    """Carfusion Dataset Packer.

    Args:
        src_data_dir: The directory path where the source data is located.
        target_data_dir: The path where the packed data will be stored.
        split_name: The name of the dataset split to be packed
                 optional: ('train', 'test').
        num_workers: The number of workers to use for parallel processing.
        pack_type: The type of packing to be performed.
        num_samples: The number of samples to pack. Defaults to None
        **kwargs: Additional keyword arguments for the packing process.
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
        self.data_root = os.path.dirname(src_data_dir)
        anno_json_file = (
            f"{src_data_dir}/simple_anno/keypoints_{split_name}.json"
        )
        with open(anno_json_file, "r") as f:
            self.anno_dict = json.load(f)
        self.img_list = list(self.anno_dict.keys())

        if num_samples is None:
            num_samples = len(self.img_list)
        super(CarfusionPacker, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def pack_data(self, idx):
        key = self.img_list[idx]
        image_path = os.path.join(self.data_root, key)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        h, w, _ = image.shape

        valid_mask = np.array(
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
        )  # No.0, No.9 无效
        valid_mask = valid_mask.astype("bool")

        keypoints = self.anno_dict[key]
        keypoints = np.array(keypoints).reshape(14, 3)
        keypoints = keypoints[valid_mask]

        gt_ldmk = keypoints[:, :2]
        gt_ldmk_attr = keypoints[:, 2]

        gt_ldmk_attr[(gt_ldmk[:, 0] < 0) | (gt_ldmk[:, 0] > w)] = 0
        gt_ldmk_attr[(gt_ldmk[:, 1] < 0) | (gt_ldmk[:, 1] > h)] = 0

        shape_data = np.asarray((h, w), dtype=np.uint16).tobytes()
        image = image.astype(np.uint8).tobytes()
        gt_ldmk = gt_ldmk.astype(np.float32).tobytes()  # 4 * 2 *12
        gt_ldmk_attr = gt_ldmk_attr.astype(np.uint8).tobytes()

        pack_data = shape_data + image + gt_ldmk + gt_ldmk_attr
        pack_block = msgpack.packb(pack_data, use_bin_type=True)

        return pack_block


@OBJECT_REGISTRY.register
class CarfusionPackData(Dataset):
    """Carfusion Dataset of packed lmdb format.

        carfusion is a car keypoints datasets, see
        http://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/cvpr2018/index.html

    Args:
        data_path: The path to the packed dataset.
        transforms: List of data transformations to apply.
        pack_type: The type of packing used for the dataset. here is "lmdb"
        pack_kwargs: Additional keyword arguments for dataset packing.
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
        # self.out_pil = out_pil
        if pack_type is not None:
            self.pack_type = PackTypeMapper[pack_type.lower()]
        else:
            self.pack_type = get_packtype_from_path(data_path)

        self.pack_file = self.pack_type(
            self.data_path, writable=False, **self.kwargs
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()
        self.ldmk_pairs = _create_ldmk_pairs(KEYPOINT_NAMES, FLIP_MAP)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raw_data = self.pack_file.read(self.samples[index])
        raw_data = msgpack.unpackb(raw_data, raw=False)
        h, w = np.frombuffer(raw_data[:4], dtype=np.uint16).astype(np.int64)

        ldmk_size = 12 * 2 * 4
        attr_size = 12
        break1 = ldmk_size + attr_size

        image = np.frombuffer(raw_data[4:-break1], dtype=np.uint8)
        image = image.reshape([h, w, 3])
        ldmk = np.frombuffer(raw_data[-break1:-attr_size], dtype=np.float32)
        ldmk = np.copy(ldmk).reshape([12, 2])
        ldmk_attr = np.frombuffer(raw_data[-attr_size:], dtype=np.uint8)
        ldmk_attr = np.copy(ldmk_attr)

        data = {
            "img": image,
            "ori_img": np.copy(image),
            "layout": "hwc",
            "img_shape": image.shape,
            "gt_ldmk": ldmk,
            "gt_ldmk_attr": ldmk_attr,
            "ldmk_pairs": self.ldmk_pairs,
            "color_space": "rgb",
        }
        if self.transforms:
            data = self.transforms(data)
        return data


@OBJECT_REGISTRY.register
class CarfusionCroppedData(Dataset):
    """Cropped Carfusion Dataset. The car instances are cropped.

        carfusion is a car keypoints datasets, see
        http://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/cvpr2018/index.html

    Args:
        data_path: The path to the dataset.
        anno_json_file: The path to the annotation JSON file in COCO format.
        transforms: List of data transformations to apply. Defaults to None.
    """

    def __init__(
        self, data_path: str, anno_json_file: str, transforms: list = None
    ):
        self.data_path = data_path
        self.anno_json_file = anno_json_file
        self.transforms = transforms
        self.ldmk_pairs = _create_ldmk_pairs(KEYPOINT_NAMES, FLIP_MAP)
        with open(anno_json_file, "r") as f:
            self.anno_dict = json.load(f)
        self.img_list = list(self.anno_dict.keys())

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        key = self.img_list[idx]
        image_path = os.path.join(self.data_path, key)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        h, w, _ = image.shape

        valid_mask = np.array(
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
        )  # No.0, No.9 无效
        valid_mask = valid_mask.astype("bool")
        keypoints = self.anno_dict[key]
        keypoints = np.array(keypoints).reshape(14, 3)
        keypoints = keypoints[valid_mask]

        gt_ldmk = keypoints[:, :2]
        gt_ldmk_attr = keypoints[:, 2]

        gt_ldmk_attr[(gt_ldmk[:, 0] < 0) | (gt_ldmk[:, 0] > w)] = 0
        gt_ldmk_attr[(gt_ldmk[:, 1] < 0) | (gt_ldmk[:, 1] > h)] = 0

        data = {
            "img": image,
            "layout": "hwc",
            "img_shape": image.shape,
            "gt_ldmk": gt_ldmk,
            "gt_ldmk_attr": gt_ldmk_attr,
            "ldmk_pairs": self.ldmk_pairs,
            "color_space": "rgb",
        }
        if self.transforms:
            data = self.transforms(data)
        return data
