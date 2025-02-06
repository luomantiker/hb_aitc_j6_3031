# Copyright (c) Horizon Robotics. All rights reserved.
import copy
from typing import List, Optional

import msgpack
import msgpack_numpy
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from hat.core.data_struct.img_structures import ImgObjDet
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type import PackTypeMapper
from hat.utils.pack_type.utils import get_packtype_from_path
from hat.utils.package_helper import require_packages
from .data_packer import Packer
from .mscoco import transforms as coco_transforms

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

try:
    from torchvision.datasets.voc import VOCDetection
except ImportError:
    VOCDetection = object


__all__ = ["PascalVOC", "VOCDetectionPacker", "VOCFromImage"]

_PASCAL_VOC_LABELS = {
    "aeroplane": (0, "Vehicle"),
    "bicycle": (1, "Vehicle"),
    "bird": (2, "Animal"),
    "boat": (3, "Vehicle"),
    "bottle": (4, "Indoor"),
    "bus": (5, "Vehicle"),
    "car": (6, "Vehicle"),
    "cat": (7, "Animal"),
    "chair": (8, "Indoor"),
    "cow": (9, "Animal"),
    "diningtable": (10, "Indoor"),
    "dog": (11, "Animal"),
    "horse": (12, "Animal"),
    "motorbike": (13, "Vehicle"),
    "person": (14, "Person"),
    "pottedplant": (15, "Indoor"),
    "sheep": (16, "Animal"),
    "sofa": (17, "Indoor"),
    "train": (18, "Vehicle"),
    "tvmonitor": (19, "Indoor"),
}


@OBJECT_REGISTRY.register
class PascalVOC(data.Dataset):  # noqa: D205,D400
    """
    PascalVOC provides the method of reading voc data
    from target pack type.

    Args:
        data_path: The path of packed file.
        transforms: Transforms of voc before using.
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
        self.root = data_path
        self.transforms = transforms
        self.kwargs = {} if pack_kwargs is None else pack_kwargs

        if pack_type is not None:
            self.pack_type = PackTypeMapper[pack_type.lower()]
        else:
            self.pack_type = get_packtype_from_path(data_path)

        self.pack_file = self.pack_type(
            self.root, writable=False, **self.kwargs
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()

    def __getitem__(self, index):
        raw_data = self.pack_file.read(self.samples[index])
        raw_data = msgpack.unpackb(
            raw_data, object_hook=msgpack_numpy.decode, raw=True
        )
        sample = raw_data[b"image"].astype(np.uint8)

        labels = copy.deepcopy(raw_data[b"label"])
        data = {}
        data["img"] = sample
        data["ori_img"] = sample
        data["layout"] = "hwc"
        data["color_space"] = "rgb"
        h, w, _ = sample.shape
        data["img_height"], data["img_width"] = h, w
        data["img_shape"] = sample.shape
        data["gt_bboxes"] = labels[:, :4]
        data["gt_classes"] = labels[:, 4]
        data["gt_difficult"] = labels[:, 5]
        data["img_id"] = np.array([index])

        if self.transforms is not None:
            data = self.transforms(data)

            data["gt_labels"] = torch.cat(
                (data["gt_bboxes"], data["gt_classes"].unsqueeze(-1)), -1
            )

        data["structure"] = ImgObjDet(
            img=data["resized_ori_img"]
            if self.transforms is not None
            else data["ori_img"],
            img_id=index,
            layout="hwc",
            color_space=data["color_space"],
            img_height=data["img_height"],
            img_width=data["img_width"],
        )
        return data

    def __len__(self):
        return len(self.samples)


class VOCDetectionPacker(Packer):
    """
    VOCDetectionPacker is used for packing voc dataset to target format.

    Args:
        src_data_dir: Dir of original voc data.
        target_data_dir: Path for packed file.
        split_name: Split name of data, such as trainval and test.
        num_workers: Num workers for reading data using multiprocessing.
        pack_type: The file type for packing.
        num_samples: the number of samples you want to pack. You
            will pack all the samples if num_samples is None.
    """

    @require_packages("torchvision")
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
        if split_name == "trainval":
            ds_2007 = VOCDetection(
                root=src_data_dir,
                year="2007",
                image_set=split_name,
                transforms=coco_transforms,
            )
            ds_2012 = VOCDetection(
                root=src_data_dir,
                year="2012",
                image_set=split_name,
                transforms=coco_transforms,
            )
            self.dataset = data.dataset.ConcatDataset([ds_2007, ds_2012])
        elif split_name == "test":
            self.dataset = VOCDetection(
                root=src_data_dir,
                year="2007",
                image_set="test",
                transforms=coco_transforms,
            )
        else:
            raise NameError(
                "split name must be trainval or test, but get %s"
                % (split_name)
            )

        if num_samples is None:
            num_samples = len(self.dataset)
        super(VOCDetectionPacker, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def pack_data(self, idx):
        image, label = self.dataset[idx]
        image = image

        h, w, c = image.shape
        label = label["annotation"]
        labels = []
        for obj in label["object"]:
            labels.append(
                [
                    (float(obj["bndbox"]["xmin"]) - 1),
                    (float(obj["bndbox"]["ymin"]) - 1),
                    (float(obj["bndbox"]["xmax"]) - 1),
                    (float(obj["bndbox"]["ymax"]) - 1),
                    float(_PASCAL_VOC_LABELS[obj["name"]][0]),
                    float(obj["difficult"]),
                ]
            )
        labels = np.array(labels)
        return msgpack.packb(
            {"image": image, "label": labels}, default=msgpack_numpy.encode
        )


@OBJECT_REGISTRY.register
class VOCFromImage(VOCDetection):
    """VOC from image by torchvision.

    The params of VOCFromImage is same as params of
    torchvision.dataset.VOCDetection.
    """

    @require_packages("torchvision")
    def __init__(self, size=416, *args, **kwargs):
        super(VOCFromImage, self).__init__(*args, **kwargs)
        self.size = size

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert("RGB")
        label = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())

        w, h = img.size
        label = label["annotation"]
        labels = []
        for obj in label["object"]:
            labels.append(
                [
                    (float(obj["bndbox"]["xmin"]) - 1),
                    (float(obj["bndbox"]["ymin"]) - 1),
                    (float(obj["bndbox"]["xmax"]) - 1),
                    (float(obj["bndbox"]["ymax"]) - 1),
                    float(_PASCAL_VOC_LABELS[obj["name"]][0]),
                    float(obj["difficult"]),
                ]
            )
        labels = np.array(labels)

        gt_bboxes = labels[:, :4]
        gt_bboxes[:, 0] = gt_bboxes[:, 0] * self.size / w
        gt_bboxes[:, 1] = gt_bboxes[:, 1] * self.size / h
        gt_bboxes[:, 2] = gt_bboxes[:, 2] * self.size / w
        gt_bboxes[:, 3] = gt_bboxes[:, 3] * self.size / h
        gt_labels = labels[:, 4]
        gt_difficults = labels[:, 5]

        data = {
            "img": img,
            "ori_img": np.array(img),
            "gt_bboxes": gt_bboxes,
            "gt_classes": gt_labels,
            "gt_difficult": gt_difficults,
            "layout": "hwc",
            "color_space": "rgb",
        }

        if self.transforms is not None:
            data = self.transforms(data)

        return data
