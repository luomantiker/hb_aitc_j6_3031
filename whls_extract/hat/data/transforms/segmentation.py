# Copyright (c) Horizon Robotics. All rights reserved.

import random
from numbers import Real
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from shapely.geometry.polygon import Polygon
from torch import Tensor
from torch.nn import functional as torchFunctional

try:
    from torchvision.transforms import InterpolationMode, RandomAffine, Resize
    from torchvision.transforms import functional as visionFunctional
except ImportError:
    RandomAffine = object
    Resize = object
    InterpolationMode = None
    visionFunctional = None

try:
    import pycocotools.mask as mask_util
except ImportError:
    mask_util = None

try:
    from torchvision.transforms.functional import get_image_size
except ImportError:
    from torchvision.transforms.functional import (
        _get_image_size as get_image_size,
    )

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages
from .functional_img import imresize_warp_when_nearest
from .functional_target import one_hot

__all__ = [
    "SegRandomCrop",
    "SegReWeightByArea",
    "LabelRemap",
    "SegOneHot",
    "SegResize",
    "SegResizeAffine",
    "SegRandomAffine",
    "Scale",
    "FlowRandomAffineScale",
    "SegRandomCutOut",
    "ReformatLanePolygon",
    "PolygonToMask",
]


@OBJECT_REGISTRY.register
class SegRandomCrop(object):  # noqa: D205,D400
    """Random crop on data with gt_seg label, can only be used for segmentation
     task.

    .. note::
        Affected keys: 'img', 'img_shape', 'pad_shape', 'layout', 'gt_seg'.

    Args:
        size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float, optional): The maximum ratio that single category
            could occupy.
        ignore_index (int, optional): When considering the cat_max_ratio
            condition, the area corresponding to ignore_index will be ignored.
    """

    def __init__(self, size, cat_max_ratio=1.0, ignore_index=255):
        assert size[0] > 0 and size[1] > 0
        self.size = size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, data):
        """Randomly get a crop bounding box."""
        assert data["layout"] in ["hwc", "chw", "hw"]
        if data["layout"] == "chw":
            h, w = data["img"].shape[1:]
        else:
            h, w = data["img"].shape[:2]

        margin_h = max(h - self.size[0], 0)
        margin_w = max(w - self.size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        crop_y1, crop_y2 = offset_h, offset_h + self.size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox, layout):
        assert layout in ["hwc", "chw", "hw"]
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        if layout in ["hwc", "hw"]:
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        else:
            img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        return img

    def crop_polygons(self, polygons, crop_bbox):
        crop_y1, _, crop_x1, _ = crop_bbox
        for polygon in polygons:
            polygon[:, 0] -= crop_x1
            polygon[:, 1] -= crop_y1
        return polygons

    def __call__(self, data):
        # find the right crop_bbox
        crop_bbox = self.get_crop_bbox(data)
        if self.cat_max_ratio < 1.0:
            # repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(data["gt_seg"], crop_bbox, "hw")
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if (
                    len(cnt) > 1
                    and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio
                ):
                    break
                crop_bbox = self.get_crop_bbox(data)

        # crop the image
        img = self.crop(data["img"], crop_bbox, data["layout"])
        data["img"] = img
        data["img_shape"] = img.shape
        data["pad_shape"] = img.shape
        # crop semantic seg
        if "gt_seg" in data:
            data["gt_seg"] = self.crop(data["gt_seg"], crop_bbox, "hw")
        if "gt_polygons" in data:
            data["gt_polygons"] = self.crop_polygons(
                data["gt_polygons"], crop_bbox
            )
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"size={self.size}, "
        repr_str += f"cat_max_ratio={self.cat_max_ratio}, "
        repr_str += f"ignore_index={self.ignore_index}"
        return repr_str


@OBJECT_REGISTRY.register
class SegRandomCenterCrop(SegRandomCrop):  # noqa: D205,D400
    """Random center crop on data with gt_seg label,
    can only be used for segmentation task.

    Args:
        cx_ratio_range: The ration range of crop_offset in x axis.
        cy_ratio_range: The ration range of crop_offset in y axis.
    """

    def __init__(
        self,
        cx_ratio_range: Tuple[float, float],
        cy_ratio_range: Tuple[float, float],
        **kwargs,
    ):
        super(SegRandomCenterCrop, self).__init__(**kwargs)
        assert (
            len(cx_ratio_range) == 2 and len(cy_ratio_range) == 2
        ), "crop center range param is error."
        self.cx_ratio_range = cx_ratio_range
        self.cy_ratio_range = cy_ratio_range

    def get_crop_bbox(self, data):
        """Randomly get a crop bounding box."""
        assert data["layout"] in ["hwc", "chw", "hw"]
        if data["layout"] == "chw":
            img_h, img_w = data["img"].shape[1:]
        else:
            img_h, img_w = data["img"].shape[:2]
        crop_size = (
            list(self.size) if self.size is not None else [img_h, img_w]
        )

        lower_lim_x1 = max(
            0, int(img_w * self.cx_ratio_range[0] - crop_size[1] / 2)
        )
        upper_lim_x1 = max(
            0, int(img_w * self.cx_ratio_range[1] - crop_size[1] / 2)
        )
        lower_lim_y1 = max(
            0, int(img_h * self.cy_ratio_range[0] - crop_size[0] / 2)
        )
        upper_lim_y1 = max(
            0, int(img_h * self.cy_ratio_range[1] - crop_size[0] / 2)
        )

        offset_h = np.random.randint(lower_lim_y1, upper_lim_y1 + 1)
        offset_w = np.random.randint(lower_lim_x1, upper_lim_x1 + 1)

        crop_y1, crop_y2 = offset_h, offset_h + self.size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2


@OBJECT_REGISTRY.register
class ReformatLanePolygon(object):
    def __init__(
        self,
        add_attribute_num: int = 2,
        class_mapping: Optional[Dict] = None,
        iou_thr: float = 0.1,
        class_position: int = 1,
        converts: Optional[Sequence[Dict[str, int]]] = None,
        ignore_index: int = 255,
        remove_ignore: bool = False,
    ):
        """Reformat lane polygons.

        .. note::
            Affected keys: 'gt_polygons', 'gt_labels'.

        Args:
            add_attribute_num: The number of added attributions.
            class_mapping: To convert class indexes.
            iou_thr: IoU threshold to recognize multi-layer lane polygons.
            class_position: The position of class indexes in gt_labels.
            converts: Conditions for converting, consists of 'src_position',
                'src_value', 'target_position' and "target_value'.
            ignore_index: Index of ignore regions.
            remove_ignore: Whether to remove ignore regions.
        """

        self.class_position = class_position
        self.add_attribute_num = add_attribute_num
        self.iou_thr = iou_thr
        self.ignore_index = ignore_index
        self.remove_ignore = remove_ignore

        if class_mapping is None:
            self.class_mapping = {
                0: 0,  # "solid",
                1: 1,  # "dashed",
                2: 2,  # "wide_solid",
                3: 3,  # "wide_dashed",
                4: 0,  # "deceleration_lane",
                5: 0,  # "tidal_lane",
                6: 4,  # "mixed",
                7: 5,  # "Road_teeth",
                8: 6,  # "botts_dots",
                255: 255,
            }
        else:
            self.class_mapping = class_mapping
        # for instance_i, if gt_labels[i][src_position] == src_value,
        # then find the first instance_j that
        #     is_intersect(instance_i, instance_j), and:
        # 1) set gt_labels[j][target_position] == target_value,
        # 2) delete instance_i
        if converts is None:
            self.converts = [
                {
                    "src_position": 1,
                    "src_value": 4,
                    "target_position": -2,
                    "target_value": 1,
                },
                {
                    "src_position": 1,
                    "src_value": 5,
                    "target_position": -1,
                    "target_value": 1,
                },
                {
                    "src_position": 2,
                    "src_value": 1,
                    "target_position": 2,
                    "target_value": 1,
                },
            ]
        else:
            self.converts = converts

    def to_shapely_polygon(self, polygon):
        if len(polygon) <= 2:
            return None
        return Polygon(polygon).buffer(0)

    def is_intersect(self, polygon1, polygon2):
        inter = polygon1.intersection(polygon2).area
        if inter > 0:
            # union = polygon1.union(polygon2).area
            min_area = min(polygon1.area, polygon2.area)
            return 1.0 * inter / min_area > self.iou_thr
        return False

    def is_interest(self, gt_label):
        for convert in self.converts:
            if gt_label[convert["src_position"]] == convert["src_value"]:
                return True
        return False

    def __call__(self, data: Dict) -> Dict:
        """.

        Args:
            data["gt_polygons"] is in the format of
            [
            [[x11,y11],[x12,y12],...],
            [[x21,y21],[x22,y22],...],
            ...,
            [[xn1,yn1],[xn2,yn2],...],
            ]
            data["gt_labels"] is in the format of
            [
            [index1, attr11, ..., attr1m],
            [index2, attr21, ..., attr2m],
            ...,
            [indexn, attrn1, ..., attrnm],
            ]
        """

        gt_polygons = data["gt_polygons"]
        gt_labels = data["gt_labels"]
        ins_num = len(gt_polygons)
        assert ins_num == len(gt_labels), (ins_num, len(gt_labels))
        if ins_num == 0:
            return data
        gt_labels = [
            np.append(lab, [0 for _ in range(self.add_attribute_num)])
            for lab in gt_labels
        ]
        gt_shapely_polygons = [self.to_shapely_polygon(p) for p in gt_polygons]
        keep = [p is not None for p in gt_shapely_polygons]
        if self.remove_ignore:
            keep = [
                k and (gt_label_i[0] != self.ignore_index)
                for k, gt_label_i in zip(keep, gt_labels)
            ]
        for i in range(ins_num):
            if not keep[i]:
                continue
            gt_label_i = gt_labels[i]
            if gt_label_i[0] == self.ignore_index:
                continue
            if not self.is_interest(gt_label_i):
                continue
            gt_shapely_polygon_i = gt_shapely_polygons[i]
            for j in range(i - 1, -1, -1):
                if not keep[j]:
                    continue
                gt_label_j = gt_labels[j]
                if gt_label_j[0] == self.ignore_index:
                    continue
                gt_shapely_polygon_j = gt_shapely_polygons[j]
                if self.is_intersect(
                    gt_shapely_polygon_i, gt_shapely_polygon_j
                ):
                    for convert in self.converts:
                        if (
                            gt_label_i[convert["src_position"]]
                            == convert["src_value"]
                        ):
                            gt_label_j[convert["target_position"]] = convert[
                                "target_value"
                            ]
                        keep[i] = False

        data["gt_polygons"] = [p for p, k in zip(gt_polygons, keep) if k]
        data["gt_labels"] = [lab for lab, k in zip(gt_labels, keep) if k]
        if self.class_mapping is not None:
            for lab in data["gt_labels"]:
                lab[self.class_position] = self.class_mapping[
                    lab[self.class_position]
                ]
        return data


@OBJECT_REGISTRY.register
class PolygonToMask(object):
    @require_packages("pycocotools")
    def __init__(
        self,
        filter_emtpy: bool = True,
        replace_gt_seg: bool = False,
        replace_orig_gt_seg: bool = False,
        add_bbox: bool = False,
    ):
        """Convert ground truth polygons to masks or gt_seg.

        .. note::
            Affected keys: 'gt_polygons', 'gt_masks', 'gt_seg', 'orig_gt_seg',
                'gt_labels' 'gt_bboxes'.

        Args:
            filter_emtpy: Remove polygons with zero area. Default is True.
            replace_gt_seg: Overwrite gt_seg.
            replace_orig_gt_seg: Overwrite orig_gt_seg.
            add_bbox: Add gt_bboxes with boxes generated from polygons.
        """
        self.filter_emtpy = filter_emtpy
        self.replace_gt_seg = replace_gt_seg
        self.replace_orig_gt_seg = replace_orig_gt_seg
        if self.replace_orig_gt_seg:
            assert self.replace_gt_seg
        self.add_bbox = add_bbox

    def _rle_to_mask(self, rle):
        mask = mask_util.decode(rle)
        return mask

    def _rle_to_bbox(self, rle):
        bbox = mask_util.toBbox(rle)
        return bbox

    def _rle_to_area(self, rle):
        if rle is None:
            return 0
        area = mask_util.area(rle)
        return area

    def _polygon_to_rle(self, polygon, h, w):
        if len(polygon) <= 2:
            return None
        rles = mask_util.frPyObjects([polygon.reshape(-1)], h, w)
        rle = mask_util.merge(rles)
        return rle

    def __call__(self, data: Dict) -> Dict:
        assert data["layout"] in ["hwc", "chw", "hw"]
        if data["layout"] == "chw":
            h, w = data["img"].shape[1:]
        else:
            h, w = data["img"].shape[:2]

        gt_rles = [self._polygon_to_rle(p, h, w) for p in data["gt_polygons"]]
        if self.filter_emtpy:
            areas = [self._rle_to_area(r) for r in gt_rles]
            gt_rles = [p for p, a in zip(gt_rles, areas) if a > 0]
            data["gt_polygons"] = [
                p for p, a in zip(data["gt_polygons"], areas) if a > 0
            ]
            if "gt_labels" in data:
                assert len(areas) == len(data["gt_labels"])
                data["gt_labels"] = [
                    lb for lb, a in zip(data["gt_labels"], areas) if a > 0
                ]
        if self.add_bbox:
            data["gt_bboxes"] = np.array(
                [self._rle_to_bbox(rle) for rle in gt_rles]
            )
        if self.replace_gt_seg:
            gt_seg = data["gt_seg"]
            gt_seg *= 0
            for rle, label in zip(gt_rles, data["gt_labels"]):
                mask = self._rle_to_mask(rle)
                index = label[0]
                gt_seg[mask == 1] = index  # set foreground pixels to label[0]
            data["gt_seg"] = gt_seg
            if self.replace_orig_gt_seg:
                data["orig_gt_seg"] = gt_seg
        else:
            data["gt_masks"] = np.array(
                [self._rle_to_mask(rle) for rle in gt_rles]
            )
        return data


@OBJECT_REGISTRY.register
class SegReWeightByArea(object):  # noqa: D205,D400
    """Calculate the weight of each category according to the area of each
    category.

    For each category, the calculation formula of weight is as follows:
    weight = max(1.0 - seg_area / total_area, lower_bound)

    .. note::
        Affected keys: 'gt_seg', 'gt_seg_weight'.

    Args:
        seg_num_classes (int): Number of segmentation categories.
        lower_bound (float): Lower bound of weight.
        ignore_index (int): Index of ignore class.
    """

    def __init__(
        self,
        seg_num_classes,
        lower_bound: int = 0.5,
        ignore_index: int = 255,
    ):
        self.seg_num_classes = seg_num_classes
        self.lower_bound = lower_bound
        self.ignore_index = ignore_index

    def _reweight_by_area(self, gt_seg):
        """Private function to generate weights based on area of semantic."""
        H, W = gt_seg.shape[0], gt_seg.shape[1]
        gt_seg_weight = np.zeros((H, W), dtype=np.float32)
        total_area = (gt_seg != self.ignore_index).sum()
        for ind in range(self.seg_num_classes):
            seg_area = (gt_seg == ind).sum()
            if seg_area > 0:
                gt_seg_weight[gt_seg == ind] = max(
                    1.0 - seg_area / total_area, self.lower_bound
                )
        return gt_seg_weight

    def __call__(self, data):
        if "gt_seg" in data:
            gt_seg_weight = self._reweight_by_area(data["gt_seg"])
            data["gt_seg_weight"] = gt_seg_weight
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"seg_num_classes={self.seg_num_classes}, "
        repr_str += f"lower_bound={self.lower_bound}"
        return repr_str


@OBJECT_REGISTRY.register
class LabelRemap(object):
    r"""
    Remap labels.

    .. note::
        Affected keys: 'gt_seg'.

    Args:
        mapping (Sequence): Mapping from input to output.
    """

    def __init__(self, mapping: Sequence):
        super(LabelRemap, self).__init__()
        if not isinstance(mapping, Sequence):
            raise TypeError(
                "mapping should be a sequence. Got {}".format(type(mapping))
            )
        self.mapping = mapping

    def __call__(self, data: Tensor):
        label = data["gt_seg"]
        if isinstance(label, torch.Tensor):
            mapping = torch.tensor(
                self.mapping, dtype=label.dtype, device=label.device
            )
            data["gt_seg"] = mapping[label.to(dtype=torch.long)]
        else:
            mapping = np.array(self.mapping, dtype=label.dtype)
            data["gt_seg"] = mapping[label]
        return data


@OBJECT_REGISTRY.register
class SegOneHot(object):
    r"""
    OneHot is used for convert layer to one-hot format.

    .. note::
        Affected keys: 'gt_seg'.

    Args:
        num_classes (int): Num classes.
    """

    def __init__(self, num_classes: int):
        super(SegOneHot, self).__init__()
        self.num_classes = num_classes

    def __call__(self, data):
        ndim = data["gt_seg"].ndim
        if ndim == 3 or ndim == 2:
            data["gt_seg"] = torch.unsqueeze(data["gt_seg"], 0)
        data["gt_seg"] = one_hot(data["gt_seg"], self.num_classes)
        if ndim == 3 or ndim == 2:
            data["gt_seg"] = data["gt_seg"][0]
        return data


@OBJECT_REGISTRY.register
class SegResize(Resize):
    """
    Apply resize for both image and label.

    .. note::
        Affected keys: 'img', 'gt_seg'.

    Args:
        size: target size of resize.
        interpolation: interpolation method of resize.

    """

    @require_packages("torchvision")
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        super(SegResize, self).__init__(size, interpolation)

    def forward(self, data):
        data["img"] = super(SegResize, self).forward(data["img"])
        if "gt_seg" in data:
            data["gt_seg"] = super(SegResize, self).forward(data["gt_seg"])

        return data


@OBJECT_REGISTRY.register
class SegResizeAffine(object):
    """Resize image & seg.

    .. note::
        Affected keys: 'img', 'img_shape', 'pad_shape', 'resized_shape',
        'scale_factor', 'gt_seg', 'gt_polygons'.

    Args:
        img_scale: (height, width) or a list of
            [(height1, width1), (height2, width2), ...] for image resize.
        max_scale: The max size of image. If the image's shape > max_scale,
            The image is resized to max_scale
        multiscale_mode: Value must be one of "range" or "value".
            This transform resizes the input image and bbox to same scale
            factor.
            There are 3 multiscale modes:
            'ratio_range' is not None: randomly sample a ratio from the ratio
            range and multiply with the image scale.
            e.g. Resize(img_scale=(400, 500)), multiscale_mode='range',
            ratio_range=(0.5, 2.0)
            'ratio_range' is None and 'multiscale_mode' == "range": randomly
            sample a scale from a range, the length of img_scale[tuple] must be
            2, which represent small img_scale and large img_scale.
            e.g. Resize(img_scale=((100, 200), (400,500)),
            multiscale_mode='range')
            'ratio_range' is None and 'multiscale_mode' == "value": randomly
            sample a scale from multiple scales.
            e.g. Resize(img_scale=((100, 200), (300, 400), (400, 500)),
            multiscale_mode='value')))
        ratio_range: Scale factor range like (min_ratio, max_ratio).
        keep_ratio: Whether to keep the aspect ratio when resizing the image.
    """

    def __init__(
        self,
        img_scale: Union[Sequence[int], Sequence[Sequence[int]]] = None,
        max_scale: Union[Sequence[int], Sequence[Sequence[int]]] = None,
        multiscale_mode: str = "range",
        ratio_range: Tuple[float, float] = None,
        keep_ratio: bool = True,
    ):
        if img_scale is None:
            self.img_scale = img_scale
        else:
            if isinstance(img_scale, (list, tuple)):
                if isinstance(img_scale[0], (tuple, list)):
                    self.img_scale = img_scale
                else:
                    self.img_scale = [img_scale]
            else:
                self.img_scale = [img_scale]
            for value in self.img_scale:
                assert isinstance(value, (tuple, list)), (
                    "you should set img_scale like a tupe/list or a list of "
                    "tuple/list"
                )
        self.max_scale = max_scale
        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range", "max_size"]

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long), max(img_scale_long) + 1
        )
        short_edge = np.random.randint(
            min(img_scale_short), max(img_scale_short) + 1
        )
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, (tuple, list)) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, 0

    @staticmethod
    def max_size(max_scale, origin_shape):
        if max(origin_shape) > max(max_scale):
            resize_scale = max_scale
        else:
            resize_scale = origin_shape

        return resize_scale, 0

    def _random_scale(self, data):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range
            )
        elif self.multiscale_mode == "max_size":
            scale, scale_idx = self.max_size(
                self.max_scale, data["img_shape"][:2]
            )
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        data["scale"] = scale
        data["scale_idx"] = scale_idx

    def _resize_img(self, data):
        h = data["scale"][0]
        w = data["scale"][1]
        img = data["img"]

        resized_img, w_scale, h_scale = imresize_warp_when_nearest(
            img,
            w,
            h,
            data["layout"],
            keep_ratio=self.keep_ratio,
            return_scale=True,
        )
        data["scale_factor"] = np.array(
            [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
        )

        data["img"] = resized_img
        data["img_shape"] = resized_img.shape
        data["resized_shape"] = resized_img.shape
        data["pad_shape"] = resized_img.shape
        data["keep_ratio"] = self.keep_ratio

    def _resize_seg(self, data):
        """Resize semantic segmentation map with ``data['scale']``."""
        h = data["scale"][0]
        w = data["scale"][1]
        resized_seg = imresize_warp_when_nearest(
            data["gt_seg"],
            w,
            h,
            "hw",
            keep_ratio=self.keep_ratio,
            return_scale=False,
            interpolation="nearest",
        )
        data["gt_seg"] = resized_seg

    def _resize_polygons(self, data):
        scale_factor = data["scale_factor"]
        w_scale, h_scale = scale_factor[0], scale_factor[1]
        gt_polygons = data["gt_polygons"]
        for gt_polygon in gt_polygons:
            gt_polygon[:, 0] *= w_scale
            gt_polygon[:, 1] *= h_scale
        data["gt_polygons"] = gt_polygons

    def inverse_transform(
        self,
        inputs: np.ndarray,
        task_type: str,
        inverse_info: Dict[str, Any],
    ):
        """Inverse option of transform to map the prediction to the original image.

        Args:
            inputs: Prediction.
            task_type: support `segmentation` only.
            inverse_info: The transform keyword is the key,
                and the corresponding value is the value.

        """
        assert task_type == "segmentation", task_type
        scale_factor = inverse_info["scale_factor"][:2]
        if isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.detach().cpu().numpy()
        elif isinstance(scale_factor, (tuple, list)):
            scale_factor = np.array(scale_factor)
        else:
            assert isinstance(scale_factor, np.ndarray)
        before_resize_shape = inputs.shape / scale_factor
        out_height, out_width = before_resize_shape
        out_img = cv2.resize(
            inputs,
            (int(out_width), int(out_height)),
            interpolation=cv2.INTER_NEAREST,
        )
        return out_img

    def __call__(self, data):
        self._random_scale(data)
        self._resize_img(data)
        if "gt_seg" in data:
            self._resize_seg(data)
        if "gt_polygons" in data:
            self._resize_polygons(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"img_scale={self.img_scale}, "
        repr_str += f"multiscale_mode={self.multiscale_mode}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"keep_ratio={self.keep_ratio}"
        return repr_str


@OBJECT_REGISTRY.register
class SegRandomAffine(RandomAffine):
    """
    Apply random for both image and label.

    Please refer to :class:`~torchvision.transforms.RandomAffine` for details.

    .. note::
        Affected keys: 'img', 'gt_flow', 'gt_seg'.

    Args:
        label_fill_value (tuple or int, optional): Fill value for label.
            Defaults to -1.
        translate_p: Translate flip probability, range between [0, 1].
        scale_p: Scale flip probability, range between [0, 1].
    """

    @require_packages("torchvision")
    def __init__(
        self,
        degrees: Union[Sequence, float] = 0,
        translate: Tuple = None,
        scale: Tuple = None,
        shear: Union[Sequence, float] = None,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Union[tuple, int] = 0,
        label_fill_value: Union[tuple, int] = -1,
        rotate_p: float = 1.0,
        translate_p: float = 1.0,
        scale_p: float = 1.0,
    ):
        super(SegRandomAffine, self).__init__(
            degrees, translate, scale, shear, interpolation, fill
        )

        self.label_fill_value = label_fill_value
        self.rotate_p = rotate_p
        self.translate_p = translate_p
        self.scale_p = scale_p

    def __call__(self, data: Dict[str, Tensor]):
        img = data["img"]
        img_size = get_image_size(img)

        rotate_flag = np.random.choice(
            [False, True], p=[1 - self.rotate_p, self.rotate_p]
        )
        translate_flag = np.random.choice(
            [False, True], p=[1 - self.translate_p, self.translate_p]
        )
        scale_flag = np.random.choice(
            [False, True], p=[1 - self.scale_p, self.scale_p]
        )
        params = [
            self.degrees,
            self.translate,
            self.scale,
            self.shear,
            img_size,
        ]
        if not rotate_flag:
            params[0] = (0.0, 0.0)
        if not translate_flag:
            params[1] = (0.0, 0.0)
        if not scale_flag:
            params[2] = (1.0, 1.0)
        ret = self.get_params(*params)

        if "gt_flow" in data:
            if translate_flag:
                params[2] = (1.0, 1.0)
                ret = self.get_params(*params)
                data["img"][3:] = visionFunctional.affine(
                    img[3:],
                    *ret,
                    interpolation=self.interpolation,
                    fill=self.fill,
                )
                data["gt_flow"][0, ...] += ret[1][0]
                data["gt_flow"][1, ...] += ret[1][1]
            if scale_flag:
                params[1] = (0.0, 0.0)
                params[2] = self.scale
                ret = self.get_params(*params)

                data["img"] = visionFunctional.affine(
                    data["img"],
                    *ret,
                    interpolation=self.interpolation,
                    fill=self.fill,
                )
                data["gt_flow"] = visionFunctional.affine(
                    data["gt_flow"],
                    *ret,
                    interpolation=self.interpolation,
                    fill=self.label_fill_value,
                )
                data["gt_flow"] *= ret[2]
        else:
            data["img"] = visionFunctional.affine(
                img,
                *ret,
                interpolation=self.interpolation,
                fill=self.fill,
            )

            if "gt_seg" in data:
                if len(data["gt_seg"].shape) == 2:
                    data["gt_seg"] = data["gt_seg"].unsqueeze(0)
                data["gt_seg"] = visionFunctional.affine(
                    data["gt_seg"],
                    *ret,
                    interpolation=InterpolationMode.NEAREST,
                    fill=self.label_fill_value,
                )
                data["gt_seg"] = data["gt_seg"].squeeze(0)
        return data

    def __repr__(self):
        s = super(SegRandomAffine, self).__repr__()[:-1]
        if self.label_fill_value != 0:
            s += ", label_fill_value={label_fill_value}"
        s += ")"
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)


@OBJECT_REGISTRY.register
class Scale(object):
    r"""
    Scale input according to a scale list.

    .. note::
        Affected keys: 'img', 'gt_flow', 'gt_ori_flow', 'gt_seg'.

    Args:
        scales (Union[Real, Sequence]): The scales to apply on input.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'bilinear'`` | ``'area'``. Default: ``'nearest'``
        mul_scale(bool): Whether to multiply the scale coefficient.
    """

    def __init__(
        self,
        scales: Union[Real, Sequence],
        mode: str = "nearest",
        mul_scale: bool = False,
    ):
        super(Scale, self).__init__()
        if isinstance(scales, Real):
            self.scales = [scales]
        elif isinstance(scales, Sequence):
            self.scales = scales
        else:
            raise TypeError(
                "scales should be number or sequence. Got {}".format(
                    type(scales)
                )
            )
        self.mode = mode
        self.mul_scale = mul_scale

    def _scale(self, data: Tensor):
        scaled_data = []
        for scale in self.scales:
            scaled_tmp_data = torchFunctional.interpolate(
                data.to(dtype=torch.float),
                scale_factor=scale,
                mode=self.mode,
                recompute_scale_factor=True,
            ).to(dtype=data.dtype)
            scaled_data.append(
                scaled_tmp_data * scale if self.mul_scale else scaled_tmp_data
            )
        return scaled_data

    def __call__(self, data: dict):
        if "gt_seg" in data:
            data["gt_seg"] = self._scale(data["gt_seg"])
        if "gt_flow" in data:
            data["gt_ori_flow"] = data["gt_flow"]
            data["gt_flow"] = self._scale(data["gt_flow"])
        return data


@OBJECT_REGISTRY.register
class FlowRandomAffineScale(object):
    def __init__(
        self,
        scale_p: float = 0.5,
        scale_r: float = 0.05,
    ):  # noqa: D205,D400,D401,D403
        """
        RandomAffineScale using Opencv, the results are slightly different from
        ~torchvision.transforms.RandomAffine with scale.

        .. note::
            Affected keys: 'img', 'gt_flow'.

        Args:
        scale_p: Scale flip probability, range between [0, 1].
        scale_r: The scale transformation range is (1-scale_r, 1 + scale_r).

        """
        self.scale_p = scale_p
        self.scale_r = scale_r

    def cvscale(self, img, zoom_factor):  # noqa: D205,D400,D401
        """
        Center zoom in/out of the given image and returning
        an enlarged/shrinked view of the image without changing dimensions

        - Scipy rotate and zoom an image without changing its dimensions
        https://stackoverflow.com/a/48097478
        Written by Mohamed Ezz
        License: MIT License

        Args:
            img : Image array
            zoom_factor : amount of zoom as a ratio (0 to Inf)
        """
        height, width = img.shape[:2]  # It's also the final desired shape
        new_height, new_width = int(height * zoom_factor), int(
            width * zoom_factor
        )

        # Crop only the part that will remain in the result (more efficient)
        # Centered bbox of the final desired size in resized \
        # (larger/smaller) image coordinates
        y1, x1 = (
            max(0, new_height - height) // 2,
            max(0, new_width - width) // 2,
        )
        y2, x2 = y1 + height, x1 + width
        bbox = np.array([y1, x1, y2, x2])

        # Map back to original image coordinates
        bbox = (bbox / zoom_factor).astype(np.int64)
        y1, x1, y2, x2 = bbox
        cropped_img = img[y1:y2, x1:x2]

        # Handle padding when downscaling
        resize_height, resize_width = min(new_height, height), min(
            new_width, width
        )
        pad_height1, pad_width1 = (height - resize_height) // 2, (
            width - resize_width
        ) // 2
        pad_height2, pad_width2 = (height - resize_height) - pad_height1, (
            width - resize_width
        ) - pad_width1
        pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [
            (0, 0)
        ] * (img.ndim - 2)

        result = cv2.resize(cropped_img, (resize_width, resize_height))
        result = np.pad(result, pad_spec, mode="constant")

        assert result.shape[0] == height and result.shape[1] == width
        return result

    def __call__(self, data: Dict):
        assert data["img"].size()[0] == 6
        assert data["img"].ndim == 3
        assert "gt_flow" in data
        image1 = np.copy(data["img"].permute((1, 2, 0)).numpy()[..., :3])
        image2 = np.copy(data["img"].permute((1, 2, 0)).numpy()[..., 3:])
        flow = np.copy(data["gt_flow"].permute((1, 2, 0)).numpy())
        if self.scale_p > 0.0:
            rand = random.random()
            if rand < self.scale_p:
                ratio = random.uniform(1.0 - self.scale_r, 1.0 + self.scale_r)
                image1 = self.cvscale(image1, ratio)
                image2 = self.cvscale(image2, ratio)
                flow = self.cvscale(flow, ratio)
                flow *= ratio
        imgs = np.concatenate((image1, image2), axis=2)
        imgs_chw_np = np.ascontiguousarray(imgs.transpose((2, 0, 1)))
        flow_chw_np = np.ascontiguousarray(flow.transpose((2, 0, 1)))
        imgs_chw_tensor = torch.from_numpy(imgs_chw_np.copy())
        flow_chw_tensor = torch.from_numpy(flow_chw_np.copy())
        data["img"] = imgs_chw_tensor
        data["gt_flow"] = flow_chw_tensor
        return data


@OBJECT_REGISTRY.register
class SegRandomCutOut(object):
    """CutOut operation for segmentation task.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.

    Args:
        prob: Cutout probability.
        n_holes: Number of regions to be dropped. If it is given as a list,
        number of holes will be randomly selected from the closed interval
            [`n_holes[0]`, `n_holes[1]`].
        cutout_shape: The candidate shape of dropped regions. It can be
            `tuple[int, int]` to use a fixed cutout shape, or
            `list[tuple[int, int]]` to randomly choose shape from the list.
        cutout_ratio: The candidate ratio of dropped regions. It can be
            `tuple[float, float]` to use a fixed ratio or
            `list[tuple[float, float]]` to randomly choose ratio from the list.
            Please note that `cutout_shape` and `cutout_ratio` cannot be both
            given at the same time.
        fill_in: The value of pixel to fill in the dropped regions. Default is
            (0, 0, 0).
        seg_fill_in: The labels of pixel to fill in the dropped regions.
            If seg_fill_in is None, skip. Default is None.
    """

    def __init__(
        self,
        prob: float,
        n_holes: Union[int, Tuple[int, int]],
        cutout_shape: Optional[
            Union[Tuple[int, int], Tuple[Tuple[int, int], ...]]
        ] = None,
        cutout_ratio: Optional[
            Union[Tuple[int, int], Tuple[Tuple[int, int], ...]]
        ] = None,
        fill_in: Tuple[float, float, float] = (0, 0, 0),
        seg_fill_in: Optional[int] = None,
    ):
        assert 0 <= prob and prob <= 1
        assert (cutout_shape is None) ^ (
            cutout_ratio is None
        ), "Either cutout_shape or cutout_ratio should be specified."
        assert isinstance(cutout_shape, (list, tuple)) or isinstance(
            cutout_ratio, (list, tuple)
        )
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        if seg_fill_in is not None:
            assert (
                isinstance(seg_fill_in, int)
                and 0 <= seg_fill_in
                and seg_fill_in <= 255
            )
        self.prob = prob
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.seg_fill_in = seg_fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    def __call__(self, data):
        """Call function to drop some regions of image."""
        cutout = True if np.random.rand() < self.prob else False
        if cutout is False:
            return data
        layout = data["layout"]
        img = data["img"]
        assert layout in ["hwc", "chw", "hw"]
        if layout == "hwc":
            h, w, c = img.shape
        elif layout == "chw":
            c, h, w = img.shape
        else:
            h, w = img.shape
        if layout == "chw":
            # opencv only supports hwc layout
            img = np.ascontiguousarray(img.transpose((1, 2, 0)))  # chw > hwc
        n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        for _ in range(n_holes):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            index = np.random.randint(0, len(self.candidates))
            if not self.with_ratio:
                cutout_w, cutout_h = self.candidates[index]
            else:
                cutout_w = int(self.candidates[index][0] * w)
                cutout_h = int(self.candidates[index][1] * h)

            x2 = np.clip(x1 + cutout_w, 0, w)
            y2 = np.clip(y1 + cutout_h, 0, h)
            data["img"][y1:y2, x1:x2, :] = self.fill_in
        if layout == "chw":
            # change to the original layout
            img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        if self.seg_fill_in is not None:
            if "gt_seg" in data:
                data["gt_seg"][y1:y2, x1:x2] = self.seg_fill_in
            if "gt_depth" in data:
                data["gt_depth"][y1:y2, x1:x2] = self.seg_fill_in
        return data
