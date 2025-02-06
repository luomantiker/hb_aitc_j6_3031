# Copyright (c) Horizon Robotics. All rights reserved.
# Source code reference to mmdetection, gluoncv, gluonnas.

import copy
import inspect
import logging
import math
import random
from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

from hat.core.box_utils import bbox_overlaps, is_center_of_bboxes_in_roi
from hat.data.transforms.functional_img import (
    image_normalize,
    image_pad,
    imresize,
    imresize_pad_to_keep_ratio,
    random_flip,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages
from .affine import (
    AffineAugMat,
    AffineMat2DGenerator,
    AffineMatFromROIBoxGenerator,
    AlphaImagePyramid,
    ImageAffineTransform,
    LabelAffineTransform,
    _pad_array,
    get_affine_image_resize,
    resize_affine_mat,
)
from .bbox import (
    clip_bbox,
    filter_bbox,
    remap_bbox_label_by_area,
    remap_bbox_label_by_clip_area_ratio,
)
from .classification import BgrToYuv444, BgrToYuv444V2
from .common import Cast
from .gaze import eye_ldmk_mirror

try:
    import albumentations
except ImportError:
    albumentations = None

try:
    from torchvision.transforms import ColorJitter
except ImportError:
    ColorJitter = object

logger = logging.getLogger(__name__)

__all__ = [
    "Resize",
    "Resize3D",
    "RandomFlip",
    "Pad",
    "Normalize",
    "RandomCrop",
    "ToTensor",
    "Batchify",
    "FixedCrop",
    "PresetCrop",
    "ColorJitter",
    "RandomExpand",
    "MinIoURandomCrop",
    "AugmentHSV",
    "get_dynamic_roi_from_camera",
    "IterableDetRoITransform",
    "PadDetData",
    "ToFasterRCNNData",
    "ToMultiTaskFasterRCNNData",
    "PadTensorListToBatch",
    "PlainCopyPaste",
    "HueSaturationValue",
    "RGBShift",
    "MeanBlur",
    "MedianBlur",
    "RandomBrightnessContrast",
    "ShiftScaleRotate",
    "RandomResizedCrop",
    "AlbuImageOnlyTransform",
    "BoxJitter",
    "RandomSizeCrop",
    "DetYOLOv5MixUp",
    "DetYOLOXMixUp",
    "DetMosaic",
    "DetAffineAugTransformer",
    "IterableDetRoIListTransform",
]


def _transform_bboxes(
    gt_boxes,
    ig_regions,
    img_roi,
    affine_aug_param,
    clip=True,
    min_valid_area=8,
    min_valid_clip_area_ratio=0.5,
    min_edge_size=2,
    complete_boxes=False,
):
    bbox_ts = LabelAffineTransform(label_type="box")

    ts_gt_boxes = bbox_ts(
        gt_boxes, affine_aug_param.mat, flip=affine_aug_param.flipped
    )
    if clip:
        clip_gt_boxes = clip_bbox(ts_gt_boxes, img_roi, need_copy=True)
    else:
        clip_gt_boxes = ts_gt_boxes
    clip_gt_boxes = remap_bbox_label_by_area(clip_gt_boxes, min_valid_area)
    clip_gt_boxes = remap_bbox_label_by_clip_area_ratio(
        ts_gt_boxes, clip_gt_boxes, min_valid_clip_area_ratio
    )
    if clip and complete_boxes:
        mask = filter_bbox(
            clip_gt_boxes,
            img_roi,
            allow_outside_center=True,
            min_edge_size=min_edge_size,
            return_mask=True,
        )
        to_be_hard_flag = np.logical_and(
            clip_gt_boxes[:, 4] < 0, ts_gt_boxes[:, 4] > 0
        )
        ts_gt_boxes[to_be_hard_flag, 4] *= -1
        clip_gt_boxes = ts_gt_boxes[mask]
    else:
        clip_gt_boxes = filter_bbox(
            clip_gt_boxes,
            img_roi,
            allow_outside_center=True,
            min_edge_size=min_edge_size,
        )

    if ig_regions is not None:
        ts_ig_regions = bbox_ts(
            ig_regions, affine_aug_param.mat, flip=affine_aug_param.flipped
        )
        if clip:
            clip_ig_regions = clip_bbox(ts_ig_regions, img_roi)
        else:
            clip_ig_regions = ts_ig_regions
    else:
        clip_ig_regions = None

    return clip_gt_boxes, clip_ig_regions


@OBJECT_REGISTRY.register
class Resize(object):
    """Resize image & bbox & mask & seg.

    .. note::
        Affected keys: 'img', 'ori_img', 'img_shape', 'pad_shape',
        'resized_shape', 'pad_shape', 'scale_factor', 'gt_bboxes',
        'gt_seg', 'gt_ldmk'.

    Args:
        img_scale: See above.
        max_scale: The max size of image. If the image's shape > max_scale,
            The image is resized to max_scale
        multiscale_mode: Value must be one of "max_size", "range" or "value".
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
        pad_to_keep_ratio: Whether to pad image to keep the same shape
             and aspect ratio when resizing the image to target shape.
        interpolation: Interpolation method of image scaling, candidate
            value is ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos'].
        raw_scaler_enable: Whether to enable raw scaler when resize the image.
        sample1c_enable: Whether to sample one channel after resize the image.
        divisor: Width and height are rounded to multiples of `divisor`.
        rm_neg_coords: Whether to rm negative coordinates.
    """

    def __init__(
        self,
        img_scale: Union[Sequence[int], Sequence[Sequence[int]]] = None,
        max_scale: Union[Sequence[int], Sequence[Sequence[int]]] = None,
        multiscale_mode: str = "range",
        ratio_range: Tuple[float, float] = None,
        keep_ratio: bool = True,
        pad_to_keep_ratio: bool = False,
        interpolation: str = "bilinear",
        raw_scaler_enable: bool = False,
        sample1c_enable: bool = False,
        divisor: int = 1,
        rm_neg_coords: bool = True,
        split_transform: bool = False,
        split_trans_w: int = 256,
        split_trans_h: int = 256,
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
        self.pad_to_keep_ratio = pad_to_keep_ratio

        assert interpolation in [
            "nearest",
            "bilinear",
            "bicubic",
            "area",
            "lanczos",
        ], (
            "Currently interpolation only supports "
            "['nearest', 'bilinear', 'bicubic', 'area', 'lanczos'], "
            f"but got {interpolation}."
        )
        self.interpolation = interpolation

        self.raw_scaler_enable = raw_scaler_enable
        self.sample1c_enable = sample1c_enable
        self.divisor = divisor
        self.rm_neg_coords = rm_neg_coords
        self.split_transform = split_transform
        self.split_trans_w = split_trans_w
        self.split_trans_h = split_trans_h

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
        me_ds_img = None
        if self.pad_to_keep_ratio and self.keep_ratio:
            resized_img, R = imresize_pad_to_keep_ratio(
                img, (h, w), data["layout"], keep_ratio=True
            )
            data["scale_factor"] = R
        else:
            resized_img, me_ds_img, w_scale, h_scale = imresize(
                img,
                w,
                h,
                data["layout"],
                keep_ratio=self.keep_ratio,
                return_scale=True,
                interpolation=self.interpolation,
                raw_scaler_enable=self.raw_scaler_enable,
                raw_pattern=data.get("cur_pattern"),
                sample1c_enable=self.sample1c_enable,
                divisor=self.divisor,
                split_transform=self.split_transform,
                split_trans_w=self.split_trans_w,
                split_trans_h=self.split_trans_h,
            )
            data["scale_factor"] = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
            )

        if "ori_img" in data:
            ori_img = data["ori_img"]
            if self.pad_to_keep_ratio:
                resized_ori_img, _ = imresize_pad_to_keep_ratio(
                    ori_img, (h, w), data["layout"], keep_ratio=self.keep_ratio
                )
            else:
                resized_ori_img, _, _, _ = imresize(
                    ori_img,
                    w,
                    h,
                    data["layout"],
                    keep_ratio=self.keep_ratio,
                    return_scale=True,
                    divisor=self.divisor,
                )
            # No direct replacement to prevent ori_img used.
            data["resized_ori_img"] = resized_ori_img

        data["img"] = resized_img
        if me_ds_img is not None:
            data["me_in_img"] = me_ds_img
        data["img_shape"] = resized_img.shape
        data["resized_shape"] = resized_img.shape
        data[
            "pad_shape"
        ] = resized_img.shape  # in case that there is no padding  # noqa
        data["keep_ratio"] = self.keep_ratio

    def _resize_bbox(self, data):
        if not data["gt_bboxes"].any():
            return
        if data["layout"] == "hwc":
            h, w = data["img_shape"][:2]
        else:
            h, w = data["img_shape"][1:]
        bboxes = data["gt_bboxes"]
        if self.pad_to_keep_ratio and self.keep_ratio:
            n = bboxes.shape[0]
            R = data["scale_factor"]
            xy = np.ones((n * 4, 3))
            xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2
            )  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ R.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = (
                np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)))
                .reshape(4, n)
                .T
            )
            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, w - 1)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, h - 1)
            bboxes = xy.astype(np.float32)
        else:
            bboxes = bboxes * data["scale_factor"]
            scale_offset = data.get("scale_offset", (0, 0))
            bboxes[:, 0::2] += scale_offset[1]
            bboxes[:, 1::2] += scale_offset[0]
            if self.rm_neg_coords:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w - 1)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h - 1)
        data["gt_bboxes"] = bboxes

    def _resize_seg(self, data):
        """Resize semantic segmentation map with ``data['scale']``."""
        if self.pad_to_keep_ratio and self.keep_ratio:
            raise NotImplementedError
        h = data["scale"][0]
        w = data["scale"][1]
        resized_seg = imresize(
            data["gt_seg"],
            w,
            h,
            "hw",
            keep_ratio=self.keep_ratio,
            return_scale=False,
            interpolation="nearest",
        )
        data["gt_seg"] = resized_seg

    def _resize_ldmk(self, data):
        scale_factor = data["scale_factor"]
        w_scale, h_scale = scale_factor[0], scale_factor[1]

        ldmk = data["gt_ldmk"]

        assert (
            ldmk.shape[-1] == 2 or ldmk.shape[-1] == 3
        ), "ldmk coordinates should be 2 or 3 dimensions"

        ldmk[..., 0] = ldmk[..., 0] * w_scale
        ldmk[..., 1] = ldmk[..., 1] * h_scale
        data["gt_ldmk"] = ldmk

    def _resize_mask(self, data):
        raise NotImplementedError

    def _resize_lines(self, data):
        scale_factor = data["scale_factor"]
        w_scale, h_scale = scale_factor[0], scale_factor[1]
        gt_lines = data["gt_lines"]
        for gt_line in gt_lines:
            gt_line[:, 0] *= w_scale
            gt_line[:, 1] *= h_scale
        data["gt_lines"] = gt_lines

    def inverse_transform(
        self,
        inputs: Union[np.ndarray, torch.Tensor],
        task_type: str,
        inverse_info: Dict,
    ):
        """Inverse option of transform to map the prediction to the original image.

        Args:
            inputs: Prediction.
            task_type (str): `detection` or `segmentation`.
            inverse_info (dict): The transform keyword is the key,
                and the corresponding value is the value.

        """
        if task_type == "detection":
            scale_factor = inverse_info["scale_factor"]
            if not isinstance(scale_factor, torch.Tensor):
                scale_factor = inputs.new_tensor(scale_factor)
            inputs[:, :4] = inputs[:, :4] / scale_factor
            return inputs
        elif task_type == "segmentation":
            scale_factor = inverse_info["scale_factor"][:2]
            if isinstance(scale_factor, torch.Tensor):
                scale_factor = scale_factor.detach().cpu().numpy()
            elif isinstance(scale_factor, (tuple, list)):
                scale_factor = np.array(scale_factor)
            else:
                assert isinstance(scale_factor, np.ndarray)
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.detach().cpu().numpy()
            before_resize_shape = inputs.shape / scale_factor
            out_height, out_width = before_resize_shape
            out_img = cv2.resize(
                inputs,
                (int(out_width), int(out_height)),
                interpolation=cv2.INTER_NEAREST,
            )
            return out_img
        else:
            raise Exception(
                "error task_type, your task_type[{}],"
                " we need segmentation or detection".format(task_type)
            )

    def __call__(self, data):
        self._random_scale(data)
        self._resize_img(data)
        if "gt_bboxes" in data:
            self._resize_bbox(data)
        if "gt_seg" in data:
            self._resize_seg(data)
        if "gt_ldmk" in data:
            self._resize_ldmk(data)
        if "gt_lines" in data:
            self._resize_lines(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"img_scale={self.img_scale}, "
        repr_str += f"multiscale_mode={self.multiscale_mode}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"keep_ratio={self.keep_ratio}"
        repr_str += f"raw_scaler_enable={self.raw_scaler_enable}"
        repr_str += f"sample1c_enable={self.sample1c_enable}"
        return repr_str


@OBJECT_REGISTRY.register
class Resize3D(Resize):
    """Resize 3D labels.

    Different from 2D Resize, we accept img_scale=None and ratio_range is not
    None. In that case we will take the input img scale as the ori_scale for
    rescaling with ratio_range.

    Args:
        img_scale: Images scales for resizing.
        multiscale_mode: Either "range" or "value".
        ratio_range: (min_ratio, max_ratio).
        keep_ratio: Whether to keep the aspect ratio when resizing the image.
        bbox_clip_border: Whether to clip the objects outside
            the border of the image.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice.
    """

    def __init__(
        self,
        img_scale=None,
        multiscale_mode="range",
        ratio_range=None,
        keep_ratio=True,
        bbox_clip_border=True,
        backend="cv2",
        interpolation="nearest",
        override=False,
        cam2img_keep_ratio=False,
    ):
        super(Resize3D, self).__init__(
            img_scale=img_scale,
            multiscale_mode=multiscale_mode,
            ratio_range=ratio_range,
            keep_ratio=keep_ratio,
        )

        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]

        if ratio_range is None:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range"]

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.interpolation = interpolation
        self.override = override
        self.bbox_clip_border = bbox_clip_border
        self.cam2img_keep_ratio = cam2img_keep_ratio

    def _random_scale(self, results):
        # ori_scale = results['img'].shape[:2]
        # consider the ori_scale can be specified by self.img_scale
        if self.img_scale is not None:
            ori_scale = self.img_scale[0]
        else:
            ori_scale = results["img"].shape[:2]
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                ori_scale, self.ratio_range
            )
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results["scale"] = scale
        results["scale_idx"] = scale_idx

    def _resize_3d(self, results):
        if "centers2d" in results and results["centers2d"].size(0) > 0:
            results["centers2d"] *= results["scale_factor"][:2]
        # resize image equals to change focal length and
        # camera intrinsic
        results["cam2img"][0] *= results["scale_factor"][0].repeat(
            len(results["cam2img"][0])
        )
        if self.cam2img_keep_ratio:
            results["cam2img"][1] *= results["scale_factor"][0].repeat(
                len(results["cam2img"][1])
            )
        else:
            results["cam2img"][1] *= results["scale_factor"][1].repeat(
                len(results["cam2img"][1])
            )

    def __call__(self, results):
        super(Resize3D, self).__call__(results)
        self._resize_3d(results)
        return results


@OBJECT_REGISTRY.register
class RandomFlip(object):
    """Flip image & bbox & mask & seg & flow.

    .. note::
        Affected keys: 'img', 'ori_img', 'img_shape', 'pad_shape',
        'gt_bboxes', 'gt_tanalphas', 'gt_seg', 'gt_flow',
        'gt_mask', 'gt_ldmk', 'ldmk_pairs'.

    Args:
        px: Horizontal flip probability, range between [0, 1].
        py: Vertical flip probability, range between [0, 1].
    """

    def __init__(self, px: Optional[float] = 0.5, py: Optional[float] = 0):
        assert px >= 0 and px <= 1, "px must range between [0, 1]"
        assert py >= 0 and py <= 1, "py must range between [0, 1]"
        self.px = px
        self.py = py

    def _flip_img(self, data):
        raw_pattern = data.get("cur_pattern")
        data["img"], (flip_x, flip_y), raw_pattern = random_flip(
            data["img"], data["layout"], self.px, self.py, raw_pattern
        )
        if raw_pattern is not None:
            data["cur_pattern"] = raw_pattern

        return flip_x, flip_y

    def _flip_bbox(self, data):
        if "img_shape" not in data:
            img_shape = (data["img_height"], data["img_width"])
        else:
            img_shape = data["img_shape"]
            if "pad_shape" in data:
                img_shape = data["pad_shape"]
        bboxes = data["gt_bboxes"]  # shape is Nx4, format is (x1, y1, x2, y2)
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        h, w = img_shape[:2]
        if self.flip_x:
            flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
            flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        if self.flip_y:
            flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
            flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        data["gt_bboxes"] = flipped
        if "gt_tanalphas" in data:
            data["gt_tanalphas"] = -data["gt_tanalphas"]

    def _flip_seg(self, data):
        flipped_seg, _, _ = random_flip(
            data["gt_seg"], "hw", self.flip_x, self.flip_y
        )
        data["gt_seg"] = flipped_seg

    def _flip_flow(self, data):
        flipped_flow, _, _ = random_flip(
            data["gt_flow"], "hwc", self.flip_x, self.flip_y
        )
        if isinstance(flipped_flow, np.ndarray):
            flipped_flow_clone = np.copy(flipped_flow)
        else:
            flipped_flow_clone = flipped_flow.clone()
        if self.flip_x:
            flipped_flow_clone[:, :, 0] = flipped_flow_clone[:, :, 0] * -1
        if self.flip_y:
            flipped_flow_clone[:, :, 1] = flipped_flow_clone[:, :, 1] * -1
        data["gt_flow"] = flipped_flow_clone

    def _flip_mask(self, data):
        flipped_mask, _, _ = random_flip(
            data["gt_mask"], data["layout"], self.flip_x, self.flip_y
        )
        data["gt_mask"] = flipped_mask

    def _flip_target_img(self, data):
        """Gt image for reconstruction loss.

        data["gt_img"] is used for superviseing reconstructed image.
        It may be different from data["img] in image size or color space.
        """
        flipped_gt_img, _, _ = random_flip(
            data["gt_img"], data["layout"], self.flip_x, self.flip_y
        )
        data["gt_img"] = flipped_gt_img

    def _flip_ldmk(self, data):
        if "img_shape" not in data:
            height, width = (data["img_height"], data["img_width"])
        else:
            height, width, _ = data["img_shape"]
        ldmk = data["gt_ldmk"].copy()
        assert "ldmk_pairs" in data
        assert ldmk.ndim in [2, 3], "ldmk layout should be (K, D) or (N, K, D)"
        # last dimension must (x, y) or (x, y, z).
        assert (
            ldmk.shape[-1] == 2 or ldmk.shape[-1] == 3
        ), "ldmk coordinates should be 2 or 3 dimensions"  # noqa
        pairs = data["ldmk_pairs"]

        if self.flip_x:
            if pairs is not None:
                for pair in pairs:
                    # (num_ldmk, coord)
                    if ldmk.ndim == 2:
                        temp = ldmk[pair[0]].copy()
                        ldmk[pair[0]] = ldmk[pair[1]]
                        ldmk[pair[1]] = temp
                    # (num_box, num_ldmk, coord)
                    elif ldmk.ndim == 3:
                        temp = ldmk[:, pair[0]].copy()
                        ldmk[:, pair[0]] = ldmk[:, pair[1]]
                        ldmk[:, pair[1]] = temp
            ldmk[..., 0] = width - ldmk[..., 0] - 1
        if self.flip_y:
            ldmk[..., 1] = height - ldmk[..., 1] - 1
        data["gt_ldmk"] = ldmk

    def _flip_gaze(self, data):
        # for eye gaze task
        if self.flip_y:
            raise NotImplementedError
        if not self.flip_x:
            return
        gaze_label = data["gaze_label"]
        # flip position map
        if "horizon_img" in data.keys() and "vertical_img" in data.keys():
            assert (
                "mirror_horizon_img" in data.keys()
                and "mirror_vertical_img" in data.keys()
            ), "\
                    key of 'mirror_horizon_img' and 'mirror_vertical_img'\
                    should in input"
            data["horizon_img"] = data["mirror_horizon_img"]
            data["vertical_img"] = data["mirror_vertical_img"]
        # flip gaze
        if "gt_gaze" in gaze_label.keys():
            gt_gaze = gaze_label["gt_gaze"]
            gaze_label["gt_gaze"] = np.array(
                [
                    gt_gaze[2],
                    -gt_gaze[3] if gt_gaze[3] > -1000 else -1000,
                    gt_gaze[0],
                    -gt_gaze[1] if gt_gaze[1] > -1000 else -1000,
                ]
            )
        # flip head pose
        if "gt_head_pose" in gaze_label.keys():
            gaze_label["gt_head_pose"] *= [1, -1, -1]
        # flip normed_eye_ldmk, to do in offline augm pipline
        if "gt_normed_eye_ldmk" in gaze_label.keys():
            gaze_label["gt_normed_eye_ldmk"] = eye_ldmk_mirror(
                gaze_label["gt_normed_eye_ldmk"]
            )
        # rotate 3d augm pipline
        if "intrinsics_K" in gaze_label.keys():
            intrinsics_K = gaze_label["intrinsics_K"]
            ctr_x = intrinsics_K[0, 2]
            # flip eye_bbox
            if "gt_eye_bbox" in gaze_label.keys():
                x1, y1, x2, y2 = gaze_label["gt_eye_bbox"]
                x1_ = 2 * ctr_x - x1 + 1
                x2_ = 2 * ctr_x - x2 + 1
                x1, x2 = (x2_, x1_)
                gaze_label["gt_eye_bbox"] = np.array([x1, y1, x2, y2])
            # face_ldmk only for roi crop
            if "gt_face_ldmks" in gaze_label.keys():
                face_ldmks = gaze_label["gt_face_ldmks"]
                mask = face_ldmks[:, 0] > -1000
                face_ldmks[mask, 0] = 2 * ctr_x - face_ldmks[mask, 0]

                eye_ldmks = gaze_label["gt_eye_ldmk"]
                mask = eye_ldmks[:, 0] > -1000
                eye_ldmks[mask, 0] = 2 * ctr_x - eye_ldmks[mask, 0]
                gaze_label["gt_face_ldmks"] = face_ldmks
                gaze_label["gt_eye_ldmk"] = eye_ldmk_mirror(
                    eye_ldmks, normd=False
                )
        if "gt_gazemap" in gaze_label.keys():
            gaze_label["gt_gazemap"] = gaze_label["gt_gazemap"][:, ::-1, :]
        data["gaze_label"] = gaze_label

    def _flip_point_sets(self, data, key):
        if "img_shape" not in data:
            height, width = (data["img_height"], data["img_width"])
        else:
            height, width, _ = data["img_shape"]

        gt_point_sets = data[key]
        for gt_point_set in gt_point_sets:
            if self.flip_x:
                gt_point_set[:, 0] = (width - 1) - gt_point_set[:, 0]
            if self.flip_y:
                gt_point_set[:, 1] = (height - 1) - gt_point_set[:, 1]
        data[key] = gt_point_sets

    def _flip_eye_status(self, data):
        if self.flip_x:
            if "eye_status" in data.keys():
                eye_status = data["eye_status"].copy()
                l_status, r_status = np.split(eye_status, 2, axis=0)
                eye_status = np.concatenate([r_status, l_status])
                data["eye_status"] = eye_status
            if "eye_vis_labels" in data.keys():
                eye_vis_labels = data["eye_vis_labels"].copy()
                data["eye_vis_labels"] = np.array(
                    [eye_vis_labels[i] for i in (0, 3, 4, 1, 2)]
                )
            if "gt_eye_cls_labels" in data.keys():
                eye_cls_labels = data["gt_eye_cls_labels"].copy()
                l_labels, r_labels = np.split(eye_cls_labels, 2, axis=0)
                eye_cls_labels = np.concatenate([r_labels, l_labels])
                data["gt_eye_cls_labels"] = eye_cls_labels
            if "gt_ldmk_attr" in data.keys():
                pairs = data["ldmk_pairs"]
                ldmk_attr = data["gt_ldmk_attr"].copy()
                for pair in pairs:
                    temp = ldmk_attr[pair[0]].copy()
                    ldmk_attr[pair[0]] = ldmk_attr[pair[1]]
                    ldmk_attr[pair[1]] = temp
                data["gt_ldmk_attr"] = ldmk_attr

    def _flip_ellipse_param(self, data):
        height, width, _ = data["img"].shape
        if self.flip_y:
            if "gt_pupil_ellipse_param" in data.keys():
                data["gt_pupil_ellipse_param"][1] = (
                    height - data["gt_pupil_ellipse_param"][1]
                )
                data["gt_pupil_ellipse_param"][-1] = (
                    180 - data["gt_pupil_ellipse_param"][-1]
                )
        if self.flip_x:
            if "gt_pupil_ellipse_param" in data.keys():
                data["gt_pupil_ellipse_param"][0] = (
                    width - data["gt_pupil_ellipse_param"][0]
                )
                data["gt_pupil_ellipse_param"][-1] = (
                    180 - data["gt_pupil_ellipse_param"][-1]
                )

    def __call__(self, data):
        flip_x, flip_y = self._flip_img(data)
        # bbox & mask & seg do the same flip operation as img
        self.flip_x = flip_x
        self.flip_y = flip_y
        if "gt_bboxes" in data:
            self._flip_bbox(data)
        if "gt_seg" in data:
            self._flip_seg(data)
        if "gt_flow" in data:
            self._flip_flow(data)
        if "gt_ldmk" in data:
            self._flip_ldmk(data)
        if "gt_mask" in data:
            self._flip_mask(data)
        if "gt_img" in data:
            self._flip_target_img(data)
        if "gaze_label" in data:
            self._flip_gaze(data)
        if "gt_lines" in data:
            self._flip_point_sets(data, "gt_lines")
        if "gt_polygons" in data:
            self._flip_point_sets(data, "gt_polygons")
        if "gt_eye_cls_labels" in data:
            self._flip_eye_status(data)
        if "gt_pupil_ellipse_param" in data:
            self._flip_ellipse_param(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"px={self.px}, py={self.py}"
        return repr_str


@OBJECT_REGISTRY.register
class Pad(object):
    def __init__(
        self,
        size: Tuple = None,
        divisor: int = 1,
        pad_val: int = 0,
        seg_pad_val: int = 255,
    ):
        """Pad image & mask & seg.

        .. note::
            Affected keys: 'img', 'layout', 'pad_shape', 'gt_seg'.

        Args:
            size (Optional[tuple]): Expected padding size, meaning of dimension
                is the same as img, if layout of img is `hwc`, shape must be
                (pad_h, pad_w) or (pad_h, pad_w, c).
            divisor (int): Padded image edges will be multiple to divisor.
            pad_val: Values to be filled in
                padding areas for img, single value or a list of values with
                len c. E.g. : pad_val = 10, or pad_val = [10, 20, 30].
            seg_pad_val: Value to be filled in padding areas
                for gt_seg.
        """
        self.size = size
        self.divisor = divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def _pad_img(self, data):
        if data["layout"] == "chw":
            data["before_pad_shape"] = np.array(data["img"].shape[1:])
        else:
            data["before_pad_shape"] = np.array(data["img"].shape[:2])
        padded_img = image_pad(
            data["img"], data["layout"], self.size, self.divisor, self.pad_val
        )
        data["img"] = padded_img
        data["padded_img"] = padded_img
        data["pad_shape"] = padded_img.shape

    def _pad_seg(self, data):
        if data["layout"] == "chw":
            size = data["pad_shape"][1:]
        else:
            size = data["pad_shape"][:2]
        padded_seg = image_pad(data["gt_seg"], "hw", size, 1, self.seg_pad_val)
        data["gt_seg"] = padded_seg

    def _pad_disp(self, data):
        if data["layout"] == "chw":
            size = data["pad_shape"][1:]
        else:
            size = data["pad_shape"][:2]
        padded_disp = image_pad(data["gt_disp"], "hw", size, 1, -1)
        data["gt_disp"] = padded_disp

    def __call__(self, data):
        self._pad_img(data)
        if "gt_seg" in data:
            self._pad_seg(data)
        if "gt_disp" in data:
            self._pad_disp(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"size={self.size}, "
        repr_str += f"divisor={self.divisor}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"seg_pad_val={self.seg_pad_val}"
        return repr_str


@OBJECT_REGISTRY.register
class Normalize(object):
    """
    Normalize image.

    .. note::
        Affected keys: 'img', 'layout'.

    Args:
        mean: mean of normalize.
        std: std of normalize.
        raw_norm (bool) : Whether to open raw_norm.
    """

    def __init__(
        self,
        mean: Union[float, Sequence[float]],
        std: Union[float, Sequence[float]],
        raw_norm: bool = False,
        split_transform: bool = False,
    ):
        self.mean = mean
        self.std = std
        self.raw_norm = raw_norm
        self.split_transform = split_transform

    def __call__(self, data):
        layout = data.get("layout", "chw")
        if self.raw_norm:
            bit_nums_lower = data["bit_nums_lower"][0]
            in_img = data["img"] / 2 ** bit_nums_lower
        else:
            in_img = data["img"]
        data["img"] = image_normalize(in_img, self.mean, self.std, layout)

        if self.split_transform:
            if self.raw_norm:
                bit_nums_lower = data["bit_nums_lower"][0]
                me_in_img = data["me_in_img"] / 2 ** bit_nums_lower
            else:
                me_in_img = data["me_in_img"]
            data["me_in_img"] = image_normalize(
                me_in_img, self.mean, self.std, layout
            )
            data["me_in_img"] = np.transpose(data["me_in_img"], (2, 0, 1))
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += (
            f"mean={self.mean}, std={self.std}, raw_norm={self.raw_norm}"
        )
        return repr_str


@OBJECT_REGISTRY.register
class RandomCrop(object):
    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        min_area: float = -1,
        min_iou: float = -1,
        center_crop_prob: float = 0.0,
        center_shake: Optional[Tuple[float, float]] = None,
        truncate_gt: bool = True,
        h_ratio_range: Optional[Tuple[float, float]] = None,
        w_ratio_range: Optional[Tuple[float, float]] = None,
        wh_ratio_range: Optional[Tuple[float, float]] = None,
        discriminate_ignore_classes: bool = False,
        keep_raw_pattern: bool = False,
        without_background: bool = False,
        crop_around_gt: bool = False,
        repeat_times: int = 1,
        inclusion_rate: float = 0.0,
        filter_area: bool = True,
        rm_neg_coords: bool = True,
        discard_truncate: bool = False,
    ):  # noqa: D205,D400
        """

        .. note::
            Affected keys: 'img', 'img_shape', 'pad_shape', 'layout',
            'gt_bboxes', 'gt_classes', 'gt_seg', 'gt_flow'.

        Args:
            size: Expected size after cropping, (h, w). If not given, it will
                be changed to shape of each image.
            min_area: If min_area > 0, boxes whose areas are less than
                min_area will be ignored.
            min_iou: If min_iou > 0, boxes whose iou between before and
                after truncation < min_iou will be ignored.
            center_crop_prob: The center_crop_prob is the center crop
                probability.
            center_shake: The list is the center shake's top, bottom, left,
                right range.
            truncate_gt: if True, truncate the gt_boxes when the gt_boxes
                exceed the crop area. Default True.
            h_ratio_range: lower & upper bound to randomly change the height
                of target size.
            w_ratio_range: lower & upper bound to randomly change the width
                of target size.
            wh_ratio_range: lower & upper bound to randomly change the width
                and height of target size simultaneously.
            discriminate_ignore_classes: if True, ignored area by min_iou
                retain the original class info. And class id should greater
                than 0. Default False. Support class id starts from 0.
            keep_raw_pattern: if True, limit the crop coordinates to keep raw
                pattern when crop raw image.
            without_background: if True, the class label starts from 0, else 0
                represents background.
            crop_around_gt: if Ture, will do the image crop around the gt
                using the function _crop_img_around_gt() instead of
                _crop_img(). This helps generate more effective training
                examples. Default False.
            repeat_times: When one crop has no gt bbox or few gt bboxes, e.g.
                the number of gt_bboxes in this crop is less than
                total_num_of_gt_bboxes x inclusion_rate in it,
                _crop_img_around_gt() will do repeat_times random crop so that
                there are required inclusion_rate gt_bboxes in this crop.
                Default 1.
            inclusion_rate: The minimum proportion of gt bboxes you want one
                crop to include. Default 0.0.
            filter_area: Whether to filter boxes with small area.
            rm_neg_coords: Whether to remove negative coordinates.
            discard_truncate: Whether to remove truncate coordinates.
        """
        self.size = size
        self.min_area = min_area
        self.min_iou = min_iou
        assert 0 <= center_crop_prob <= 1
        self.center_crop_prob = center_crop_prob
        self.center_shake = center_shake
        self.truncate_gt = truncate_gt
        assert size is None or (
            isinstance(size, tuple) and len(size) == 2
        ), "size must be tuple and of size 2"
        assert h_ratio_range is None or (
            isinstance(h_ratio_range, tuple) and len(h_ratio_range) == 2
        ), "h_ratio_range must be tuple and of size 2"
        assert w_ratio_range is None or (
            isinstance(w_ratio_range, tuple) and len(w_ratio_range) == 2
        ), "w_ratio_range must be tuple and of size 2"
        assert wh_ratio_range is None or (
            isinstance(wh_ratio_range, tuple) and len(wh_ratio_range) == 2
        ), "w_ratio_range must be tuple and of size 2"
        self.w_ratio_range = w_ratio_range
        self.h_ratio_range = h_ratio_range
        self.wh_ratio_range = wh_ratio_range
        self.discriminate_ignore_classes = discriminate_ignore_classes
        self.keep_raw_pattern = keep_raw_pattern
        self.without_background = without_background
        self.crop_around_gt = crop_around_gt
        assert repeat_times >= 1
        self.repeat_times = repeat_times
        assert 0 <= inclusion_rate <= 1
        self.inclusion_rate = inclusion_rate
        self.filter_area = filter_area
        self.rm_neg_coords = rm_neg_coords
        self.discard_truncate = discard_truncate

    def _union_of_bboxes(self, data: Dict):
        """Calculate union of bounding boxes.

        Args:
            data (dict): A dict that includes all information of an image.

        Returns:
            tuple: A minimum bounding box `(x_min, y_min, x_max, y_max)` that
                includes all gt bboxes.
        """
        if data["layout"] == "hwc":
            h, w = data["img_shape"][:2]
        else:
            h, w = data["img_shape"][1:]
        x1, y1 = w, h
        x2, y2 = 0, 0
        bboxes = data["gt_bboxes"]
        classes = data["gt_classes"]
        # if there is any gt bboxes
        if (classes >= 0).any():
            for bbox, cls in zip(bboxes, classes):
                if cls < 0:
                    continue
                x_min, y_min, x_max, y_max = bbox[:4]
                x1, y1 = int(np.min([x1, x_min])), int(np.min([y1, y_min]))
                x2, y2 = int(np.max([x2, x_max])), int(np.max([y2, y_max]))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
            return np.array([x1, y1, x2, y2], dtype=int)
        return np.zeros((0, 4), dtype=int)

    def _is_good_enough(self, data: Dict, crop_bbox: np.ndarray):
        """Test if one crop has enough gt bboxes in it.

        Args:
            data (dict): A dict that includes all information of an image.
            crop_bbox (np.array): The cropped box.

        Returns:
            bool: If the cropped box has enough gt bboxes in it.
        """

        bboxes = data["gt_bboxes"].copy()
        classes = data["gt_classes"].copy()
        valid_indexes = classes >= 0
        if valid_indexes.any():
            bboxes = bboxes[valid_indexes, :].reshape(-1, 4)
            crop_bbox = crop_bbox.reshape(1, -1)
            # Note the order of bboxes and crop_bbox in this function, which
            # can't be swapped
            iofs = bbox_overlaps(bboxes, crop_bbox, mode="iof")
            rate = iofs.sum() / bboxes.shape[0]
            if rate >= self.inclusion_rate:
                return True
            else:
                return False
        else:
            return True

    @staticmethod
    def _get_rand_ratio(ratio_range):
        if ratio_range is None:
            return 1
        rrange = ratio_range[1] - ratio_range[0]
        assert rrange >= 0
        rratio = np.random.rand() * rrange + ratio_range[0]
        return rratio

    def _get_crop_region(self, data, iter_i=0):
        img_shape = data["img_shape"]
        layout = data["layout"]
        assert layout in ["hwc", "chw"], (
            "layout of img must be `chw` or " "`hwc`"
        )
        do_center_crop = np.random.rand() < self.center_crop_prob

        if layout == "chw":
            img_shape = (img_shape[1], img_shape[2], img_shape[0])
        img_h, img_w = img_shape[:2]
        crop_size = (
            list(self.size) if self.size is not None else [img_h, img_w]
        )
        rratio = self._get_rand_ratio(self.wh_ratio_range)
        crop_size = [s * rratio for s in crop_size]
        crop_size[0] *= self._get_rand_ratio(self.h_ratio_range)
        crop_size[1] *= self._get_rand_ratio(self.w_ratio_range)

        margin_h = max(img_shape[0] - crop_size[0], 1)
        margin_w = max(img_shape[1] - crop_size[1], 1)
        lower_lim_x1, upper_lim_x1 = 0, margin_w
        lower_lim_y1, upper_lim_y1 = 0, margin_h

        if self.crop_around_gt and not do_center_crop:
            # get the union box of all gt boxes
            union_box = self._union_of_bboxes(data)
            if union_box.any():
                union_x1, union_y1, union_x2, union_y2 = union_box
            else:
                union_x1, union_y1, union_x2, union_y2 = (
                    0,
                    0,
                    img_w - 1,
                    img_h - 1,
                )
            # get the range of the crop's up-left corner
            lower_lim_x1 = max(0, int(min(union_x1, union_x2 - crop_size[1])))
            lower_lim_y1 = max(0, int(min(union_y1, union_y2 - crop_size[0])))
            upper_lim_x1 = min(
                img_w, int(max(union_x1, union_x2 - crop_size[1]))
            )
            upper_lim_y1 = min(
                img_h, int(max(union_y1, union_y2 - crop_size[0]))
            )

            # avoid the cropped box exceeding the image boundary
            upper_lim_x1 = min(upper_lim_x1, int(max(0, img_w - crop_size[1])))
            upper_lim_y1 = min(upper_lim_y1, int(max(0, img_h - crop_size[0])))
            lower_lim_x1, upper_lim_x1 = min(lower_lim_x1, upper_lim_x1), max(
                lower_lim_x1, upper_lim_x1
            )
            lower_lim_y1, upper_lim_y1 = min(lower_lim_y1, upper_lim_y1), max(
                lower_lim_y1, upper_lim_y1
            )

            # randomly generate upper left corner coordinates and extend the
            # limits by 15 pixels
            if lower_lim_x1 < union_x1:
                lower_lim_x1 = lower_lim_x1 + 15
                upper_lim_x1 = max(0, upper_lim_x1 - 15)
                # in case lower_lim_x1 > upper_lim_x1 after the extension above
                if lower_lim_x1 > upper_lim_x1:
                    lower_lim_x1, upper_lim_x1 = upper_lim_x1, lower_lim_x1
            else:
                lower_lim_x1 = max(0, lower_lim_x1 - 15)
                upper_lim_x1 = upper_lim_x1 + 15
            if lower_lim_y1 < union_y1:
                lower_lim_y1 = lower_lim_y1 + 15
                upper_lim_y1 = max(0, upper_lim_y1 - 15)
                # in case lower_lim_y1 > upper_lim_y1 after the extension above
                if lower_lim_y1 > upper_lim_y1:
                    lower_lim_y1, upper_lim_y1 = upper_lim_y1, lower_lim_y1
            else:
                lower_lim_y1 = max(0, lower_lim_y1 - 15)
                upper_lim_y1 = upper_lim_y1 + 15
            if iter_i == self.repeat_times - 2:
                # randomly crop upper left part or lower right part if a good
                # crop can't be achieved in repeat_times-1 times
                if np.random.choice([0, 1]) == 0:
                    upper_lim_x1 = min(lower_lim_x1 + 1, img_w)
                    upper_lim_y1 = min(lower_lim_y1 + 1, img_h)
                else:
                    lower_lim_x1 = max(0, upper_lim_x1 - 1)
                    lower_lim_y1 = max(0, upper_lim_y1 - 1)

        elif do_center_crop:
            if self.center_shake is not None:
                center_shake = copy.deepcopy(list(self.center_shake))
                if "scale_factor" in data.keys():
                    scale_factor_h = data["scale_factor"][1]
                    scale_factor_w = data["scale_factor"][0]
                    center_shake[0] = int(center_shake[0] * scale_factor_h)
                    center_shake[1] = int(center_shake[1] * scale_factor_h)
                    center_shake[2] = int(center_shake[2] * scale_factor_w)
                    center_shake[3] = int(center_shake[3] * scale_factor_w)
                center_h = np.floor(margin_h / 2)
                center_w = np.floor(margin_w / 2)
                lower_lim_y1 = max(center_h - center_shake[0], 0)
                upper_lim_y1 = min(center_h + center_shake[1], margin_h)
                lower_lim_x1 = max(center_w - center_shake[2], 0)
                upper_lim_x1 = min(center_w + center_shake[3], margin_w)
            else:
                lower_lim_y1 = upper_lim_y1 = np.floor(margin_h / 2)
                lower_lim_x1 = upper_lim_x1 = np.floor(margin_w / 2)

        offset_h = np.random.randint(lower_lim_y1, upper_lim_y1 + 1)
        offset_w = np.random.randint(lower_lim_x1, upper_lim_x1 + 1)

        if self.keep_raw_pattern:
            cur_pattern = data["cur_pattern"]
            tar_pattern = data["raw_pattern"]
            if tar_pattern == cur_pattern:
                x_offset, y_offset = 0, 0
            elif (
                cur_pattern[0] == tar_pattern[2]
                and cur_pattern[1] == tar_pattern[3]
            ):
                # rggb->gbrg, bggr->grbg
                x_offset, y_offset = 1, 0
            elif (
                cur_pattern[0] == tar_pattern[1]
                and cur_pattern[2] == tar_pattern[3]
            ):
                # rggb->grbg, bggr->gbrg
                x_offset, y_offset = 0, 1
            else:  # This is not happening in ["RGGB", "BGGR", "GRBG", "GBRG"]
                raise RuntimeError(
                    "Unexpected pair of input and target bayer pattern!"
                )
            if y_offset < margin_h + 1:
                offset_h = random.randrange(y_offset, margin_h + 1, 2)
            else:
                offset_h = y_offset
            if x_offset < margin_w + 1:
                offset_w = random.randrange(x_offset, margin_w + 1, 2)
            else:
                offset_w = x_offset
        crop_y1, crop_y2 = offset_h, min(offset_h + crop_size[0], img_h)
        crop_x1, crop_x2 = offset_w, min(offset_w + crop_size[1], img_w)
        if layout == "hwc":
            data["img_shape"] = (
                int(crop_y2) - int(crop_y1),
                int(crop_x2) - int(crop_x1),
                img_shape[2],
            )
        else:
            data["img_shape"] = (
                img_shape[2],
                int(crop_y2) - int(crop_y1),
                int(crop_x2) - int(crop_x1),
            )
        return (
            offset_h,
            offset_w,
            np.array([crop_y1, crop_y2, crop_x1, crop_x2]),
        )

    def _crop_img(self, data, offset_h, offset_w, crop_bbox):
        if "limit_box" in data:
            limit_box = data["limit_box"]
            crop_bbox[2] = min(crop_bbox[2], limit_box[0])
            crop_bbox[0] = min(crop_bbox[0], limit_box[1])
            crop_bbox[3] = max(crop_bbox[3], limit_box[2])
            crop_bbox[1] = max(crop_bbox[1], limit_box[3])
        # crop the image
        if data["layout"] == "hwc":
            crop_img = data["img"][
                int(crop_bbox[0]) : int(crop_bbox[1]),
                int(crop_bbox[2]) : int(crop_bbox[3]),
                :,
            ]
        else:
            crop_img = data["img"][
                :,
                int(crop_bbox[0]) : int(crop_bbox[1]),
                int(crop_bbox[2]) : int(crop_bbox[3]),
            ]
        img_shape = crop_img.shape
        data["img"] = crop_img
        data["img_shape"] = img_shape
        data["pad_shape"] = img_shape
        data["crop_offset"] = [offset_w, offset_h, offset_w, offset_h]
        data["crop_bbox"] = crop_bbox

    def _crop_bbox(self, data, offset_h, offset_w):
        # crop bboxes and clip to the image boundary
        img_shape = data["img_shape"]
        if data["layout"] == "chw":
            img_shape = (img_shape[1], img_shape[2], img_shape[0])
        bbox_offset = np.array(
            [offset_w, offset_h, offset_w, offset_h], dtype=np.float32
        )
        if data["gt_bboxes"].any():
            bboxes = data["gt_bboxes"] - bbox_offset
            boxes_real = bboxes.copy()
            classes = data["gt_classes"].copy()
            classes_real = classes.copy()
            gt_ids_flag = "gt_ids" in data
            if gt_ids_flag:
                gt_ids = data["gt_ids"].copy()
            else:
                gt_ids = data["gt_classes"].copy()
            if self.rm_neg_coords:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            # default setttings, ignore the tiny boxes
            area = np.maximum(0, bboxes[:, 2] - bboxes[:, 0]) * np.maximum(
                0, bboxes[:, 3] - bboxes[:, 1]
            )
            if self.filter_area:
                area_ignore_index = (area < 1e-3) & (classes >= 0)
                if self.discriminate_ignore_classes:
                    classes[area_ignore_index] = -abs(
                        classes[area_ignore_index]
                        + int(self.without_background)
                    )
                    gt_ids[area_ignore_index] = -abs(
                        gt_ids[area_ignore_index]
                        + int(self.without_background)
                    )
                else:
                    classes[area_ignore_index] = -1
                    gt_ids[area_ignore_index] = -1
            # user settings, ignore boxes whose area is less than min_area
            if self.min_area > 0:
                area_ignore_index = (area < self.min_area) & (classes >= 0)
                if self.discriminate_ignore_classes:
                    classes[area_ignore_index] = -abs(
                        classes[area_ignore_index]
                        + int(self.without_background)
                    )
                    gt_ids[area_ignore_index] = -abs(
                        gt_ids[area_ignore_index]
                        + int(self.without_background)
                    )
                else:
                    classes[area_ignore_index] = -1
                    gt_ids[area_ignore_index] = -1
            # ignore the regions where the iou between before and after
            # truncation < min_iou
            if self.min_iou > 0:
                if self.discard_truncate:
                    discard_ids = [True] * boxes_real.shape[0]
                for i in range(boxes_real.shape[0]):
                    iou = bbox_overlaps(
                        bboxes[i].reshape((1, -1)),
                        boxes_real[i].reshape((1, -1)),
                    )[0][0]
                    if iou < self.min_iou and classes[i] >= 0:
                        if self.discard_truncate:
                            discard_ids[i] = False
                        elif self.discriminate_ignore_classes:
                            classes[i] = -1 * (
                                abs(classes[i]) + int(self.without_background)
                            )
                            gt_ids[i] = -1 * (
                                abs(gt_ids[i]) + int(self.without_background)
                            )
                        else:
                            classes[i] = -1
                            gt_ids[i] = -1
                if self.discard_truncate:
                    classes = classes[discard_ids]
                    gt_ids = gt_ids[discard_ids]
                    bboxes = bboxes[discard_ids]
                    boxes_real = boxes_real[discard_ids]
                classes_real = classes.copy()
                gt_ids_real = gt_ids.copy()

            # filter out the gt bboxes that are completely cropped
            if self.rm_neg_coords:
                valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                    bboxes[:, 3] > bboxes[:, 1]
                )
            else:
                cropped_boxes = bboxes.reshape(-1, 2, 2)
                max_size = np.array([img_shape[1], img_shape[0]])
                cropped_boxes = np.minimum(
                    cropped_boxes.reshape(-1, 2, 2), max_size
                )
                cropped_boxes = np.clip(cropped_boxes, 0, 1000000)
                valid_inds = np.all(
                    cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1
                )
            bboxes = bboxes[valid_inds]
            classes = classes[valid_inds]
            gt_ids = gt_ids[valid_inds]
            # if no gt bbox remains after cropping, set bboxes shape (0, 4)
            # keep dtype consistency with origin bboxes and classes
            if not np.any(valid_inds):
                boxes_real = bboxes = np.zeros(
                    (0, 4), dtype=data["gt_bboxes"].dtype
                )
                classes_real = classes = np.zeros(
                    (0,), dtype=data["gt_classes"].dtype
                )
                if gt_ids_flag:
                    gt_ids_real = gt_ids = np.zeros(
                        (0,), dtype=data["gt_ids"].dtype
                    )
                else:
                    gt_ids_real = gt_ids = np.zeros(
                        (0,), dtype=data["gt_classes"].dtype
                    )
                if "gt_tanalphs" in data:
                    data["gt_tanalphs"] = np.zeros((0,), dtype=np.float32)
            data["gt_bboxes"] = bboxes if self.truncate_gt else boxes_real
            data["gt_classes"] = classes if self.truncate_gt else classes_real
            if gt_ids_flag:
                data["gt_ids"] = gt_ids if self.truncate_gt else gt_ids_real

    def _crop_seg(self, data, crop_bbox):
        gt_seg = data["gt_seg"]
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        crop_gt_seg = gt_seg[
            int(crop_y1) : int(crop_y2), int(crop_x1) : int(crop_x2)
        ]
        data["gt_seg"] = crop_gt_seg

    def _crop_flow(self, data, crop_bbox):
        gt_flow = data["gt_flow"]
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        crop_gt_flow = gt_flow[
            int(crop_y1) : int(crop_y2), int(crop_x1) : int(crop_x2)
        ]
        data["gt_flow"] = crop_gt_flow

    def _crop_disp(self, data, crop_bbox):
        gt_disp = data["gt_disp"]
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        crop_gt_disp = gt_disp[
            int(crop_y1) : int(crop_y2), int(crop_x1) : int(crop_x2)
        ]
        data["gt_disp"] = crop_gt_disp

    def _crop_lines(self, data, offset_h, offset_w):
        gt_lines = data["gt_lines"]
        for gt_line in gt_lines:
            gt_line[:, 0] = gt_line[:, 0] - offset_w
            gt_line[:, 1] = gt_line[:, 1] - offset_h

        data["gt_lines"] = gt_lines

    def inverse_transform(
        self, inputs: torch.Tensor, task_type: str, inverse_info: Dict
    ):
        """Inverse option of transform to map the prediction to the original image.

        Args:
            inputs (array): Prediction
            task_type (str): `detection` or `segmentation`.
            inverse_info (dict): The transform keyword is the key,
                and the corresponding value is the value.

        """
        if task_type == "detection":
            crop_offset = inverse_info["crop_offset"]
            if not isinstance(crop_offset, torch.Tensor):
                crop_offset = inputs.new_tensor(crop_offset)
            crop_offset = crop_offset.reshape((1, 4))
            inputs[:, :4] = inputs[:, :4] + crop_offset
            return inputs
        elif task_type == "segmentation":
            before_crop_shape = inverse_info["before_crop_shape"][1:]
            crop_offset_x, crop_offset_y = inverse_info["crop_offset"][:2]
            crop_h, crop_w = inputs.shape
            out_img = np.full(before_crop_shape, 255)
            out_img[
                crop_offset_y : crop_offset_y + crop_h,
                crop_offset_x : crop_offset_x + crop_w,
            ] = inputs.cpu().numpy()
            return out_img
        else:
            raise Exception(
                "error task_type, your task_type[{}],"
                " we need segmentation or detection".format(task_type)
            )

    def __call__(self, data):
        data["crop_offset"] = [0, 0, 0, 0]
        data["pad_shape"] = data["img"].shape
        data_without_img = copy.deepcopy(data)
        image = data_without_img.pop("img")
        if self.crop_around_gt:
            assert "gt_bboxes" in data
        for i in range(self.repeat_times):
            data_this = copy.deepcopy(data_without_img)
            offset_h, offset_w, crop_bbox = self._get_crop_region(data_this, i)
            if "gt_bboxes" in data and self._is_good_enough(data, crop_bbox):
                break
        data = data_this
        data["img"] = image
        self._crop_img(data, offset_h, offset_w, crop_bbox)
        if "gt_bboxes" in data:
            self._crop_bbox(data_this, offset_h, offset_w)
        if "gt_seg" in data:
            self._crop_seg(data, crop_bbox)
        elif "gt_flow" in data:
            self._crop_flow(data, crop_bbox)
        elif "gt_disp" in data:
            self._crop_disp(data, crop_bbox)
        elif "gt_lines" in data:
            self._crop_lines(data, offset_h, offset_w)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"size={self.size}, "
        repr_str += f"min_area={self.min_area}, "
        repr_str += f"min_iou={self.min_iou}"
        repr_str += f"keep_raw_pattern={self.keep_raw_pattern}"
        return repr_str


@OBJECT_REGISTRY.register
class FixedCrop(RandomCrop):
    """Crop image with fixed position and size.

    .. note::
        Affected keys: 'img', 'img_shape', 'pad_shape', 'layout',
        'before_crop_shape', 'crop_offset', 'gt_bboxes', 'gt_classes'.

    """

    def __init__(
        self,
        size: Tuple[int] = None,
        min_area: int = -1,
        min_iou: int = -1,
        dynamic_roi_params: Dict = None,
        discriminate_ignore_classes: Optional[bool] = False,
        allow_smaller: bool = False,
    ):  # noqa: D205,D400
        """

        Args:
            size (Tuple): Expected size after cropping, (w, h) or
                (x1, y1, w, h).
            min_area (Optional[int]): If min_area > 0, boxes whose areas are
                less than min_area will be ignored.
            min_iou (Optional[float]): If min_iou > 0, boxes whose iou between
                before and after truncation < min_iou will be ignored.
            dynamic_roi_param (Dict): Dynamic ROI parameters contains keys
                {'w', 'h', 'fp_x', 'fp_y'}
            discriminate_ignore_classes (Optional[bool]): if True, ignored area
                by min_iou retain the original class info. And class id should
                greater than 0.
                Default False. Support class id starts from 0.
            allow_smaller: Weather to allow the image shape smaller
                than roi after resize.
        """
        if not dynamic_roi_params:
            assert size is not None
            if len(size) == 2:
                size = (0, 0, size[0], size[1])
            else:
                assert len(size) == 4
        self.size = size
        self.min_area = min_area
        self.min_iou = min_iou
        self.dynamic_roi_params = dynamic_roi_params
        self.discriminate_ignore_classes = discriminate_ignore_classes
        self.allow_smaller = allow_smaller

    def _crop_img(self, data):
        img = data["img"]
        if self.dynamic_roi_params:
            crop_roi, crop_roi_on_orig = get_dynamic_roi_from_camera(
                camera_info=data["camera_info"],
                dynamic_roi_params=self.dynamic_roi_params,
                img_hw=data["img_shape"][:2],
                infer_model_type=data["infer_model_type"],
            )
            data["crop_roi"] = crop_roi_on_orig
            x1, y1, x2, y2 = crop_roi
        else:
            x1, y1, w, h = self.size
            x2, y2 = x1 + w, y1 + h
            assert w > 0 and h > 0

        if not self.allow_smaller:
            assert x2 <= img.shape[1] and y2 <= img.shape[0]
        offset_w, offset_h = x1, y1

        # crop the image
        crop_img = img[y1:y2, x1:x2, :]
        data["img"] = crop_img
        data["img_shape"] = crop_img.shape
        data["before_crop_shape"] = img.shape
        data["crop_offset"] = [x1, y1, x1, y1]
        return offset_h, offset_w

    def _crop_bbox(self, data, offset_h, offset_w):
        # crop bboxes and clip to the image boundary
        img_shape = data["img_shape"]
        bbox_offset = np.array(
            [offset_w, offset_h, offset_w, offset_h], dtype=np.float32
        )
        if data["gt_bboxes"].any():
            bboxes = data["gt_bboxes"] - bbox_offset
            boxes_real = bboxes.copy()
            classes = data["gt_classes"].copy()
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            # default setttings, ignore the tiny boxes
            area = np.maximum(0, bboxes[:, 2] - bboxes[:, 0]) * np.maximum(
                0, bboxes[:, 3] - bboxes[:, 1]
            )
            area_ignore_index = (area < 1e-3) & (classes >= 0)
            if self.discriminate_ignore_classes:
                classes[area_ignore_index] = -abs(classes[area_ignore_index])
            else:
                classes[area_ignore_index] = -1
            # user settings, ignore boxes whose area is less than min_area
            if self.min_area > 0:
                area_ignore_index = (area < self.min_area) & (classes >= 0)
                if self.discriminate_ignore_classes:
                    classes[area_ignore_index] = -abs(
                        classes[area_ignore_index]
                    )
                else:
                    classes[area_ignore_index] = -1
            # ignore the regions where the iou between before and after
            # truncation < min_iou
            if self.min_iou > 0:
                for i in range(boxes_real.shape[0]):
                    iou = bbox_overlaps(
                        bboxes[i].reshape((1, -1)),
                        boxes_real[i].reshape((1, -1)),
                    )[0][0]
                    if iou < self.min_iou and classes[i] >= 0:
                        if self.discriminate_ignore_classes:
                            classes[i] = -1 * abs(classes[i])
                        else:
                            classes[i] = -1

            # filter out the gt bboxes that are completely cropped
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1]
            )
            # if no gt bbox remains after cropping, set bboxes shape (0, 4)
            if not np.any(valid_inds):
                bboxes = np.zeros((0, 4), dtype=np.float32)
                classes = np.zeros((0,), dtype=np.int64)
            data["gt_bboxes"] = bboxes
            data["gt_classes"] = classes

    def _crop_lines(self, data, offset_h, offset_w):
        gt_lines = data["gt_lines"]
        for gt_line in gt_lines:
            gt_line[:, 0] = gt_line[:, 0] - offset_w
            gt_line[:, 1] = gt_line[:, 1] - offset_h

        data["gt_lines"] = gt_lines

    def inverse_transform(
        self, inputs: torch.Tensor, task_type: str, inverse_info: Dict
    ):
        """Inverse option of transform to map the prediction to the original image.

        Args:
            inputs (array): Prediction
            task_type (str): `detection` or `segmentation`.
            inverse_info (dict): The transform keyword is the key,
                and the corresponding value is the value.

        """
        if task_type == "detection":
            crop_offset = inverse_info["crop_offset"]
            if not isinstance(crop_offset, torch.Tensor):
                crop_offset = inputs.new_tensor(crop_offset)
            crop_offset = crop_offset.reshape((1, 4))
            inputs[:, :4] = inputs[:, :4] + crop_offset
            return inputs
        elif task_type == "segmentation":
            crop_offset_x, crop_offset_y = inverse_info["crop_offset"][:2]
            crop_h, crop_w = inputs.shape
            before_crop_shape = (
                inverse_info["before_crop_shape"][:2]
                if inverse_info["before_crop_shape"][0] > 3
                else inverse_info["before_crop_shape"][1:]
            )
            if before_crop_shape:
                out_img = np.full(before_crop_shape, 255)
                out_img[
                    crop_offset_y : crop_offset_y + crop_h,
                    crop_offset_x : crop_offset_x + crop_w,
                ] = inputs.cpu().numpy()
            else:
                out_img = inputs.cpu().numpy()
            return out_img
        elif task_type == "cone_invasion":
            return inputs
        else:
            raise Exception(
                "error task_type, your task_type[{}],"
                " we need segmentation or detection".format(task_type)
            )

    def __call__(self, data):
        offset_h, offset_w = self._crop_img(data)
        if "gt_bboxes" in data:
            self._crop_bbox(data, offset_h, offset_w)
        if "gt_lines" in data:
            self._crop_lines(data, offset_h, offset_w)
        return data


@OBJECT_REGISTRY.register
class PresetCrop(object):
    """Crop image with preset roi param."""

    def __init__(
        self,
        crop_top: int = 220,
        crop_bottom: int = 128,
        crop_left: int = 0,
        crop_right: int = 0,
        min_area: float = -1,
        min_iou: float = -1,
        truncate_gt: bool = True,
    ):  # noqa: D205,D400
        """Crop image and labels with preset roi param.

        Args:
            crop_top: crop size from top boundary
            crop_bottom: crop size from bottom boundary
            crop_left: crop size from left boundary
            crop_right: crop size from right boundary
            min_area (Optional[int]): If min_area > 0, boxes whose areas are
                less than min_area will be ignored.
            min_iou (Optional[float]): If min_iou > 0, boxes whose iou between
                before and after truncation < min_iou will be ignored.
            truncate_gt (optional[bool]): If truncate gt bbox, set
                `truncate_gt = True`.
        """

        self.crop_top = int(crop_top)
        self.crop_bottom = int(crop_bottom)
        self.crop_left = int(crop_left)
        self.crop_right = int(crop_right)
        self.min_area = min_area
        self.min_iou = min_iou
        self.truncate_gt = truncate_gt

    def _crop_img(self, data):
        img = data["img"]
        x1, y1 = self.crop_left, self.crop_top
        x2 = img.shape[1] - self.crop_right
        y2 = img.shape[0] - self.crop_bottom

        assert x2 <= img.shape[1] and y2 <= img.shape[0]
        assert x2 > x1 and y2 > y1

        # crop the image
        crop_img = img[y1:y2, x1:x2, :]
        data["img"] = crop_img
        data["img_shape"] = crop_img.shape
        data["before_crop_shape"] = img.shape
        data["crop_offset"] = [
            self.crop_left,
            self.crop_top,
            self.crop_left,
            self.crop_top,
        ]

    def _crop_bbox(self, data):
        # crop bboxes and clip to the image boundary
        img_shape = data["img_shape"]
        bbox_offset = np.array(data["crop_offset"], dtype=np.float32)
        if data["gt_bboxes"].any():
            bboxes = data["gt_bboxes"] - bbox_offset
            boxes_real = bboxes.copy()
            classes = data["gt_classes"].copy()
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            # default setttings, ignore the tiny boxes
            area = np.maximum(0, bboxes[:, 2] - bboxes[:, 0]) * np.maximum(
                0, bboxes[:, 3] - bboxes[:, 1]
            )
            area_ignore_index = (area < 1e-3) & (classes >= 0)
            classes[area_ignore_index] = -1
            # user settings, ignore boxes whose area is less than min_area
            if self.min_area > 0:
                area_ignore_index = (area < self.min_area) & (classes >= 0)
                classes[area_ignore_index] = -1
            # ignore the regions where the iou between before and after
            # truncation < min_iou
            if self.min_iou > 0:
                for i in range(boxes_real.shape[0]):
                    iou = bbox_overlaps(
                        bboxes[i].reshape((1, -1)),
                        boxes_real[i].reshape((1, -1)),
                    )[0][0]
                    if iou < self.min_iou and classes[i] >= 0:
                        classes[i] = -1

            # filter out the gt bboxes that are completely cropped
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1]
            )
            # if no gt bbox remains after cropping, set bboxes shape (0, 4)
            if not np.any(valid_inds):
                bboxes = np.zeros((0, 4), dtype=np.float32)
                classes = np.zeros((0,), dtype=np.int64)
                if "gt_tanalphs" in data:
                    data["gt_tanalphs"] = np.zeros((0,), dtype=np.float32)
            data["gt_bboxes"] = bboxes if self.truncate_gt else boxes_real
            data["gt_classes"] = classes

    def inverse_transform(
        self, inputs: torch.Tensor, task_type: str, inverse_info: Dict
    ):
        """Inverse option of transform to map the prediction to the original image.

        Args:
            inputs: Prediction
            task_type: `detection` or `segmentation`.
            inverse_info: not used yet.

        """
        if task_type == "detection":
            crop_offset = np.array(
                [self.crop_left, self.crop_top, self.crop_left, self.crop_top],
                dtype=np.float32,
            )
            if not isinstance(crop_offset, torch.Tensor):
                crop_offset = inputs.new_tensor(crop_offset)
            crop_offset = crop_offset.reshape((1, 4))
            inputs[:, :4] = inputs[:, :4] + crop_offset
            return inputs
        else:
            raise Exception(
                "error task_type, your task_type[{}],"
                " we need segmentation or detection".format(task_type)
            )

    def __call__(self, data):
        self._crop_img(data)
        if "gt_bboxes" in data:
            self._crop_bbox(data)
        return data


@OBJECT_REGISTRY.register
class RandomCenterCrop(RandomCrop):  # noqa: D205,D400
    """Random center crop on data with gt label,
    can only be used for detection task.

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
        super(RandomCenterCrop, self).__init__(**kwargs)
        assert (
            len(cx_ratio_range) == 2 and len(cy_ratio_range) == 2
        ), "crop center range param is error."
        self.cx_ratio_range = cx_ratio_range
        self.cy_ratio_range = cy_ratio_range

    def _get_crop_region(self, data, iter_i=0):
        img_shape = data["img_shape"]
        layout = data["layout"]
        assert layout in ["hwc", "chw"], (
            "layout of img must be `chw` or " "`hwc`"
        )

        if layout == "chw":
            img_shape = (img_shape[1], img_shape[2], img_shape[0])
        img_h, img_w = img_shape[:2]
        crop_size = (
            list(self.size) if self.size is not None else [img_h, img_w]
        )
        rratio = self._get_rand_ratio(self.wh_ratio_range)
        crop_size = [s * rratio for s in crop_size]
        crop_size[0] *= self._get_rand_ratio(self.h_ratio_range)
        crop_size[1] *= self._get_rand_ratio(self.w_ratio_range)

        margin_h = max(img_shape[0] - crop_size[0], 1)
        margin_w = max(img_shape[1] - crop_size[1], 1)
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

        if self.crop_around_gt:
            # get the union box of all gt boxes
            union_box = self._union_of_bboxes(data)
            if union_box.any():
                union_x1, union_y1, union_x2, union_y2 = union_box
            else:
                union_x1, union_y1, union_x2, union_y2 = (
                    0,
                    0,
                    img_w - 1,
                    img_h - 1,
                )
            # get the range of the crop's up-left corner
            lower_lim_x1 = max(0, int(min(union_x1, union_x2 - crop_size[1])))
            lower_lim_y1 = max(0, int(min(union_y1, union_y2 - crop_size[0])))
            upper_lim_x1 = min(
                img_w, int(max(union_x1, union_x2 - crop_size[1]))
            )
            upper_lim_y1 = min(
                img_h, int(max(union_y1, union_y2 - crop_size[0]))
            )

            # avoid the cropped box exceeding the image boundary
            upper_lim_x1 = min(upper_lim_x1, int(max(0, img_w - crop_size[1])))
            upper_lim_y1 = min(upper_lim_y1, int(max(0, img_h - crop_size[0])))
            lower_lim_x1, upper_lim_x1 = min(lower_lim_x1, upper_lim_x1), max(
                lower_lim_x1, upper_lim_x1
            )
            lower_lim_y1, upper_lim_y1 = min(lower_lim_y1, upper_lim_y1), max(
                lower_lim_y1, upper_lim_y1
            )

            # randomly generate upper left corner coordinates and extend the
            # limits by 15 pixels
            if lower_lim_x1 < union_x1:
                lower_lim_x1 = lower_lim_x1 + 15
                upper_lim_x1 = max(0, upper_lim_x1 - 15)
                # in case lower_lim_x1 > upper_lim_x1 after the extension above
                if lower_lim_x1 > upper_lim_x1:
                    lower_lim_x1, upper_lim_x1 = upper_lim_x1, lower_lim_x1
            else:
                lower_lim_x1 = max(0, lower_lim_x1 - 15)
                upper_lim_x1 = upper_lim_x1 + 15
            if lower_lim_y1 < union_y1:
                lower_lim_y1 = lower_lim_y1 + 15
                upper_lim_y1 = max(0, upper_lim_y1 - 15)
                # in case lower_lim_y1 > upper_lim_y1 after the extension above
                if lower_lim_y1 > upper_lim_y1:
                    lower_lim_y1, upper_lim_y1 = upper_lim_y1, lower_lim_y1
            else:
                lower_lim_y1 = max(0, lower_lim_y1 - 15)
                upper_lim_y1 = upper_lim_y1 + 15
            if iter_i == self.repeat_times - 2:
                # randomly crop upper left part or lower right part if a good
                # crop can't be achieved in repeat_times-1 times
                if np.random.choice([0, 1]) == 0:
                    upper_lim_x1 = min(lower_lim_x1 + 1, img_w)
                    upper_lim_y1 = min(lower_lim_y1 + 1, img_h)
                else:
                    lower_lim_x1 = max(0, upper_lim_x1 - 1)
                    lower_lim_y1 = max(0, upper_lim_y1 - 1)

        offset_h = np.random.randint(lower_lim_y1, upper_lim_y1 + 1)
        offset_w = np.random.randint(lower_lim_x1, upper_lim_x1 + 1)

        if self.keep_raw_pattern:
            cur_pattern = data["cur_pattern"]
            tar_pattern = data["raw_pattern"]
            if tar_pattern == cur_pattern:
                x_offset, y_offset = 0, 0
            elif (
                cur_pattern[0] == tar_pattern[2]
                and cur_pattern[1] == tar_pattern[3]
            ):
                # rggb->gbrg, bggr->grbg
                x_offset, y_offset = 1, 0
            elif (
                cur_pattern[0] == tar_pattern[1]
                and cur_pattern[2] == tar_pattern[3]
            ):
                # rggb->grbg, bggr->gbrg
                x_offset, y_offset = 0, 1
            else:  # This is not happening in ["RGGB", "BGGR", "GRBG", "GBRG"]
                raise RuntimeError(
                    "Unexpected pair of input and target bayer pattern!"
                )
            if y_offset < margin_h + 1:
                offset_h = random.randrange(y_offset, margin_h + 1, 2)
            else:
                offset_h = y_offset
            if x_offset < margin_w + 1:
                offset_w = random.randrange(x_offset, margin_w + 1, 2)
            else:
                offset_w = x_offset
        crop_y1, crop_y2 = offset_h, min(offset_h + crop_size[0], img_h)
        crop_x1, crop_x2 = offset_w, min(offset_w + crop_size[1], img_w)
        if layout == "hwc":
            data["img_shape"] = (
                int(crop_y2) - int(crop_y1),
                int(crop_x2) - int(crop_x1),
                img_shape[2],
            )
        else:
            data["img_shape"] = (
                img_shape[2],
                int(crop_y2) - int(crop_y1),
                int(crop_x2) - int(crop_x1),
            )
        return (
            offset_h,
            offset_w,
            np.array([crop_y1, crop_y2, crop_x1, crop_x2]),
        )


@OBJECT_REGISTRY.register
class RandomSizeCrop(RandomCrop):
    def __init__(
        self,
        min_size: int,
        max_size: int,
        **kwargs,
    ):
        super(RandomSizeCrop, self).__init__(
            size=(min_size, max_size),
            **kwargs,
        )
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, data):
        if data["layout"] == "hwc":
            img_h, img_w = data["img_shape"][:2]
        else:
            img_h, img_w = data["img_shape"][1:]
        assert img_h > self.min_size and img_w > self.min_size, (
            f"img shape should be larger than {self.min_size},"
            "but got {img_h}x{img_w}"
        )
        w = random.randint(self.min_size, min(img_w, self.max_size))
        h = random.randint(self.min_size, min(img_h, self.max_size))
        self.size = (h, w)
        offset_h, offset_w, crop_bbox = self._get_crop_region(data)
        self._crop_img(data, offset_h, offset_w, crop_bbox)
        if "gt_bboxes" in data:
            self._crop_bbox(data, offset_h, offset_w)
        elif "gt_seg" in data:
            self._crop_seg(data, crop_bbox)
        elif "gt_flow" in data:
            self._crop_flow(data, crop_bbox)
        return data


@OBJECT_REGISTRY.register
class ToTensor(object):  # noqa: D205,D400
    """Convert objects of various python types to torch.Tensor and convert the
    img to yuv444 format if to_yuv is True.

    Supported types are: numpy.ndarray, torch.Tensor, Sequence, int, float.

    .. note::
        Affected keys: 'img', 'img_shape', 'pad_shape', 'layout', 'gt_bboxes',
        'gt_seg', 'gt_seg_weights', 'gt_flow', 'color_space'.

    Args:
        to_yuv: If true, convert the img to yuv444 format.
        use_yuv_v2: If true, use BgrToYuv444V2 when convert img to yuv format.

    """

    def __init__(
        self,
        to_yuv: bool = False,
        use_yuv_v2: bool = True,
        split_transform: bool = False,
    ):
        self.to_yuv = to_yuv
        self.use_yuv_v2 = use_yuv_v2
        self.split_transform = split_transform

    @staticmethod
    def _convert_layout(img: Union[torch.Tensor, np.ndarray], layout: str):
        # convert the layout from hwc to chw
        assert layout in ["hwc", "chw"]
        if layout == "chw":
            return img, layout

        if isinstance(img, torch.Tensor):
            img = img.permute((2, 0, 1))  # HWC > CHW
        elif isinstance(img, np.ndarray):
            img = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC > CHW
        else:
            raise TypeError
        return img, "chw"

    @staticmethod
    def _to_tensor(
        data: Union[torch.Tensor, np.ndarray, Sequence, int, float]
    ):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data.copy())
        elif isinstance(data, Sequence) and not isinstance(data, str):
            return torch.tensor(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError(
                f"type {type(data)} cannot be converted to tensor."
            )

    @staticmethod
    def _convert_hw_to_even(data):  # noqa: D205,D400
        """Convert hw of img and img-like labels to even because BgrToYuv444
        requires that the h and w of img are even numbers.
        """
        if data["layout"] == "hwc":
            h, w, c = data["img"].shape
        else:
            c, h, w = data["img"].shape
            if "before_crop_shape" in data:
                data["before_crop_shape"] = (
                    data["before_crop_shape"][2],
                    data["before_crop_shape"][0],
                    data["before_crop_shape"][1],
                )

        crop_h = h if (h % 2) == 0 else h - 1
        crop_w = w if (w % 2) == 0 else w - 1

        if data["layout"] == "hwc":
            data["img"] = data["img"][:crop_h, :crop_w, :]
        else:
            data["img"] = data["img"][:, :crop_h, :crop_w]

        if "gt_seg" in data:
            data["gt_seg"] = data["gt_seg"][:crop_h, :crop_w]
        if "gt_seg_weights" in data:
            data["gt_seg_weights"] = data["gt_seg_weights"][:crop_h, :crop_w]

        if "gt_flow" in data:
            data["gt_flow"] = data["gt_flow"][:, :crop_h, :crop_w]
        if "gt_depth" in data:
            data["gt_depth"] = data["gt_depth"][:crop_h, :crop_w]
        # update img_shape
        data["img_shape"] = np.array(data["img"].shape)
        # update pad_shape
        data["pad_shape"] = np.array(data["img"].shape)
        return data

    def __call__(self, data):
        # step1: convert the layout from hwc to chw
        data_layout = data["layout"]
        data["img"], data["layout"] = self._convert_layout(
            data["img"], data["layout"]
        )
        # update img_shape
        data["img_shape"] = np.array(data["img"].shape)
        # update pad_shape
        data["pad_shape"] = np.array(data["img"].shape)
        # step2: convert to tensor
        data["img"] = self._to_tensor(data["img"])
        if self.split_transform:
            data["me_in_img"] = self._to_tensor(data["me_in_img"])
        if "gt_bboxes" in data:
            data["gt_bboxes"] = self._to_tensor(data["gt_bboxes"])
        if "ig_bboxes" in data:
            data["ig_bboxes"] = self._to_tensor(data["ig_bboxes"])
        if "gt_tanalphas" in data:
            data["gt_tanalphas"] = self._to_tensor(data["gt_tanalphas"])
        if "gt_classes" in data:
            data["gt_classes"] = self._to_tensor(data["gt_classes"])
        if "gt_labels" in data:
            data["gt_labels"] = self._to_tensor(data["gt_labels"])
        if "gt_seg" in data:
            data["gt_seg"] = self._to_tensor(data["gt_seg"])
        if "gt_seg_weights" in data:
            data["gt_seg_weights"] = self._to_tensor(data["gt_seg_weights"])
        if "gt_flow" in data:
            data["gt_flow"], _ = self._convert_layout(
                data["gt_flow"], data_layout
            )
            data["gt_flow"] = self._to_tensor(data["gt_flow"])
        if "gt_depth" in data:
            data["gt_depth"] = self._to_tensor(data["gt_depth"])
        if "gt_ids" in data:
            data["gt_ids"] = self._to_tensor(data["gt_ids"])
        if "uv_map" in data:
            data["uv_map"] = self._to_tensor(data["uv_map"])
        if "gt_invasion_status" in data:
            data["gt_invasion_status"] = self._to_tensor(
                data["gt_invasion_status"]
            )
            data["gt_beside_valid"] = self._to_tensor(data["gt_beside_valid"])
            data["gt_invasion_scale"] = self._to_tensor(
                data["gt_invasion_scale"]
            )
        # step3: convert to yuv color_space, if necessary
        if self.to_yuv:
            data = self._convert_hw_to_even(data)
            color_space = data.get("color_space", None)
            if color_space is None:
                color_space = "bgr"
                logger.warning(
                    "current color_space is unknown, treat as bgr "
                    "by default"
                )
            rgb_input = True if color_space.lower() == "rgb" else False

            if self.use_yuv_v2:
                yuv_converter = BgrToYuv444V2(rgb_input)
            else:
                yuv_converter = BgrToYuv444(rgb_input=rgb_input)

            data["img"] = yuv_converter(data["img"])
            data["color_space"] = "yuv"

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"to_yuv={self.to_yuv}"
        return repr_str


@OBJECT_REGISTRY.register
class Batchify(object):
    def __init__(
        self,
        size: Sequence,
        divisor: int = 1,
        pad_val: Union[float, Sequence[float]] = 0,
        seg_pad_val: Optional[float] = 255,
        repeat: int = 1,
    ):
        """Collate the image-like data to the expected size.

        .. note::
            Affected keys: 'img', 'img_shape', 'layout', 'gt_seg'.

        Args:
            size (Tuple): The expected size of collated images, (h, w).
            divisor (int): Padded image edges will be multiple to divisor.
            pad_val: Values to be filled in
                padding areas for img, single value or a list of values with
                len c. E.g. : pad_val = 10, or pad_val = [10, 20, 30].
            seg_pad_val: Value to be filled in padding areas
                for gt_seg.
            repeat (int): The returned imgs will consist of repeat img.

        """
        size = list(size)
        size[0] = int(np.ceil(size[0] / divisor)) * divisor
        size[1] = int(np.ceil(size[1] / divisor)) * divisor
        self.size = size
        self.divisor = divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.repeat = repeat

    def __call__(self, data):
        # get max-shape
        short, long = min(self.size), max(self.size)
        if data["layout"] == "hwc":
            h, w = data["img"].shape[:2]
        else:
            h, w = data["img"].shape[1:]

        if w > h:
            max_shape = (short, long)
        else:
            max_shape = (long, short)
        # pad img
        padded_img = image_pad(
            data["img"], data["layout"], max_shape, pad_val=self.pad_val
        )
        data["imgs"] = [padded_img for _ in range(self.repeat)]
        data["pad_shape"] = np.array(padded_img.shape)
        # pad seg
        if "gt_seg" in data:
            padded_seg = image_pad(
                data["gt_seg"], "hw", max_shape, pad_val=self.seg_pad_val
            )
            data["gt_seg"] = padded_seg
        # delete useless key-value
        del data["img"]

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"size={self.size}, "
        repr_str += f"divisor={self.divisor}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"seg_pad_val={self.seg_pad_val}, "
        repr_str += f"repeat={self.repeat}"
        return repr_str


@OBJECT_REGISTRY.register
class ColorJitter(ColorJitter):  # noqa: D205,D400
    """Randomly change the brightness, contrast, saturation and
    hue of an image.

    For det and dict input are the main differences
    with ColorJitter in torchvision and the default settings have been
    changed to the most common settings.

    .. note::
        Affected keys: 'img'.

    Args:
        brightness (float or tuple of float (min, max)):
            How much to jitter brightness.
        contrast (float or tuple of float (min, max)):
            How much to jitter contrast.
        saturation (float or tuple of float (min, max)):
            How much to jitter saturation.
        hue (float or tuple of float (min, max)):
            How much to jitter hue.
    """

    @require_packages("torchvision")
    def __init__(
        self,
        brightness: Union[float, Tuple[float]] = 0.5,
        contrast: Union[float, Tuple[float]] = (0.5, 1.5),
        saturation: Union[float, Tuple[float]] = (0.5, 1.5),
        hue: float = 0.1,
    ):
        super(ColorJitter, self).__init__(
            brightness, contrast, saturation, hue
        )

    def __call__(self, data):
        assert "img" in data.keys()
        img = data["img"]
        img = super().__call__(img)
        data["img"] = img
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"brightness={self.brightness}, "
        repr_str += f"contrast={self.contrast}, "
        repr_str += f"saturation={self.saturation}, "
        repr_str += f"hue={self.hue}. "
        return repr_str


@OBJECT_REGISTRY.register
class RandomExpand(object):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    .. note::
        Affected keys: 'img', 'gt_bboxes'.

    Args:
        ratio_range (tuple): range of expand ratio.
        prob (float): probability of applying this transformation
    """

    def __init__(
        self,
        mean: Tuple = (0, 0, 0),
        ratio_range: Tuple = (1, 4),
        prob: float = 0.5,
    ):
        self.mean = mean
        self.ratio_range = ratio_range
        self.min_ratio, self.max_ratio = ratio_range
        self.prob = prob

    def __call__(self, data: Dict) -> Dict:
        """Call function to expand images, bounding boxes.

        Args:
            data (dict): Result dict from loading pipeline.

        Returns:
            Result dict with images, bounding boxes expanded
        """

        if random.uniform(0, 1) > self.prob:
            return data

        img = data["img"]

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        # speedup expand when meets large image
        if np.all(self.mean == self.mean[0]):
            expand_img = np.empty(
                (int(h * ratio), int(w * ratio), c), img.dtype
            )
            expand_img.fill(self.mean[0])
        else:
            expand_img = np.full(
                (int(h * ratio), int(w * ratio), c), self.mean, dtype=img.dtype
            )
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top : top + h, left : left + w] = img

        data["img"] = expand_img

        # expand bboxes
        data["gt_bboxes"] = data["gt_bboxes"] + np.tile((left, top), 2).astype(
            data["gt_bboxes"].dtype
        )

        # TODO(zhigang.yang, 0.5): expand segs
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        return repr_str


@OBJECT_REGISTRY.register
class MinIoURandomCrop(object):  # noqa: D205,D400
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    .. note::
        Affected keys: 'img', 'gt_bboxes', 'gt_classes', 'gt_difficult'.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).
        bbox_clip_border (bool): Whether clip the objects outside
            the border of the image. Defaults to True.
        repeat_num (float): Max repeat num for finding avaiable bbox.

    """

    def __init__(
        self,
        min_ious: Tuple[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size: float = 0.3,
        bbox_clip_border: bool = True,
        repeat_num: int = 50,
    ):
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border
        self.repeat_num = repeat_num

    def __call__(self, data):
        img = data["img"]
        assert "gt_bboxes" in data
        boxes = data["gt_bboxes"]
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                data["img_shape"] = img.shape
                data["img_height"] = img.shape[0]
                data["img_width"] = img.shape[1]
                return data

            min_iou = mode
            for _i in range(self.repeat_num):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(0, w - new_w)
                top = random.uniform(0, h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h))
                )
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)
                ).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    mask = is_center_of_bboxes_in_roi(boxes, patch)
                    if not mask.any():
                        continue

                    boxes = data["gt_bboxes"].copy()
                    mask = is_center_of_bboxes_in_roi(boxes, patch)
                    boxes = boxes[mask]
                    if self.bbox_clip_border:
                        boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                        boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                    boxes -= np.tile(patch[:2], 2)

                    data["gt_bboxes"] = boxes
                    # labels
                    if "gt_classes" in data:
                        data["gt_classes"] = data["gt_classes"][mask]
                    if "gt_difficult" in data:
                        data["gt_difficult"] = data["gt_difficult"][mask]

                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1] : patch[3], patch[0] : patch[2]]
                data["img"] = img
                data["img_shape"] = img.shape
                data["img_height"] = img.shape[0]
                data["img_width"] = img.shape[1]

                # TODO(zhigang.yang, 0.5): add seg mask
                return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(min_ious={self.min_ious}, "
        repr_str += f"min_crop_size={self.min_crop_size}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str


@OBJECT_REGISTRY.register
class AugmentHSV(object):
    """Random add color disturbance.

    Convert RGB img to HSV, and then randomly change the hue,
    saturation and value.

    .. note::
        Affected keys: 'img'.

    Args:
        hgain (float): Gain of hue.
        sgain (float): Gain of saturation.
        vgain (float): Gain of value.
        p (float): Prob.
    """

    def __init__(
        self,
        hgain: float = 0.5,
        sgain: float = 0.5,
        vgain: float = 0.5,
        p: float = 1.0,
    ):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.p = p

    def __call__(self, data):
        do_augment = np.random.choice([False, True], p=[1 - self.p, self.p])
        if do_augment:
            img = data["img"]
            if data.get("layout", None) == "chw":
                img = np.transpose(img, (1, 2, 0))

            r = (
                np.random.uniform(-1, 1, 3)
                * [self.hgain, self.sgain, self.vgain]
                + 1
            )
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
            dtype = img.dtype

            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            img_hsv = cv2.merge(
                (
                    cv2.LUT(hue, lut_hue),
                    cv2.LUT(sat, lut_sat),
                    cv2.LUT(val, lut_val),
                )
            ).astype(dtype)
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            if data.get("layout", None) == "chw":
                img = np.transpose(img, (2, 0, 1))

            data["img"] = img

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(hgain={self.hgain}, "
        repr_str += f"sgain={self.sgain}, "
        repr_str += f"vgain={self.vgain})"
        return repr_str


def get_dynamic_roi_from_camera(
    camera_info: Dict,
    dynamic_roi_params: Dict,
    img_hw: Union[List[int], Tuple[int, int]],
    infer_model_type: str,
) -> Union[List[int], Tuple[int]]:
    """
    Get dynamic roi from camera info.

    Args:
        camera_info (Dict): Camera info.
        dynamic_roi_params (Dict): Must contains keys
            {'w', 'h', 'fp_x', 'fp_y'}
        img_hw:  of 2 int height and width of the image.

    Returns:
        dynamic_roi: dynamic ROI coordinate [x1, y1, x2, y2]
            of the image.
    """
    from crop_roi.merge.utils import get_camera, get_fanishing_point

    assert isinstance(dynamic_roi_params, dict)
    assert (
        "w" in dynamic_roi_params
        and "h" in dynamic_roi_params
        and "fp_x" in dynamic_roi_params
        and "fp_y" in dynamic_roi_params
    ), f"{dynamic_roi_params}"
    cam_info = None if camera_info == 0 else camera_info
    fp_x = dynamic_roi_params["fp_x"]
    fp_y = dynamic_roi_params["fp_y"]
    w = dynamic_roi_params["w"]
    h = dynamic_roi_params["h"]
    if cam_info:
        cam = get_camera(cam_info)
        fanishing_point = get_fanishing_point(cam.gnd2img)
        if infer_model_type == "crop_with_resize_quarter":
            fanishing_point = [loc / 2 for loc in fanishing_point]
        x1, y1 = (
            fanishing_point[0] - fp_x,
            fanishing_point[1] - fp_y,
        )
        x2, y2 = x1 + w, y1 + h
        dynamic_roi = [x1, y1, x2, y2]
    else:
        fanishing_point = (img_hw[1] / 2, img_hw[0] / 2)
        x1, y1 = (
            fanishing_point[0] - fp_x,
            fanishing_point[1] - fp_y,
        )
        x2, y2 = (
            x1 + w,
            y1 + h,
        )
        dynamic_roi = [x1, y1, x2, y2]
    if infer_model_type == "crop_with_resize_quarter":
        dynamic_roi_on_orig = [coord * 2 for coord in dynamic_roi]
    else:
        dynamic_roi_on_orig = dynamic_roi
    dynamic_roi = [int(i) for i in dynamic_roi]
    dynamic_roi_on_orig = [int(i) for i in dynamic_roi_on_orig]
    return dynamic_roi, dynamic_roi_on_orig


def pad_detection_data(
    img,
    gt_boxes,
    ig_regions=None,
):
    """
    Pad detection data.

    Parameters
    ----------
    img : array
        With shape (H, W, 1) or (H, W, 3), required that
        H <= target_wh[1] and W <= target_wh[0]
    gt_boxes : array
        With shape (B, K), required that B <= max_gt_boxes_num
    ig_regions : array, optional
        With shape (IB, IK), required that IB <= max_ig_regions_num,
        by default None
    """
    im_hw = np.array(img.shape[:2]).reshape((2,))

    cast = Cast(np.float32)

    if ig_regions is None:
        return {
            "img": img,
            "im_hw": cast(im_hw),
            "gt_boxes": cast(gt_boxes),
        }

    return {
        "img": img,
        "im_hw": cast(im_hw),
        "gt_boxes": cast(gt_boxes),
        "ig_regions": cast(ig_regions),
    }


@OBJECT_REGISTRY.register
class IterableDetRoITransform:
    """
    Iterable transformer base on rois for object detection.

    Parameters
    ----------
    resize_wh : list/tuple of 2 int, optional
        Resize input image to target size, by default None
    complete_boxes : bool, optional
            Using the uncliped boxes, by default False.
    **kwargs :
        Please see :py:class:`AffineMatFromROIBoxGenerator` and
        :py:class:`ImageAffineTransform`
    """

    # TODO(alan): No need to use resize_wh.

    def __init__(
        self,
        target_wh,
        flip_prob,
        img_scale_range=(0.5, 2.0),
        roi_scale_range=(0.8, 1.0 / 0.8),
        min_sample_num=1,
        max_sample_num=1,
        center_aligned=True,
        inter_method=10,
        use_pyramid=True,
        pyramid_min_step=0.7,
        pyramid_max_step=0.8,
        pixel_center_aligned=True,
        min_valid_area=8,
        min_valid_clip_area_ratio=0.5,
        min_edge_size=2,
        rand_translation_ratio=0,
        rand_aspect_ratio=0,
        rand_rotation_angle=0,
        reselect_ratio=0,
        clip_bbox=True,
        rand_sampling_bbox=True,
        resize_wh=None,
        keep_aspect_ratio=False,
        complete_boxes=False,
    ):
        self._roi_ts = AffineMatFromROIBoxGenerator(
            target_wh=target_wh,
            scale_range=roi_scale_range,
            min_sample_num=min_sample_num,
            max_sample_num=max_sample_num,
            min_valid_edge=min_edge_size,
            min_valid_area=min_valid_area,
            center_aligned=center_aligned,
            rand_scale_range=img_scale_range,
            rand_translation_ratio=rand_translation_ratio,
            rand_aspect_ratio=rand_aspect_ratio,
            rand_rotation_angle=rand_rotation_angle,
            flip_prob=flip_prob,
            rand_sampling_bbox=rand_sampling_bbox,
            reselect_ratio=reselect_ratio,
        )
        self._img_ts = ImageAffineTransform(
            dst_wh=target_wh,
            inter_method=inter_method,
            border_value=0,
            use_pyramid=use_pyramid,
            pyramid_min_step=pyramid_min_step,
            pyramid_max_step=pyramid_max_step,
            pixel_center_aligned=pixel_center_aligned,
        )
        self._bbox_ts_kwargs = {
            "clip": clip_bbox,
            "min_valid_area": min_valid_area,
            "min_valid_clip_area_ratio": min_valid_clip_area_ratio,
            "min_edge_size": min_edge_size,
            "complete_boxes": complete_boxes,
        }
        self._resize_wh = resize_wh
        self._use_pyramid = use_pyramid
        self._pyramid_min_step = pyramid_min_step
        self._pyramid_max_step = pyramid_max_step
        self._keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, data):
        assert isinstance(data, (dict))
        assert "img" in data.keys()
        assert "gt_boxes" in data.keys()
        img = data.get("img")
        gt_boxes = data.get("gt_boxes")
        ig_regions = data.get("ig_regions", None)

        if self._keep_aspect_ratio and self._resize_wh:
            origin_wh = img.shape[:2][::-1]
            resize_wh_ratio = float(self._resize_wh[0]) / float(
                self._resize_wh[1]
            )  # noqa
            origin_wh_ratio = float(origin_wh[0]) / float(origin_wh[1])
            affine = np.array([[1.0, 0, 0], [0, 1.0, 0]])

            if resize_wh_ratio > origin_wh_ratio:
                new_wh = (
                    int(origin_wh[1] * resize_wh_ratio),
                    origin_wh[1],
                )  # noqa
                img = cv2.warpAffine(img, affine, new_wh, 0)
            elif resize_wh_ratio < origin_wh_ratio:
                new_wh = (
                    origin_wh[0],
                    int(origin_wh[0] / resize_wh_ratio),
                )  # noqa
                img = cv2.warpAffine(img, affine, new_wh, 0)
        else:
            if self._use_pyramid:
                img = AlphaImagePyramid(
                    img,
                    scale_step=np.random.uniform(
                        self._pyramid_min_step, self._pyramid_max_step
                    ),
                )

        roi = gt_boxes.copy()

        if self._resize_wh is None:
            img_wh = img.shape[:2][::-1]
            affine_mat = AffineMat2DGenerator.identity()
        else:
            img_wh = self._resize_wh
            affine_mat = resize_affine_mat(
                img.shape[:2][::-1], self._resize_wh
            )
            roi = LabelAffineTransform(label_type="box")(
                roi, affine_mat, flip=False
            )

        for affine_aug_param in self._roi_ts(roi, img_wh):  # noqa
            new_affine_mat = AffineMat2DGenerator.stack_affine_transform(
                affine_mat, affine_aug_param.mat
            )[:2]
            affine_aug_param = AffineAugMat(
                mat=new_affine_mat, flipped=affine_aug_param.flipped
            )
            ts_img = self._img_ts(img, affine_aug_param.mat)

            ts_img_wh = ts_img.shape[:2][::-1]
            ts_gt_boxes, ts_ig_regions = _transform_bboxes(
                gt_boxes,
                ig_regions,
                (0, 0, ts_img_wh[0], ts_img_wh[1]),
                affine_aug_param,
                **self._bbox_ts_kwargs,
            )

            data = pad_detection_data(
                ts_img,
                ts_gt_boxes,
                ts_ig_regions,
            )
            data["img"] = data["img"].transpose(2, 0, 1)

            return data


@OBJECT_REGISTRY.register
class PadDetData(object):
    def __init__(self, max_gt_boxes_num=100, max_ig_regions_num=100):
        self.max_gt_boxes_num = max_gt_boxes_num
        self.max_ig_regions_num = max_ig_regions_num

    def __call__(self, data):
        pad_shape = list(data["gt_boxes"].shape)
        pad_shape[0] = self.max_gt_boxes_num
        data["gt_boxes_num"] = (
            np.array(data["gt_boxes"].shape[0])
            .reshape((1,))
            .astype(np.float32)
        )
        data["gt_boxes"] = _pad_array(
            data["gt_boxes"], pad_shape, "gt_boxes"
        ).astype(np.float32)

        if "ig_regions" in data:
            pad_shape = list(data["ig_regions"].shape)
            pad_shape[0] = self.max_ig_regions_num
            data["ig_regions_num"] = (
                np.array(data["ig_regions"].shape[0])
                .reshape((1,))
                .astype(np.float32)
            )
            data["ig_regions"] = _pad_array(
                data["ig_regions"], pad_shape, "ig_regions"
            ).astype(np.float32)

        return data


@OBJECT_REGISTRY.register
class DetAffineAugTransformer(object):
    """
    Affine augmentation for object detection.

    Args:
        resize_wh:
            Resize input image to target size, by default None
        complete_boxes:
            Using the uncliped boxes, by default False.
        **kwargs :
            Please see :py:func:`get_affine_image_resize` and
            :py:class:`ImageAffineTransform`
    """

    def __init__(
        self,
        target_wh,
        flip_prob,
        scale_type="W",
        inter_method=10,
        use_pyramid=True,
        pyramid_min_step=0.7,
        pyramid_max_step=0.8,
        pixel_center_aligned=True,
        center_aligned=False,
        rand_scale_range=(1.0, 1.0),
        rand_translation_ratio=0.0,
        rand_aspect_ratio=0.0,
        rand_rotation_angle=0.0,
        norm_wh=None,
        norm_scale=None,
        resize_wh: Union[Tuple[int, int], List[int]] = None,
        min_valid_area=8,
        min_valid_clip_area_ratio=0.5,
        min_edge_size=2,
        clip_bbox=True,
        keep_aspect_ratio=False,
        complete_boxes: bool = False,
    ):
        self._img_ts = ImageAffineTransform(
            dst_wh=target_wh,
            inter_method=inter_method,
            border_value=0,
            use_pyramid=use_pyramid,
            pyramid_min_step=pyramid_min_step,
            pyramid_max_step=pyramid_max_step,
            pixel_center_aligned=pixel_center_aligned,
        )
        self._bbox_ts = LabelAffineTransform(label_type="box")
        self._affine_kwargs = {
            "target_wh": target_wh,
            "scale_type": scale_type,
            "center_aligned": center_aligned,
            "rand_scale_range": rand_scale_range,
            "rand_translation_ratio": rand_translation_ratio,
            "rand_aspect_ratio": rand_aspect_ratio,
            "rand_rotation_angle": rand_rotation_angle,
            "flip_prob": flip_prob,
            "norm_wh": norm_wh,
            "norm_scale": norm_scale,
        }
        self._bbox_ts_kwargs = {
            "clip": clip_bbox,
            "min_valid_area": min_valid_area,
            "min_valid_clip_area_ratio": min_valid_clip_area_ratio,
            "min_edge_size": min_edge_size,
            "complete_boxes": complete_boxes,
        }
        self._resize_wh = resize_wh
        self._use_pyramid = use_pyramid
        self._pyramid_min_step = pyramid_min_step
        self._pyramid_max_step = pyramid_max_step
        self._keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, data):
        assert isinstance(data, (dict))
        assert "img" in data.keys()
        assert "gt_boxes" in data.keys()
        img = data.get("img")
        gt_boxes = data.get("gt_boxes")
        ig_regions = data.get("ig_regions", None)
        if self._keep_aspect_ratio and self._resize_wh:
            origin_wh = img.shape[:2][::-1]
            resize_wh_ratio = float(self._resize_wh[0]) / float(
                self._resize_wh[1]
            )  # noqa
            origin_wh_ratio = float(origin_wh[0]) / float(origin_wh[1])
            affine = np.array([[1.0, 0, 0], [0, 1.0, 0]])

            if resize_wh_ratio > origin_wh_ratio:
                new_wh = (
                    int(origin_wh[1] * resize_wh_ratio),
                    origin_wh[1],
                )  # noqa
                img = cv2.warpAffine(img, affine, new_wh, 0)
            elif resize_wh_ratio < origin_wh_ratio:
                new_wh = (
                    origin_wh[0],
                    int(origin_wh[0] / resize_wh_ratio),
                )  # noqa
                img = cv2.warpAffine(img, affine, new_wh, 0)
        else:
            if self._use_pyramid:
                img = AlphaImagePyramid(
                    img,
                    scale_step=np.random.uniform(
                        self._pyramid_min_step, self._pyramid_max_step
                    ),
                )

        if self._resize_wh is None:
            img_wh = img.shape[:2][::-1]
            affine_mat = AffineMat2DGenerator.identity()
        else:
            img_wh = self._resize_wh
            affine_mat = resize_affine_mat(
                img.shape[:2][::-1], self._resize_wh
            )

        affine_aug_param = get_affine_image_resize(
            img_wh, **self._affine_kwargs
        )

        affine_mat = AffineMat2DGenerator.stack_affine_transform(
            affine_mat, affine_aug_param.mat
        )[:2]
        affine_aug_param = AffineAugMat(
            mat=affine_mat, flipped=affine_aug_param.flipped
        )

        ts_img = self._img_ts(img, affine_aug_param.mat)
        ts_img_wh = ts_img.shape[:2][::-1]

        ts_gt_boxes, ts_ig_regions = _transform_bboxes(
            gt_boxes,
            ig_regions,
            (0, 0, ts_img_wh[0], ts_img_wh[1]),
            affine_aug_param,
            **self._bbox_ts_kwargs,
        )

        data = pad_detection_data(
            ts_img,
            ts_gt_boxes,
            ts_ig_regions,
        )
        data["img"] = data["img"].transpose(2, 0, 1)

        return data


@OBJECT_REGISTRY.register
class DetInputPadding(object):
    def __init__(self, input_padding: Tuple[int]):
        assert len(input_padding) == 4
        self.input_padding = input_padding

    def __call__(self, data):
        im_hw = data["im_hw"]
        im_hw[0] += self.input_padding[2] + self.input_padding[3]
        im_hw[1] += self.input_padding[0] + self.input_padding[1]

        data["gt_boxes"][..., :4:2] += self.input_padding[0]
        data["gt_boxes"][..., 1:4:2] += self.input_padding[2]

        if "ig_regions" in data:
            data["ig_regions"][..., :4:2] += self.input_padding[0]
            data["ig_regions"][..., 1:4:2] += self.input_padding[2]

        return data


@OBJECT_REGISTRY.register
class ToFasterRCNNData(object):
    """Prepare faster-rcnn input data.

    Convert ``gt_bboxes`` (n, 4) & ``gt_classes`` (n, ) to ``gt_boxes`` (n, 5),
    ``gt_boxes_num`` (1, ), ``ig_regions`` (m, 5), ``ig_regions_num`` (m, );
    If ``gt_ids`` exists, it will be concated into ``gt_boxes``, resulting in
    ``gt_boxes`` array shape expanding from nx5 to nx6.

    Convert key ``img_shape`` to ``im_hw``;
    Convert image Layout to ``chw``;

    Args:
        max_gt_boxes_num (int): Max gt bboxes number in one image,
            Default 500.
        max_ig_regions_num (int): Max ignore regions number in one image,
            Default 500.

    Returns:
        dict: Result dict with
            ``gt_boxes`` (max_gt_boxes_num, 5 or 6),
            ``gt_boxes_num`` (1, ),
            ``ig_regions`` (max_ig_regions_num, 5 or 6),
            ``ig_regions_num`` (1, ),
            ``im_hw`` (2,)
            ``layout`` convert to "chw".
    """

    def __init__(
        self,
        max_gt_boxes_num: int = 500,
        max_ig_regions_num: int = 500,
    ):
        self.max_gt_boxes_num = max_gt_boxes_num
        self.max_ig_regions_num = max_ig_regions_num
        self.faster_rcnn_input_keys = [
            "img",
            "im_hw",
            "gt_boxes",
            "gt_boxes_num",
            "ig_regions",
            "ig_regions_num",
            "layout",
            "img_name",
            "data_desc",
            "scale_factor",
            "ori_img",
            "frame_index",  # for track
            "camera",  # for person position
        ]

    def _cvt_bbox(self, data):
        bboxes = data["gt_bboxes"]
        # shape is Nx4, format is (x1, y1, x2, y2)
        assert bboxes.shape[-1] == 4
        classes = data["gt_classes"]
        gt_ids = data.get("gt_ids", None)

        all_boxes = np.hstack((bboxes, classes.reshape(-1, 1)))
        cls_indx = bboxes.shape[-1]
        if gt_ids is not None:
            # (x1, y1, x2, y2, cls, id)
            all_boxes = np.hstack((all_boxes, gt_ids.reshape(-1, 1)))
        gt_inds = all_boxes[:, cls_indx] > 0

        # gt_boxes shape should be padded to same shape
        # in order to collate to batch_data,
        # these value will be used in pytorch pulgin.

        # Just discard all gt bbox and ignore the whole image
        # for simple the code.
        gt_num = gt_inds.sum()
        if gt_num > self.max_gt_boxes_num:
            # ignore the whole image if it has too many bboxes
            all_boxes[gt_inds, cls_indx] *= -1
            gt_inds[...] = False
            gt_num = 0
        data["gt_boxes_num"] = np.array([gt_num], dtype=np.float32)
        data["gt_boxes"] = np.zeros(
            (self.max_gt_boxes_num, all_boxes.shape[1]), dtype=np.float32
        )
        data["gt_boxes"][0:gt_num] = all_boxes[gt_inds]

        # cls = 0 in all_boxes also be treat as ignore boxes.
        ig_inds = ~gt_inds
        ig_num = ig_inds.sum()
        if ig_num > self.max_ig_regions_num:
            ign_cls = set(abs(all_boxes[ig_inds, cls_indx])) - {0}
            ig_boxes_list = []
            for cls in ign_cls:
                ig_boxes_list.append(
                    [0, 0, data["im_hw"][1] - 1, data["im_hw"][0] - 1, -cls]
                )
            all_boxes = np.array(ig_boxes_list, dtype=np.float64)
            ig_num = all_boxes.shape[0]
            ig_inds = np.array([True] * ig_num)
            if gt_ids is not None:
                ig_gt_ids = np.array([[-1] * ig_num], dtype=np.float64)
                all_boxes = np.hstack([all_boxes, ig_gt_ids])
        data["ig_regions_num"] = np.array([ig_num], dtype=np.float32)
        data["ig_regions"] = np.zeros(
            (self.max_ig_regions_num, all_boxes.shape[1]), dtype=np.float32
        )
        data["ig_regions"][0:ig_num] = all_boxes[ig_inds]

    def _remove_redundance_keys(self, data):
        for key in list(data.keys()):
            if key not in self.faster_rcnn_input_keys:
                data.pop(key)

    def _cvt_img(self, data):
        img_shape = data["img_shape"]
        if data["layout"] == "hwc":
            assert len(img_shape) == 3
            data["im_hw"] = np.array(img_shape[:2], dtype=np.int32)
            data["layout"] = "chw"
            data["img"] = (data["img"]).transpose((2, 0, 1))
        elif data["layout"] == "chw":
            assert len(img_shape) == 3
            data["im_hw"] = np.array(img_shape[1:], dtype=np.int32)
        else:
            raise TypeError("Not Support Layout:{}".format(data["layout"]))

    def __call__(self, data):
        self._cvt_img(data)
        self._cvt_bbox(data)
        self._remove_redundance_keys(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"max_gt_boxes_num={self.max_gt_boxes_num}, "
        repr_str += f"max_ig_regions_num={self.max_ig_regions_num})"
        repr_str += "cvt img, cvt bbox"
        return repr_str


@OBJECT_REGISTRY.register
class ToLdmkRCNNData(ToFasterRCNNData):
    """Transform dataset to RCNN input need.

    This class is used to stack landmark with boxes, and typically used to
    facilitate landmark and boxes matching in anchor-based model.

    Args:
        num_ldmk: Number of landmark. Defaults to 15.
        max_gt_boxes_num: Max gt bboxes number in one image. Defaults to 1000.
        max_ig_regions_num: Max ignore regions number in one image.
            Defaults to 1000.
    """

    def __init__(
        self,
        num_ldmk=15,
        max_gt_boxes_num=1000,
        max_ig_regions_num=1000,
    ):
        super().__init__(max_gt_boxes_num, max_ig_regions_num)
        self.num_ldmk = num_ldmk

    def _cvt_ldmk(self, data):
        # stack labels
        _classes = data["gt_classes"]
        _bboxes = data["gt_bboxes"]
        _ldmk = data["gt_ldmk"]
        _ldmk = _ldmk.reshape(-1, self.num_ldmk * 3)
        gt_bboxes = np.column_stack((_bboxes, _classes, _ldmk))
        cls_indx = _bboxes.shape[-1]
        num_value_per_ldmk_bbox = gt_bboxes.shape[-1]
        # padding labels
        gt_inds = gt_bboxes[:, cls_indx] > 0

        # gt_boxes shape should be same in batch,
        # in order to concatate in collate.
        # these value will be used in faster-rcnn.
        gt_num = gt_inds.sum()
        assert (
            gt_num <= self.max_gt_boxes_num
        ), f"gt_num {gt_num} is exceed than \
                the max_gt_boxes_num {self.max_gt_boxes_num}"
        data["gt_boxes_num"] = np.array([gt_num], dtype=np.float32)
        data["gt_boxes"] = np.zeros(
            (self.max_gt_boxes_num, num_value_per_ldmk_bbox), dtype=np.float32
        )
        data["gt_boxes"][0:gt_num] = gt_bboxes[gt_inds]

        ig_inds = ~gt_inds
        ig_num = ig_inds.sum()
        assert (
            ig_num <= self.max_ig_regions_num
        ), f"ig_num {ig_num} is exceed than \
                the max_ig_boxes_num {self.max_ig_regions_num}"
        data["ig_regions_num"] = np.array([ig_num], dtype=np.float32)
        data["ig_regions"] = np.zeros(
            (self.max_ig_regions_num, num_value_per_ldmk_bbox),
            dtype=np.float32,
        )
        data["ig_regions"][0:ig_num] = gt_bboxes[ig_inds]

    def __call__(self, data):
        assert "img" in data
        assert "gt_ldmk" in data
        assert "gt_bboxes" in data
        assert "gt_classes" in data

        super()._cvt_img(data)
        self._cvt_ldmk(data)

        super()._remove_redundance_keys(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"num_ldmk={self.num_ldmk}"
        repr_str += f"max_gt_boxes_num={self.max_gt_boxes_num}"
        repr_str += f"max_ig_regions_num={self.max_ig_regions_num}"
        return repr_str


@OBJECT_REGISTRY.register
class ToMultiTaskFasterRCNNData(ToLdmkRCNNData):
    """Convert multi-classes detection data to multi-task data.

    Each class will be convert to a detection task.

    Args:
        taskname_clsidx_map: {cls1: cls_idx1, cls2: cls_idx2}.
        max_gt_boxes_num: Same as ToFasterRCNNData. Defaults to 500.
        max_ig_regions_num: Same as ToFasterRCNNData. Defaults to 500.
        num_ldmk: Number of human ldmk. Defaults to 15.

    Returns:
        dict: Result dict with
            "task1": FasterRCNNDataDict1,
            "task2": FasterRCNNDataDict2,

    """

    def __init__(
        self,
        taskname_clsidx_map: Dict[str, int],
        max_gt_boxes_num: int = 500,
        max_ig_regions_num: int = 500,
        num_ldmk: int = 15,
    ):
        super().__init__(num_ldmk, max_gt_boxes_num, max_ig_regions_num)
        self.taskname_clsidx_map = taskname_clsidx_map
        self.faster_rcnn_input_keys += list(taskname_clsidx_map.keys())

    def _cvt_data(self, data):
        bboxes = data["gt_bboxes"]
        # shape is Nx4, format is (x1, y1, x2, y2)
        assert bboxes.shape[-1] == 4
        classes = data["gt_classes"]
        gt_ids = data.get("gt_ids", None)
        im_hw = data["im_hw"]
        multi_task_data = {}
        for task_name, cls_idx in self.taskname_clsidx_map.items():
            _idx = (classes == cls_idx) | (classes == -cls_idx)
            _bboxes = bboxes[_idx]
            _classes = classes[_idx]
            # cvt to sing cls
            _classes[_classes > 0] = 1
            _classes[_classes < 0] = -1
            if gt_ids is not None:
                _gt_ids = gt_ids[_idx]
            else:
                _gt_ids = None
            task_data = {
                "gt_bboxes": _bboxes,
                "gt_classes": _classes,
                "gt_ids": _gt_ids,
                "im_hw": im_hw,
            }
            if ("ldmk" in task_name) or ("kps" in task_name):
                _gt_ldmk = data["gt_ldmk"]
                task_data["gt_ldmk"] = _gt_ldmk
                super()._cvt_ldmk(task_data)
            else:
                super()._cvt_bbox(task_data)
            super()._remove_redundance_keys(task_data)
            multi_task_data[task_name] = task_data
        data.update(multi_task_data)

    def __call__(self, data):
        super()._cvt_img(data)
        self._cvt_data(data)
        super()._remove_redundance_keys(data)
        return data


@OBJECT_REGISTRY.register
class PadTensorListToBatch(object):
    """List of image tensor to be stacked vertically.

    Used for diff shape tensors list.

    Args:
        pad_val: Values to be filled in padding areas for img.
            Default to 0.
        seg_pad_val: Value to be filled in padding areas
            for gt_seg.
            Default to 255.
    """

    def __init__(
        self,
        pad_val: int = 0,
        seg_pad_val: Optional[int] = 255,
    ):
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def pad_tensor_list_to_batch(self, tensor_list, pad_val):
        batch = len(tensor_list)
        ndim = len(tensor_list[0].shape)
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        dimensions = [batch]
        for dim in range(0, ndim):
            dimensions.append(
                max([tensor.shape[dim] for tensor in tensor_list])
            )

        all_tensor = pad_val * torch.ones(
            tuple(dimensions), dtype=dtype, device=device
        )

        if ndim == 1:
            for ind, tensor in enumerate(tensor_list):
                all_tensor[ind, 0] = tensor
        elif ndim == 2:
            for ind, tensor in enumerate(tensor_list):
                all_tensor[ind, : tensor.shape[0], : tensor.shape[1]] = tensor
        elif ndim == 3:
            for ind, tensor in enumerate(tensor_list):
                all_tensor[
                    ind,
                    : tensor.shape[0],
                    : tensor.shape[1],
                    : tensor.shape[2],
                ] = tensor
        else:
            raise Exception("NotImplementedError.")
        return all_tensor

    def __call__(self, data):
        # pad img
        img_tensor_list = data["img"]
        assert isinstance(img_tensor_list, list) and isinstance(
            img_tensor_list[0], torch.Tensor
        )
        padded_img = self.pad_tensor_list_to_batch(
            img_tensor_list, self.pad_val
        )
        data["img"] = padded_img
        data["pad_shape"] = torch.tensor(
            [padded_img.shape[-2:]] * padded_img.shape[0],
            dtype=torch.int,
            device=padded_img.device,
        )

        # update im_hw to padded shape,
        # used to clip rois in roi proposal,
        # in order to add more empty regions rois
        # as negative rois.
        # NOTE: Just to align with gluonperson.
        im_hw = torch.tensor(
            [padded_img.shape[-2:]] * padded_img.shape[0],
            dtype=torch.int,
            device=padded_img.device,
        )
        # recurrent update to support multitask data dict.
        for key in data:
            if key == "im_hw":
                data[key] = im_hw
            elif isinstance(data[key], Dict):
                for sub_key in data[key]:
                    if sub_key == "im_hw":
                        data[key][sub_key] = im_hw

        # pad seg
        if "gt_seg" in data:
            seg_tensor_list = data["gt_seg"]
            assert isinstance(seg_tensor_list, list) and isinstance(
                seg_tensor_list[0], torch.Tensor
            )
            padded_seg = self.pad_tensor_list_to_batch(
                img_tensor_list, self.seg_pad_val
            )
            data["gt_seg"] = padded_seg

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"seg_pad_val={self.seg_pad_val}, "
        return repr_str


@OBJECT_REGISTRY.register
class PlainCopyPaste(object):
    """Copy and paste instances plainly.

    Args:
        min_ins_num: Min instances num of the image after paste.
        cp_prob: Probability of applying this transformation.
    """

    def __init__(self, min_ins_num: int = 1, cp_prob: float = 0.0):
        self.min_ins_num = min_ins_num
        self.cp_prob = cp_prob
        self.cache_instances = []

    def __call__(self, data):
        gt_bboxes = data["gt_bboxes"]
        gt_classes = data["gt_classes"]
        img = data["img"]

        if random.random() < self.cp_prob:
            cp_num = self.min_ins_num - sum(gt_classes != -1)
            cp_ins = copy.copy(self.cache_instances)
            while len(cp_ins) > 0 and cp_num > 0:
                cp_bbox, cp_cls, cp_img = cp_ins.pop(0)
                left_top = np.maximum(
                    cp_bbox[None, None, :2], gt_bboxes[None, :, :2]
                )
                r_d = np.minimum(
                    cp_bbox[None, None, 2:4], gt_bboxes[None, :, 2:4]
                )
                areas = np.prod(r_d - left_top, axis=2) * (r_d > left_top).all(
                    axis=2
                )
                if (areas == 0).all():
                    x1, y1, x2, y2 = list(map(int, cp_bbox.tolist()))
                    img[y1:y2, x1:x2, :] = cp_img
                    gt_bboxes = np.concatenate([gt_bboxes, cp_bbox[None]])
                    gt_classes = np.concatenate(
                        [gt_classes, np.array([cp_cls])]
                    )
                    cp_num -= 1
            data["gt_bboxes"] = gt_bboxes
            data["gt_classes"] = gt_classes

        # update self.cache_instances
        if len(gt_classes) > 0 and sum(gt_classes != 0) > 0:
            for idx, gt_cls in enumerate(gt_classes):
                if gt_cls == -1:
                    continue
                gt_bbox = gt_bboxes[idx]
                x1, y1, x2, y2 = map(int, gt_bbox)
                crop_img = img[y1:y2, x1:x2, :]
                self.cache_instances.append([gt_bbox, gt_cls, crop_img])
                if len(self.cache_instances) > 5 * self.min_ins_num:
                    self.cache_instances.pop(0)
        return data


@OBJECT_REGISTRY.register
class HueSaturationValue(object):  # noqa: D205,D400
    """Randomly change hue, saturation and value of the input image.

    Used for unit8 np.ndarray, RGB image input. Unlike AugmentHSV,
    this transform uses addition to shift value. This transform is same as
    albumentations.augmentations.transforms.HueSaturationValue

    Args:
        hue_range: range for changing hue. Default: (-20, 20).
        sat_range: range for changing saturation. Default: (-30, 30).
        val_range: range for changing value. Default: (-20, 20).
        p: probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self,
        hue_range: Tuple[float, float] = (-20, 20),
        sat_range: Tuple[float, float] = (-30, 30),
        val_range: Tuple[float, float] = (-20, 20),
        p: float = 0.5,
    ):
        self.hue_range = hue_range
        self.sat_range = sat_range
        self.val_range = val_range
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            img = data["img"]
            r = [
                random.uniform(self.hue_range[0], self.hue_range[1]),
                random.uniform(self.sat_range[0], self.sat_range[1]),
                random.uniform(self.val_range[0], self.val_range[1]),
            ]

            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
            dtype = img.dtype

            x = np.arange(0, 256, dtype=np.int16)

            lut_hue = ((x + r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x + r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x + r[2], 0, 255).astype(dtype)
            img_hsv = cv2.merge(
                (
                    cv2.LUT(hue, lut_hue),
                    cv2.LUT(sat, lut_sat),
                    cv2.LUT(val, lut_val),
                )
            ).astype(dtype)
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            data["img"] = img
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(hue_range={self.hue_range}, "
        repr_str += f"sat_range={self.sat_range}, "
        repr_str += f"val_range={self.val_range})"
        return repr_str


@OBJECT_REGISTRY.register
class RGBShift(object):  # noqa: D205,D400
    """Randomly shift values for each channel of the input image.

    Used for np.ndarray. This transform is same as
    albumentations.augmentations.transforms.RGBShift.

    Args:
        r_shift_limit: range for changing values for the red channel.
            Default: (-20, 20).
        g_shift_limit: range for changing values for the green channel.
            Default: (-20, 20).
        b_shift_limit: range for changing values for the blue channel.
            Default: (-20, 20).
        p: probability of applying the transform. Default: 0.5.

    """

    def __init__(
        self,
        r_shift_limit: Tuple[float, float] = (-20, 20),
        g_shift_limit: Tuple[float, float] = (-20, 20),
        b_shift_limit: Tuple[float, float] = (-20, 20),
        p: float = 0.5,
    ):
        self.r_shift_limit = r_shift_limit
        self.g_shift_limit = g_shift_limit
        self.b_shift_limit = b_shift_limit
        self.p = p

    def get_params(self):
        r_shift = random.uniform(self.r_shift_limit[0], self.r_shift_limit[1])
        g_shift = random.uniform(self.g_shift_limit[0], self.g_shift_limit[1])
        b_shift = random.uniform(self.b_shift_limit[0], self.b_shift_limit[1])

        return r_shift, g_shift, b_shift

    def __call__(self, data):
        if random.random() < self.p:
            r_shift, g_shift, b_shift = self.get_params()

            img = data["img"]
            color_space = data.get("color_space", None)

            if color_space is None:
                color_space = "bgr"
                logger.warning(
                    "current color_space is unknown, treat as bgr "
                    "by default"
                )
            assert color_space.lower() in [
                "rgb",
                "bgr",
            ], "color_space must one of rgb and bgr using RGBShift"
            rgb_input = True if color_space.lower() == "rgb" else False

            layout = data.get("layout", None)
            if layout is None:
                layout = "hwc"
                logger.warning(
                    "current layout is unknown, treat as hwc " "by default"
                )
            assert layout.lower() in [
                "hwc",
                "chw",
            ], "layout must one of hwc and chw using RGBShift"

            hwc_input = True if layout.lower() == "hwc" else False
            if not hwc_input:
                img = np.ascontiguousarray(img.transpose((1, 2, 0)))
            shift_rgb_img = self.shift_rgb(
                img=img,
                r_shift=r_shift,
                g_shift=g_shift,
                b_shift=b_shift,
                rgb_input=rgb_input,
            )
            if not hwc_input:
                shift_rgb_img = np.ascontiguousarray(
                    shift_rgb_img.transpose((2, 0, 1))
                )
            data["img"] = shift_rgb_img
        return data

    def shift_rgb(self, img, r_shift, g_shift, b_shift, rgb_input):
        if not rgb_input:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        if img.dtype == np.uint8:
            shift_img = self._shift_rgb_uint8(img, r_shift, g_shift, b_shift)
        else:
            shift_img = self._shift_rgb_non_uint8(
                img, r_shift, g_shift, b_shift
            )
        if not rgb_input:
            cv2.cvtColor(shift_img, cv2.COLOR_RGB2BGR, shift_img)
        return shift_img

    def _shift_image_uint8(self, img, value):
        max_value = 255
        lut = np.arange(0, max_value + 1).astype("float32")
        lut += value
        lut = np.clip(lut, 0, max_value).astype(img.dtype)
        return cv2.LUT(img, lut)

    def _shift_rgb_uint8(self, img, r_shift, g_shift, b_shift):
        h, w, c = img.shape
        if r_shift == g_shift == b_shift:
            img = img.reshape([h, w * c])
            result_img = self._shift_image_uint8(img, r_shift)
        else:
            result_img = np.empty_like(img)
            shifts = [r_shift, g_shift, b_shift]
            for i, shift in enumerate(shifts):
                result_img[..., i] = self._shift_image_uint8(
                    img[..., i], shift
                )

        return result_img.reshape([h, w, c])

    def _shift_rgb_non_uint8(self, img, r_shift, g_shift, b_shift):
        if r_shift == g_shift == b_shift:
            return img + r_shift

        result_img = np.empty_like(img)
        shifts = [r_shift, g_shift, b_shift]
        for i, shift in enumerate(shifts):
            result_img[..., i] = img[..., i] + shift
        return result_img


@OBJECT_REGISTRY.register
class MeanBlur(object):  # noqa: D205,D400
    """Apply mean blur to the input image using a fix-sized kernel.

    Used for np.ndarray.

    Args:
        ksize: maximum kernel size for blurring the input image.
            Default: 3.
        p: probability of applying the transform. Default: 0.5.

    """

    def __init__(self, ksize: int = 3, p: float = 0.5):
        self.ksize = ksize
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            blur_img = cv2.blur(data["img"], ksize=(self.ksize, self.ksize))
            data["img"] = blur_img
        return data


@OBJECT_REGISTRY.register
class MedianBlur(object):  # noqa: D205,D400
    """Apply median blur to the input image using a fix-sized kernel.

    Used for np.ndarray.

    Args:
        ksize: maximum kernel size for blurring the input image.
            Default: 3.
        p: probability of applying the transform. Default: 0.5.

    """

    def __init__(self, ksize: int = 3, p: float = 0.5):
        if ksize % 2 != 1:
            raise ValueError("MedianBlur supports only odd blur limits.")
        self.ksize = ksize
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            median_blur_img = cv2.medianBlur(data["img"], ksize=self.ksize)
            data["img"] = median_blur_img
        return data


@OBJECT_REGISTRY.register
class RandomBrightnessContrast(object):
    """Randomly change brightness and contrast of the input image.

    Used for unit8 np.ndarray. This transform is same as
    albumentations.augmentations.transforms.RandomBrightnessContrast.

    Args:
        brightness_limit: factor range for changing brightness.
            Default: (-0.2, 0.2).
        contrast_limit: factor range for changing contrast.
            Default: (-0.2, 0.2).
        brightness_by_max: If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p: probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self,
        brightness_limit: Tuple[float, float] = (-0.2, 0.2),
        contrast_limit: Tuple[float, float] = (-0.2, 0.2),
        brightness_by_max: bool = True,
        p=0.5,
    ):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.brightness_by_max = brightness_by_max
        self.p = p

    def _brightness_contrast_adjust(
        self, img, alpha=1, beta=0, beta_by_max=False
    ):
        dtype = np.dtype("uint8")
        max_value = 255
        lut = np.arange(0, max_value + 1).astype("float32")
        if alpha != 1:
            lut *= alpha
        if beta != 0:
            if beta_by_max:
                lut += beta * max_value
            else:
                lut += beta * np.mean(img)

        lut = np.clip(lut, 0, max_value).astype(dtype)
        img = cv2.LUT(img, lut)
        return img

    def __call__(self, data):
        if random.random() < self.p:
            img = data["img"]
            alpha = 1.0 + random.uniform(
                self.contrast_limit[0],
                self.contrast_limit[1],
            )
            beta = random.uniform(
                self.brightness_limit[0],
                self.brightness_limit[1],
            )

            result_img = self._brightness_contrast_adjust(
                img, alpha, beta, self.brightness_by_max
            )
            data["img"] = result_img
        return data


@OBJECT_REGISTRY.register
class ShiftScaleRotate(object):
    """Randomly apply affine transforms: translate, scale and rotate the input.

    Used for np.ndarray hwc img. This transform is same as
    albumentations.augmentations.transforms.ShiftScaleRotate.

    Args:
        shift_limit: shift factor range for both height and width.
                     Absolute values for lower and upper bounds should lie in
                     range [0, 1]. Default: (-0.0625, 0.0625).
        scale_limit: scaling factor range. Default: (-0.1, 0.1).
        rotate_limit: rotation range. Default: (-45, 45).
        interpolation: flag that is used to specify the
                       interpolation algorithm. Should be one of:
                       cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
                       cv2.INTER_AREA, cv2.INTER_LANCZOS4.
                       Default: cv2.INTER_LINEAR.
        border_mode: flag that is used to specify the pixel
                     extrapolation method. Should be one of:
                     cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE,
                     cv2.BORDER_REFLECT, cv2.BORDER_WRAP,
                     cv2.BORDER_REFLECT_101.
                     Default: cv2.BORDER_REFLECT_101
        value: padding value if border_mode is cv2.BORDER_CONSTANT.
        p: probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self,
        shift_limit: Tuple[float, float] = (-0.0625, 0.0625),
        scale_limit: Tuple[float, float] = (-0.1, 0.1),
        rotate_limit: Tuple[float, float] = (-45.0, 45.0),
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: Optional[int] = None,
        p: float = 0.5,
    ):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.p = p

    def _shift_scale_rotate_lines(self, data, matrix):
        gt_lines = data["gt_lines"]
        for i in range(len(gt_lines)):
            gt_lines[i] = cv2.transform(
                np.expand_dims(gt_lines[i], axis=0), matrix
            ).squeeze()
        data["gt_lines"] = gt_lines

    def _get_rotation_matrix(self, height, width, angle, scale, dx, dy):
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += dx * width
        matrix[1, 2] += dy * height
        return matrix

    def __call__(self, data):
        if random.random() < self.p:
            height, width = data["img"].shape[:2]
            angle = random.uniform(self.rotate_limit[0], self.rotate_limit[1])
            scale = random.uniform(self.scale_limit[0], self.scale_limit[1])
            dx = random.uniform(self.shift_limit[0], self.shift_limit[1])
            dy = random.uniform(self.shift_limit[0], self.shift_limit[1])

            mat = self._get_rotation_matrix(
                height, width, angle, scale, dx, dy
            )

            data["img"] = cv2.warpAffine(
                src=data["img"],
                M=mat,
                dsize=(width, height),
                flags=self.interpolation,
                borderMode=self.border_mode,
                borderValue=self.value,
            )

            if "gt_lines" in data:
                self._shift_scale_rotate_lines(data, matrix=mat)
        return data


@OBJECT_REGISTRY.register
class RandomResizedCrop(object):  # noqa: D205,D400
    """Torchvision's variant of crop a random part of the input,
    and rescale it to some size.

    Used for np.ndarray. This transform is same as
    albumentations.augmentations.transforms.RandomResizedCrop.

    Args:
        height: height after crop and resize.
        width: width after crop and resize.
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped.
        interpolation: flag that is used to specify the interpolation
            algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR,
            cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p: probability of applying the transform. Default: 1.
    """

    def __init__(
        self,
        height: int,
        width: int,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation: int = cv2.INTER_LINEAR,
        p: float = 1.0,
    ):
        self.height = height
        self.width = width
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.p = p

    def get_params_dependent_on_targets(self, data):
        img = data["img"]
        area = img.shape[0] * img.shape[1]

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= img.shape[1] and 0 < h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return {
                    "crop_height": h,
                    "crop_width": w,
                    "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
                    "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
                }

        # Fallback to central crop
        in_ratio = img.shape[1] / img.shape[0]
        if in_ratio < min(self.ratio):
            w = img.shape[1]
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = img.shape[0]
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = img.shape[1]
            h = img.shape[0]
        i = (img.shape[0] - h) // 2
        j = (img.shape[1] - w) // 2
        return {
            "crop_height": h,
            "crop_width": w,
            "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
            "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
        }

    def _get_random_crop_coords(
        self, height, width, crop_height, crop_width, h_start, w_start
    ):
        y1 = int((height - crop_height) * h_start)
        y2 = y1 + crop_height
        x1 = int((width - crop_width) * w_start)
        x2 = x1 + crop_width
        return x1, y1, x2, y2

    def _crop_img(self, img, x1, y1, x2, y2):
        img = img[y1:y2, x1:x2]
        return img

    def _crop_scale_lines(self, gt_lines, x1, y1, scale_x, scale_y):
        for gt_line in gt_lines:
            gt_line[:, 0] = (gt_line[:, 0] - x1) * scale_x
            gt_line[:, 1] = (gt_line[:, 1] - y1) * scale_y
        return gt_lines

    def __call__(self, data):
        if random.random() < self.p:
            img = data["img"]
            img_height = img.shape[0]
            img_width = img.shape[1]
            params = self.get_params_dependent_on_targets(data)

            crop_height = params["crop_height"]
            crop_width = params["crop_width"]
            h_start = params["h_start"]
            w_start = params["w_start"]

            x1, y1, x2, y2 = self._get_random_crop_coords(
                img_height,
                img_width,
                crop_height,
                crop_width,
                h_start,
                w_start,
            )
            croped_img = self._crop_img(img, x1, y1, x2, y2)
            resized_img = cv2.resize(
                croped_img,
                dsize=(self.width, self.height),
                interpolation=self.interpolation,
            )
            data["img"] = resized_img

            if "gt_lines" in data:
                gt_lines = data["gt_lines"]
                scale_x = self.width / crop_width
                scale_y = self.height / crop_height

                data["gt_lines"] = self._crop_scale_lines(
                    gt_lines,
                    x1,
                    y1,
                    scale_x,
                    scale_y,
                )
        return data


@OBJECT_REGISTRY.register
class AlbuImageOnlyTransform(object):
    """AlbuImageOnlyTransform used on img only.

    Composed by list of albu `ImageOnlyTransform`.

    Args:
        albu_params: List of albu iamge only transform.

    Examples::

        dict(
            type="AlbuImageOnlyTransform",
            albu_params=[
                dict(
                    name="RandomBrightnessContrast",
                    p=0.3,
                ),
                dict(
                    name="GaussNoise",
                    var_limit=50.0,
                    p=0.5,
                ),
                dict(
                    name="Blur",
                    p=0.2,
                    blur_limit=(3, 15),
                ),
                dict(
                    name="ToGray",
                    p=0.2,
                ),
            ],
        )

    """

    @require_packages("albumentations")
    def __init__(self, albu_params: List[Dict]):
        self.albu_params = albu_params
        self.albu_transform = self.compose_albu_transform()

    def __call__(self, data):
        # input should be rgb image
        img = data["img"]
        img = self.albu_transform(image=img)["image"]
        data["img"] = img
        return data

    def check_transform(self, transform):
        """Check transform is ImageOnlyTransform.

        only support ImageOnlyTransform till now.

        """
        base_list = inspect.getmro(type(transform))
        if albumentations.ImageOnlyTransform not in base_list:
            raise ValueError("%s is not ImageOnlyTransform" % transform)

    def compose_albu_transform(self):
        transforms = []
        if isinstance(self.albu_params, list):
            for albu_param in self.albu_params:
                param = copy.deepcopy(albu_param)
                name = param.pop("name")
                transform = getattr(albumentations, name)(**param)
                self.check_transform(transform)
                transforms.append(transform)
        else:
            raise ValueError
        return albumentations.Compose(transforms)


@OBJECT_REGISTRY.register
class BoxJitter(object):
    """Jitter box to simulate the box predicted by the model.

    Usually used in tasks that use ground truth boxes for training.

    Args:
        exp_ratio: Ratio of the expansion of box. Defaults to 1.0.
        exp_jitter: Jitter of expansion ratio . Defaults to 0.0.
        center_shift: Box center shift range. Defaults to 0.0.
    """

    def __init__(
        self,
        exp_ratio: float = 1.0,
        exp_jitter: float = 0.0,
        center_shift: float = 0.0,
    ):
        self.exp_ratio = exp_ratio
        self.exp_jitter = exp_jitter
        self.center_shift = center_shift

    def __call__(self, data):
        img_shape = data["pad_shape"]
        boxes = data["gt_bboxes"]

        boxes = self._box_jitter(boxes, img_shape)
        data["gt_bboxes"] = boxes
        return data

    def _box_jitter(self, boxes, img_shape):
        if self.exp_ratio == 1.0 and self.center_shift == 0.0:
            return boxes

        img_h, img_w = img_shape[:2]
        # calcluate expand ratio
        scale = self.exp_ratio + np.clip(
            np.random.normal(0, 0.05, (len(boxes),)),
            -self.exp_jitter,
            self.exp_jitter,
        )  # noqa

        center_x = boxes[:, 0::2].mean(axis=1)  # N,
        center_y = boxes[:, 1::2].mean(axis=1)  # N,
        box_h = boxes[:, 3] - boxes[:, 1]  # N,
        box_w = boxes[:, 2] - boxes[:, 0]  # N,

        # shift center
        shifts = np.clip(
            np.random.normal(0, 0.1, (2, len(boxes))),
            -self.center_shift,
            self.center_shift,
        )  # noqa
        center_x = center_x + shifts[0] * box_w
        center_y = center_y + shifts[1] * box_h

        # expand the box
        x1 = np.clip(center_x - scale * 0.5 * box_w, 0, img_w)
        x2 = np.clip(center_x + scale * 0.5 * box_w, 0, img_w)
        y1 = np.clip(center_y - scale * 0.5 * box_h, 0, img_h)
        y2 = np.clip(center_y + scale * 0.5 * box_h, 0, img_h)
        new_boxes = np.vstack([x1, y1, x2, y2]).T
        return new_boxes


class BaseMixImageTransform(object):
    """Base class for mix image transforms.

    Args:
        p: Probability of applying this transformation.
            Defaults to 1.0.
        use_cached: Whether to use cache. Defaults to False.
        max_cached_images: The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop: Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch: The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Defaults to 15.
    """

    def __init__(
        self,
        p: float = 1.0,
        use_cached: bool = True,
        max_cached_images: int = 40,
        random_pop: bool = True,
        max_refetch: int = 15,
    ):
        self.max_refetch = max_refetch
        self.p = p

        self.use_cached = use_cached
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.data_cache = []

    @abstractmethod
    def get_indexes(self, dataset):
        """Create indexes of selected images in dataset."""
        raise NotImplementedError

    @abstractmethod
    def mix_img_transform(self, data):
        """Do data transform."""
        raise NotImplementedError

    def __call__(self, data):
        if random.uniform(0, 1) > self.p:
            return data

        if self.use_cached:
            # Be careful: deep copying can be very time-consuming
            # if results includes dataset.
            gt_bboxes = data.get("gt_bboxes", np.array([]))
            gt_lines = data.get("gt_lines", np.array([]))

            if len(gt_bboxes) != 0 or len(gt_lines) != 0:
                self.data_cache.append(copy.deepcopy(data))

            if len(self.data_cache) > self.max_cached_images:
                if self.random_pop:
                    index = random.randint(0, len(self.data_cache) - 1)
                else:
                    index = 0
                self.data_cache.pop(index)

            if len(self.data_cache) <= 4:
                return data

            for _ in range(self.max_refetch):
                # get index of one or three other images
                if self.use_cached:
                    indexes = self.get_indexes(self.data_cache)
                    mix_datas = [
                        copy.deepcopy(self.data_cache[i]) for i in indexes
                    ]

                if None not in mix_datas:
                    data["mix_datas"] = mix_datas
                    break
                print("Repeated calculation")

        if "mix_datas" in data:
            if gt_bboxes.shape[0] > 0 or gt_lines.shape[0] > 0:
                data = self.mix_img_transform(data)
            data.pop("mix_datas")

        return data


@OBJECT_REGISTRY.register
class DetYOLOv5MixUp(BaseMixImageTransform):
    """MixUp augmentation.

    Args:
        alpha: parameter of beta distribution to get mixup ratio.
            Defaults to 32.
        beta: parameter of beta distribution to get mixup ratio.
            Defaults to 32.
        p: Probability of applying this transformation.
            Defaults to 1.0.
        use_cached: Whether to use cache. Defaults to False.
        max_cached_images: The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop: Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch: The maximum number of iterations. If the number of
            iterations is greater than `max_refetch`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
    """

    def __init__(
        self,
        alpha: float = 32.0,
        beta: float = 32.0,
        p: float = 1.0,
        use_cached: bool = True,
        max_cached_images: int = 20,
        random_pop: bool = True,
        max_refetch: int = 15,
    ):
        super(DetYOLOv5MixUp, self).__init__(
            p, use_cached, max_cached_images, random_pop, max_refetch
        )
        self.alpha = alpha
        self.beta = beta

    def get_indexes(self, dataset):
        return [random.randint(0, len(dataset) - 1)]

    def mix_img_transform(self, data):
        assert "mix_datas" in data
        retrieve_results = data["mix_datas"][0]
        ori_img = data["img"]
        if data["img"].shape != retrieve_results["img"].shape:
            retrieve_results = Resize(
                (ori_img.shape[0], ori_img.shape[1]), keep_ratio=False
            )(retrieve_results)
        retrieve_img = retrieve_results["img"]
        assert ori_img.shape == retrieve_img.shape

        ratio = np.random.beta(self.alpha, self.beta)
        mixup_img = ori_img * ratio + retrieve_img * (1 - ratio)

        retrieve_gt_bboxes = retrieve_results["gt_bboxes"]
        retrieve_gt_classes = retrieve_results["gt_classes"]

        mixup_gt_bboxes = np.concatenate(
            (data["gt_bboxes"], retrieve_gt_bboxes), axis=0
        )
        mixup_gt_classes = np.concatenate(
            (data["gt_classes"], retrieve_gt_classes), axis=0
        )

        data["img"] = mixup_img.astype(np.uint8)
        data["img_shape"] = mixup_img.shape
        data["gt_bboxes"] = mixup_gt_bboxes
        data["gt_classes"] = mixup_gt_classes
        if "gt_tanalphas" in data:
            retrieve_gt_tanalphas = retrieve_results["gt_tanalphas"]
            mixup_gt_tanalphas = np.concatenate(
                (data["gt_tanalphas"], retrieve_gt_tanalphas), axis=0
            )
            data["gt_tanalphas"] = mixup_gt_tanalphas

        return data


@OBJECT_REGISTRY.register
class DetYOLOXMixUp(BaseMixImageTransform):
    """MixUp data augmentation for YOLOX.

                +---------------+--------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                +---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      +-----------------+     |
                |             pad              |
                +------------------------------+

    Args:
        img_scale: Image output size after mixup pipeline.
            The shape order should be (height, width). Defaults to (640, 640).
        ratio_range: Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio: Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val: Pad value. Defaults to 114.
        bbox_clip_border: Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pre_transform: Sequence of transform object or
            config dict to be composed.
        p: Probability of applying this transformation.
            Defaults to 1.0.
        use_cached: Whether to use cache. Defaults to False.
        max_cached_images: The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop: Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch: The maximum number of iterations. If the number of
            iterations is greater than `max_refetch`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
    """

    def __init__(
        self,
        img_scale: Tuple[int, int] = (640, 640),
        ratio_range: Tuple[float, float] = (0.5, 1.5),
        flip_ratio: float = 0.5,
        pad_val: float = 114.0,
        bbox_clip_border: bool = True,
        p: float = 1.0,
        use_cached: bool = True,
        max_cached_images: int = 20,
        random_pop: bool = True,
        max_refetch: int = 15,
    ):
        super(DetYOLOXMixUp, self).__init__(
            p, use_cached, max_cached_images, random_pop, max_refetch
        )
        self.img_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.bbox_clip_border = bbox_clip_border

    def get_indexes(self, dataset):
        return [random.randint(0, len(dataset) - 1)]

    def mix_img_transform(self, data):
        assert "mix_datas" in data
        retrieve_data = data["mix_datas"][0]
        retrieve_img = retrieve_data["img"]

        jit_factor = random.uniform(*self.ratio_range)
        is_flip = random.uniform(0, 1) < self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = (
                np.ones(
                    (self.img_scale[0], self.img_scale[1], 3),
                    dtype=retrieve_img.dtype,
                )
                * self.pad_val
            )
        else:
            out_img = (
                np.ones(self.img_scale, dtype=retrieve_img.dtype)
                * self.pad_val
            )

        # 1. keep_ratio resize
        scale_ratio = min(
            self.img_scale[0] / retrieve_img.shape[0],
            self.img_scale[1] / retrieve_img.shape[1],
        )
        retrieve_img = cv2.resize(
            retrieve_img,
            (
                int(retrieve_img.shape[1] * scale_ratio),
                int(retrieve_img.shape[0] * scale_ratio),
            ),
        )

        # 2. paste
        out_img[
            : retrieve_img.shape[0], : retrieve_img.shape[1]
        ] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = cv2.resize(
            out_img,
            (
                int(out_img.shape[1] * jit_factor),
                int(out_img.shape[0] * jit_factor),
            ),
        )

        # 4. flip
        if is_flip:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = data["img"]
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3)
        ).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[
            y_offset : y_offset + target_h, x_offset : x_offset + target_w
        ]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_data["gt_bboxes"]
        retrieve_gt_bboxes[:, 0::2] = retrieve_gt_bboxes[:, 0::2] * scale_ratio
        retrieve_gt_bboxes[:, 1::2] = retrieve_gt_bboxes[:, 1::2] * scale_ratio
        if self.bbox_clip_border:
            retrieve_gt_bboxes[:, 0::2] = np.clip(
                retrieve_gt_bboxes[:, 0::2], 0, origin_w
            )
            retrieve_gt_bboxes[:, 1::2] = np.clip(
                retrieve_gt_bboxes[:, 1::2], 0, origin_h
            )

        if is_flip:
            retrieve_gt_bboxes[:, 0::2] = (
                origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1]
            )

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        cp_retrieve_gt_bboxes[:, 0::2] = (
            cp_retrieve_gt_bboxes[:, 0::2] - x_offset
        )
        cp_retrieve_gt_bboxes[:, 1::2] = (
            cp_retrieve_gt_bboxes[:, 1::2] - y_offset
        )
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes[:, 0::2] = np.clip(
                cp_retrieve_gt_bboxes[:, 0::2], 0, target_w
            )
            cp_retrieve_gt_bboxes[:, 1::2] = np.clip(
                cp_retrieve_gt_bboxes[:, 1::2], 0, target_h
            )

        # 8. mix up
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img
        retrieve_gt_classes = retrieve_data["gt_classes"]

        mixup_gt_bboxes = np.concatenate(
            (data["gt_bboxes"], cp_retrieve_gt_bboxes), axis=0
        )
        mixup_gt_classes = np.concatenate(
            (data["gt_classes"], retrieve_gt_classes), axis=0
        )

        # remove outside bbox
        def find_inside_bboxes(bboxes, img_h, img_w):
            inside_inds = (
                (bboxes[:, 0] < img_w)
                & (bboxes[:, 2] > 0)
                & (bboxes[:, 1] < img_h)
                & (bboxes[:, 3] > 0)
            )
            return inside_inds

        inside_inds = find_inside_bboxes(mixup_gt_bboxes, target_h, target_w)
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_classes = mixup_gt_classes[inside_inds]

        data["img"] = mixup_img.astype(np.uint8)
        data["img_shape"] = mixup_img.shape
        data["gt_bboxes"] = mixup_gt_bboxes
        data["gt_classes"] = mixup_gt_classes
        if "gt_tanalphas" in data:
            retrieve_gt_tanalphas = retrieve_data["gt_tanalphas"]
            assert not self.bbox_clip_border
            if is_flip:
                retrieve_gt_tanalphas = -retrieve_gt_tanalphas
            mixup_gt_tanalphas = np.concatenate(
                (data["gt_tanalphas"], retrieve_gt_tanalphas), axis=0
            )
            mixup_gt_tanalphas = mixup_gt_tanalphas[inside_inds]
            data["gt_tanalphas"] = mixup_gt_tanalphas

        return data


@OBJECT_REGISTRY.register
class DetMosaic(BaseMixImageTransform):
    """Mosaic augmentation for detection task.

    Args:
        img_scale: Image size after mosaic pipeline of
            a single image. The size of the output image is four times
            that of a single image. The output image comprises 4 single images.
            Default: (640, 640).
        center_ratio_range: Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border: Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val: Pad value. Defaults to 114.
        p: Probability of applying this transformation.
            Defaults to 1.0.
        use_cached: Whether to use cache. Defaults to False.
        max_cached_images: The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop: Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch: The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Defaults to 15.
    """

    def __init__(
        self,
        img_scale: Tuple[int, int] = (640, 640),
        center_ratio_range: Tuple[float, float] = (0.5, 1.5),
        bbox_clip_border: bool = True,
        pad_val: float = 114.0,
        p: float = 1.0,
        use_cached: bool = True,
        max_cached_images: int = 40,
        random_pop: bool = True,
        max_refetch: int = 15,
    ):
        super(DetMosaic, self).__init__(
            p, use_cached, max_cached_images, random_pop, max_refetch
        )
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val

    def get_indexes(self, dataset):
        indexes = [random.randint(0, len(dataset) - 1) for _ in range(3)]
        return indexes

    def mix_img_transform(self, data):
        assert "mix_datas" in data

        mosaic_bboxes = []
        mosaic_lines = []
        mosaic_classes = []
        if "gt_tanalphas" in data:
            mosaic_tanalphas = []
        if len(data["img"].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=data["img"].dtype,
            )
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=data["img"].dtype,
            )

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1]
        )
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0]
        )
        center_position = (center_x, center_y)

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        for i, loc in enumerate(loc_strs):
            if loc == "top_left":
                results_patch = copy.deepcopy(data)
            else:
                results_patch = copy.deepcopy(data["mix_datas"][i - 1])
            img_idx = results_patch["img"]
            h_i, w_i = img_idx.shape[:2]

            # keep_ratio resize
            scale_ratio_i = min(
                self.img_scale[0] / h_i, self.img_scale[1] / w_i
            )
            img_idx = cv2.resize(
                img_idx, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i))
            )

            if "gt_bboxes" in results_patch:
                # compute the combine parameters
                paste_coord, crop_coord = self._mosaic_combine(
                    loc, center_position, img_idx.shape[:2][::-1]
                )
                x1_p, y1_p, x2_p, y2_p = paste_coord
                x1_c, y1_c, x2_c, y2_c = crop_coord

                # crop and paste image
                mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_idx[
                    y1_c:y2_c, x1_c:x2_c
                ]

                # adjust coordinate
                gt_bboxes_i = results_patch["gt_bboxes"]
                gt_classes_i = results_patch["gt_classes"]
                if "gt_tanalphas" in results_patch:
                    gt_tanalphas_i = results_patch["gt_tanalphas"]

                if gt_bboxes_i.shape[0] > 0:
                    padw = x1_p - x1_c
                    padh = y1_p - y1_c
                    gt_bboxes_i[:, 0::2] = (
                        scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                    )
                    gt_bboxes_i[:, 1::2] = (
                        scale_ratio_i * gt_bboxes_i[:, 1::2] + padh
                    )

                    mosaic_bboxes.append(gt_bboxes_i)
                    mosaic_classes.append(gt_classes_i)
                    if "gt_tanalphas" in data:
                        mosaic_tanalphas.append(gt_tanalphas_i)
            elif "gt_lines" in results_patch:
                # compute the combine parameters
                paste_coord, crop_coord = self._mosaic_combine(
                    loc, center_position, img_idx.shape[:2][::-1]
                )
                x1_p, y1_p, x2_p, y2_p = paste_coord
                x1_c, y1_c, x2_c, y2_c = crop_coord

                # crop and paste image
                mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_idx[
                    y1_c:y2_c, x1_c:x2_c
                ]

                # adjust coordinate
                gt_lines_i = results_patch["gt_lines"]

                if gt_lines_i.shape[0] > 0:
                    padw = x1_p - x1_c
                    padh = y1_p - y1_c
                    gt_lines_i[..., 0:1] = (
                        scale_ratio_i * gt_lines_i[..., 0:1] + padw
                    )
                    gt_lines_i[..., 1:2] = (
                        scale_ratio_i * gt_lines_i[..., 1:2] + padh
                    )

                    mosaic_lines.append(gt_lines_i)
            else:
                raise NotImplementedError

        def find_inside_bboxes(bboxes, img_h, img_w):
            # Find bboxes as long as a part of bboxes is inside the image.
            inside_inds = (
                (bboxes[:, 0] < img_w)
                & (bboxes[:, 2] > 0)
                & (bboxes[:, 1] < img_h)
                & (bboxes[:, 3] > 0)
            )
            return inside_inds

        def find_inside_lines(lines, img_h, img_w):
            # Find pts  is inside the image.
            inside_inds = (
                (lines[..., 0] < img_w)
                & (lines[..., 0] > 0)
                & (lines[..., 1] < img_h)
                & (lines[..., 1] > 0)
            )
            return inside_inds

        if "gt_bboxes" in results_patch:
            if len(mosaic_classes) > 0:
                mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
                mosaic_classes = np.concatenate(mosaic_classes, 0)

                if "gt_tanalphas" in data:
                    mosaic_tanalphas = np.concatenate(mosaic_tanalphas, 0)
                if self.bbox_clip_border:
                    mosaic_bboxes[:, 0::2] = np.clip(
                        mosaic_bboxes[:, 0::2], 0, 2 * self.img_scale[1]
                    )
                    mosaic_bboxes[:, 1::2] = np.clip(
                        mosaic_bboxes[:, 1::2], 0, 2 * self.img_scale[0]
                    )
                inside_inds = find_inside_bboxes(
                    mosaic_bboxes, 2 * self.img_scale[0], 2 * self.img_scale[1]
                )
                mosaic_bboxes = mosaic_bboxes[inside_inds]
                mosaic_classes = mosaic_classes[inside_inds]
                if "gt_tanalphas" in data:
                    mosaic_tanalphas = mosaic_tanalphas[inside_inds]
            else:
                mosaic_bboxes = np.array(mosaic_bboxes)
                mosaic_classes = np.array(mosaic_classes)
                if "gt_tanalphas" in data:
                    mosaic_tanalphas = np.array(mosaic_tanalphas)

        elif "gt_lines" in results_patch:
            if len(mosaic_lines) > 0:
                mosaic_lines = np.concatenate(mosaic_lines, 1)

                if self.bbox_clip_border:
                    mosaic_lines[..., 0:1] = np.clip(
                        mosaic_lines[..., 0:1], 0, 2 * self.img_scale[1]
                    )
                    mosaic_lines[..., 1:2] = np.clip(
                        mosaic_lines[..., 1:2], 0, 2 * self.img_scale[0]
                    )
                mosaic_lines_new = []
                for i, mosaic_line in enumerate(mosaic_lines):
                    inside_inds = find_inside_lines(
                        mosaic_line,
                        2 * self.img_scale[0],
                        2 * self.img_scale[1],
                    )
                    mosaic_lines_new.append(mosaic_lines[i][inside_inds])
                mosaic_lines = mosaic_lines_new
            else:
                mosaic_lines = np.array(mosaic_lines)

        else:
            raise NotImplementedError

        data["img"] = mosaic_img
        data["img_shape"] = mosaic_img.shape

        if "gt_bboxes" in data:
            data["gt_bboxes"] = mosaic_bboxes
        if "gt_classes" in data:
            data["gt_classes"] = mosaic_classes
        if "gt_lines" in data:
            data["gt_lines"] = mosaic_lines
        if "gt_tanalphas" in data:
            data["gt_tanalphas"] = mosaic_tanalphas
        return data

    def _mosaic_combine(
        self,
        loc: str,
        center_position_xy: Sequence[float],
        img_shape_wh: Sequence[int],
    ) -> Tuple[Tuple[int], Tuple[int]]:
        """Calculate global coordinate of mosaic image and local coordinate.

        Args:
            loc: Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy: Mixing center for 4 images, (x, y).
            img_shape_wh: Width and height of sub-image
        Returns:
            tuple: Corresponding coordinate of pasting and
                cropping
                - paste_coord: paste corner coordinate in mosaic image.
                - crop_coord: crop corner coordinate in mosaic image.
        """
        assert loc in ("top_left", "top_right", "bottom_left", "bottom_right")
        if loc == "top_left":
            # index0 to top left part of image
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                max(center_position_xy[1] - img_shape_wh[1], 0),
                center_position_xy[0],
                center_position_xy[1],
            )
            crop_coord = (
                img_shape_wh[0] - (x2 - x1),
                img_shape_wh[1] - (y2 - y1),
                img_shape_wh[0],
                img_shape_wh[1],
            )

        elif loc == "top_right":
            # index1 to top right part of image
            x1, y1, x2, y2 = (
                center_position_xy[0],
                max(center_position_xy[1] - img_shape_wh[1], 0),
                min(
                    center_position_xy[0] + img_shape_wh[0],
                    self.img_scale[1] * 2,
                ),
                center_position_xy[1],
            )
            crop_coord = (
                0,
                img_shape_wh[1] - (y2 - y1),
                min(img_shape_wh[0], x2 - x1),
                img_shape_wh[1],
            )

        elif loc == "bottom_left":
            # index2 to bottom left part of image
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                center_position_xy[1],
                center_position_xy[0],
                min(
                    self.img_scale[0] * 2,
                    center_position_xy[1] + img_shape_wh[1],
                ),
            )
            crop_coord = (
                img_shape_wh[0] - (x2 - x1),
                0,
                img_shape_wh[0],
                min(y2 - y1, img_shape_wh[1]),
            )

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = (
                center_position_xy[0],
                center_position_xy[1],
                min(
                    center_position_xy[0] + img_shape_wh[0],
                    self.img_scale[1] * 2,
                ),
                min(
                    self.img_scale[0] * 2,
                    center_position_xy[1] + img_shape_wh[1],
                ),
            )
            crop_coord = (
                0,
                0,
                min(img_shape_wh[0], x2 - x1),
                min(y2 - y1, img_shape_wh[1]),
            )

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord


@OBJECT_REGISTRY.register
class Mosaic(object):
    """Mosaic augmentation for detection task.

    Args:
        image_size: Image size after mosaic pipeline. Default: (512, 512).
        degrees: Rotation degree. Defaults to 10.
        translate: translate value for warpPerspective. Defaults to 0.1.
        scale: Random scale value. Defaults to 0.1.
        shear: Shear value for warpPerspective. Defaults to 10.
        perspective: perspective value for warpPerspective. Defaults to 0.0.
        mixup: Whether use mixup. Defaults to True.
    """

    def __init__(
        self,
        image_size: int = 512,
        degrees: int = 10,
        translate: float = 0.1,
        scale: float = 0.1,
        shear: int = 10,
        perspective: float = 0.0,
        mixup: bool = True,
    ):
        self.image_size = image_size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = [-image_size // 2, -image_size // 2]
        self.mixup = mixup

    def _fix_labels(self, labels, offsets):
        new_labels = labels.copy()
        new_labels[:, 0] = labels[:, 0] + offsets[1]
        new_labels[:, 1] = labels[:, 1] + offsets[0]
        new_labels[:, 2] = labels[:, 2] + offsets[1]
        new_labels[:, 3] = labels[:, 3] + offsets[0]
        return new_labels

    def _mixup(self, data1, data2):
        img1 = data1["img"]
        bboxes1 = data1["gt_bboxes"]
        labels1 = data1["gt_classes"]

        img2 = data2["img"]
        bboxes2 = data2["gt_bboxes"]
        labels2 = data2["gt_classes"]

        img2, bboxes2, _ = self._random_scale(img2, bboxes2)
        img2, bboxes2, labels2 = self._crop(img2, bboxes2, labels2)
        img2 = self._pad(img2)

        r = 0.5
        img1 = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        bboxes1 = np.concatenate((bboxes1, bboxes2), 0)
        labels1 = np.concatenate((labels1, labels2), 0)

        data1["img"] = img1
        data1["gt_bboxes"] = bboxes1
        data1["gt_classes"] = labels1
        return data1

    def _random_scale(self, img, bboxes):
        min_ratio, max_ratio = 1 - self.scale, 1 + self.scale
        scale = random.uniform(min_ratio, max_ratio)

        max_side = self.image_size * scale

        h, w, _ = img.shape
        largest_side = max(h, w)
        scale = max_side / largest_side
        img = cv2.resize(
            img,
            (int(round(w * scale)), int(round((h * scale)))),
            interpolation=cv2.INTER_LINEAR,
        )

        bboxes = bboxes * scale
        return img, bboxes, scale

    def _crop(self, img, bboxes, labels):
        margin_h = max(img.shape[0] - self.image_size, 0)
        margin_w = max(img.shape[1] - self.image_size, 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.image_size
        crop_x1, crop_x2 = offset_w, offset_w + self.image_size
        # crop the image
        crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        bbox_offset = np.array(
            [offset_w, offset_h, offset_w, offset_h], dtype=np.float32
        )
        bboxes[:, :4] -= bbox_offset
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, crop_img.shape[1] - 1)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, crop_img.shape[0] - 1)

        valid_ind = (bboxes[:, 2] > bboxes[:, 0]) & (
            bboxes[:, 3] > bboxes[:, 1]
        )
        bboxes = bboxes[valid_ind]
        labels = labels[valid_ind]
        return crop_img, bboxes, labels

    def _pad(self, img):
        h, w, c = img.shape
        h_padded = self.image_size
        w_padded = self.image_size

        new_image = np.zeros((h_padded, w_padded, c), dtype=np.uint8)
        new_image[:h, :w, :] = img.astype(np.uint8)
        return new_image

    def _box_candidates(self, box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
        # box1(4,n), box2(4,n)
        # Compute candidate boxes: box1 before augment,
        # box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
        return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
            & (ar < ar_thr)
        )

    def _random_rerspective(self, data):
        image = data["img"]
        bboxes = data["gt_bboxes"]
        labels = data["gt_classes"]
        height = image.shape[0] + self.border[0] * 2  # shape(h,w,c)
        width = image.shape[1] + self.border[1] * 2
        # Center
        C = np.eye(3)
        C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -image.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(
            -self.perspective, self.perspective
        )  # x perspective (about y)
        P[2, 1] = random.uniform(
            -self.perspective, self.perspective
        )  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)

        s = random.uniform(1.0, 1 + self.scale)
        direction = random.uniform(0, 1)
        s = s if direction > 0.5 else 1 / s
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )  # x shear (deg)
        S[1, 0] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
        )  # x translation (pixels)
        T[1, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate) * height
        )  # y translation (pixels)

        # Combined rotation matrix
        M = (
            T @ S @ R @ P @ C
        )  # order of operations (right to left) is IMPORTANT
        if (
            (self.border[0] != 0)
            or (self.border[1] != 0)
            or (M != np.eye(3)).any()
        ):  # image changed
            if self.perspective:
                image = cv2.warpPerspective(
                    image,
                    M,
                    dsize=(width, height),
                    borderValue=(114, 114, 114),
                )
            else:  # affine
                image = cv2.warpAffine(
                    image,
                    M[:2],
                    dsize=(width, height),
                    borderValue=(114, 114, 114),
                )

        # Transform label coordinates
        n = len(bboxes)
        if n:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2
            )  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            if self.perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = (
                np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)))
                .reshape(4, n)
                .T
            )

            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width - 1)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height - 1)

            # filter candidates
            i = self._box_candidates(box1=bboxes[:, :4].T * s, box2=xy.T)
            bboxes = bboxes[i]
            bboxes[:, :4] = xy[i]
            labels = labels[i]

        data["img"] = image
        data["gt_bboxes"] = bboxes
        data["gt_classes"] = labels
        data["scale_factor"] = np.asarray([s])

    @property
    def images_needed(self):
        return 4 if self.mixup is not True else 5

    def __call__(self, data_list):
        _, _, c = data_list[0]["img"].shape
        s = self.image_size
        img4 = np.full((s * 2, s * 2, c), 114, dtype=np.uint8)
        # top left
        data1 = data_list[0].copy()
        image = data1["img"]
        oh, ow, _ = image.shape
        bboxes = data1["gt_bboxes"]
        labels = data1["gt_classes"]
        h = min(oh, s)
        w = min(ow, s)
        img4[s - h : s, s - w : s, :] = image[oh - h : oh, ow - w : ow, :]
        new_bboxes = self._fix_labels(bboxes, ((s - oh), (s - ow)))
        new_labels = labels

        # top right
        data2 = data_list[1].copy()
        image = data2["img"]
        oh, ow, _ = image.shape
        bboxes = data2["gt_bboxes"]
        labels = data2["gt_classes"]
        h = min(oh, s)
        w = min(ow, s)

        img4[s - h : s, s : s + w, :] = image[oh - h : oh, :w, :]
        bboxes = self._fix_labels(bboxes, ((s - oh), s))
        new_bboxes = np.concatenate((new_bboxes, bboxes), axis=0)
        new_labels = np.concatenate((new_labels, labels), axis=0)

        # bottom left
        data3 = data_list[2].copy()
        image = data3["img"]
        oh, ow, _ = image.shape
        bboxes = data3["gt_bboxes"]
        labels = data3["gt_classes"]
        h = min(oh, s)
        w = min(ow, s)
        img4[s : s + h, s - w : s, :] = image[:h, ow - w : ow, :]
        bboxes = self._fix_labels(bboxes, (s, s - ow))
        new_bboxes = np.concatenate((new_bboxes, bboxes), axis=0)
        new_labels = np.concatenate((new_labels, labels), axis=0)

        # bottom right
        data4 = data_list[3].copy()
        image = data4["img"]
        oh, ow, _ = image.shape
        bboxes = data4["gt_bboxes"]
        labels = data4["gt_classes"]
        h = min(oh, s)
        w = min(ow, s)

        img4[s : s + h, s : s + w, :] = image[:h, :w, :]
        bboxes = self._fix_labels(bboxes, (s, s))
        new_bboxes = np.concatenate((new_bboxes, bboxes), axis=0)
        new_labels = np.concatenate((new_labels, labels), axis=0)

        new_bboxes[:, 0:4] = np.clip(new_bboxes[:, 0:4], 0, (2 * s - 1))
        data1["img"] = img4
        data1["gt_bboxes"] = new_bboxes
        data1["gt_classes"] = new_labels
        self._random_rerspective(data1)
        if self.mixup:
            data5 = data_list[4].copy()
            data1 = self._mixup(data1, data5)
        data1["img_shape"] = data1["img"].shape
        data1["pad_shape"] = data1["img"].shape
        return data1


@OBJECT_REGISTRY.register
class ToPositionFasterRCNNData(ToFasterRCNNData):
    """Transform person potion dataset to RCNN input need.

    This class is used to stack position label with boxes and camera type,
    and typically used to facilitate position label and boxes
    matching in anchor-based model.
    """

    def _cvt_bbox(self, data):
        bboxes = data["gt_bboxes"]

        all_boxes = bboxes
        gt_inds = all_boxes[:, 4] > 0

        # gt_boxes shape should be same shape in batch,
        # in order to concatate in collate.
        # these value will be used in pytorch pulgin.
        gt_num = gt_inds.sum()
        data["gt_boxes_num"] = np.array([gt_num], dtype=np.float32)
        data["gt_boxes"] = np.zeros(
            (self.max_gt_boxes_num, 7), dtype=np.float32
        )
        data["gt_boxes"][0:gt_num] = bboxes[gt_inds]

        ig_inds = ~gt_inds
        ig_num = ig_inds.sum()
        data["ig_regions_num"] = np.array([ig_num], dtype=np.float32)
        data["ig_regions"] = np.zeros(
            (self.max_ig_regions_num, 7), dtype=np.float32
        )
        data["ig_regions"][0:ig_num] = all_boxes[ig_inds]

    def _stack_label(self, data):
        gt_classes = data["gt_classes"]
        gt_bboxes = data["gt_bboxes"]
        gt_position_dms = data["gt_position_dms"]
        gt_position_oms = data["gt_position_oms"]
        # 1 for oms, 0 for dms
        if data["camera"] == "OMS":
            camera = np.zeros(gt_position_oms.shape) + 1
            gt_position = gt_position_oms
        if data["camera"] == "DMS":
            camera = np.zeros(gt_position_dms.shape)
            gt_position = gt_position_dms

        gt_bboxes = np.column_stack(
            (gt_bboxes, gt_classes, camera, gt_position)
        )

        data["gt_bboxes"] = gt_bboxes

    def __call__(self, data):
        assert "img" in data
        assert "gt_bboxes" in data
        assert "gt_position_dms" in data
        assert "gt_position_oms" in data
        assert "camera" in data

        self._cvt_img(data)
        self._stack_label(data)
        self._cvt_bbox(data)
        self._remove_redundance_keys(data)
        return data


@OBJECT_REGISTRY.register
class IterableDetRoIListTransform(IterableDetRoITransform):
    """
    Iterable transformer base on roi list for object detection.

    Parameters
    ----------
    resize_wh : list/tuple of 2 int, optional
        Resize input image to target size, by default None
    roi_list : ndarray, optional
        Transform the specified image region
    append_gt : bool, optional
        Append the groundtruth to roi_list
    complete_boxes : bool, optional
        Using the uncliped boxes, by default False.
    **kwargs :
        Please see :py:class:`AffineMatFromROIBoxGenerator` and
        :py:class:`ImageAffineTransform`
    """

    # TODO(alan): No need to use resize_wh.

    def __init__(
        self,
        target_wh,
        flip_prob,
        img_scale_range=(0.5, 2.0),
        roi_scale_range=(0.8, 1.0 / 0.8),
        min_sample_num=1,
        max_sample_num=1,
        center_aligned=True,
        inter_method=10,
        use_pyramid=True,
        pyramid_min_step=0.7,
        pyramid_max_step=0.8,
        pixel_center_aligned=True,
        min_valid_area=8,
        min_valid_clip_area_ratio=0.5,
        min_edge_size=2,
        rand_translation_ratio=0,
        rand_aspect_ratio=0,
        rand_rotation_angle=0,
        reselect_ratio=0,
        clip_bbox=True,
        rand_sampling_bbox=True,
        resize_wh=None,
        keep_aspect_ratio=False,
        roi_list=None,
        append_gt=False,
        complete_boxes=False,
    ):
        super().__init__(
            target_wh,
            flip_prob,
            img_scale_range,
            roi_scale_range,
            min_sample_num,
            max_sample_num,
            center_aligned,
            inter_method,
            use_pyramid,
            pyramid_min_step,
            pyramid_max_step,
            pixel_center_aligned,
            min_valid_area,
            min_valid_clip_area_ratio,
            min_edge_size,
            rand_translation_ratio,
            rand_aspect_ratio,
            rand_rotation_angle,
            reselect_ratio,
            clip_bbox,
            rand_sampling_bbox,
            resize_wh,
            keep_aspect_ratio,
            complete_boxes,
        )

        if roi_list is not None:
            self._roi_list = np.concatenate(
                [roi_list, np.ones([roi_list.shape[0], 1])], axis=-1
            )
        else:
            self._roi_list = None
        self.append_gt = append_gt

    def __call__(self, data):
        assert isinstance(data, (dict))
        assert "img" in data.keys()
        assert "gt_boxes" in data.keys()
        img = data.get("img")
        gt_boxes = data.get("gt_boxes")
        ig_regions = data.get("ig_regions", None)

        if self._keep_aspect_ratio and self._resize_wh:
            origin_wh = img.shape[:2][::-1]
            resize_wh_ratio = float(self._resize_wh[0]) / float(
                self._resize_wh[1]
            )  # noqa
            origin_wh_ratio = float(origin_wh[0]) / float(origin_wh[1])
            affine = np.array([[1.0, 0, 0], [0, 1.0, 0]])

            if resize_wh_ratio > origin_wh_ratio:
                new_wh = (
                    int(origin_wh[1] * resize_wh_ratio),
                    origin_wh[1],
                )  # noqa
                img = cv2.warpAffine(img, affine, new_wh, 0)
            elif resize_wh_ratio < origin_wh_ratio:
                new_wh = (
                    origin_wh[0],
                    int(origin_wh[0] / resize_wh_ratio),
                )  # noqa
                img = cv2.warpAffine(img, affine, new_wh, 0)
        else:
            if self._use_pyramid:
                img = AlphaImagePyramid(
                    img,
                    scale_step=np.random.uniform(
                        self._pyramid_min_step, self._pyramid_max_step
                    ),
                )

        if self._roi_list is not None:
            roi = self._roi_list.copy()
            if self.append_gt and gt_boxes.shape[0]:
                roi = np.concatenate([roi, gt_boxes], axis=0)
        else:
            roi = gt_boxes.copy()

        if self._resize_wh is None:
            img_wh = img.shape[:2][::-1]
            affine_mat = AffineMat2DGenerator.identity()
        else:
            img_wh = self._resize_wh
            affine_mat = resize_affine_mat(
                img.shape[:2][::-1], self._resize_wh
            )
            roi = LabelAffineTransform(label_type="box")(
                roi, affine_mat, flip=False
            )

        for affine_aug_param in self._roi_ts(roi, img_wh):  # noqa
            new_affine_mat = AffineMat2DGenerator.stack_affine_transform(
                affine_mat, affine_aug_param.mat
            )[:2]
            affine_aug_param = AffineAugMat(
                mat=new_affine_mat, flipped=affine_aug_param.flipped
            )
            ts_img = self._img_ts(img, affine_aug_param.mat)

            ts_img_wh = ts_img.shape[:2][::-1]
            ts_gt_boxes, ts_ig_regions = _transform_bboxes(
                gt_boxes,
                ig_regions,
                (0, 0, ts_img_wh[0], ts_img_wh[1]),
                affine_aug_param,
                **self._bbox_ts_kwargs,
            )

            data = pad_detection_data(
                ts_img,
                ts_gt_boxes,
                ts_ig_regions,
            )
            data["img"] = data["img"].transpose(2, 0, 1)
            return data
