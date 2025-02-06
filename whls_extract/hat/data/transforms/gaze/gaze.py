# Copyright (c) Horizon Robotics. All rights reserved.
import math
import random
import warnings
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages
from .utils import (
    calc_roi_with_ldmk,
    eyeldmk_transform,
    fixed_crop,
    generate_pos_map,
    rotate_gaze,
    transform_ldmk,
    warp_crop_image,
)

try:
    import albumentations
    from albumentations.augmentations.transforms import ColorJitter
except ImportError:
    pass


__all__ = [
    "GazeYUVTransform",
    "GazeRandomCropWoResize",
    "Clip",
    "RandomColorJitter",
    "GazeRotate3DWithCrop",
]

numeric_types = (float, int, np.generic)


def _do_equalize_hist(img, method, **kwargs):
    if method is None or method == "calc":
        return cv2.equalizeHist(img.astype(np.uint8))
    elif method == "fix":
        raise NotImplementedError
    elif method == "fix_circle":
        raise NotImplementedError
    elif method == "fix_cuberoot":
        return np.array(np.cbrt(img * 255 ** 2), dtype=np.uint8)
    elif method == "fix_circle_squeeze":
        thresh = kwargs.get("thresh", 127.5)
        cross_length = np.sqrt(255 ** 2 + thresh ** 2)
        base_x, base_y = thresh / 2.0, 255.0 / 2
        base_angle = np.arctan(255.0 / thresh)
        alpha = np.pi / 2 - (base_angle - np.pi / 4)
        distance = cross_length / 2.0 * np.tan(alpha)
        delta_x, delta_y = distance * np.sin(base_angle), distance * np.cos(
            base_angle
        )
        x0, y0 = base_x + delta_x, base_y - delta_y
        R = distance / np.sin(alpha)
        # Note: x = x / 2.
        img /= 255.0 / thresh
        return np.array(np.sqrt(R ** 2 - (img - x0) ** 2) + y0, dtype=np.uint8)
    elif method == "fix_cuberoot_squeeze":
        thresh = kwargs.get("thresh", 127.5)
        x_scale = 3 ** (3.0 / 2)
        x0 = 255.0 / x_scale  # where y' = 1
        y0 = np.cbrt(x0 * 255.0 ** 2)
        y_scale = 255.0 / y0
        img /= 255.0 / thresh
        img /= thresh / x0
        return np.array(y_scale * np.cbrt(img * (255 ** 2)), dtype=np.uint8)
    else:
        raise NotImplementedError(
            f"equalize_hist_method {method} is not supported."
        )


@OBJECT_REGISTRY.register
class GazeYUVTransform:
    """YUVTransform for Gaze Task.

    This pipeline: bgr_to_yuv444 -> equalizehist -> yuv444_to_yuv444_int8
    Args:
        rgb_data: whether input data is rgb format
        nc: output channels of data
        equalize_hist: do histogram equalization or not
        equalize_hist_method: method for histogram equalization
    Inputs:
        - **data**: input tensor with (H x W x C) shape.
    Outputs:
        - **out**: output tensor with same shape as `data`.
    """

    def __init__(
        self,
        rgb_data=False,
        nc=3,
        equalize_hist=True,
        equalize_hist_method=None,
    ):
        self._rgb_data = rgb_data
        self._nc = nc
        self._equalize_hist = equalize_hist
        self._equalize_hist_method = equalize_hist_method

    def bgr2yuv444(self, img_bgr):
        img_h = img_bgr.shape[0]
        img_w = img_bgr.shape[1]
        uv_start_idx = img_h * img_w
        v_size = int(img_h * img_w / 4)

        def _trans(img_uv):
            img_uv = img_uv.reshape(
                (int(math.ceil(img_h / 2.0)), int(math.ceil(img_w / 2.0)), 1)
            )
            img_uv = np.repeat(img_uv, 2, axis=0)
            img_uv = np.repeat(img_uv, 2, axis=1)
            return img_uv

        img_yuv420sp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
        img_yuv420sp = img_yuv420sp.flatten()
        img_y = img_yuv420sp[:uv_start_idx].reshape((img_h, img_w, 1))
        img_u = img_yuv420sp[uv_start_idx : uv_start_idx + v_size]
        img_v = img_yuv420sp[uv_start_idx + v_size : uv_start_idx + 2 * v_size]
        img_u = _trans(img_u)
        img_v = _trans(img_v)
        img_yuv444 = np.concatenate((img_y, img_u, img_v), axis=2)
        return img_yuv444

    def _histo_equal(self, img_yuv444: np.ndarray):
        """Apply equalizehist to img_yuv444's y_channel.

        Args:
            img_yuv444: image of yuv444 format
        """
        if self._equalize_hist_method == "calc":
            data = np.array(img_yuv444[:, :, 0])
        else:
            data = np.array(img_yuv444[:, :, 0], dtype=np.float32)
        eye_img_y_equaled = _do_equalize_hist(
            img=data, method=self._equalize_hist_method
        )
        img_yuv444[:, :, 0] = eye_img_y_equaled
        return img_yuv444

    def _pyr_down_up(self, img):
        h, w, _ = img.shape
        img_down = cv2.resize(img, (w // 2, h // 2))
        # if random.random() < 0.5:
        #     img_down = cv2.resize(img_down, (w//4, h//4))
        img_up = cv2.resize(img_down, (w, h))
        return img_up

    def __call__(self, data):
        img = data["img"].copy()
        assert isinstance(
            img, (torch.Tensor, np.ndarray)
        ), " \
            GazeYUVTransform not support {}".format(
            type(img)
        )
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img = img.astype(np.uint8)

        if self._rgb_data:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8)
        eye_img, pos_x, pos_y = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        eye_img_stacked = np.stack([eye_img] * 3).transpose((1, 2, 0))

        # bgr2yuv444
        img_yuv444 = self.bgr2yuv444(eye_img_stacked)
        # histo_equal
        if self._equalize_hist:
            img_yuv444 = self._histo_equal(img_yuv444)

        # yuv444 --> yuv444_int8
        # img_yuv444_int32 = img_yuv444.astype(np.int32)
        # img_yuv444_int32 = img_yuv444_int32 - 128
        # img_yuv444_int8 = img_yuv444_int32.astype(np.int8)
        # img_yuv444_int8[:, :, 1] = pos_x - 128
        # img_yuv444_int8[:, :, 2] = pos_y - 128

        img_yuv444[:, :, 1] = pos_x
        img_yuv444[:, :, 2] = pos_y
        data["img"] = img_yuv444[:, :, : self._nc]
        return data


@OBJECT_REGISTRY.register
class GazeRandomCropWoResize:
    """Random crop without resize.

    More notes ref to https://horizonrobotics.feishu.cn/docx/LKhddopAeoXJmXxa6KocbwJdnSg.  # noqa
    """

    def __init__(
        self,
        size=(192, 320),
        area=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        prob: float = 1.0,
        is_train: bool = True,
    ):
        self.prob = prob
        self._crop_args = (size, area, ratio)
        self.is_train = is_train

    def _trans_eye_ldmks(self, eye_ldmk: np.ndarray, roi_trans_info: Tuple):
        """trans_eye_ldmks label in random crop.

        Args:
            eye_ldmk: eye_ldmk ratio data
            roi_trans_info: (x0, y0, new_w, new_h, w, h)
        """
        x0, y0, new_w, new_h, w, h = roi_trans_info
        ldmks_extended = np.concatenate(
            [eye_ldmk, np.ones((eye_ldmk.shape[0], 1))], 1
        )
        T_ldmk_trans = np.array(
            [[w / new_w, 0], [0, h / new_h], [-x0 / new_w, -y0 / new_h]]
        )

        ldmks_auged = np.dot(ldmks_extended, T_ldmk_trans)
        # set ldmks out of croped region to (-1000, -1000) to avoid gradient
        # valid_mask = (ldmks_auged >= 0).min(axis=1).reshape(-1, 1)
        valid_mask = ldmks_auged >= 0
        # valid_mask = np.stack(valid_mask, valid_mask).T
        ldmks_auged = ldmks_auged * valid_mask + (1 - valid_mask) * -1000
        return ldmks_auged

    def _scale_down(self, src_size, size):
        w, h = size
        sw, sh = src_size
        if sh < h:
            w, h = float(w * sh) / h, sh
        if sw < w:
            w, h = sw, float(h * sw) / w
        return int(w), int(h)

    def _center_crop(
        self,
        src: Union[np.ndarray, Tuple],
        size=None,
        is_val=False,
        return_crop_roi=True,
    ):
        """Crops the image `src` to the given `size`.

        Crop by trimming on all four sides and preserving the center of the
        image, upsamples if `src` is smaller than `size`.
        """
        if isinstance(src, tuple):
            h, w, _ = src
            return_crop_roi = False
        elif isinstance(src, np.ndarray):
            h, w, _ = src.shape
        else:
            raise NotImplementedError
        if not is_val:
            assert (
                size is not None
            ), "when is_val==False, size must not be None"
            new_w, new_h = self._scale_down((w, h), size)
        else:
            new_w, new_h = w, h
        # refer to https://mxnet.apache.org/versions/1.8.0/api/python/docs/_modules/mxnet/image/image.html#scale_down  # noqa
        new_w = int(new_w // 2 * 2)
        new_h = int(new_h // 2 * 2)
        if not is_val:
            x0 = int((w - new_w) / 2)
            y0 = int((h - new_h) / 2)
        else:
            x0 = 0
            y0 = 0
        if return_crop_roi:
            out = fixed_crop(src, x0, y0, new_w, new_h)
            return out, (x0, y0, new_w, new_h, w, h)
        else:
            return (x0, y0, new_w, new_h)

    def _random_crop(
        self,
        src: Union[np.ndarray, Tuple] = None,
        size: Union[List, Tuple] = None,
        area: Union[float, Tuple[float, float]] = None,
        ratio: Tuple = None,
        return_crop_roi: bool = True,
        **kwargs,
    ):
        """Random crop wrapping for actually random operation.

        1. For offline augm baseline, this function require return
        crop roi and roi coordination.
        2. For online augm baseline, this function only require return
        roi coordination.

        Args:
            src: In offline augm, src:image. In online augm, src:ori_shape
            size: Size of the crop formatted as (width, height).
            area: float in (0, 1] or tuple of (float, float),
                If tuple, minimum area and maximum area to be
                maintained after cropping
                If float, minimum area to be maintained after
                cropping, maximum area is set to 1.0
            ratio: Aspect ratio range as (min_aspect_ratio,
                max_aspect_ratio)
            return_crop_roi: In offline augm baseline, require return crop roi.
        """
        if isinstance(src, tuple):
            h, w, _ = src
            return_crop_roi = False
        elif isinstance(src, np.ndarray):
            h, w, _ = src.shape
        else:
            raise NotImplementedError
        src_area = h * w
        if "min_area" in kwargs:
            warnings.warn(
                "`min_area` is deprecated. " "Please use `area` instead.",
                DeprecationWarning,
            )
            area = kwargs.pop("min_area")
        assert not kwargs, (
            "unexpected keyword arguments " "for `random_size_crop`."
        )

        for _ in range(10):
            target_area = random.uniform(area[0], area[1]) * src_area
            log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
            new_ratio = np.exp(random.uniform(*log_ratio))

            # new_w is even
            new_w = int(round(np.sqrt(target_area * new_ratio))) // 2 * 2
            # new_h is even
            new_h = int(round(np.sqrt(target_area / new_ratio))) // 2 * 2

            if new_w <= w and new_h <= h:
                x0 = random.randint(0, w - new_w)
                y0 = random.randint(0, h - new_h)
                if return_crop_roi:
                    out = fixed_crop(src, x0, y0, new_w, new_h)
                    return out, (x0, y0, new_w, new_h, w, h)
                else:
                    return (x0, y0, new_w, new_h)

        # fall back to center_crop
        return self._center_crop(src, size)

    def __call__(self, data):
        # offline augm
        if isinstance(data, dict):
            gaze_input = data["img"].copy()
            gaze_input[:, :, 1] = data["horizon_img"]
            gaze_input[:, :, 2] = data["vertical_img"]
            if self.is_train and random.random() < self.prob:
                ret_roi, roi_trans_info = self._random_crop(
                    gaze_input, *self._crop_args
                )
            else:
                ret_roi, roi_trans_info = self._center_crop(
                    gaze_input, is_val=True
                )
            x0, y0, new_w, new_h, _, _ = roi_trans_info
            _gazemap_crop = data["gaze_label"]["gt_gazemap"][
                y0 : y0 + new_h, x0 : x0 + new_w, :
            ]
            eye_ldmk = self._trans_eye_ldmks(
                eye_ldmk=data["gaze_label"]["gt_normed_eye_ldmk"],
                roi_trans_info=roi_trans_info,
            )
            data["img"] = ret_roi
            data["gaze_label"]["gt_normed_eye_ldmk"] = eye_ldmk
            data["gaze_label"]["gt_gazemap"] = _gazemap_crop
            return data
        # online augm(rotate 3d augm)
        elif isinstance(data, tuple):
            ori_shape = data
            if self.is_train and random.random() < self.prob:
                return self._random_crop(ori_shape, *self._crop_args)
            else:
                return self._center_crop(ori_shape, is_val=True)
        else:
            raise NotImplementedError


@OBJECT_REGISTRY.register
class Clip(object):
    """Clip Data to [minimum, maximum].

    Args:
        minimum: The minimum number of data. Defaults 0.
        maximum: The maximum number of data. Defaults 255.
    """

    def __init__(self, minimum=0.0, maximum=255.0):
        self.min = minimum
        self.max = maximum

    def __call__(self, data):
        image = data["img"]
        if isinstance(data, torch.Tensor):
            image = image.numpy()
        image = np.clip(image, self.min, self.max)
        data["img"] = image
        return data


@OBJECT_REGISTRY.register
class RandomColorJitter(object):  # noqa: D205,D400
    """Randomly change the brightness, contrast, saturation and hue of an image.  # noqa

    More notes ref to https://horizonrobotics.feishu.cn/docx/LKhddopAeoXJmXxa6KocbwJdnSg.  # noqa
    """

    @require_packages("albumentations")
    def __init__(
        self,
        brightness=0.5,
        contrast=(0.5, 1.5),
        saturation=(0.5, 1.5),
        hue=0.1,
        prob=0.5,
    ):
        super(RandomColorJitter, self).__init__()
        self.prob = prob
        self.transform = albumentations.Compose(
            [
                ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            ]
        )

    def __call__(self, data):
        if np.random.rand() >= self.prob:
            return data
        assert "img" in data.keys()
        transformed = self.transform(image=data["img"])
        data["img"] = transformed["image"]
        if "gt_img" in data:
            data["gt_img"] = data["img"].copy()
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        return repr_str


@OBJECT_REGISTRY.register
class GazeRotate3DWithCrop(object):
    """Random rotate image, calculate ROI and random crop if necessary.

    Meanwhile, pos map is generated.

    Args:
        is_train: To apply 3d rotate augm in train mod or test mod.
            Defaults to True.
        head_pose_type: Type of head pose. Defaults to "euler z-xy degree".
        rand_crop_scale: Scale of rand crop. Defaults to (0.85, 1.0).
        rand_crop_ratio: Ratio of rand crop. Defaults to (1.25, 2).
        rand_crop_cropper_border: Expanded pixel size. Defaults to 5.
        rotate_type: 3D rotate augm type. Defaults to "pos_map_uniform".
        rotate_augm_prob: Prob to do 3d rotate augm. Defaults to 1.
        pos_map_range_pitch: Rotate range in pitch dimension.
        pos_map_range_yaw: Rotate range in yaw dimension.
        pos_map_range_roll: Rotate range in roll dimension.
        delta_rpy_range: _description_.
        seperate_ldmk: _description_. Defaults to False.
        seperate_ldmk_roll_range: _description_. Defaults to (0, 0).
        crop_size: Crop size. Defaults to (256, 128).
        to_yuv420sp: Whether transform to yuv420sp. Defaults to True.
        standard_focal: Standard focal of camera. Defaults to 600.
        cropping_ratio: Ratio of crop when calc crop roi with
            rotated face ldmks.
        rand_inter_type: Whether use rand inter type. Defaults to False.
    """

    def __init__(
        self,
        is_train=True,
        head_pose_type="euler z-xy degree",
        rand_crop_scale=(0.85, 1.0),
        rand_crop_ratio=(1.25, 2),
        rand_crop_cropper_border=5,
        rotate_type="pos_map_uniform",
        rotate_augm_prob: float = 1,
        pos_map_range_pitch=(-17, 17),
        pos_map_range_yaw=(-20, 20),
        pos_map_range_roll=(-20, 20),
        delta_rpy_range=([-0, 0], [-0, 0], [-0, 0]),
        seperate_ldmk=False,
        seperate_ldmk_roll_range=(0, 0),
        crop_size=(256, 128),
        to_yuv420sp=True,
        standard_focal=600,
        cropping_ratio=0.25,
        rand_inter_type=False,
    ):
        super(GazeRotate3DWithCrop, self).__init__()
        self.rotate_type = rotate_type
        self.crop_size = crop_size
        self.to_yuv420sp = to_yuv420sp
        std_focal = standard_focal
        self.std_intrin = np.array(
            [[std_focal, 0, 1280 / 2], [0, std_focal, 960 / 2], [0, 0, 1]]
        )
        self.cropping_ratio = cropping_ratio
        self.head_pose_type = head_pose_type
        self.rand_crop_cropper_border = rand_crop_cropper_border
        self.cropper = GazeRandomCropWoResize(
            area=rand_crop_scale,
            ratio=rand_crop_ratio,
            size=self.crop_size,
        )
        self.rand_inter_type = rand_inter_type
        self.is_train = is_train
        self.rand_rotate_prob = rotate_augm_prob
        if rotate_type is None or rotate_type.lower() == "none":
            self.generate_random_R = self._generate_R_no_rotate
        elif rotate_type == "pos_map_uniform":
            self.lo, self.hi = np.array(
                [list(pos_map_range_pitch), list(pos_map_range_yaw)]
            ).T
            self.roll_range = list(pos_map_range_roll)
            self.generate_random_R = self._generate_R_pos_map_uniform
        elif rotate_type == "delta_rpy_uniform":
            self.lo, self.hi = np.array(list(delta_rpy_range)).T
            self.generate_random_R = self._generate_R_delta_rpy_uniform
        else:
            raise NotImplementedError(rotate_type)
        self.is_seperate = seperate_ldmk
        self.roll_lo, self.roll_hi = seperate_ldmk_roll_range
        self.cropping_ldmks = np.array([6, 7, 8, 9, 11, 12, 13, 14])
        self.cropping_ldmks_68 = np.arange(36, 48)
        self.EPS = 10 ** -8

    def _generate_R_no_rotate(self, *args, **argv):
        return np.eye(3, dtype=np.float32)

    def _generate_R_pos_map_uniform(self, face_ldmks, K):
        cropping_ldmks = (
            self.cropping_ldmks_68
            if len(face_ldmks) == 68
            else self.cropping_ldmks
        )
        R_2_ctr = self._get_R_to_ctr(
            face_ldmks[cropping_ldmks].mean(0, keepdims=True), K
        )
        euler_roll = [np.random.uniform(*self.roll_range), 0, 0]
        R_roll = Rotation.from_euler("zxy", euler_roll, True).as_matrix()
        random_pitch_yaw = np.random.uniform(self.lo, self.hi, 2)
        euler_py = [0, random_pitch_yaw[0], random_pitch_yaw[1]]
        R_pitch_yaw = Rotation.from_euler("zxy", euler_py, True).as_matrix()
        R = R_pitch_yaw @ R_roll @ R_2_ctr
        return R.astype(np.float32)

    def _generate_R_delta_rpy_uniform(self, face_ldmks, K):
        cropping_ldmks = (
            self.cropping_ldmks_68
            if len(face_ldmks) == 68
            else self.cropping_ldmks
        )
        R_2_ctr = self._get_R_to_ctr(
            face_ldmks[cropping_ldmks].mean(0, keepdims=True), K
        )
        random_rpy = np.random.uniform(self.lo, self.hi, 3)
        R_delta = Rotation.from_euler("zxy", random_rpy, True).as_matrix()
        R = R_2_ctr.T @ R_delta @ R_2_ctr
        return R.astype(np.float32)

    def _generate_R_delta_image_roll(self, *args, **argv):
        roll = np.random.uniform(self.roll_lo, self.roll_hi)
        R = Rotation.from_euler("zxy", [roll, 0, 0], True).as_matrix()
        return R.astype(np.float32)

    def _get_R_to_ctr(self, eye_ctr, K):
        if isinstance(eye_ctr, torch.Tensor):
            eye_ctr = eye_ctr.numpy()
        if isinstance(K, torch.Tensor):
            K = K.numpy()
        eye_ctr = eye_ctr.reshape(1, 2)
        K_inv = np.linalg.inv(K)
        eye_ctr_h = np.concatenate(
            (eye_ctr, np.ones_like(eye_ctr)[:, :1]), axis=1
        )
        eye_ctr_vec = eye_ctr_h @ K_inv.T
        eye_ctr_vec = eye_ctr_vec / np.linalg.norm(eye_ctr_vec, 2)
        axis = np.cross(eye_ctr_vec.reshape(-1), np.array([0, 0, 1]))
        axis_norm = np.linalg.norm(axis)
        if axis_norm < self.EPS:
            return np.eye(3)
        axis = axis / axis_norm
        theta = np.arccos(eye_ctr_vec[0, -1])
        rotvec = axis * theta
        return Rotation.from_rotvec(rotvec).as_matrix()

    def _check_gaze(self, _gaze):
        if isinstance(_gaze, torch.Tensor):
            gaze = _gaze.numpy()
        else:
            gaze = _gaze
        return gaze.min() > -500

    def _transform_headpose(self, pose, R):
        if self.head_pose_type == "euler z-xy degree":
            r, p, y = pose
            pose_R = Rotation.from_euler("zxy", [r, -p, y], True).as_matrix()
            pose_R = R @ pose_R
            r, p, y = Rotation.from_matrix(pose_R).as_euler("zxy", True)
            return np.array([r, -p, y])
        else:
            raise NotImplementedError(self.head_pose_type)

    def _gen_data_with_rotate(self, R, image, label):
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        head_pose = label["gt_head_pose"]
        gaze = label["gt_gaze"]
        eye_ldmks = label["gt_eye_ldmk"]
        eye_bbox = label["gt_eye_bbox"]
        loss_weight = label["gt_loss_weight"]
        face_ldmks = label["gt_face_ldmks"]
        K = label["intrinsics_K"]

        # (head_pose, gaze, eye_ldmks, eye_bbox, loss_weight,
        #     glass, face_ldmks, K) = \
        #     [_.asnumpy() for _ in label]

        K_inv = np.linalg.inv(K)
        # 1. rotate face ldmks
        cropping_ldmks = (
            self.cropping_ldmks_68
            if len(face_ldmks) == 68
            else self.cropping_ldmks
        )
        face_ldmks = transform_ldmk(
            face_ldmks[cropping_ldmks], K_inv.T @ R.T @ K.T
        )
        # 2. calc crop roi with rotated face ldmks
        left, top, right, bottom = calc_roi_with_ldmk(
            face_ldmks,
            self.cropping_ratio,
            self.crop_size,
            self.rand_crop_cropper_border,
            self.is_train,
        )
        # 3. calc random crop roi
        h, w = bottom - top, right - left
        if self.is_train:
            x1, y1, x2, y2 = self.cropper((h, w, 3))
            left, top = left + x1, top + y1
            right, bottom = left + x2, top + y2
        else:
            right = left + int(w // 2 * 2)
            bottom = top + int(h // 2 * 2)
        eye_bbox_new = [left, top, right, bottom]
        # 4. calc map for warping and warp or cropped image in val
        image_crop = warp_crop_image(
            image,
            eye_bbox,
            eye_bbox_new,
            K_inv,
            R,
            K,
            self.rand_inter_type,
            self.is_train,
        )
        # 5. calc pos map
        horizon_pos_map, vertical_pos_map = generate_pos_map(
            eye_bbox_new,
            image_crop.shape,
            K_inv,
            self.std_intrin,
            self.to_yuv420sp,
        )
        # 6. calc new eye ldmk
        valid_mask = (eye_ldmks > 0).min(axis=1)
        if self.is_train:
            eye_ldmks_valid = transform_ldmk(
                eye_ldmks[valid_mask], K_inv.T @ R.T @ K.T
            )
            eye_ldmks[valid_mask] = eye_ldmks_valid
        eye_ldmks = eyeldmk_transform(eye_ldmks, eye_bbox_new)
        # 7. calc new gaze
        if self.is_train and np.max(gaze) > -300:
            gaze = rotate_gaze(gaze, R)
        # 8. calc new head pose
        if self.is_train:
            head_pose = self._transform_headpose(head_pose, R)

        # image_crop = (image_crop, horizon_pos_map, vertical_pos_map)
        image_crop[:, :, 1] = horizon_pos_map
        image_crop[:, :, 2] = vertical_pos_map

        label["gt_head_pose"] = head_pose
        label["gt_gaze"] = gaze
        label["gt_eye_ldmk"] = eye_ldmks
        label["gt_eye_bbox"] = eye_bbox_new
        label["gt_loss_weight"] = loss_weight

        return image_crop, label

    def __call__(self, data):
        image = data["img"]
        label = data["gaze_label"]
        if isinstance(image, tuple):
            (image,) = image
        gaze = label["gt_gaze"]
        face_ldmks = label["gt_face_ldmks"]
        intrinsics_K = label["intrinsics_K"]
        # (head_pose, gaze, eye_ldmks, eye_bbox, loss_weight,
        #     glass, face_ldmks, intrinsics_K) = label
        if self.is_train and np.random.rand() < self.rand_rotate_prob:
            if self.is_seperate and not self._check_gaze(gaze):
                R = self._generate_R_delta_image_roll()
            else:
                R = self.generate_random_R(face_ldmks, intrinsics_K)
        else:
            R = np.eye(3)
        image_crop, label_crop = self._gen_data_with_rotate(R, image, label)
        data["img"] = image_crop
        data["gaze_label"] = label_crop
        data["gaze_label"]["R"] = R
        return data
