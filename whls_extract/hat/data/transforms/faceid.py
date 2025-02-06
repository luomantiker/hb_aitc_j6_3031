# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import math
import random
from typing import Optional, Tuple

import cv2
import numpy as np

from hat.data.transforms.functional_img import imresize
from hat.data.utils import encode_decode_img
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = [
    "RandomGray",
    "JPEGCompress",
    "RandomDownSample",
    "GaussianBlur",
    "MotionBlur",
    "SpatialVariantBrightness",
    "Contrast",
]


@OBJECT_REGISTRY.register
class RandomGray(object):
    """
    Transform RGB or BGR format into Gray format.

    .. note::
        Affected keys: 'img'.

    Args:
        p : prob
        rgb_data : Default=True
            Whether the input data is in RGB format. If not, it should be
            in BGR format.
        only_one_channel: If ture, the returned gray image contains
            only one channel. Default to False.

    """

    def __init__(
        self,
        p: float = 0.08,
        rgb_data: bool = True,
        only_one_channel: bool = False,
    ):
        self.p = p
        self.rgb_data = rgb_data
        self.only_one_channel = only_one_channel

    def _do_gray(self, data):
        do_gray = np.random.choice([False, True], p=[1 - self.p, self.p])

        if do_gray:
            img = data["img"]
            if self.rgb_data:
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.only_one_channel:
                data["img"] = gray_img[:, :, np.newaxis].astype(img.dtype)
            else:
                new_img = np.zeros(img.shape)
                for i in range(new_img.shape[-1]):
                    new_img[:, :, i] = gray_img
                data["img"] = new_img.astype(img.dtype)

    def __call__(self, data):
        self._do_gray(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"p_dogray={self.p}"
        return repr_str


@OBJECT_REGISTRY.register
class JPEGCompress(object):
    """
    Do JPEG compression to downgrade image quality.

    .. note::
        Affected keys: 'img'.

    Args:
        p : prob
        max_quality : (0, 100]
            JPEG compression highest quality
        min_quality : (0, 100]
            JPEG compression lowest quality
    """

    def __init__(
        self,
        p: float = 0.08,
        max_quality: int = 95,
        min_quality: int = 30,
    ):
        self.p = p
        self.max_quality = max_quality
        self.min_quality = min_quality

    def _jpeg_compress_aug(self, data):
        do_compress = np.random.choice([False, True], p=[1 - self.p, self.p])

        if do_compress:
            img = data["img"]
            scale = random.random()
            quality = (
                scale * (self.max_quality - self.min_quality)
                + self.min_quality
            )
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            if np.random.uniform() < 0.5:
                decimg = encode_decode_img(img, ".jpg", encode_param)
            else:
                img_bgr = img[:, :, [2, 1, 0]]
                decimg = encode_decode_img(img_bgr, ".jpg", encode_param)
                decimg = decimg[:, :, [2, 1, 0]]
            data["img"] = decimg

    def __call__(self, data):
        self._jpeg_compress_aug(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"p_docompress={self.p}"
        repr_str += f"max_quality={self.max_quality}"
        repr_str += f"min_quality={self.min_quality}"
        return repr_str


@OBJECT_REGISTRY.register
class SpatialVariantBrightness(object):
    """
    Spatial variant brightness, Enhanced Edition.

    .. note::
        Affected keys: 'img'.

    Args:
        p : prob
        brightness : default is 0.6
            Brightness ratio for this augmentation, the value choice
            in Uniform ~ [-brightness, brigheness].
        max_template_type : default is 3
            Max number of template type in once process. Note,
            the selection process is repeated.
        online_template : default is False
            Template generated online or offline.
            "False" is recommended to get fast speed.
    """

    def __init__(
        self,
        p: float = 0.08,
        brightness: float = 0.6,
        max_template_type: int = 3,
        online_template: bool = False,
    ):
        self.p = p
        self.brightness = brightness
        self.max_template_type = max_template_type
        self.online_template = online_template
        self.template_in_cache = False

    def _get_line_coeff(self, angle):
        sin_x = math.sin(angle) ** 2 * (2 * (math.sin(angle) > 0) - 1)
        cos_x = math.cos(angle) ** 2 * (2 * (math.cos(angle) > 0) - 1)
        return sin_x, cos_x

    def _normalize_template(self, template_h):
        min_value, max_value = np.min(template_h), np.max(template_h)
        template_h = (template_h - min_value) / (max_value - min_value)  # noqa
        return template_h

    def _linear_template(self, h, w, angle):
        template_h = np.ones((h, w))
        sin_x, cos_x = self._get_line_coeff(angle)
        for x in range(w):
            for y in range(h):
                template_h[y, x] = (sin_x * x / w + cos_x * y / h) / 2
        return self._normalize_template(template_h)

    def _qudratical_template(self, h, w, angle):
        template_h = np.ones((h, w))
        sin_x, cos_x = self._get_line_coeff(angle)
        for x in range(w):
            for y in range(h):
                template_h[y, x] = (sin_x * x / w + cos_x * y / h) ** 2
        return self._normalize_template(template_h)

    def _parabola_template(self, h, w, angle):
        template_h = np.ones((h, w))
        sin_x, cos_x = self._get_line_coeff(angle)
        for x in range(w):
            for y in range(h):
                template_h[y, x] = (
                    (sin_x * x / w + cos_x * y / h) - 0.5
                ) ** 2  # noqa
        return self._normalize_template(template_h)

    def _cubic_template(self, h, w, angle):
        template_h = np.ones((h, w))
        sin_x, cos_x = self._get_line_coeff(angle)
        for x in range(w):
            for y in range(h):
                template_h[y, x] = (sin_x * x / w + cos_x * y / h) ** 3  # noqa
        return self._normalize_template(template_h)

    def _sinwave_template(self, h, w, angle, frequency, theta):
        # sin (fx + theta)
        template_h = np.ones((h, w))
        sin_x, cos_x = self._get_line_coeff(angle)
        for x in range(w):
            for y in range(h):
                template_h[y, x] = (
                    math.sin(
                        (sin_x * x / w + cos_x * y / h) * frequency * np.pi
                        + theta * np.pi / 180.0  # noqa
                    )
                    + 1
                ) / 2  # noqa
        return self._normalize_template(template_h)

    def generate_template(self, h, w):
        # `sinwave` has a bigger proportion than others.
        temp_types = [
            "parabola",
            "linear",
            "qudratical",
            "cubic",
            "sinwave",
            "sinwave",
        ]
        idxs = np.random.randint(
            0, len(temp_types), random.randint(1, self.max_template_type)
        )
        temp_type_list = [temp_types[i] for i in idxs]
        template_h_list = []
        for temp_type in temp_type_list:
            template_h = np.ones((h, w))
            angle = random.randint(0, 360) * np.pi / 180.0
            if temp_type == "parabola":
                template_h = self._parabola_template(h, w, angle)
            elif temp_type == "linear":
                template_h = self._linear_template(h, w, angle)
            elif temp_type == "qudratical":
                template_h = self._qudratical_template(h, w, angle)
            elif temp_type == "cubic":
                template_h = self._cubic_template(h, w, angle)
            elif temp_type == "sinwave":
                frequency = random.choice([0.5, 1, 1.5, 2, 3])
                theta = random.choice([0, 30, 60, 90])
                self._sinwave_template(h, w, angle, frequency, theta)
            template_h_list.append(template_h)
        return np.mean(np.dstack(template_h_list), axis=2, keepdims=True)

    def generate_template_offline(self, h, w):
        if self.template_in_cache is False:
            pi = 3.14159
            line_angle_list = np.arange(0, 360, 10) * pi / 180.0
            template_h_list = []
            # Parabola
            for angle in line_angle_list:
                template_h_list.append(self._parabola_template(h, w, angle))

            # Linearly Light Change
            for angle in line_angle_list:
                template_h_list.append(self._linear_template(h, w, angle))
            # Qudratically Light Change
            for angle in line_angle_list:
                template_h_list.append(self._qudratical_template(h, w, angle))
            # Cubicly Light Change
            for angle in line_angle_list:
                template_h_list.append(self._cubic_template(h, w, angle))
            # Sinwave Light Change
            frequency_list = [0.5, 1, 1.5, 2, 3]
            theta_list = [0, 30, 60, 90]
            for frequency in frequency_list:
                for theta in theta_list:
                    for angle in line_angle_list:
                        template_h_list.append(
                            self._sinwave_template(
                                h, w, angle, frequency, theta
                            )
                        )
            self.template_list = template_h_list
            self.template_in_cache = True
            self.cache_template_height = h
            self.cache_template_width = w

        assert (
            self.cache_template_height == h and self.cache_template_width == w
        ), (
            "image shape change detected, please use online "
            "spatial-variant-brightness"
            "or write code to support tmplate resize"
        )
        selected_template_num = np.random.randint(self.max_template_type) + 1
        choice_list = np.random.choice(
            len(self.template_list), selected_template_num
        )
        r_template = self.template_list[choice_list[0]].copy()
        for i in range(selected_template_num - 1):
            r_template += self.template_list[choice_list[i + 1]]
        r_template = r_template / selected_template_num
        return r_template

    def _do_brightness(self, data):
        do_brightness = np.random.choice([False, True], p=[1 - self.p, self.p])

        if do_brightness:
            img = data["img"]
            src_dtype = img.dtype
            h, w = img.shape[:2]
            if self.online_template:
                template_h = self.generate_template(h, w).reshape((h, w, 1))
            else:
                template_h = self.generate_template_offline(h, w).reshape(
                    (h, w, 1)
                )
            template_r = np.broadcast_to(
                template_h,
                (template_h.shape[0], template_h.shape[1], img.shape[2]),
            )
            c = np.random.uniform(-self.brightness, self.brightness)
            img = img * (1 + template_r * c)
            img = np.clip(img, 0, 255)
            data["img"] = img.astype(src_dtype)

    def __call__(self, data):
        self._do_brightness(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"p_dospatialbrightness={self.p}"
        repr_str += f"brightness={self.brightness}"
        repr_str += f"max_template_type={self.max_template_type}"
        repr_str += f"online_template={self.online_template}"
        return repr_str


@OBJECT_REGISTRY.register
class Contrast(object):
    """
    Randomly jitters image contrast with a factor.

    .. note::
        Affected keys: 'img'.

    Args:
        p : prob
        contrast : How much to jitter contrast.
        The contrast jitter ratio range, [0, 1]
    """

    def __init__(
        self,
        p: float = 0.08,
        contrast: float = 0.5,
    ):
        self.p = p
        self.contrast = contrast
        self.coef = np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)

    def _do_contrast(self, data):
        do_contrast = np.random.choice([False, True], p=[1 - self.p, self.p])

        if do_contrast:
            img = data["img"]
            alpha = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            gray = img * self.coef
            gray = (3.0 * (1.0 - alpha) / float(gray.size)) * np.sum(gray)
            img = img.astype(np.float32) * alpha
            img += gray
            img = np.clip(img, 0, 255)
            data["img"] = img.astype(np.uint8)

    def __call__(self, data):
        self._do_contrast(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"p_docontrast={self.p}"
        repr_str += f"contrast={self.contrast}"
        return repr_str


@OBJECT_REGISTRY.register
class GaussianBlur(object):
    """
    Randomly add guass blur on an image.

    .. note::
        Affected keys: 'img'.

    Args:
        p : prob
        kernel_size_min : min size of guass kernel
        kernel_size_max : max size of guass kernel
        sigma_min : min sigma of guass kernel
        sigma_max : max sigma of guass kernel
    """

    def __init__(
        self,
        p: float = 0.08,
        kernel_size_min: int = 2,
        kernel_size_max: int = 9,
        sigma_min: float = 0.0,
        sigma_max: float = 0.0,
    ):
        self.p = p
        self.kernel_size_min = kernel_size_min
        self.kernel_size_max = kernel_size_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _do_gaussian_blur(self, data):
        do_gaussian_blur = np.random.choice(
            [False, True], p=[1 - self.p, self.p]
        )

        if do_gaussian_blur:
            img = data["img"]
            k = np.random.randint(self.kernel_size_min, self.kernel_size_max)
            if k % 2 == 0:
                if np.random.rand() > 0.5:
                    k += 1
                else:
                    k -= 1
            s = np.random.uniform(self.sigma_min, self.sigma_max)
            img_blur = cv2.GaussianBlur(src=img, ksize=(k, k), sigmaX=s)
            img_blur = np.clip(img_blur, 0, 255)
            data["img"] = img_blur.astype(np.uint8)

    def __call__(self, data):
        self._do_gaussian_blur(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"p_dogaussianblur={self.p}"
        repr_str += f"kernel_size_min={self.kernel_size_min}"
        repr_str += f"kernel_size_max={self.kernel_size_max}"
        repr_str += f"sigma_min={self.sigma_min}"
        repr_str += f"sigma_max={self.sigma_max}"
        return repr_str


@OBJECT_REGISTRY.register
class MotionBlur(object):
    """
    Randomly add motion blur on an image.

    .. note::
        Affected keys: 'img'.

    Args:
        p : prob
        length_min : min size of motion blur
        length_max : max size of motion blur
        angle_min : min angle of motion blur
        angle_max : max angle of motion blur
    """

    def __init__(
        self,
        p: float = 0.08,
        length_min: int = 9,
        length_max: int = 18,
        angle_min: float = 1,
        angle_max: float = 359,
    ):
        self.p = p
        self.length_min = length_min
        self.length_max = length_max
        self.angle_min = angle_min
        self.angle_max = angle_max

    def _do_motion_blur(self, data):
        do_motion_blur = np.random.choice(
            [False, True], p=[1 - self.p, self.p]
        )

        if do_motion_blur:
            img = data["img"]
            length = np.random.randint(self.length_min, self.length_max)
            angle = np.random.randint(self.angle_min, self.angle_max)
            if angle in [0, 90, 180, 270, 360]:
                angle += 1

            half = length / 2
            EPS = np.finfo(float).eps
            alpha = (angle - math.floor(angle / 180) * 180) / 180 * math.pi
            cosalpha = math.cos(alpha)
            sinalpha = math.sin(alpha)
            if cosalpha < 0:
                xsign = -1
            elif angle == 90:
                xsign = 0
            else:
                xsign = 1
            psfwdt = 1

            # blur kernel size
            sx = int(
                math.fabs(length * cosalpha + psfwdt * xsign - length * EPS)
            )
            sy = int(math.fabs(length * sinalpha + psfwdt - length * EPS))
            psf1 = np.zeros((sy, sx))

            # psf1 is getting small when (x, y) move from left-top
            # to right-bottom at this moment (x, y) is moving from
            # right-bottom to left-top
            for i in range(0, sy):
                for j in range(0, sx):
                    psf1[i][j] = i * math.fabs(cosalpha) - j * sinalpha
                    rad = math.sqrt(i * i + j * j)
                    if rad >= half and math.fabs(psf1[i][j]) <= psfwdt:
                        temp = half - math.fabs(
                            (j + psf1[i][j] * sinalpha) / cosalpha
                        )
                        psf1[i][j] = math.sqrt(
                            psf1[i][j] * psf1[i][j] + temp * temp
                        )
                    psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j])
                    if psf1[i][j] < 0:
                        psf1[i][j] = 0

            # anchor is (0, 0) when (x, y) is moving towards left-top
            anchor = (0, 0)
            # anchor is (width, heigth) when (x, y) is moving towards right-top
            if angle < 90 and angle > 0:  # flip kernel at this moment
                psf1 = np.fliplr(psf1)
                anchor = (psf1.shape[1] - 1, 0)
            elif angle > -90 and angle < 0:  # moving towards right-bottom
                psf1 = np.flipud(psf1)
                psf1 = np.fliplr(psf1)
                anchor = (psf1.shape[1] - 1, psf1.shape[0] - 1)
            elif angle < -90:  # moving towards left-bottom
                psf1 = np.flipud(psf1)
                anchor = (0, psf1.shape[0] - 1)
            psf1 = psf1 / psf1.sum()

            img_blur = cv2.filter2D(
                src=img, ddepth=-1, kernel=psf1, anchor=anchor
            )

            img_blur = np.clip(img_blur, 0, 255)
            data["img"] = img_blur.astype(np.uint8)

    def __call__(self, data):
        self._do_motion_blur(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"p_domotionblur={self.p}"
        repr_str += f"length_min={self.length_min}"
        repr_str += f"length_max={self.length_max}"
        repr_str += f"angle_min={self.angle_min}"
        repr_str += f"angle_max={self.angle_max}"
        return repr_str


@OBJECT_REGISTRY.register
class RandomDownSample(object):
    """
    First downsample and upsample to original size.

    .. note::
        Affected keys: 'img'.

    Args:
        p : prob
        data_shape :
            C, H, W
        min_downsample_width :
            minimum downsample width
        inter_method :
            interpolation method index
    """

    def __init__(
        self,
        p: float = 0.2,
        data_shape: Optional[Tuple] = (3, 112, 112),
        min_downsample_width: int = 60,
        inter_method: int = 1,
    ):
        self.p = p
        self.data_shape = data_shape
        self.min_downsample_width = min_downsample_width
        self.inter_method = inter_method

    def _do_random_downsample(self, data):
        do_random_downsample = np.random.choice(
            [False, True], p=[1 - self.p, self.p]
        )

        def _get_inter_method(inter_method, sizes=()):
            if inter_method == 9:
                if sizes:
                    assert len(sizes) == 4
                    oh, ow, nh, nw = sizes
                    if nh > oh and nw > ow:
                        return "bicubic"
                    elif nh < oh and nw < ow:
                        return "area"
                    else:
                        return "bilinear"
                else:
                    return "bicubic"
            if inter_method == 10:
                inter_method = np.random.randint(0, 4)

            map_dict = {
                0: "nearest",
                1: "bilinear",
                2: "bicubic",
                3: "area",
                4: "lanczos",
            }
            if inter_method not in map_dict:
                raise AssertionError(
                    "error inter method {}".format(inter_method)
                )
            return map_dict[inter_method]

        if do_random_downsample:
            img = data["img"]
            scale = np.random.random()
            new_w = int(
                scale * (self.data_shape[2] - self.min_downsample_width)
                + self.min_downsample_width
            )
            new_h = int(new_w * self.data_shape[2] / self.data_shape[1])
            org_w = int(self.data_shape[2])
            org_h = int(self.data_shape[1])
            interpolation_method = _get_inter_method(
                self.inter_method,
                (self.data_shape[1], self.data_shape[2], new_h, new_w),
            )
            img = imresize(
                img,
                new_w,
                new_h,
                data["layout"],
                interpolation=interpolation_method,
            )
            img = imresize(
                img,
                org_w,
                org_h,
                data["layout"],
                interpolation=interpolation_method,
            )
            data["img"] = img

    def __call__(self, data):
        self._do_random_downsample(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"p_dogaussianblur={self.p}"
        repr_str += f"data_shape={self.data_shape}"
        repr_str += f"min_downsample_width={self.min_downsample_width}"
        repr_str += f"inter_method={self.inter_method}"
        return repr_str
