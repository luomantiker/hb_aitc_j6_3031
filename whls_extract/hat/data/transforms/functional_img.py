# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as nnf
from torch import Tensor

from hat.data.transforms.affine import AffineMat2DGenerator
from hat.utils.package_helper import require_packages

try:
    import hat_sim
except Exception:
    hat_sim = None


__all__ = [
    "demosaic",
    "imresize",
    "random_flip",
    "image_pad",
    "image_normalize",
    "imresize_pad_to_keep_ratio",
    "imresize_warp_when_nearest",
]

cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


@require_packages("hat_sim")
def demosaic(img: np.ndarray, img_h: int, img_w: int, raw_pattern: str):
    """Demosaic one channel raw data to three channel rgb data.

    Args:
        img : Input image width dtype np.float32.
        img_h : Image height.
        img_w : Image width.
        raw_pattern : Raw pattern of image.

    Returns:
        np.ndarray: output image.
    """
    assert img.dtype == np.float32
    assert raw_pattern in ["GRBG", "RGGB", "BGGR", "GBRG"]

    if raw_pattern == "GRBG":
        demosaic_pattern = 0
    elif raw_pattern == "RGGB":
        demosaic_pattern = 1
    elif raw_pattern == "BGGR":
        demosaic_pattern = 2
    elif raw_pattern == "GBRG":
        demosaic_pattern = 3
    else:
        raise NotImplementedError

    out_img = np.zeros((img_h, img_w, 3), dtype=np.float32)
    hat_sim.linear_demosaic(img_h, img_w, img, out_img, demosaic_pattern)
    out_img = out_img.reshape(img_h, img_w, 3)
    return out_img


def imresize_warp_when_nearest(
    img: np.ndarray,
    w: int,
    h: int,
    layout: str,
    divisor: int = 1,
    keep_ratio: bool = False,
    return_scale: bool = False,
    interpolation: str = "bilinear",
) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
    """Resize image with OpenCV.

    If keep_ratio=False, the image will be scaled to the maximum size that
    does not exceed wh and is divisible by divisor, otherwise resize shorter
    side to min(w, h) if the long side does not exceed max(w, h), otherwise
    resize the long side to max(w, h).

    Args:
        w: Width of resized image.
        h: Height of resized image.
        layout: Layout of img, `hwc` or `chw` or `hw`.
        divisor: Width and height are rounded to multiples of
            `divisor`, usually used in FPN-like structure.
        keep_ratio: If True, resize img to target size while keeping w:h
            ratio.
        return_scale: Whether to return `w_scale` and `h_scale`.
        interpolation: Interpolation method of image scaling, candidate
            value is ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos'].
    Returns:
        tuple: resized_img + (w_scale, h_scale)
    """
    assert layout in ["hwc", "chw", "hw"]
    if layout == "hwc":
        img_h, img_w, img_c = img.shape
    elif layout == "chw":
        img_c, img_h, img_w = img.shape
    else:
        img_h, img_w = img.shape

    if layout == "chw":
        # opencv only supports hwc layout
        img = np.ascontiguousarray(img.transpose((1, 2, 0)))  # chw > hwc
    if keep_ratio:
        short = min(w, h)
        max_size = max(w, h)
        im_size_min, im_size_max = (
            (img_h, img_w) if img_w > img_h else (img_w, img_h)
        )  # noqa
        scale = float(short) / float(im_size_min)
        if np.floor(scale * im_size_max / divisor) * divisor > max_size:
            # fit in max_size
            scale = (
                float(np.floor(max_size / divisor) * divisor) / im_size_max
            )  # noqa
        new_w, new_h = (
            int(np.floor(img_w * scale / divisor) * divisor),
            int(np.floor(img_h * scale / divisor) * divisor),
        )
    else:
        new_w, new_h = (
            int(np.floor(w / divisor) * divisor),
            int(np.floor(h / divisor) * divisor),
        )

    def _imresize_nearest_align(labelmap, h, w):
        img_h, img_w = labelmap.shape[:2]
        stride_h, stride_w = 1.0 * img_h / h, 1.0 * img_w / w
        ax, ay, bx, by = (
            stride_w,
            stride_h,
            (stride_w - 1.0) / 2,
            (stride_h - 1.0) / 2,
        )
        dst = np.float32([[0.0, 0.0], [0.0, h], [w, h]])
        src = np.float32(
            [[bx, by], [bx, h * ay + by], [w * ax + bx, h * ay + by]]
        )
        M = cv2.getAffineTransform(src, dst)
        return cv2.warpAffine(labelmap, M, (w, h), flags=cv2.INTER_NEAREST)

    if interpolation == "nearest":
        resized_img = _imresize_nearest_align(img, new_h, new_w)
    else:
        resized_img = cv2.resize(
            img, (new_w, new_h), interpolation=cv2_interp_codes[interpolation]
        )
    if layout == "chw":
        # change to the original layout
        resized_img = np.ascontiguousarray(resized_img.transpose((2, 0, 1)))
    w_scale = float(new_w / img_w)
    h_scale = float(new_h / img_h)
    if return_scale:
        return resized_img, w_scale, h_scale
    else:
        return resized_img


def imresize(
    img: np.ndarray,
    w,
    h,
    layout,
    divisor=1,
    keep_ratio=False,
    return_scale=False,
    interpolation="bilinear",
    raw_scaler_enable=False,
    sample1c_enable=True,
    raw_pattern=None,
    split_transform: bool = False,
    split_trans_h=256,
    split_trans_w=256,
) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
    """Resize image with OpenCV.

    If keep_ratio=False, the image will be scaled to the maximum size that
    does not exceed wh and is divisible by divisor, otherwise resize shorter
    side to min(w, h) if the long side does not exceed max(w, h), otherwise
    resize the long side to max(w, h).

    Args:
        w (int): Width of resized image.
        h (int): Height of resized image.
        layout (str): Layout of img, `hwc` or `chw` or `hw`.
        divisor (int): Width and height are rounded to multiples of
            `divisor`, usually used in FPN-like structure.
        keep_ratio (bool): If True, resize img to target size while keeping w:h
            ratio.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method of image scaling, candidate
            value is ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos'].
        raw_scaler_enable (bool): Whether to enable raw scaler.
        sample1c_enable (bool): Whether to sample one channel after resize
            image.
        raw_pattern (str): Current raw pattern.

    Returns:
        tuple: resized_img + (w_scale, h_scale)
    """
    assert layout in ["hwc", "chw", "hw"]
    if layout == "hwc":
        img_h, img_w, img_c = img.shape
    elif layout == "chw":
        img_c, img_h, img_w = img.shape
    else:
        img_h, img_w = img.shape

    if layout == "chw":
        # opencv only supports hwc layout
        img = np.ascontiguousarray(img.transpose((1, 2, 0)))  # chw > hwc
    if keep_ratio:
        short = min(w, h)
        max_size = max(w, h)
        im_size_min, im_size_max = (
            (img_h, img_w) if img_w > img_h else (img_w, img_h)
        )  # noqa
        scale = float(short) / float(im_size_min)
        if np.floor(scale * im_size_max / divisor) * divisor > max_size:
            # fit in max_size
            scale = (
                float(np.floor(max_size / divisor) * divisor) / im_size_max
            )  # noqa
        new_w, new_h = (
            int(np.floor(img_w * scale / divisor) * divisor),
            int(np.floor(img_h * scale / divisor) * divisor),
        )
    else:
        new_w, new_h = (
            int(np.floor(w / divisor) * divisor),
            int(np.floor(h / divisor) * divisor),
        )

    me_ds_img = None
    if raw_scaler_enable:
        img = img.astype(np.float32)

        if layout == "hw" or img_c == 1:
            img = img.ravel()
            out_img = demosaic(img, img_h, img_w, raw_pattern)
            out_img = np.reshape(out_img, (img_h, img_w, 3))  # h, w, c, n
            out_img = np.transpose(
                out_img[..., np.newaxis], (3, 2, 0, 1)
            )  # n, c, h, w
        else:
            out_img = np.transpose(img[np.newaxis, ...], (0, 3, 1, 2))
        out_img_tensor = torch.from_numpy(out_img)
        resized_tensor = nnf.interpolate(
            out_img_tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        resized_img = resized_tensor.cpu().detach().numpy()
        resized_img = resized_img.transpose(2, 3, 1, 0)[:, :, :, 0]

        if split_transform:
            me_ds_tensor = nnf.interpolate(
                out_img_tensor,
                size=(split_trans_h, split_trans_w),
                mode="bilinear",
                align_corners=False,
            )
            me_ds_img = me_ds_tensor.cpu().detach().numpy()
            me_ds_img = me_ds_img.transpose(2, 3, 1, 0)[:, :, :, 0]

        if sample1c_enable:
            r_arr = resized_img[:, :, 0]
            g_arr = resized_img[:, :, 1]
            b_arr = resized_img[:, :, 2]
            resized_img = np.zeros((new_h, new_w)).astype(resized_img.dtype)
            if raw_pattern == "RGGB":
                resized_img[0::2, 0::2] = r_arr[0::2, 0::2]
                resized_img[1::2, 0::2] = g_arr[1::2, 0::2]
                resized_img[0::2, 1::2] = g_arr[0::2, 1::2]
                resized_img[1::2, 1::2] = b_arr[1::2, 1::2]
            elif raw_pattern == "BGGR":
                resized_img[0::2, 0::2] = b_arr[0::2, 0::2]
                resized_img[1::2, 0::2] = g_arr[1::2, 0::2]
                resized_img[0::2, 1::2] = g_arr[0::2, 1::2]
                resized_img[1::2, 1::2] = r_arr[1::2, 1::2]
            elif raw_pattern == "GBRG":
                resized_img[0::2, 0::2] = g_arr[0::2, 0::2]
                resized_img[1::2, 0::2] = b_arr[1::2, 0::2]
                resized_img[0::2, 1::2] = r_arr[0::2, 1::2]
                resized_img[1::2, 1::2] = g_arr[1::2, 1::2]
            elif raw_pattern == "GRBG":
                resized_img[0::2, 0::2] = g_arr[0::2, 0::2]
                resized_img[1::2, 0::2] = r_arr[1::2, 0::2]
                resized_img[0::2, 1::2] = b_arr[0::2, 1::2]
                resized_img[1::2, 1::2] = r_arr[1::2, 1::2]
            resized_img = resized_img[..., np.newaxis]
        if layout == "hw":
            resized_img = resized_img[..., 0]
    else:
        resized_img = cv2.resize(
            img, (new_w, new_h), interpolation=cv2_interp_codes[interpolation]
        )
    if layout == "chw":
        # change to the original layout
        resized_img = np.ascontiguousarray(resized_img.transpose((2, 0, 1)))
    w_scale = float(new_w / img_w)
    h_scale = float(new_h / img_h)
    if return_scale:
        return resized_img, me_ds_img, w_scale, h_scale
    else:
        return resized_img


def imresize_pad_to_keep_ratio(
    img: np.ndarray,
    target_hw,
    layout,
    keep_ratio=True,
):
    """Resize image with padding to keep ratio.

    Get resize matrix for resizing raw img to input size
    Args:
        img: image in numpy array.
        target_hw: target height and width.
        layout: choices in "hwc", "chw", "hw".
        keep_ratio: default is True.
    Return:
        img: resized image
        Rs: 3x3 transformation matrix
    """
    assert layout in ["hwc", "chw", "hw"]
    if layout == "chw":
        # opencv only supports hwc layout
        img = np.ascontiguousarray(img.transpose((1, 2, 0)))  # chw > hwc

    R = AffineMat2DGenerator.resize(img.shape[:2], target_hw, keep_ratio)
    img = cv2.warpPerspective(img, R, dsize=target_hw[::-1])
    return img, R


def random_flip(
    img: Union[np.ndarray, torch.Tensor], layout, px=0, py=0, raw_pattern=None
) -> Tuple[Union[np.ndarray, torch.Tensor], Tuple[bool, bool]]:
    """Randomly flip image along horizontal and vertical with probabilities.

    Args:
        layout (str): Layout of img, `hwc` or `chw`.
        px (float): Horizontal flip probability, range between [0, 1].
        py (float): Vertical flip probability, range between [0, 1].
        raw_pattern (str): Current raw pattern.

    Returns:
        tuple: flipped image + (flip_x, flip_y) + raw pattern

    """
    assert layout in ["hwc", "chw", "hw"]
    assert isinstance(img, (torch.Tensor, np.ndarray))
    h_index = layout.index("h")
    w_index = layout.index("w")
    flip_x = np.random.choice([False, True], p=[1 - px, px])
    flip_y = np.random.choice([False, True], p=[1 - py, py])

    if isinstance(img, np.ndarray):
        if flip_x:
            img = np.flip(img, axis=w_index)
        if flip_y:
            img = np.flip(img, axis=h_index)
    else:
        if flip_x:
            img = img.flip(w_index)
        if flip_y:
            img = img.flip(h_index)

    if raw_pattern in ["RGGB", "BGGR", "GBRG", "GRBG"]:
        if flip_x:
            raw_pattern = (
                raw_pattern[1]
                + raw_pattern[0]
                + raw_pattern[3]
                + raw_pattern[2]
            )
        if flip_y:
            raw_pattern = (
                raw_pattern[2]
                + raw_pattern[3]
                + raw_pattern[0]
                + raw_pattern[1]
            )
    else:
        raw_pattern = None

    return img, (flip_x, flip_y), raw_pattern


def image_pad(
    img: Union[np.ndarray, torch.Tensor],
    layout,
    shape=None,
    divisor=1,
    pad_val=0,
) -> Union[np.ndarray, torch.Tensor]:
    """Pad image to a certain shape.

    Args:
        layout (str): Layout of img, `hwc` or `chw` or `hw`.
        shape (tuple): Expected padding shape, meaning of dimension is the
            same as img, if layout of img is `hwc`, shape must be (pad_h,
            pad_w) or (pad_h, pad_w, c).
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (Union[float, Sequence[float]]): Values to be filled in padding
            areas, single value or a list of values with len c.
            E.g. : pad_val = 10, or pad_val = [10, 20, 30].

    Returns:
        ndarray or torch.Tensor: padded image.
    """
    assert layout in ["hwc", "chw", "hw"]
    if isinstance(pad_val, Sequence):
        assert layout in ["hwc", "chw"]
        c_index = layout.index("c")
        assert len(pad_val) == img.shape[c_index]
        pad_val = torch.tensor(pad_val)
        if layout == "hwc":
            pad_val = pad_val.unsqueeze(0).unsqueeze(0)
        elif layout == "chw":
            pad_val = pad_val.unsqueeze(-1).unsqueeze(-1)
        if isinstance(img, np.ndarray):
            pad_val = pad_val.numpy()

    # calculate pad_h and pad_h
    if shape is None:
        shape = img.shape
        if divisor == 1:
            return img
    assert len(shape) in [2, 3]
    if layout == "chw":
        if len(shape) == 2:
            h = max(img.shape[1], shape[0])
            w = max(img.shape[2], shape[1])
        else:
            h = max(img.shape[1], shape[1])
            w = max(img.shape[2], shape[2])
    else:
        h = max(img.shape[0], shape[0])
        w = max(img.shape[1], shape[1])
    pad_h = int(np.ceil(h / divisor)) * divisor
    pad_w = int(np.ceil(w / divisor)) * divisor

    if len(shape) == 3:
        if layout == "hwc":
            shape = (pad_h, pad_w, shape[-1])
        elif layout == "chw":
            shape = (shape[0], pad_h, pad_w)
    else:
        shape = (pad_h, pad_w)

    if len(shape) < len(img.shape):
        if layout == "hwc":
            shape = tuple(shape) + (img.shape[-1],)
        elif layout == "chw":
            shape = (img.shape[0],) + tuple(shape)
    assert len(shape) == len(img.shape)
    for i in range(len(shape)):
        assert shape[i] >= img.shape[i], (
            "padded shape must greater than " "the src shape of img"
        )

    if isinstance(img, Tensor):
        pad = torch.zeros(shape, dtype=img.dtype, device=img.device)
    elif isinstance(img, np.ndarray):
        pad = np.zeros(shape, dtype=img.dtype)
    else:
        raise TypeError
    pad[...] = pad_val

    if len(img.shape) == 2:
        pad[: img.shape[0], : img.shape[1]] = img
    elif layout == "hwc":
        pad[: img.shape[0], : img.shape[1], ...] = img
    else:
        pad[:, : img.shape[1], : img.shape[2]] = img
    return pad


def image_normalize(img: Union[np.ndarray, Tensor], mean, std, layout):
    """Normalize the image with mean and std.

    Args:
        mean (Union[float, Sequence[float]]): Shared mean or sequence of means
            for each channel.
        std (Union[float, Sequence[float]]): Shared std or sequence of stds for
            each channel.
        layout (str): Layout of img, `hwc` or `chw`.

    Returns:
        np.ndarray or torch.Tensor: Normalized image.

    """
    assert layout in ["hwc", "chw"]
    c_index = layout.index("c")

    return_ndarray = False
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img.astype(np.float32))
        return_ndarray = True
    elif isinstance(img, Tensor):
        img = img.float()
    else:
        raise TypeError

    if isinstance(mean, Sequence):
        assert len(mean) == img.shape[c_index]
    else:
        mean = [mean] * img.shape[c_index]

    if isinstance(std, Sequence):
        assert len(std) == img.shape[c_index]
    else:
        std = [std] * img.shape[c_index]

    dtype = img.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=img.device)
    std = torch.as_tensor(std, dtype=dtype, device=img.device)
    if (std == 0).any():
        raise ValueError(
            "std evaluated to zero after conversion to {}, "
            "leading to division by zero.".format(dtype)
        )
    if c_index == 0:
        mean = mean[:, None, None]
        std = std[:, None, None]
    else:
        mean = mean[None, None, :]
        std = std[None, None, :]
    img.sub_(mean).div_(std)

    if return_ndarray:
        img = img.numpy()

    return img
