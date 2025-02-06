# Copyright (c) Horizon Robotics. All rights reserved.
import os
from typing import Callable, Union

import cv2
import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from hat.registry import OBJECT_REGISTRY
from .utils import colormap, plot_image, show_images

__all__ = ["SegViz"]

_cityscapes_colormap = torch.tensor(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0],
    ]
)


def _default_image_process(image: Tensor, prefix):
    show_images(image, prefix)


def _seg_target_process(target: Tensor, prefix):
    if target.size(1) > 1:
        one_hot = torch.argmax(target, dim=1, keepdim=True)
    else:
        one_hot = target
    one_hot = colormap(one_hot, _cityscapes_colormap)
    show_images(one_hot, prefix, "hwc")


@OBJECT_REGISTRY.register
class SegViz(object):
    """
    The visualize method of segmentation result.

    Args:
        image_process (Callable): Process of image.
        label_process (Callable): Process of label.
    """

    def __init__(
        self,
        is_plot: bool,
        image_process: Callable = _default_image_process,
        label_process: Callable = _seg_target_process,
    ):
        self.is_plot = is_plot
        self.image_process = image_process
        self.label_process = label_process

    def __call__(
        self,
        image: Union[numpy.ndarray, Tensor] = None,
        output: Tensor = None,
        save_path: str = None,
    ):
        output = torch.clip(output, 0, 19)
        output = colormap(output, _cityscapes_colormap)
        output = output.detach().cpu().numpy().astype(np.uint8)
        if image is not None:
            plot_image(image)
        plot_image(output)
        if self.is_plot:
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                result_path = os.path.join(save_path, "seg_pred.png")
                plt.savefig(result_path)
            else:
                plt.show()


def colorize(img, colors):
    """Colorize img based on colors."""
    assert img.ndim == 2
    img = np.concatenate(
        (
            img[:, :, np.newaxis],
            img[:, :, np.newaxis],
            img[:, :, np.newaxis],
        ),
        axis=2,
    )

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val = img[i][j][0] if img[i][j][0] != 255 else 0
            img[i][j] = colors[val]
    return img


def semantic_segmentation(
    _seg,
    colors,
    image=None,
    hwc=False,
    alpha=0.5,
    keep_shape_as_image=False,
):

    if torch and isinstance(_seg, torch.Tensor):
        _seg = _seg.cpu().numpy()

    seg = _seg.copy()
    assert isinstance(seg, np.ndarray)

    if seg.ndim == 2:
        seg = seg
    elif seg.ndim == 3:
        if hwc:
            seg = np.transpose(seg, (2, 0, 1))
        seg = np.argmax(seg, axis=0)
    else:
        raise ValueError("bad input ndim: %d" % seg.ndim)
    h, w = seg.shape

    mask_color = colorize(seg, colors)

    if image is not None:
        if keep_shape_as_image:
            image_h, image_w = image.shape[:2]
            if image_h != h or image_w != w:
                mask_color = cv2.resize(mask_color, (image_w, image_h))
            h, w = image_h, image_w
        if image.shape[0] != h:
            image_resize = cv2.resize(image, (w, h))
        else:
            image_resize = image
        blended = image_resize * alpha + mask_color * (1 - alpha)
    else:
        blended = mask_color
    return blended
