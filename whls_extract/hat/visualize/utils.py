# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import random
from typing import List, Sequence, Union

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor

from hat.registry import OBJECT_REGISTRY


def colormap(
    gray_image: Union[Tensor, np.ndarray, Image.Image],
    colormap: Union[Tensor, int],
    scale=1,
):
    """Append colormap on gray image.

    Args:
        gray_image: Input image for appending color map.
        colormap: Color map for image.
        scale: Scale of image.
    Returns:
        image: Image after colormap.
    """

    if isinstance(gray_image, Image.Image):
        gray_image = copy.deepcopy(gray_image)
        gray_image.flags.writeable = True
        gray_image = np.asarray(gray_image)
    if isinstance(gray_image, Tensor):
        assert isinstance(colormap, Tensor)
        return colormap.to(device=gray_image.device)[
            (gray_image * scale).to(dtype=torch.int64)
        ].squeeze()
    elif isinstance(gray_image, np.ndarray):
        assert isinstance(colormap, int)
        return cv2.applyColorMap(gray_image * scale, colormap)
    else:
        raise ValueError("Unsupported input type: %s" + str(type(gray_image)))


def show_images(images: Tensor, prefix: str, layout="chw", reverse_rgb=False):
    """
    Show the image from Tensor.

    Args:
        images: Images for showing.
        prefix: Prefix for showing window.
        layout: Layout of images.
        reverse_rgb: Whether to reverse channel of rgb.
    """

    if images.ndim == 4:
        if "chw" in layout:
            images = images.permute(0, 2, 3, 1)
        if reverse_rgb:
            images[:, :, :, (0, 1, 2)] = images[:, :, :, (2, 1, 0)]
        images = images.split(1)
        for i, image in enumerate(images):
            cv2.imshow(
                prefix + "_%d" % i,
                image.squeeze(0).detach().cpu().numpy().astype(np.uint8),
            )
    else:
        if "chw" in layout:
            images = images.permute(1, 2, 0)
        if reverse_rgb:
            images[:, :, (0, 1, 2)] = images[:, :, (2, 1, 0)]
        cv2.imshow(prefix, images.detach().cpu().numpy().astype(np.uint8))


def constructed_show(data, prefix, process):
    """
    Show constructed images.

    Args:
        data: Constructed images.
        prefix: Prefix for showing window.
        process: Process of images before showing.
    """

    if isinstance(data, dict):
        for k, v in data.items():
            constructed_show(v, prefix + "_" + str(k), process)
    elif isinstance(data, Sequence):
        for i, v in enumerate(data):
            constructed_show(v, prefix + "_" + str(i), process)
    elif isinstance(data, Tensor):
        process(data, prefix)
    else:
        raise TypeError("Visualization only accept dict/Sequence of Tensors")


def plot_image(img: np.array, ax=None, reverse_rgb=False):
    """Visualize image.

    Args:
        img: Image with shape `H, W, 3`.
        ax: You can reuse previous axes if provided.
        reverse_rgb: Reverse RGB<->BGR orders if `True`.
    Returns:
        The ploted axes.

    Examples:
        from matplotlib import pyplot as plt
        ax = plot_image(img)
        plt.show()
    """

    assert isinstance(img, np.ndarray)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    img = img.copy()
    if reverse_rgb:
        img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
    ax.imshow(img.astype(np.uint8))
    return ax


def plot_bbox(
    img: np.array,
    bboxes: np.array,
    scores: np.array = None,
    labels: np.array = None,
    thresh: float = 0.5,
    class_names: List[str] = None,
    colors=None,
    ax=None,
    reverse_rgb=False,
    absolute_coordinates=True,
):
    """Visualize bounding boxes.

    Args:
        img: Image with shape `H, W, 3`.
        bboxes: Bounding boxes with shape `N, 4`.
            Where `N` is the number of boxes.
        scores: Confidence scores of the provided
            `bboxes` with shape `N`.
        labels: Class labels of the provided `bboxes` with shape `N`.
        thresh: Display threshold if `scores` is provided.
            Scores with less than `thresh` will be ignored
            in display, this is visually more elegant if you
            have a large number of bounding boxes with very small scores.
        class_names: Description of parameter `class_names`.
        colors: You can provide desired colors as
            {0: (255, 0, 0), 1:(0, 255, 0), ...},
            otherwise random colors will be substituted.
        ax: You can reuse previous axes if provided.
        reverse_rgb: Reverse RGB<->BGR orders if `True`.
        absolute_coordinates: If `True`, absolute coordinates
            will be considered, otherwise coordinates are
            interpreted as in range(0, 1).

    Returns:
        The ploted axes.
    """

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError(
            "The length of labels and bboxes mismatch, {} vs {}".format(
                len(labels), len(bboxes)
            )
        )
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError(
            "The length of scores and bboxes mismatch, {} vs {}".format(
                len(scores), len(bboxes)
            )
        )

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if len(bboxes) < 1:
        return ax

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= height
        bboxes[:, (1, 3)] *= width

    # use random colors if None is provided
    if colors is None:
        colors = {}
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap("hsv")(cls_id / len(class_names))
            else:
                colors[cls_id] = (
                    random.random(),
                    random.random(),
                    random.random(),
                )
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=colors[cls_id],
            linewidth=3.5,
        )
        ax.add_patch(rect)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ""
        score = "{:.3f}".format(scores.flat[i]) if scores is not None else ""
        if class_name or score:
            ax.text(
                xmin,
                ymin - 2,
                "{:s} {:s}".format(class_name, score),
                bbox={"facecolor": colors[cls_id], "alpha": 0.5},
                fontsize=12,
                color="white",
            )
    return ax


@OBJECT_REGISTRY.register
class ANCBEVImgStitcher(object):
    """Restore multi view pictures for bev tasks.

    Args:
        camera_view_names: the camera view names of the model.
        camera_layouts: the customized layout of cameras for visualization.
            Support to visualize part of camera views. "tmp_img" denotes
            placeholder img with all zeros. Further,
            we support view expansion, if the name of the view in a
            line or a square area is the same.
        per_extra_img_size: Multiview img sizes to visualize,(w, h).
        is_bev_horizon: indicates whether the BEV image occupies the bottom
            row or the rightmost column.
        interpolation: the interpolation mode of resizing bev_img.
    """

    def __init__(
        self,
        camera_view_names: list,
        camera_layouts: list = None,
        per_extra_img_size: tuple = (960, 512),
        is_bev_horizon: bool = False,
        interpolation: int = cv2.INTER_LINEAR,
        **kwargs,
    ):
        self.camera_view_names = camera_view_names
        self.per_extra_img_size = per_extra_img_size
        self.interpolation = interpolation
        self.is_bev_horizon = is_bev_horizon
        if camera_layouts is None:
            self.camera_layouts = [
                ["camera_front_left", "camera_front", "camera_front_right"],
                ["fisheye_front", "camera_front_30fov", "fisheye_rear"],
                ["fisheye_left", "tmp_img", "fisheye_right"],
                ["camera_rear_left", "camera_rear", "camera_rear_right"],
            ]
        else:
            self.camera_layouts = camera_layouts

        self.row2idxes = self._get_row2idxes()

    def _get_row2idxes(self):
        row2idxes = []
        for row2cams in self.camera_layouts:
            row2idx = []
            for cam in row2cams:
                idx = -1
                if cam in self.camera_view_names:
                    idx = self.camera_view_names.index(cam)
                row2idx.append(idx)
            if sum(row2idx) > -3:
                row2idxes.append(row2idx)
        return row2idxes

    def _get_img_size(self, multiview_imgs, row2idxes: list):
        def _trace_one(w, h, idx):
            w_factor = 1
            h_factor = 1
            while ((w + w_factor) < len(row2idxes[h])) and (
                row2idxes[h][w + w_factor] == idx
            ):
                w_factor += 1
            while (h + h_factor) < len(row2idxes) and (
                row2idxes[h + h_factor][w] == idx
            ):
                h_factor += 1
            assert (
                h_factor == 1 and w_factor >= 1 or w_factor == h_factor
            ), "same view name area must be a line or a square."
            for x in range(w_factor):
                for y in range(h_factor):
                    assert (
                        row2idxes[h + y][w + x] == idx
                    ), "invalid layouts, area must have the same number."
            return w_factor, h_factor

        width = self.per_extra_img_size[0]
        min_h = self.per_extra_img_size[1]

        sum_h = 0
        max_w_factor = max([len(row) for row in row2idxes])
        idx2img_info = {}
        for y, row2idx in enumerate(row2idxes):
            row_img_h = min_h
            for x, idx in enumerate(row2idx):
                if idx >= 0 and idx not in idx2img_info:
                    w_factor, h_factor = _trace_one(x, y, idx)
                    view_name = self.camera_view_names[idx]
                    img = multiview_imgs[idx]
                    if "fisheye" in view_name:
                        h, w, _ = img.shape
                        img_h = int(h / w * width)
                    else:
                        img_h = min_h
                    if w_factor == max_w_factor:
                        row_img_h = img_h * w_factor
                    row_img_h = max(row_img_h, img_h)

                    img_info = {
                        "size": (w_factor * width, w_factor * img_h),
                        "origin": (sum_h, x * width),
                    }
                    idx2img_info[idx] = img_info
            sum_h += row_img_h
        return idx2img_info, sum_h, max_w_factor * width

    def __call__(self, multiview_imgs: list, bev_img: np.array):
        idx2img_info, img_h, img_w = self._get_img_size(
            multiview_imgs, self.row2idxes
        )
        stacked_img = np.zeros((img_h, img_w, 3))
        for idx, img_info in idx2img_info.items():
            img = multiview_imgs[idx]
            img = cv2.resize(
                img,
                img_info["size"],
                interpolation=self.interpolation,
            )
            w, h = img_info["size"]
            origin_h, origin_w = img_info["origin"]
            stacked_img[origin_h : origin_h + h, origin_w : origin_w + w] = img

        if self.is_bev_horizon:
            target_bev_size = (
                img_w,
                (img_w * bev_img.shape[0]) // bev_img.shape[1],
            )
        else:
            target_bev_size = (
                (img_h * bev_img.shape[1]) // bev_img.shape[0],
                img_h,
            )
        bev_img = cv2.resize(
            bev_img,
            target_bev_size,
        )
        if self.is_bev_horizon:
            stacked_img = np.vstack([stacked_img, bev_img])
        else:
            stacked_img = np.hstack([stacked_img, bev_img])
        return stacked_img


def get_flexiable_thickness(image_shape):
    """Get flexiable thickness for drawing bbox."""
    if max(image_shape) > 3000:
        thickness = 5
    elif max(image_shape) > 2000:
        thickness = 4
    elif max(image_shape) > 1000:
        thickness = 3
    elif max(image_shape) > 500:
        thickness = 2
    else:
        thickness = 1
    return thickness
