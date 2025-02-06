# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
from typing import Union

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from hat.registry import OBJECT_REGISTRY
from .utils import plot_image

__all__ = ["TrackViz"]

logger = logging.getLogger(__name__)


ID_COLORS = [
    (144, 238, 144),
    (178, 34, 34),
    (221, 160, 221),
    (0, 255, 0),
    (0, 128, 0),
    (210, 105, 30),
    (220, 20, 60),
    (192, 192, 192),
    (255, 228, 196),
    (50, 205, 50),
    (139, 0, 139),
    (100, 149, 237),
    (138, 43, 226),
    (238, 130, 238),
    (255, 0, 255),
    (0, 100, 0),
    (127, 255, 0),
    (255, 0, 255),
    (0, 0, 205),
    (255, 140, 0),
    (255, 239, 213),
    (199, 21, 133),
    (124, 252, 0),
    (147, 112, 219),
    (106, 90, 205),
    (176, 196, 222),
    (65, 105, 225),
    (173, 255, 47),
    (255, 20, 147),
    (219, 112, 147),
    (186, 85, 211),
    (199, 21, 133),
    (148, 0, 211),
    (255, 99, 71),
    (144, 238, 144),
    (255, 255, 0),
    (230, 230, 250),
    (0, 0, 255),
    (128, 128, 0),
    (189, 183, 107),
    (255, 255, 224),
    (128, 128, 128),
    (105, 105, 105),
    (64, 224, 208),
    (205, 133, 63),
    (0, 128, 128),
    (72, 209, 204),
    (139, 69, 19),
    (255, 245, 238),
    (250, 240, 230),
    (152, 251, 152),
    (0, 255, 255),
    (135, 206, 235),
    (0, 191, 255),
    (176, 224, 230),
    (0, 250, 154),
    (245, 255, 250),
    (240, 230, 140),
    (245, 222, 179),
    (0, 139, 139),
    (143, 188, 143),
    (255, 0, 0),
    (240, 128, 128),
    (102, 205, 170),
    (60, 179, 113),
    (46, 139, 87),
    (165, 42, 42),
    (178, 34, 34),
    (175, 238, 238),
    (255, 248, 220),
    (218, 165, 32),
    (255, 250, 240),
    (253, 245, 230),
    (244, 164, 96),
    (210, 105, 30),
]


@OBJECT_REGISTRY.register
class TrackViz(object):
    """
    The visiualize method of object track result.

    Args:
        is_plot (bool): Whether to plot image.
    """

    def __init__(
        self,
        is_plot: bool = True,
    ):
        self.is_plot = is_plot
        self.sample_idx = 0

    def __call__(
        self,
        image: Union[torch.Tensor, np.ndarray],
        preds: Union[torch.Tensor, np.ndarray],
        save_path: str = None,
    ):
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()

        for _, pred in preds.items():

            pred_boxes = pred["pred_boxes"].cpu().numpy()
            obj_idxes = pred["obj_idxes"].cpu().numpy()

            for i in range(len(pred_boxes)):
                boxx = pred_boxes[i].astype(np.int32)
                idxss = obj_idxes[i]
                coll = ID_COLORS[idxss % len(ID_COLORS)]
                cv2.rectangle(
                    image, (boxx[0], boxx[1]), (boxx[2], boxx[3]), coll, 2
                )
                t_size = cv2.getTextSize(
                    "{}".format(idxss), 0, fontScale=2 / 3, thickness=1
                )[0]
                c2 = boxx[0] + t_size[0], boxx[1] - t_size[1] - 3
                cv2.rectangle(image, (boxx[0], boxx[1]), c2, coll, -1)
                cv2.putText(
                    image,
                    "{}".format(idxss),
                    (boxx[0], boxx[1] - 2),
                    0,
                    2 / 3,
                    [225, 255, 255],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            if self.is_plot:
                plot_image(image, reverse_rgb=False)
                if save_path is not None:
                    os.makedirs(save_path, exist_ok=True)
                    result_path = os.path.join(
                        save_path, f"track_pred_{self.sample_idx}.png"
                    )
                    plt.savefig(result_path)
                else:
                    plt.show()
            self.sample_idx += 1
