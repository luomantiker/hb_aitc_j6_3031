# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
from typing import Optional, Union

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from hat.registry import OBJECT_REGISTRY

__all__ = ["KeypointsViz"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class KeypointsViz(object):
    """
    The viz method of Keypoints predict result.

    Args:
        threshold: threshold of ploting keypoint prediction.
        is_plot: Whether to plot image.
    """

    def __init__(self, threshold: float = 0.0, is_plot: bool = False):
        self.threshold = threshold
        self.is_plot = is_plot

    def __call__(
        self,
        image: Union[torch.Tensor, np.ndarray],
        keypoints: torch.Tensor,
        scale: Optional[tuple] = None,
        save_path: Optional[str] = None,
    ):
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()
        img_size = image.shape
        if scale is None:
            scale_factor = 1.0
        else:
            scale_factor = max(
                img_size[0] / float(scale[0]), img_size[1] / float(scale[1])
            )

        image_with_keypoints = image.copy()

        def get_color(i):
            if i < 4:
                return (255, 0, 0)
            elif i >= 4 and i < 8:
                return (0, 255, 0)
            elif i >= 8:
                return (0, 0, 255)

        for i, (x, y, conf) in enumerate(keypoints):
            x = float(x * scale_factor)
            y = float(y * scale_factor)
            if conf > self.threshold:
                color = get_color(i)
                cv2.circle(
                    image_with_keypoints,
                    (round(x), round(y)),
                    radius=3,
                    color=color,
                    thickness=-1,
                )

        if self.is_plot:
            plt.imshow(image_with_keypoints)
            plt.margins(0, 0)
            plt.axis("off")
            plt.tight_layout()
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                result_path = os.path.join(save_path, "keypoint_pred.png")
                plt.savefig(result_path)
            else:
                plt.show()
        return image_with_keypoints
