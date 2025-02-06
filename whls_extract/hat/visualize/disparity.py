# Copyright (c) Horizon Robotics. All rights reserved.

import os
from typing import Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt

from hat.registry import OBJECT_REGISTRY
from hat.visualize.utils import plot_image

__all__ = ["DispViz"]


@OBJECT_REGISTRY.register
class DispViz(object):
    """
    The visualize method of disparity or depth pred result.

    Args:
        is_plot: Whether to plot image.
    """

    def __init__(
        self,
        is_plot: bool = True,
    ):
        self.is_plot = is_plot

    def __call__(
        self,
        image: np.ndarray,
        disparity: np.ndarray,
        depth: Optional[np.ndarray] = None,
        save_path: str = None,
    ):
        disparity_color = cv2.applyColorMap(
            cv2.convertScaleAbs(disparity, alpha=11),
            cv2.COLORMAP_JET,
        )
        if depth is not None:
            num_pic = 4
            depth_color = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=11),
                cv2.COLORMAP_JET,
            )
        else:
            num_pic = 3

        fig = plt.figure()

        ax = fig.add_subplot(1, num_pic, 1)
        ax.set_title("left")
        ax = plot_image(image[..., :3], ax=ax)

        ax = fig.add_subplot(1, num_pic, 2)
        ax.set_title("right")
        ax = plot_image(image[..., 3:], ax=ax)
        ax = fig.add_subplot(1, num_pic, 3)
        ax.set_title("disparity")
        ax = plot_image(disparity_color, ax=ax)

        if depth is not None:
            ax = fig.add_subplot(1, num_pic, 4)
            ax.set_title("depth")
            ax = plot_image(depth_color, ax=ax)

        if self.is_plot:
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                result_path = os.path.join(save_path, "disp_pred.png")
                plt.savefig(result_path)
            else:
                plt.show()
