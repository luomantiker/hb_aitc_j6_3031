# Copyright (c) Horizon Robotics. All rights reserved.

import os
from typing import List, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt

from hat.registry import OBJECT_REGISTRY
from hat.visualize.utils import plot_image

__all__ = ["LanelineViz"]


@OBJECT_REGISTRY.register
class LanelineViz(object):
    """
    The visualize method of Laneline detection result.

    Args:
        is_plot: Whether to plot image.
        thickness: The thickness of lane line.
        color: The color of lane line.
    """

    def __init__(
        self,
        is_plot: bool = True,
        thickness: int = 10,
        color: Tuple[int, int, int] = (255, 0, 0),
    ):
        self.is_plot = is_plot
        self.thickness = thickness
        self.color = color

    def __call__(
        self, image: np.ndarray, lines: List[np.ndarray], save_path: str = None
    ):
        for line in lines:
            line = np.around(line).astype(np.int64)
            points_num = len(line)
            for i in range(points_num - 1):
                x1 = line[i][0]
                y1 = line[i][1]
                x2 = line[i + 1][0]
                y2 = line[i + 1][1]
                cv2.line(
                    image,
                    (x1, y1),
                    (x2, y2),
                    color=self.color,
                    thickness=self.thickness,
                )

        plot_image(image)

        if self.is_plot:
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                result_path = os.path.join(save_path, "lane_pred.png")
                plt.savefig(result_path)
            else:
                plt.show()
