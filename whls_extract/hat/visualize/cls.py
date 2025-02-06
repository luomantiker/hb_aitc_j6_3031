# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
from typing import Union

import numpy
import torch
from matplotlib import pyplot as plt

from hat.registry import OBJECT_REGISTRY

__all__ = ["ClsViz"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ClsViz(object):
    """
    The viz method of classification result.

    Args:
        is_plot (bool): Whether to plot image.
    """

    def __init__(self, is_plot: bool = False):
        self.is_plot = is_plot

    def __call__(
        self,
        image: Union[torch.Tensor, numpy.ndarray],
        preds: torch.Tensor,
        save_path: str = None,
    ):
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()

        if preds.shape[-1] > 1:
            preds = torch.argmax(preds, -1)
        preds = preds.cpu().item()

        if self.is_plot:
            plt.imshow(image)
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                result_path = os.path.join(save_path, "cls_pred.png")
                plt.savefig(result_path)
            else:
                plt.show()
        return preds
