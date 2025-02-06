# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import List

import numpy as np
import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.package_helper import require_packages
from .common import BgrToYuv444, BgrToYuv444V2

try:
    import timm
except ImportError:
    timm = None

__all__ = [
    "ConvertLayout",
    "OneHot",
    "LabelSmooth",
    "TimmTransforms",
    "TimmMixup",
    "BgrToYuv444",
    "BgrToYuv444V2",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ConvertLayout(object):
    """
    ConvertLayout is used for layout convert.

    .. note::
        Affected keys: 'img'.

    Args:
        hwc2chw (bool): Whether to convert hwc to chw.
        keys (list)：make layout convert for the data[key]
    """

    def __init__(self, hwc2chw: bool = True, keys: List = None):
        self.hwc2chw = hwc2chw
        self.keys = _as_list(keys) if keys else ["img"]

    def __call__(self, data):
        seq = (2, 0, 1) if self.hwc2chw else (1, 2, 0)

        def _permute(np_or_tensor):
            if isinstance(np_or_tensor, np.ndarray):
                return np_or_tensor.transpose
            elif isinstance(np_or_tensor, torch.Tensor):
                return np_or_tensor.permute
            else:
                raise TypeError("torch.Tensor or np.ndarray expected")

        for key in self.keys:
            assert key in data
            if isinstance(data[key], list):
                for i, image in enumerate(data[key]):
                    image = _permute(image)((seq))
                    data[key][i] = image
            else:
                image = data[key]
                image = _permute(image)((seq))
                data[key] = image
        return data


@OBJECT_REGISTRY.register
class OneHot(object):
    """
    OneHot is used for convert layer to one-hot format.

    .. note::
        Affected keys: 'labels'.

    Args:
        num_classes (int): Num classes.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, data):
        assert "labels" in data
        target = data["labels"]
        if len(target.shape) == 1:
            # for scatter
            target = torch.unsqueeze(target, dim=1)
        batch_size = target.shape[0]
        one_hot = torch.zeros(batch_size, self.num_classes).to(target.device)
        one_hot = one_hot.scatter_(1, target, 1)
        data["labels"] = one_hot
        return data


@OBJECT_REGISTRY.register
class LabelSmooth(object):
    """
    LabelSmooth is used for label smooth.

    .. note::
        Affected keys: 'labels'.

    Args:
        num_classes (int): Num classes.
        eta (float): Eta of label smooth.
    """

    def __init__(self, num_classes: int, eta: float = 0.1):
        self.num_classes = num_classes
        self.on_value = torch.tensor([1 - eta + eta / num_classes])
        self.off_value = torch.tensor([eta / num_classes])

    def __call__(self, data):
        assert "labels" in data
        target = data["labels"]
        if len(target.shape) == 1:
            # for scatter
            target = torch.unsqueeze(target, dim=1)
        batch_size = target.shape[0]
        one_hot = torch.zeros(batch_size, self.num_classes)
        one_hot = one_hot.scatter_(1, target, 1)
        target = torch.where(one_hot == 0, self.off_value, self.on_value)
        data["labels"] = target
        return data


@OBJECT_REGISTRY.register
class TimmTransforms(object):
    """
    Transforms of timm.

    .. note::
        Affected keys: 'img'.

    Args:
        args are the same as timm.data.create_transform
    """

    @require_packages("timm")
    def __init__(self, *args, **kwargs):
        self.transform = timm.data.create_transform(*args, **kwargs)

    def __call__(self, data):
        data["img"] = self.transform(data["img"])
        return data


@OBJECT_REGISTRY.register
class TimmMixup(object):
    """
    Mixup of timm.

    .. note::
        Affected keys: 'img', 'labels'.

    Args:
        args are the same as timm.data.Mixup
    """

    @require_packages("timm")
    def __init__(self, *args, **kwargs):
        self.mixup = timm.data.Mixup(*args, **kwargs)

    def __call__(self, data):
        x, target = data["img"], data["labels"]
        if len(x) % 2 == 0:
            x, target = self.mixup(x, target)
            data["img"], data["labels"] = x, target
        else:
            logger.warning(
                "Batch size should be even when using Mixup)"
                f"get batchsize={len(x)}. Skip Mixup"
            )
        return data
