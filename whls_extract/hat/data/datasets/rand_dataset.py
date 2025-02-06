# Copyright (c) Horizon Robotics. All rights reserved.

import copy
from typing import Any

import numpy as np
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY

__all__ = ["RandDataset", "SimpleDataset"]


@OBJECT_REGISTRY.register
class RandDataset(data.Dataset):
    def __init__(
        self, length: int, example: Any, clone: bool = True, flag: int = 1
    ):
        self.length = length
        self.example = example
        self.clone = clone
        self.flag = flag * np.ones(len(self), dtype=np.uint8)
        self.epoch = 0

    def __getitem__(self, index):
        if self.clone:
            return copy.deepcopy(self.example)
        else:
            return self.example

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch


@OBJECT_REGISTRY.register
class SimpleDataset(data.Dataset):
    def __init__(self, start: int, length: int, flag: int = 1):
        self.start = start
        self.length = length
        self.flag = flag * np.ones(len(self), dtype=np.uint8)

    def __getitem__(self, index):
        return index + self.start

    def __len__(self):
        return self.length
