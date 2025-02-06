# Copyright (c) Horizon Robotics. All rights reserved.
import copy
from typing import Any

from hat.registry import OBJECT_REGISTRY

__all__ = ["PassThroughDataLoader"]


@OBJECT_REGISTRY.register
class PassThroughDataLoader:
    """
    Directly pass through input example.

    Arguments:
        data: Input data
        length: Length of dataloader
        clone: Whether clone input data
    """

    def __init__(self, data: Any, *, length: int, clone: bool = False):
        assert length > 0, "required length > 0"
        self.length = length
        self.data = data
        self.clone = clone

    def __iter__(self):
        idx = 0
        while True:
            if self.clone:
                yield copy.deepcopy(self.data)
            else:
                yield self.data
            idx += 1
            if idx >= self.length:
                return

    def __len__(self):
        return self.length
