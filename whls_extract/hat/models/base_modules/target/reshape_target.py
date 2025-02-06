# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Mapping, Optional, Sequence

import torch

from hat.registry import OBJECT_REGISTRY

__all__ = ["ReshapeTarget"]


@OBJECT_REGISTRY.register
class ReshapeTarget(object):
    """Reshape target data in label_dict to specific shape.

    Args:
        data_name (str): name of original data to reshape.
        shape (Sequence): the new shape.
    """

    def __init__(self, data_name: str, shape: Optional[Sequence] = None):
        self.data_name = data_name
        self.shape = shape

    def _reshape(self, data, shape):
        if isinstance(data, torch.Tensor):
            return torch.reshape(data, shape)
        elif isinstance(data, Mapping):
            for k, v in data.items():
                valid_shape = shape[k] if isinstance(shape, Mapping) else shape
                data[k] = self._reshape(v, valid_shape)
            return data
        elif isinstance(data, Sequence):
            valid_shape_list = []
            for idx in range(len(data)):
                valid_shape = (
                    shape[idx] if isinstance(shape[0], Sequence) else shape
                )
                valid_shape_list.append(valid_shape)
            return [
                self._reshape(sub_data, sub_shape)
                for sub_data, sub_shape in zip(data, valid_shape_list)
            ]
        else:
            return data

    def __call__(self, label_dict: Mapping, pred_dict: Mapping) -> Mapping:
        shape = self.shape if self.shape else label_dict[self.data_name].shape
        label_dict[self.data_name] = self._reshape(
            label_dict[self.data_name], shape
        )
        return label_dict
