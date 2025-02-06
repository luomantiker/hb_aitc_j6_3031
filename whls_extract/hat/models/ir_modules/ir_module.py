# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch import device

from hat.registry import OBJECT_REGISTRY

__all__ = ["IrModule"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class IrModule(nn.Module):
    """Basic class of ir module.

    Args:
         model_path: Path of ir model file.
         reformat_input_func: Callable function to reformat model inputs.
         reformat_output_func: Callable function to reformat model output.
    """

    def __init__(
        self,
        model_path: str,
        reformat_input_func: Optional[Callable] = None,
        reformat_output_func: Optional[Callable] = None,
    ):
        super().__init__()
        self.model_path = model_path
        assert os.path.exists(self.model_path), "model_path not exists."
        logger.info(f"Load ir module from {self.model_path}!")

        self.reformat_input_func = reformat_input_func
        self.reformat_output_func = reformat_output_func

    @torch.no_grad()
    def forward(self, data):
        if self.reformat_input_func is not None:
            data = self.reformat_input_func(data)

        data = self.check_input_impl(data)
        result = self.forward_impl(data)
        result = self.check_output_impl(result)

        if self.reformat_output_func is not None:
            result = self.reformat_output_func(result)
        return result

    def forward_impl(self, data):
        pass

    def check_input_impl(self, data):
        return data

    def check_output_impl(self, data):
        return data

    def check_type_shape(self, name_info, shape_info, dtype_info, data):
        if name_info in data:
            if not torch.Size(shape_info) == data[name_info].shape:
                raise ValueError(
                    f"Shape of {name_info} is not matched, \
                    expect {data[name_info].shape} but get \
                    {shape_info}."
                )

            if not str(dtype_info) in str(data[name_info].dtype):
                raise TypeError(
                    f"Dtype of {name_info} is not matched, \
                    excepted {data[name_info].dtype} but get \
                    {dtype_info}."
                )
        else:
            raise KeyError(f"Cannot find {name_info} in {data}.")

    def cpu(self):
        pass

    def cuda(self, device: Optional[Union[int, device]] = None):
        pass

    def __call__(self, data):
        return self.forward(data)
