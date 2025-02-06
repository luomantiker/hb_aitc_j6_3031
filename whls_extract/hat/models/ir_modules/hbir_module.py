# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Callable, Dict, Optional, Union

import torch
from torch import device

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import convert_numpy, convert_tensor, to_cuda
from hat.utils.package_helper import require_packages
from .ir_module import IrModule
from .utils import np2torch_dtype_dict

try:
    from hbdk4.compiler import load
    from horizon_plugin_pytorch.quantization.hbdk4 import (
        get_hbir_input_flattener,
        get_hbir_output_unflattener,
    )
except ImportError:
    load = None
    get_hbir_input_flattener = None
    get_hbir_output_unflattener = None


__all__ = ["HbirModule"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class HbirModule(IrModule):
    """Inference module of hbir.

    Args:
         model_path: Path of ir model file.
         return_tensor: Whether to return torch tensor.
         reformat_input_func: Callable function to reformat model inputs.
         reformat_output_func: Callable function to reformat model output.
    """

    @require_packages("horizon_plugin_pytorch>=1.10.3", "hbdk4")
    def __init__(
        self,
        model_path: str,
        return_tensor: bool = True,
        reformat_input_func: Optional[Callable] = None,
        reformat_output_func: Optional[Callable] = None,
    ):
        super().__init__(
            model_path=model_path,
            reformat_input_func=reformat_input_func,
            reformat_output_func=reformat_output_func,
        )
        self.model = load(model_path)
        self.return_tensor = return_tensor
        self.device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )

        self.input_flattener = get_hbir_input_flattener(self.model)
        self.output_unflattener = get_hbir_output_unflattener(self.model)

    def check_input_impl(self, data):
        assert isinstance(data, Dict)
        format_data = {}
        for inp in self.model[0].inputs:
            dtype = np2torch_dtype_dict[inp.type.np_dtype]
            self.check_type_shape(inp.name, inp.type.shape, dtype, data)
            format_data[inp.name] = data[inp.name]
            self.device = data[inp.name].device
        return convert_numpy(format_data)

    def check_output_impl(self, data):
        return_data = data

        if self.return_tensor:
            return_data = convert_tensor(return_data)

        if "cuda" in self.device.type:
            return_data = to_cuda(return_data)
        return return_data

    def forward_impl(self, data):
        output = self.model.functions[0](*self.input_flattener(data))
        output = self.output_unflattener(output)
        return output

    def cpu(self):
        return self

    def cuda(self, device: Optional[Union[int, device]] = None):
        logger.warnings("Hbir does not support GPU now. Use CPU instead.")
        return self
