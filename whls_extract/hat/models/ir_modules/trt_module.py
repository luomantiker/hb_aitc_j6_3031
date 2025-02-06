# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Callable, Optional, Union

import torch
from torch import device

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages
from .ir_module import IrModule
from .utils import trt2torch_dtype_dict

try:
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    trt = None
    TRT_LOGGER = None


__all__ = ["TrtModule"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class TrtModule(IrModule):
    """Inference module of tensorrt.

    Args:
         model_path: Path of ir model file.
         return_tensor: Whether to return torch tensor.
         reformat_input_func: Callable function to reformat model inputs.
         reformat_output_func: Callable function to reformat model output.
    """

    @require_packages("tensorrt")
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
        self.return_tensor = return_tensor
        self.device = torch.cuda.current_device()

        runtime = trt.Runtime(TRT_LOGGER)
        with open(model_path, "rb") as engine_file:
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(self.device)
        self.context.set_optimization_profile_async(0, self.stream.cuda_stream)
        self._get_engine_info()
        self.return_data = self._allocate_output_buffer()

    def _get_engine_info(self):
        num_io_tensors = self.engine.num_io_tensors
        tensor_names = [
            self.engine.get_tensor_name(i) for i in range(num_io_tensors)
        ]
        num_inputs = [
            self.engine.get_tensor_mode(name) for name in tensor_names
        ].count(trt.TensorIOMode.INPUT)
        self.input_names = tensor_names[:num_inputs]
        self.output_names = tensor_names[num_inputs:]
        self.tensor_names = tensor_names

    def _allocate_output_buffer(self):
        output_data = {}
        for name in self.output_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)

            shape = torch.Size(shape)
            dtype = trt2torch_dtype_dict[dtype]

            output = torch.zeros(shape, dtype=dtype, device=self.device)
            output_data[name] = output

            # binds the output of context
            self.context.set_tensor_address(name, output.data_ptr())

        return output_data

    def check_input_impl(self, data):
        format_data = {}
        for name in self.input_names:
            dtype = trt2torch_dtype_dict[self.engine.get_tensor_dtype(name)]
            self.check_type_shape(
                name,
                self.engine.get_tensor_shape(name),
                dtype,
                data,
            )
            assert (
                "cuda" in data[name].device.type
            ), "Please use gpu tensor for input."
            format_data[name] = data[name]
        return format_data

    def check_output_impl(self, data):
        return_data = {}
        for k, v in self.return_data.items():
            return_data[k] = v.clone()

        return return_data

    def forward_impl(self, data):
        for name in self.input_names:
            self.context.set_tensor_address(name, data[name].data_ptr())

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return self.return_data

    def cpu(self):
        logger.warnings("TensorRT does not support CPU now. Use GPU instead.")
        return self

    def cuda(self, device: Optional[Union[int, device]] = None):
        return self
