# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Callable, Optional, Union

from torch import device

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import convert_numpy, convert_tensor, to_cuda
from hat.utils.package_helper import require_packages
from .ir_module import IrModule
from .utils import onnx2torch_dtype_dict

try:
    import onnxruntime as ort
except ImportError:
    ort = None


__all__ = ["OnnxModule"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class OnnxModule(IrModule):
    """Inference module of onnx.

    Args:
         model_path: Path of ir model file.
         return_tensor: Whether to return torch tensor.
         reformat_input_func: Callable function to reformat model inputs.
         reformat_output_func: Callable function to reformat model output.
    """

    @require_packages("onnxruntime")
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
        self.sess_model = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.output_names = [
            output.name for output in self.sess_model.get_outputs()
        ]
        self.device = None

    def check_input_impl(self, data):
        format_data = {}
        for inp in self.sess_model.get_inputs():
            dtype = onnx2torch_dtype_dict[inp.type]
            self.check_type_shape(inp.name, inp.shape, dtype, data)
            format_data[inp.name] = data[inp.name]
            self.device = data[inp.name].device
        return convert_numpy(format_data)

    def check_output_impl(self, data):
        return_data = {}

        for idx, v in enumerate(self.sess_model.get_outputs()):
            return_data[v.name] = data[idx]

        if self.return_tensor:
            return_data = convert_tensor(return_data)

        if "cuda" in self.device.type:
            return_data = to_cuda(return_data)
        return return_data

    def forward_impl(self, data):
        output = self.sess_model.run(self.output_names, data)
        return output

    def cpu(self):
        providers = self.sess_model.get_providers()
        assert (
            len(providers) == 1 and "CPUExecutionProvider" in providers
        ), "Please install onnxruntime instead of onnxruntime-gpu."
        return self

    def cuda(self, device: Optional[Union[int, device]] = None):
        assert (
            "CUDAExecutionProvider" in self.sess_model.get_providers()
        ), "Please install onnxruntime-gpu."
        return self
