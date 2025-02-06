from typing import Dict, Optional

import torch

from hat.utils.package_helper import require_packages

try:
    import torch_tensorrt
except ImportError:
    torch_tensorrt = None


@require_packages("torch_tensorrt")
def compile_tensorrt(
    model: torch.nn.Module,
    input_dict: Dict,
    compile_dict: Dict,
    device_id: Optional[int] = None,
) -> torch.jit.ScriptModule:
    """Compile tensorrt torchscript module from torch module.

    Args:
        model: Input torch module.
        input_dict: Dict of input.
            refer to https://pytorch.org/TensorRT/py_api/ts.html
        compile_dict: Dict for compiling.
            refer to https://pytorch.org/TensorRT/py_api/ts.html
        device_id: Target device of tensorrt.
    """

    fake_data = torch.randn(input_dict["shape"])
    model.eval()

    if device_id is not None:
        fake_data = fake_data.cuda(device_id)
        model.cuda(device_id)

    traced_model = torch.jit.trace(model, [fake_data])
    inputs = [torch_tensorrt.Input(**input_dict)]
    trt_ts_module = torch_tensorrt.ts.compile(
        traced_model, inputs=inputs, **compile_dict
    )
    return trt_ts_module
