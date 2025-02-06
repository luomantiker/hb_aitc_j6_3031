__all__ = [
    "build_model",
    "export_onnx",
    "infer_shapes",
    "version",
    "ORTExecutor",
    "load_model",
    "check_model"
]

from typing import Dict, Optional, Sequence, Union

from hmct.builder import build_model, check_model
from hmct.converter.parser.torch_parser import export_onnx
from hmct.executor import ORTExecutor
from hmct.ir.onnx_utils import ModelProto, load_model
from hmct.version import __version__ as version


def infer_shapes(
    onnx_model: Union[bytes, str, "ModelProto"],
    referenced_model: Optional[Union[bytes, str, "ModelProto"]] = None,
    input_shape: Optional[Dict[str, Sequence[int]]] = None,
) -> "ModelProto":
    """修改onnx_model并完成shape_inference, 以保证其在给定input_shape输入下正常计算.

    Args:
        onnx_model: 待修改和shape_inference的onnx模型
        referenced_model: 原始浮点onnx模型, 用于辅助完成onnx_model的修改
        input_shape: 期望的onnx模型输入shape

    Returns:
        给定input_shape下完成修改和shape_inference的onnx模型
    """
    from hmct.common import infer_shapes

    return infer_shapes(
        onnx_model=onnx_model,
        referenced_model=referenced_model,
        input_shape=input_shape,
    ).proto
