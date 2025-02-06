import copy
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Union

from hmct.ir import load_model

from .shape_modifier import modify_model_shape

if TYPE_CHECKING:
    from hmct.ir import OnnxModel
    from hmct.ir.onnx_utils import ModelProto


def infer_shapes(
    onnx_model: Union[bytes, str, "ModelProto", "OnnxModel"],
    referenced_model: Union[
        bytes,
        str,
        "ModelProto",
        "OnnxModel",
    ] = None,
    input_shape: Optional[Dict[str, Sequence[int]]] = None,
) -> "OnnxModel":
    """修改onnx_model并完成shape_inference, 以保证其在给定input_shape输入下正常计算.

    Args:
        onnx_model: 待修改和shape_inference的onnx模型
        referenced_model: 原始浮点onnx模型, 用于辅助完成onnx_model的修改
        input_shape: 期望的onnx模型输入shape

    Returns:
        给定input_shape下完成修改和shape_inference的onnx模型
    """
    onnx_model = copy.deepcopy(load_model(onnx_model))
    if referenced_model is None and input_shape is None:
        onnx_model.infer_shapes()
    else:
        if referenced_model is None or input_shape is None:
            raise ValueError(
                "referenced_model and input_shape should be given simultaneously."
            )
        if all(
            list(input_shape[name]) == list(onnx_model.graph.input_shapes[name])
            for name in input_shape
        ):
            onnx_model.infer_shapes()
        else:
            referenced_model = copy.deepcopy(load_model(referenced_model))
            onnx_model = modify_model_shape(input_shape, onnx_model, referenced_model)

    return onnx_model


def create_batch_input_shape(
    input_shape: Mapping[str, Sequence[Union[None, str, int]]],
    input_batch: Mapping[str, int],
) -> Dict[str, List[int]]:
    """将用户指定的batch_size修改到input_shape的第0维."""
    batch_input_shape = {}
    for name, shape in input_shape.items():
        if name in input_batch:
            batch_input_shape[name] = [input_batch[name]] + list(shape)[1:]
        else:
            batch_input_shape[name] = list(shape)
    return batch_input_shape
