from typing import TYPE_CHECKING, Union

from hmct.ir import OnnxModel, load_model

from ..registry import register_parse_func

if TYPE_CHECKING:
    from onnx import ModelProto


@register_parse_func(model_type="onnx")
def parse_onnx(
    onnx_model_or_proto: Union[
        bytes,
        str,
        "ModelProto",
        OnnxModel,
    ],
) -> OnnxModel:
    return load_model(onnx_model_or_proto)
