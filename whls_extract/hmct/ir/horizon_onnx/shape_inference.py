"""onnx shape inference.

Shape inference is not guaranteed to be complete.
"""

from onnx import ModelProto

from .onnx_cpp2py_export import shape_inference
from .serialize_deserialize import deserialize_proto, serialize_proto


def infer_shapes(
    onnx_proto: ModelProto,
    check_type: bool = False,
    strict_mode: bool = False,
    data_prop: bool = False,
) -> ModelProto:
    """Apply shape inference to the provided ModelProto.

    Inferred shapes are added to the value_info field of the graph.

    If the inferred values conflict with values already provided in the
    graph, that means that the provided values are invalid (or there is a
    bug in shape inference), and the result is unspecified.

    Args:
        onnx_proto: onnx model to infer shape information.
        check_type: Checks the type-equality for input and output.
        strict_mode: Stricter shape inference, it will throw errors if any;
            Otherwise, simply stop if any error.
        data_prop: Enables data propagation for limited operators to
            perform shape computation.

    Returns:
        Onnx model with inferred shape information.
    """
    if not isinstance(onnx_proto, ModelProto):
        raise ValueError(
            f"Shape inference only accepts ModelProto, "
            f"incorrect type: {type(onnx_proto)}"
        )

    model_bytes = serialize_proto(onnx_proto)
    inferred_model_str = shape_inference.infer_shapes(
        model_bytes, check_type, strict_mode, data_prop
    )
    return deserialize_proto(inferred_model_str)
