"""onnx checker.

This implements graph utilities that allows us to check whether a serialized
proto is legal.
"""

from typing import Any, Callable, TypeVar, Union

from onnx import (
    AttributeProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    ValueInfoProto,
)

from .onnx_cpp2py_export import checker, defs
from .serialize_deserialize import serialize_proto

# TODO: This thing where we reserialize the protobuf back into the
# string, only to deserialize it at the call site, is really goofy.
# Stop doing that.

ONNX_DOMAIN = ""


def onnx_opset_version():  # type: () -> int
    return defs.schema_version_map()[ONNX_DOMAIN][1]


# NB: Please don't edit this context!
DEFAULT_CONTEXT = checker.CheckerContext()
DEFAULT_CONTEXT.ir_version = 9
# TODO: Maybe ONNX-ML should also be defaulted?
DEFAULT_CONTEXT.opset_imports = {"": onnx_opset_version()}

FuncType = TypeVar("FuncType", bound=Callable[..., Any])


def _ensure_proto_type(proto, proto_type) -> None:
    if not isinstance(proto, proto_type):
        raise TypeError(
            f"The proto message needs to be of type '{proto_type.__name__}'"
        )


def check_value_info(value_info: ValueInfoProto, ctx=DEFAULT_CONTEXT) -> None:
    _ensure_proto_type(value_info, ValueInfoProto)
    return checker.check_value_info(value_info.SerializeToString(), ctx)


def check_tensor(tensor: TensorProto, ctx=DEFAULT_CONTEXT) -> None:
    _ensure_proto_type(tensor, TensorProto)
    return checker.check_tensor(tensor.SerializeToString(), ctx)


def check_attribute(attr: AttributeProto, ctx=DEFAULT_CONTEXT) -> None:
    _ensure_proto_type(attr, AttributeProto)
    return checker.check_attribute(attr.SerializeToString(), ctx)


def check_node(node: NodeProto, ctx=DEFAULT_CONTEXT) -> None:
    _ensure_proto_type(node, NodeProto)
    return checker.check_node(node.SerializeToString(), ctx)


def check_graph(graph: GraphProto, ctx=DEFAULT_CONTEXT) -> None:
    _ensure_proto_type(graph, GraphProto)
    return checker.check_graph(graph.SerializeToString(), ctx)


def check_model(model: Union[ModelProto, str]) -> None:
    if isinstance(model, str):
        checker.check_model_path(model)
    else:
        _ensure_proto_type(model, ModelProto)
        model_bytes = serialize_proto(model)
        checker.check_model(model_bytes)
