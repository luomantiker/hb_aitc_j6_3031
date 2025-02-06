from typing import Optional

import onnx

from hbdk4.compiler.onnx.adaptor import OnnxGraphAdaptor
from hbdk4.compiler.frontend.registry import OpConvertorRegistry
from hbdk4.compiler._mlir_libs._hbdk import _const_fold
from hbdk4.compiler.overlay import Module
from hbdk4.compiler import ir as mlir

import hbdk4.compiler.onnx.opset9
import hbdk4.compiler.onnx.opset10
import hbdk4.compiler.onnx.opset11
import hbdk4.compiler.onnx.opset12
import hbdk4.compiler.onnx.opset13
import hbdk4.compiler.onnx.opset14
import hbdk4.compiler.onnx.opset16
import hbdk4.compiler.onnx.opset17
import hbdk4.compiler.onnx.opset18
import hbdk4.compiler.onnx.opset19
import hbdk4.compiler.onnx.horizon  # noqa: F401
from hbdk4.compiler.numba.tools import compile_custom


def export(proto: onnx.ModelProto, *, name: Optional[str] = None) -> Module:
    """export an onnx module to hbir mlir

    Args:
        proto (onnx.ModelProto): onnx protobuf
        name (Optional[str], optional): rename the onnx function. "None" means using onnx graph name

    Returns:
        Module: a helper for mlir.Module that manages hbdk operations
    """

    # check_model(proto)
    module = mlir.Module.create()
    with mlir.InsertionPoint(module.body):
        graph = OnnxGraphAdaptor(
            proto.graph, opset=proto.opset_import[0].version, name=name
        )
        graph.emit_mlir_func_op(OpConvertorRegistry(), True)

    module = compile_custom(module).module
    _const_fold(module.operation, module.context)
    return Module(module)


def statistics(proto: onnx.ModelProto):
    """Print op statics of given onnx module.

    Args:
        proto (onnx.ModelProto): onnx protobuf
    """

    graph = OnnxGraphAdaptor(proto.graph, opset=proto.opset_import[0].version)

    print("ops encountered in onnx graph @", graph.name, "opset version", graph.opset)
    graph.statistics()
