from hbdk4.compiler.frontend.adaptor import NodeAdaptor
from hbdk4.compiler.frontend.convertor import OpConvertor
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.ops import hbir


class Opset12(OpConvertor):
    def __init__(self, name: str):
        super().__init__(name, "onnx", 12, True)


class GreaterOrEqual(Opset12):
    def __init__(self):
        super().__init__("GreaterOrEqual")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, rhs: mlir.Value
    ):
        return hbir.greater_equal(lhs, rhs, output_type=y)


GreaterOrEqual()


class LessOrEqual(Opset12):
    def __init__(self):
        super().__init__("LessOrEqual")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, rhs: mlir.Value
    ):
        return hbir.less_equal(lhs, rhs, output_type=y)


LessOrEqual()


class ArgMax(Opset12):
    def __init__(self):
        super().__init__("ArgMax")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        axis=0,
        keepdims=1,
        select_last_index=0,
    ):
        assert (
            select_last_index == 0
        ), "argmax exported via torch should have select_last_index set to false"
        return hbir.reduce_argmax(
            x,
            dims=[axis],
            keepDim=bool(keepdims),
            output_type=y,
        )


ArgMax()


class ArgMin(Opset12):
    def __init__(self):
        super().__init__("ArgMin")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        axis=0,
        keepdims=1,
        select_last_index=0,
    ):
        assert (
            select_last_index == 0
        ), "argmin exported via torch should have select_last_index set to false"
        return hbir.reduce_argmin(
            x,
            dims=[axis],
            keepDim=bool(keepdims),
            output_type=y,
        )


ArgMin()


class GatherND(Opset12):
    def __init__(self):
        super().__init__("GatherND")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        indices: mlir.Value,
        *,
        batchDims=0,
    ):
        return hbir.gather_nd(data, indices, batchDim=batchDims, output_type=y)


GatherND()
