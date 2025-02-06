from typing import Callable
from hbdk4.compiler.frontend.adaptor import NodeAdaptor
from hbdk4.compiler.frontend.convertor import OpConvertor
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.ops import hbir


class Opset16(OpConvertor):
    def __init__(self, name: str):
        super().__init__(name, "onnx", 16, True)


class ScatterElements(Opset16):
    def __init__(self):
        super().__init__("ScatterElements")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        indices: mlir.Value,
        updates: mlir.Value,
        *,
        axis=0,
        reduction="none",
    ):
        return hbir.scatter_elements(
            data,
            indices,
            updates,
            axis=axis,
            scatterReduceMode=reduction,
            output_type=y,
        )


ScatterElements()


class GridSample(Opset16):
    def __init__(self):
        super().__init__("GridSample")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        Grid: mlir.Value,
        *,
        align_corners=0,
        mode="bilinear",
        padding_mode="zeros",
    ):
        if isinstance(mode, bytes):
            mode = mode.decode()
        # Currently only supports nearest and bilinear
        if mode not in ["nearest", "bilinear"]:
            raise ValueError(
                f"Operator GridSample does not support resize mode: {mode}"
            )
        padding_mode = (
            padding_mode if type(padding_mode) == str else padding_mode.decode()
        )

        x = hbir.transpose(x, [0, 2, 3, 1])
        if padding_mode == "zeros":
            x = hbir.grid_sample(
                x, Grid, mode=mode, alignCorner=bool(align_corners), padValue=0
            )
        elif padding_mode == "border":
            x = hbir.grid_sample(
                x,
                Grid,
                mode=mode,
                alignCorner=bool(align_corners),
                expansionMode=padding_mode,
            )
        else:
            raise ValueError(
                f"Operator GridSample does not support padding_mode {padding_mode}"
            )
        return hbir.transpose(x, [0, 3, 1, 2], output_type=y)


GridSample()
