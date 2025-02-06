from hbdk4.compiler.frontend.convertor import OpConvertor
from hbdk4.compiler.frontend.adaptor import NodeAdaptor
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.ops import hbir
import numpy as np
from typing import Any, Iterable, List, Optional, Union  # noqa: F401
from hbdk4.compiler.ops.common import get_value_or_create_const
from hbdk4.compiler.utils.cext import has_dynamic_dim


class Opset10(OpConvertor):
    def __init__(self, name: str):
        super().__init__(name, "onnx", 10, True)


class Resize(Opset10):
    def __init__(self):
        super().__init__("Resize")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        scales: mlir.Value = None,
        *,
        mode="nearest",
    ):
        input_shape = np.array(adaptor.operands[0].type.shape)
        output_shape = np.array(adaptor.results[0].type.shape)
        # Only supports resizing operations with 4-dimensional features
        assert len(input_shape) == 4
        scale_tensor = adaptor.operands[1].value
        resize_scale = scale_tensor[2:]
        if isinstance(mode, bytes):
            mode = mode.decode()
        if mode == "linear":
            mode = "bilinear"
        if mode not in ["nearest", "bilinear"]:
            raise ValueError(f"Operator Resize does not support resize mode: {mode}")
        # resize 10 only supports one resize method, which is the most basic resize method
        step = 1 / resize_scale
        initial_offset = np.array([-0.5, -0.5])

        # 1. when there is a ratio parameter, if the value of ratio is negative, you need to correct the value of initialOffset;
        # 2. when there is a size parameter, if the value of step is negative, you need to correct the value of initialOffset;
        rank = len(input_shape)
        numOfResizeAxis = 2
        for i in range(numOfResizeAxis):
            axis = rank - numOfResizeAxis - 1 + i
            if step[i] < 0:
                initial_offset[i] = float(input_shape[axis])

        x = hbir.transpose(x, [0, 2, 3, 1])
        x = hbir.resize2d(
            x, step, initial_offset, mode, size=output_shape[2:], expansionMode="border"
        )
        x = hbir.transpose(x, [0, 3, 1, 2], output_type=y)
        return x


Resize()


class TopK(Opset10):
    def __init__(self):
        super().__init__("TopK")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        k: mlir.Value,
        *,
        axis: mlir.Value,
    ):
        k = adaptor.operands[1].value.tolist()[0]
        largest = bool(1)
        sorted = bool(1)
        return hbir.topk(
            data,
            k=k,
            dim=axis,
            largest=largest,
            sorted=sorted,
            values_type=y[0],
            indices_type=y[1],
        )


TopK()


class Slice(Opset10):
    def __init__(self):
        super().__init__("Slice")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        starts: mlir.Value,
        ends: mlir.Value,
        axes: Optional[mlir.Value] = None,
        steps: Optional[mlir.Value] = None,
    ):
        # case0: process dynamic slice or attr do not has static value
        # 1. has dynamic tensor
        input_has_dynamic_dim = (
            has_dynamic_dim(data) or has_dynamic_dim(starts) or has_dynamic_dim(ends)
        )
        # 2. attr do not has static value
        if not isinstance(adaptor.operands[1].value, np.ndarray):
            input_has_dynamic_dim = True
        if not isinstance(adaptor.operands[2].value, np.ndarray):
            input_has_dynamic_dim = True
        if axes is not None:
            if has_dynamic_dim(axes):
                input_has_dynamic_dim = True
            if not isinstance(adaptor.operands[3].value, np.ndarray):
                input_has_dynamic_dim = True
        if steps is not None:
            if has_dynamic_dim(steps):
                input_has_dynamic_dim = True
            if not isinstance(adaptor.operands[4].value, np.ndarray):
                input_has_dynamic_dim = True

        # if has no static value
        if (adaptor.operands[0].value is None) or (adaptor.operands[1].value is None):
            input_has_dynamic_dim = True
        if input_has_dynamic_dim:
            starts_size = adaptor.operands[1].type.shape[0]
            axes = get_value_or_create_const(
                np.array([i for i in range(starts_size)]) if axes is None else axes
            )
            steps = get_value_or_create_const(
                np.ones(starts_size, dtype=np.int64) if steps is None else steps
            )
            return hbir.dynamic_slice(data, starts, ends, axes, steps)

        # case1: process normal slice, use hbir.slice
        input_shape = adaptor.operands[0].type.shape
        input_rank = len(input_shape)
        starts = adaptor.operands[1].value
        ends = adaptor.operands[2].value
        steps = (
            np.ones(input_rank, dtype=np.int64)
            if steps is None
            else adaptor.operands[3].value
            if axes is None
            else adaptor.operands[4].value
        )
        axes = (
            [i for i in range(input_rank)]
            if axes is None
            else adaptor.operands[3].value
        )

        assert len(starts) == len(
            ends
        ), "Incompatible attributes starts and ends for Slice."

        new_start = np.zeros(input_rank, dtype=np.int64)  # start from zero
        new_end = np.array(input_shape)  # end with original shape limit
        new_step = np.ones(input_rank, dtype=np.int64)  # step default 1
        if len(starts) == len(axes):
            for idx, axis in enumerate(axes):
                new_start[axis] = starts[idx]
                new_end[axis] = ends[idx]
                new_step[axis] = steps[idx]
        elif len(starts) == 1:
            for index in range(input_rank):
                if index in axes:
                    new_start[index] = starts[0]
                    new_end[index] = ends[0]
                    new_step[index] = steps[0]
        else:
            raise ValueError("Incompatible attributes starts and axes for Slice.")

        # torch2onnx, axes is all -1. need transpose to hbir
        if (steps == -1).any():
            return hbir.flip(data, axes)

        return hbir.slice(
            data, begin=new_start, end=new_end, step=new_step, output_type=y
        )


Slice()


class Mod(Opset10):
    def __init__(self):
        super().__init__("Mod")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        lhs: mlir.Value,
        rhs: mlir.Value,
        *,
        fmod=0,
    ):
        return hbir.mod(lhs, rhs, sameSignAsDividend=(fmod != 0), output_type=y)


Mod()
