from sympy import false
from hbdk4.compiler.frontend.adaptor import NodeAdaptor
from hbdk4.compiler.frontend.convertor import OpConvertor
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.ops import hbir, qnt
from typing import Any, List, Optional, Union
import numpy as np
import math
from hbdk4.compiler.ops.common import get_value_or_create_const
from hbdk4.compiler.utils.cext import has_dynamic_dim


class Opset13(OpConvertor):
    def __init__(self, name: str):
        super().__init__(name, "onnx", 13, True)


class ReduceSum(Opset13):
    def __init__(self):
        super().__init__("ReduceSum")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        axes=None,
        *args,
        keepdims=1,
        noop_with_empty_axes=0,
    ):
        axes = adaptor.operands[1].value.tolist() if axes is not None else None

        if axes is None:
            if noop_with_empty_axes == 0:
                # Reduce all axes
                axes = list(range(mlir.ShapedType(x.type).rank))
            else:
                # act like identity operands, here convert to reshape
                return hbir.reshape(x, adaptor.operands[0].type.shape)

        return hbir.reduce_sum(
            x,
            dims=axes,
            keepDim=bool(keepdims),
            output_type=y,
        )


ReduceSum()


class GatherElements(Opset13):
    def __init__(self):
        super().__init__("GatherElements")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        indices: mlir.Value,
        *,
        axis=0,
    ):
        return hbir.gather_elements(data, indices, dim=axis)


GatherElements()


class Gather(Opset13):
    def __init__(self):
        super().__init__("Gather")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        indices: mlir.Value,
        axis=0,
    ):
        return hbir.index(x, index=indices, dim=axis, output_type=y)


Gather()


class GatherND(Opset13):
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


class Mod(Opset13):
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


class Slice(Opset13):
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

        # torch2onnx, axes is all -1
        if (steps == -1).any():
            return hbir.flip(data, axes)

        return hbir.slice(
            data, begin=new_start, end=new_end, step=new_step, output_type=y
        )


Slice()


class Unary(Opset13):
    def __init__(self, onnx_op: str, hbir_op: str):
        super().__init__(onnx_op)
        self.attr = hbir_op

    def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value):
        return getattr(hbir, self.attr)(x, output_type=y)


Unary("Abs", "abs")

TemplateUnaryList = {
    "Abs": "abs",
}


class Squeeze(Opset13):
    def __init__(self):
        super().__init__("Squeeze")

    def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, data: mlir.Value, *args):
        axes = []
        shape = adaptor.operands[0].type.shape
        rank = len(shape)
        # axes is input for op
        if len(args) != 0:
            axes = adaptor.operands[1].value.tolist()
            # axis is [-r, r-1] where r = rank(data)
            for idx in range(len(axes)):
                if axes[idx] < 0:
                    axes[idx] = axes[idx] + rank
        else:
            # to find the single dimensions axis
            for idx in range(len(shape)):
                if shape[idx] == 1:
                    axes.append(idx)

        axes = sorted(axes, reverse=True)
        for axis in axes:
            shape.pop(axis)
        return hbir.reshape(data, shape, output_type=y)


Squeeze()


class Unsqueeze(Opset13):
    def __init__(self):
        super().__init__("Unsqueeze")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, data: mlir.Value, axes: mlir.Value
    ):
        shape = adaptor.results[0].type.shape
        return hbir.reshape(data, shape, output_type=y)


Unsqueeze()


# trilu op verision is opset14
class Trilu(Opset13):
    def __init__(self):
        super().__init__("Trilu")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *args,
        upper=1,
    ):
        k = 0
        if len(args) != 0:
            k = adaptor.operands[1].value
        shape = adaptor.operands[0].type.shape
        matrix = []
        if upper == 1:
            matrix = np.triu(np.ones(shape), k)
        else:
            matrix = np.tril(np.ones(shape), k)
        fake_quant_matrix = qnt.const_fake_quant(
            matrix, [-128.0], [127.0], 8, False, axis=None
        )
        return hbir.mul(x, fake_quant_matrix)


Trilu()


class Split(Opset13):
    def __init__(self):
        super().__init__("Split")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value, *args, axis=0
    ):
        split = None
        if len(args) != 0:
            split = adaptor.operands[1].value.tolist()
        # format: ele = ceil(dim / num), the last one is dim - ele * (num -1)
        # for example, num_outputs is 3, dim is 128, get split [43, 43, 42]
        if split is None:
            split = []
            tmp_shape = adaptor.operands[0].type.shape
            num_outputs = len(adaptor.results)
            ele = int(math.ceil(tmp_shape[axis] / num_outputs))
            if tmp_shape[axis] % num_outputs == 0:
                split = [int(tmp_shape[axis] / num_outputs)] * num_outputs
            else:
                split = [ele] * (num_outputs - 1)
                split.append(tmp_shape[axis] - ele * (num_outputs - 1))

        ret_list = []
        shape = adaptor.operands[0].type.shape
        dim = len(shape)
        asum = 0
        for i in range(len(split)):
            begin = np.array([0 if i != axis else asum for i in range(dim)])
            asum += split[i]
            end = np.array([shape[i] if i != axis else asum for i in range(dim)])
            step = np.ones(dim, dtype=np.int64)
            output_type = y[i]
            ret_list.append(
                hbir.slice(x, begin=begin, end=end, step=step, output_type=output_type)
            )
        return ret_list


Split()


class QuantizeLinear(Opset13):
    def __init__(self):
        super().__init__("QuantizeLinear")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Type,
        y_scale: mlir.Value,
        y_zero_point=None,
        *,
        axis=1,
    ):
        # only support int8/uint8 quantize linear
        y_scale = adaptor.operands[1].value
        y_zero_point = adaptor.operands[2].value
        return qnt.quantize(
            x,
            scales=y_scale,
            zeros=y_zero_point,
            output_type=y,
            axis=axis,
            narrowRange=False,
        )


QuantizeLinear()
