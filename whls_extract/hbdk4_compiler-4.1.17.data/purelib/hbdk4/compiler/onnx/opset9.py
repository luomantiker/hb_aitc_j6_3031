from typing import Any, Iterable, List, Optional, Union  # noqa: F401
from hbdk4.compiler.frontend.adaptor import NodeAdaptor
from hbdk4.compiler.frontend.common import nchw_to_nhwc, nhwc_to_nchw
from hbdk4.compiler.frontend.convertor import OpConvertor
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.ops import hbir, qnt
import numpy as np
from hbdk4.compiler.ops.common import get_value_or_create_const
from hbdk4.compiler.utils.cext import has_dynamic_dim


class Opset9(OpConvertor):
    def __init__(self, name: str):
        super().__init__(name, "onnx", 9, True)


class Conv(Opset9):
    def __init__(self):
        super().__init__("Conv")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        w: mlir.Value,
        b: Optional[mlir.Value] = None,
        *,
        dilations=(1, 1),
        group=1,
        kernel_shape,
        pads=(0, 0, 0, 0),
        strides=(1, 1),
    ):
        x = nchw_to_nhwc(x)
        w = nchw_to_nhwc(w)
        x = hbir.conv(x, w, strides, pads, dilations, group, bias=b)
        x = nhwc_to_nchw(x)
        return x


Conv()


class Add(Opset9):
    def __init__(self):
        super().__init__("Add")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, rhs: mlir.Value
    ):
        return hbir.add(lhs, rhs, output_type=y)


Add()


class And(Opset9):
    def __init__(self):
        super().__init__("And")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, rhs: mlir.Value
    ):
        # lhs = adaptor.operands[0].value.astype(np.int8)
        # rhs = adaptor.operands[1].value.astype(np.int8)
        return hbir.logical_and(lhs, rhs, output_type=y)


And()


class Or(Opset9):
    def __init__(self):
        super().__init__("Or")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, rhs: mlir.Value
    ):
        # lhs = adaptor.operands[0].value.astype(np.int8)
        # rhs = adaptor.operands[1].value.astype(np.int8)
        return hbir.logical_or(lhs, rhs, output_type=y)


Or()


class Xor(Opset9):
    def __init__(self):
        super().__init__("Xor")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, rhs: mlir.Value
    ):
        return hbir.logical_xor(lhs, rhs, output_type=y)


Xor()


class Greater(Opset9):
    def __init__(self):
        super().__init__("Greater")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, rhs: mlir.Value
    ):
        return hbir.greater(lhs, rhs, output_type=y)


Greater()


class Less(Opset9):
    def __init__(self):
        super().__init__("Less")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, rhs: mlir.Value
    ):
        return hbir.less(lhs, rhs, output_type=y)


Less()


class Pow(Opset9):
    def __init__(self):
        super().__init__("Pow")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, rhs: mlir.Value
    ):
        return hbir.pow(lhs, rhs, output_type=y)


Pow()


class Equal(Opset9):
    def __init__(self):
        super().__init__("Equal")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, rhs: mlir.Value
    ):
        return hbir.equal(lhs, rhs, output_type=y)


Equal()


class Sub(Opset9):
    def __init__(self):
        super().__init__("Sub")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, rhs: mlir.Value
    ):
        return hbir.sub(lhs, rhs, output_type=y)


Sub()


class Mul(Opset9):
    def __init__(self):
        super().__init__("Mul")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, rhs: mlir.Value
    ):
        return hbir.mul(lhs, rhs, output_type=y)


Mul()


class Max(Opset9):
    def __init__(self):
        super().__init__("Max")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        lhs: mlir.Value,
        rhs: mlir.Value,
        *args,
    ):
        if len(args) != 0:
            raise ValueError(f"Operator Max expects 2 inputs, but got {len(args)+2}")
        return hbir.max(lhs, rhs, output_type=y)


Max()


class Min(Opset9):
    def __init__(self):
        super().__init__("Min")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        lhs: mlir.Value,
        rhs: mlir.Value,
        *args,
    ):
        if len(args) != 0:
            raise ValueError(f"Operator Min expects 2 inputs, but got {len(args)+2}")
        return hbir.min(lhs, rhs, output_type=y)


Min()


class Sum(Opset9):
    def __init__(self):
        super().__init__("Sum")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        lhs: mlir.Value,
        rhs: mlir.Value,
        *args,
    ):
        if len(args) != 0:
            raise ValueError(f"Operator Sum expects 2 inputs, but got {len(args)+2}")
        return hbir.add(lhs, rhs, output_type=y)


Sum()


class Gemm(Opset9):
    def __init__(self):
        super().__init__("Gemm")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        input_a: mlir.Value,
        input_b: mlir.Value,
        input_c: mlir.Value,
        *,
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=0,
    ):
        assert len(adaptor.operands[0].type.shape) == 2
        assert len(adaptor.operands[1].type.shape) == 2
        input_a = adaptor.operands[0].value
        input_b = adaptor.operands[1].value

        if transA:
            input_a = hbir.transpose(input_a, (1, 0))
        if transB:
            input_b = hbir.transpose(input_b, (1, 0))

        # res = alpha * (a * b) + beta * c
        res = hbir.mul(alpha, hbir.matmul(input_a, input_b))
        res = hbir.add(res, hbir.mul(beta, input_c), output_type=y)
        return res


Gemm()


class GatherND(Opset9):
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


class GatherElements(Opset9):
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


class Softmax(Opset9):
    def __init__(self):
        super().__init__("Softmax")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value, *, axis=-1
    ):
        return hbir.softmax(x, axis, output_type=y)


Softmax()


class MaxPool(Opset9):
    def __init__(self):
        super().__init__("MaxPool")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        auto_pad="NOTSET",
        ceil_mode=0,
        dilations=[1],
        kernel_shape,
        pads=[0, 0],
        storage_order=0,
        strides=[1],
    ):
        # auto_pad is a DEPRECATED attribute, so it is ignored here.
        # onnx maxpool supports a second output, and it is the index of the first output. storage_order is used to control whether it is row-major order or column-major order. Since second output is not supported, so attribute storage_order is ignored here.
        assert len(adaptor.results) == 1
        kernel_dim = len(kernel_shape)
        if kernel_dim < 1 or kernel_dim > 3:
            raise ValueError(
                f"Operator MaxPool does not support kernel_dim {kernel_dim}"
            )
        if kernel_dim == 1:
            x = hbir.transpose(x, [0, 2, 1])
        elif kernel_dim == 2:
            x = hbir.transpose(x, [0, 2, 3, 1])
            if dilations == [1]:
                dilations = dilations * kernel_dim
            if strides == [1]:
                strides = strides * kernel_dim
            if pads == [0, 0]:
                pads = pads * kernel_dim
        elif kernel_dim == 3:
            x = hbir.transpose(x, [0, 2, 3, 4, 1])
            if dilations == [1]:
                dilations = dilations * kernel_dim
            if strides == [1]:
                strides = strides * kernel_dim
            if pads == [0, 0]:
                pads = pads * kernel_dim
        x = hbir.max_pool(
            x,
            kernel=kernel_shape,
            stride=strides,
            pad=pads,
            dilation=dilations,
            ceilMode=bool(ceil_mode),
        )
        if kernel_dim == 1:
            return hbir.transpose(x, [0, 2, 1], output_type=y)
        elif kernel_dim == 3:
            return hbir.transpose(x, [0, 4, 1, 2, 3], output_type=y)
        return hbir.transpose(x, [0, 3, 1, 2], output_type=y)


MaxPool()


class Concat(Opset9):
    def __init__(self):
        super().__init__("Concat")

    def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, *args, axis):
        return hbir.concat(args, dim=axis, output_type=y)


Concat()


class Reshape(Opset9):
    def __init__(self):
        super().__init__("Reshape")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value, shape: mlir.Value
    ):
        out_shape = adaptor.results[0].type.shape
        return hbir.reshape(x, out_shape)


Reshape()


class Transpose(Opset9):
    def __init__(self):
        super().__init__("Transpose")

    def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value, perm):
        return hbir.transpose(x, perm, output_type=y)


Transpose()


class GlobalAveragePool(Opset9):
    def __init__(self):
        super().__init__("GlobalAveragePool")

    def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value):
        return hbir.reduce_mean(x, dims=[-2, -1], keepDim=True, output_type=y)


GlobalAveragePool()


class Clip(Opset9):
    def __init__(self):
        super().__init__("Clip")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        max=3.4028234663852886e38,
        min=-3.4028234663852886e38,
    ):
        return hbir.clip(x, min, max, output_type=y)


Clip()


class AveragePool(Opset9):
    def __init__(self):
        super().__init__("AveragePool")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        kernel_shape,
        pads=[0, 0],
        strides=[1],
        auto_pad="NOTSET",
        ceil_mode=0,
        count_include_pad=0,
    ):
        if auto_pad != "NOTSET":
            raise ValueError(
                "Operator AveragePool does not support attribute auto_pad. It is a deprecated attribute. "
            )
        kernel_dim = len(kernel_shape)
        if kernel_dim < 1 or kernel_dim > 2:
            raise ValueError(
                f"Operator AveragePool does not support kernel_dim {kernel_dim}"
            )
        dilations = [1]
        if kernel_dim == 1:
            x = hbir.transpose(x, [0, 2, 1])
        elif kernel_dim == 2:
            x = hbir.transpose(x, [0, 2, 3, 1])
            if dilations == [1]:
                dilations = dilations * kernel_dim
            if strides == [1]:
                strides = strides * kernel_dim
            if pads == [0, 0]:
                pads = pads * kernel_dim
        # input trans: nchw->nhwc
        # output trans: nhwc->nchw
        x = hbir.avg_pool(
            x, kernel_shape, strides, pads, dilation=dilations, ceilMode=bool(ceil_mode)
        )

        if kernel_dim == 1:
            return hbir.transpose(x, [0, 2, 1], output_type=y)
        return hbir.transpose(x, [0, 3, 1, 2], output_type=y)


AveragePool()


# class GlobalAveragePool(Opset9):
#     def __init__(self):
#         super().__init__("GlobalAveragePool")

#     def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value):
#         shape = mlir.ShapedType(x.type).shape
#         kernel = [shape[-2], shape[-1]]
#         x = hbir.transpose(x, dims=[0, 2, 3, 1])
#         x = hbir.avg_pool(x, kernel=kernel)
#         x = hbir.transpose(x, dims=[0, 3, 1, 2], output_type=y)
#         return x


# GlobalAveragePool()


class Pad(Opset9):
    def __init__(self):
        super().__init__("Pad")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        pads,
        mode="constant",
        value=0.0,
    ):
        pad_length = len(pads)
        if mlir.IntegerType.isinstance(y.element_type):
            value = int(value)
        else:
            value = float(value)
        return hbir.pad(
            x,
            begin=pads[: pad_length // 2],
            end=pads[pad_length // 2 :],
            padValue=value,
            output_type=y,
        )


Pad()


class Slice(Opset9):
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
            else (
                adaptor.operands[3].value if axes is None else adaptor.operands[4].value
            )
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

        return hbir.slice(
            data, begin=new_start, end=new_end, step=new_step, output_type=y
        )


Slice()


class ArgMax(Opset9):
    def __init__(self):
        super().__init__("ArgMax")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value, *, axis=0, keepdims=1
    ):
        return hbir.reduce_argmax(
            x,
            dims=[axis],
            keepDim=bool(keepdims),
            output_type=y,
        )


ArgMax()


class LeakyReLu(Opset9):
    def __init__(self):
        super().__init__("LeakyRelu")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value, *, alpha: float = 0.01
    ):
        return hbir.leaky_relu(x, slop=alpha, output_type=y)


LeakyReLu()


class ELU(Opset9):
    def __init__(self):
        super().__init__("Elu")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value, *, alpha: float = 1.0
    ):
        return hbir.elu(x, alpha=alpha, output_type=y)


ELU()


class ReduceMax(Opset9):
    def __init__(self):
        super().__init__("ReduceMax")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        axes=None,
        keepdims=1,
    ):
        if axes is None:
            axes = list(range(mlir.ShapedType(x.type).rank))
        return hbir.reduce_max(
            x,
            dims=axes,
            keepDim=bool(keepdims),
            output_type=y,
        )


ReduceMax()


class ReduceSum(Opset9):
    def __init__(self):
        super().__init__("ReduceSum")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        axes=None,
        keepdims=1,
    ):
        if axes is None:
            axes = list(range(mlir.ShapedType(x.type).rank))
        return hbir.reduce_sum(
            x,
            dims=axes,
            keepDim=bool(keepdims),
            output_type=y,
        )


ReduceSum()


class ReduceMin(Opset9):
    def __init__(self):
        super().__init__("ReduceMin")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        axes=None,
        keepdims=1,
    ):
        if axes is None:
            axes = list(range(mlir.ShapedType(x.type).rank))
        return hbir.reduce_min(
            x,
            dims=axes,
            keepDim=bool(keepdims),
            output_type=y,
        )


ReduceMin()


class Exp(Opset9):
    def __init__(self):
        super().__init__("Exp")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
    ):
        return hbir.exp(
            x,
            output_type=y,
        )


Exp()


class Reciprocal(Opset9):
    def __init__(self):
        super().__init__("Reciprocal")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
    ):
        return hbir.reciprocal(
            x,
            output_type=y,
        )


Reciprocal()


class Split(Opset9):
    def __init__(self):
        super().__init__("Split")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value, *args, axis=0, split
    ):
        ret_list = []
        shape = adaptor.operands[0].type.shape
        dim = len(shape)
        axis = axis if axis >= 0 else (dim + axis)
        asum = 0
        for i in range(len(split)):
            begin = np.array([0 if i != axis else asum for i in range(dim)])
            asum += split[i]
            end = np.array([shape[i] if i != axis else asum for i in range(dim)])
            step = np.ones(shape=dim, dtype=np.int64)
            output_type = y[i] if isinstance(y, list) else y
            ret_list.append(
                hbir.slice(x, begin=begin, end=end, step=step, output_type=output_type)
            )
        return ret_list


Split()


class MatMul(Opset9):
    def __init__(self):
        super().__init__("MatMul")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, output_type: mlir.Type, a: mlir.Value, b: mlir.Value
    ):
        lhs_shape = np.array(adaptor.operands[0].type.shape)
        rhs_shape = np.array(adaptor.operands[0].type.shape)
        lhs_rank = len(lhs_shape)
        rhs_rank = len(rhs_shape)
        if lhs_rank == 1 and rhs_rank == 1:
            # the output of 1D matmul is a scalar, the out_shape is none, so add reshape
            out_shape = adaptor.results[0].type.shape
            return hbir.reshape(hbir.matmul(a, b), out_shape, output_type=output_type)
        else:
            return hbir.matmul(a, b, output_type=output_type)


MatMul()


class ReduceMean(Opset9):
    def __init__(self):
        super().__init__("ReduceMean")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        axes=None,
        keepdims=1,
    ):
        if axes is None:
            axes = list(range(mlir.ShapedType(x.type).rank))
        return hbir.reduce_mean(x, dims=axes, keepDim=bool(keepdims), output_type=y)


ReduceMean()


class Gather(Opset9):
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


class Div(Opset9):
    def __init__(self):
        super().__init__("Div")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, output_type: mlir.Type, a: mlir.Value, b: mlir.Value
    ):
        return hbir.div(a, b, output_type=output_type)


Div()


class Expand(Opset9):
    def __init__(self):
        super().__init__("Expand")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, data: mlir.Value, shape: mlir.Value
    ):
        original_shape = adaptor.operands[0].type.shape
        new_shape = adaptor.operands[1].value
        original_shape = [
            1,
        ] * (len(new_shape) - len(original_shape)) + original_shape

        repeat_list = []
        for i in range(len(original_shape)):
            if (
                original_shape[i] != new_shape[i]
                and original_shape[i] != 1
                and new_shape[i] != 1
            ):
                raise ValueError(
                    f"Operator Expand does not support this shape. original_shape: {original_shape}, new_shape: {new_shape}"
                )
            if original_shape[i] == 1:
                repeat_list.append(new_shape[i])
            else:
                repeat_list.append(1)
        data = hbir.reshape(
            data,
            original_shape,
            output_type=mlir.UnrankedTensorType.get(mlir.ShapedType(y).element_type),
        )
        return hbir.tile(data, repeat_list, output_type=y)


Expand()


class Flatten(Opset9):
    def __init__(self):
        super().__init__("Flatten")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        *,
        axis=1,
    ):
        original_shape = adaptor.operands[0].type.shape
        new_shape = [1, 1]
        for i in range(axis):
            new_shape[0] *= original_shape[i]
        for i in range(axis, len(original_shape)):
            new_shape[1] *= original_shape[i]
        return hbir.reshape(data, shape=new_shape, output_type=y)


Flatten()


class Tile(Opset9):
    def __init__(self):
        super().__init__("Tile")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        repeats: mlir.Value,
    ):
        repeats = adaptor.operands[1].value
        return hbir.tile(data, repeats, output_type=y)


Tile()


class Unary(Opset9):
    def __init__(self, onnx_op: str, hbir_op: str):
        super().__init__(onnx_op)
        self.attr = hbir_op

    def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value):
        return getattr(hbir, self.attr)(x, output_type=y)


Unary("Relu", "relu")
Unary("Sigmoid", "sigmoid")
Unary("Sin", "sin")
Unary("Cos", "cos")
Unary("Tan", "tan")
Unary("Sinh", "sinh")
Unary("Cosh", "cosh")
Unary("Tanh", "tanh")
Unary("Asin", "asin")
Unary("Acos", "acos")
Unary("Atan", "atan")
Unary("Asinh", "asinh")
Unary("Acosh", "acosh")
Unary("Atanh", "atanh")
Unary("Erf", "erf")
Unary("Sqrt", "sqrt")
Unary("Log", "log")
Unary("Abs", "abs")
Unary("Floor", "floor")
Unary("Ceil", "ceil")
Unary("Not", "logical_not")
Unary("Sign", "sign")

TemplateUnaryList = {
    "Relu": "relu",
    "Sigmoid": "sigmoid",
    "Sin": "sin",
    "Cos": "cos",
    "Tan": "tan",
    "Sinh": "sinh",
    "Cosh": "cosh",
    "Tanh": "tanh",
    "Asin": "asin",
    "Acos": "acos",
    "Atan": "atan",
    "Asinh": "asinh",
    "Acosh": "acosh",
    "Atanh": "atanh",
    "Erf": "erf",
    "Sqrt": "sqrt",
    "Log": "log",
    "Abs": "abs",
    "Floor": "floor",
    "Ceil": "ceil",
    "Not": "logical_not",
    "Sign": "sign",
}


class OptimizedOP(Opset9):
    def __init__(self, onnx_op: str):
        super().__init__(onnx_op)
        self.onnx_op = onnx_op

    def emit_mlir_op(self, *args, **kwargs):
        raise ValueError(
            f"Operator {self.onnx_op} should be optimized. This operator shall not appear in PTQ model."
        )


OptimizedOP("ConstantOfShape")
OptimizedOP("Range")
OptimizedOP("Constant")
OptimizedOP("RandomUniformLike")
OptimizedOP("RandomUniform")
OptimizedOP("Shape")
OptimizedOP("Size")
OptimizedOP("OneHot")


class Softplus(Opset9):
    def __init__(self):
        super().__init__("Softplus")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
    ):
        return hbir.softplus(x, 1.0, 20.0, output_type=y)


Softplus()


class Cast(Opset9):
    def __init__(self):
        super().__init__("Cast")

    def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value, *, to):
        return hbir.cast_type(x, output_type=y)


Cast()


class Squeeze(Opset9):
    def __init__(self):
        super().__init__("Squeeze")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        *,
        axes: Optional[mlir.Value] = None,
    ):
        shape = adaptor.results[0].type.shape
        return hbir.reshape(data, shape, output_type=y)


Squeeze()


class Unsqueeze(Opset9):
    def __init__(self):
        super().__init__("Unsqueeze")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        *,
        axes: mlir.Value,
    ):
        shape = adaptor.results[0].type.shape
        return hbir.reshape(data, shape, output_type=y)


Unsqueeze()


class BatchNormalization(Opset9):
    def __init__(self):
        super().__init__("BatchNormalization")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        scale: mlir.Value,
        bias: mlir.Value,
        mean: mlir.Value,
        var: mlir.Value,
        *,
        epsilon: float = 1e-05,
        momentum: float = 0.9,
    ):
        input_len = len(adaptor.operands[0].type.shape)
        axis = [i for i in range(input_len)]
        permutes_c_last = axis[0:1] + axis[2:] + axis[1:2]
        permutes_c_ahead = (
            axis[0:1] + axis[input_len - 1 : input_len] + axis[1 : input_len - 1]
        )
        data = hbir.transpose(data, permutes_c_last)
        data = hbir.batchnorm(
            data, weight=scale, bias=bias, mean=mean, var=var, eps=epsilon
        )
        data = hbir.transpose(data, permutes_c_ahead, output_type=y)
        return data


BatchNormalization()


class SpaceToDepth(Opset9):
    def __init__(self):
        super().__init__("SpaceToDepth")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        blocksize,
    ):
        input_dim = len(adaptor.operands[0].type.shape)
        if input_dim != 4:
            raise ValueError(
                f"Operator SpaceToDepth does not support input_dim not euqal to 4, got {input_dim}"
            )
        n, c, h, w = adaptor.operands[0].type.shape
        if (h % blocksize != 0) or (w % blocksize != 0):
            raise ValueError(
                "The height and width of the input shape must be divisible by blocksize."
            )
        x = hbir.reshape(
            x, (n, c, h // blocksize, blocksize, w // blocksize, blocksize)
        )
        x = hbir.transpose(x, [0, 3, 5, 1, 2, 4])
        return hbir.reshape(
            x,
            (n, c * (blocksize * blocksize), h // blocksize, w // blocksize),
            output_type=y,
        )


SpaceToDepth()


class DepthToSpace(Opset9):
    def __init__(self):
        super().__init__("DepthToSpace")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        blocksize,
    ):
        input_dim = len(adaptor.operands[0].type.shape)
        if input_dim != 4:
            raise ValueError(
                f"Operator DepthToSpace does not support input_dim not euqal to 4, got {input_dim}"
            )
        n, c, h, w = adaptor.operands[0].type.shape
        if c % (blocksize**2) != 0:
            raise ValueError(
                "The channel of the input shape must be divisible by the square of blocksize."
            )
        x = hbir.reshape(x, (n, blocksize, blocksize, c // (blocksize**2), h, w))
        x = hbir.transpose(x, [0, 3, 4, 1, 5, 2])
        return hbir.reshape(
            x,
            (n, c // (blocksize**2), h * blocksize, w * blocksize),
            output_type=y,
        )


DepthToSpace()


class LogSoftmax(Opset9):
    def __init__(self):
        super().__init__("LogSoftmax")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        *,
        axis=-1,
    ):
        return hbir.log_softmax(data, dim=axis, output_type=y)


LogSoftmax()


class TopK(Opset9):
    def __init__(self):
        super().__init__("TopK")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        *,
        k: mlir.Value,
        axis: mlir.Value,
    ):
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


class GlobalMaxPool(Opset9):
    def __init__(self):
        super().__init__("GlobalMaxPool")

    def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value):
        # do not use transpose, transpose dims directly
        input_shape = adaptor.operands[0].type.shape
        input_rank = len(input_shape)
        if input_rank < 3:
            raise ValueError("input size can not be less than 3")
        dims = []
        for i in range(input_rank - 2):
            dims.append(-(i + 1))
        return hbir.reduce_max(x, dims=dims, keepDim=True, output_type=y)


GlobalMaxPool()


class Where(Opset9):
    def __init__(self):
        super().__init__("Where")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        condition: mlir.Value,
        X: mlir.Value,
        Y: mlir.Value,
    ):
        return hbir.where(condition, X, Y, output_type=y)


Where()


class ThresholdedRelu(Opset9):
    def __init__(self):
        super().__init__("ThresholdedRelu")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        *,
        alpha=1.0,
    ):
        max = 3.4028234663852886e38
        return hbir.clip(data, alpha, max, output_type=y)


ThresholdedRelu()


class PRelu(Opset9):
    def __init__(self):
        super().__init__("PRelu")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, data: mlir.Value, slope: mlir.Value
    ):
        return hbir.prelu(data, slope=slope, output_type=y)


PRelu()


class DequantizeLinear(Opset9):
    def __init__(self):
        super().__init__("DequantizeLinear")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        x_scale: mlir.Value,
        x_zero_point: mlir.Value,
        *,
        axis: Optional[int] = None,
    ):
        x_scale = adaptor.operands[1].value
        if len(x_scale.shape) == 0:  # handle rank0 tensor
            x_scale = x_scale.reshape([1])
        x_zero_point = adaptor.operands[2].value
        if len(x_zero_point.shape) == 0:  # handle rank0 tensor
            x_zero_point = x_zero_point.reshape([1])

        if isinstance(x_scale, mlir.Value):
            raise ValueError("Operator DequantizeLinear scale must be constant")
        if isinstance(x_zero_point, mlir.Value):
            raise ValueError("Operator DequantizeLinear zero must be constant")

        return qnt.dequantize(
            x, scales=x_scale, zeros=x_zero_point, output_type=y, axis=axis
        )


DequantizeLinear()


class InstanceNormalization(Opset9):
    def __init__(self):
        super().__init__("InstanceNormalization")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        scale: mlir.Value,
        bias: mlir.Value,
        *,
        epsilon=1e-05,
    ):
        input_shape = adaptor.operands[0].type.shape
        dim = list(range(1, len(input_shape) - 1))
        x = nchw_to_nhwc(x)
        x = hbir.layernorm(x, dims=dim, eps=epsilon)
        x = hbir.mul(x, scale)
        x = hbir.add(x, bias)
        x = nhwc_to_nchw(x)
        return x


InstanceNormalization()


class ArgMin(Opset9):
    def __init__(self):
        super().__init__("ArgMin")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value, *, axis=0, keepdims=1
    ):
        return hbir.reduce_argmin(
            x,
            dims=[axis],
            keepDim=bool(keepdims),
            output_type=y,
        )


ArgMin()

# class ScatterND(Opset9):
#     def __init__(self):
#         super().__init__("ScatterND")

#     def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type,
#                      data: mlir.Value, indices: mlir.Value,
#                      updates: mlir.Value):

#         return hbir.scatter_nd(
#             data,
#             indices,
#             updates,
#             output_type=y,
#         )

# ScatterND()

# class ScatterElements(Opset9):
#     def __init__(self):
#         super().__init__("ScatterElements")

#     def emit_mlir_op(self,
#                      adaptor: NodeAdaptor,
#                      y: mlir.Type,
#                      data: mlir.Value,
#                      indices: mlir.Value,
#                      updates: mlir.Value,
#                      *,
#                      axis=0,
#                      reduction="none"):

#         reduction = reduction if type(reduction) == str else reduction.decode()
#         return hbir.scatter_elements(
#             data,
#             indices,
#             updates,
#             axis=axis,
#             scatterReduceMode=reduction,
#             output_type=y,
#         )

# ScatterElements()


class Scatter(Opset9):
    def __init__(self):
        super().__init__("Scatter")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        indices: mlir.Value,
        updates: mlir.Value,
        axis: int,
    ):
        return hbir.scatter_elements(data, indices, updates, axis, output_type=y)


Scatter()


class Neg(Opset9):
    def __init__(self):
        super().__init__("Neg")

    def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value):
        return hbir.neg(x)


class NonZero(Opset9):
    def __init__(self):
        super().__init__("NonZero")

    def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value):
        return hbir.nonzero(x)


NonZero()


# ConvTranspose op is defined in opset1 and opset11 in ONNX,
# but this is same for hbdk, so just keep this op in opset9
class ConvTranspose(Opset9):
    def __init__(self):
        super().__init__("ConvTranspose")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        w: mlir.Value,
        b: Optional[mlir.Value] = None,
        *,
        auto_pad="NOTSET",
        dilations=(1, 1),
        group=1,
        kernel_shape=None,
        output_padding=None,
        output_shape=None,
        pads=(0, 0, 0, 0),
        strides=(1, 1),
    ):
        auto_pad = auto_pad if type(auto_pad) == str else auto_pad.decode()
        if auto_pad != "NOTSET":
            raise ValueError(
                f"Operator ConvTranspose does not support auto_pad with value: {auto_pad}"
            )

        if output_padding is not None:
            for idx, p in enumerate(output_padding):
                pads[(len(pads) // 2) + idx] -= p

        kernel_dim = len(kernel_shape)
        if (kernel_dim != 1) and (kernel_dim != 2) and (kernel_dim != 3):
            raise ValueError(
                f"Only support ConvTranspose 1D, ConvTranspose 2D and 3d Operator, got kernel_dim={kernel_dim}"
            )
        # input trans: nc(d)hw->n(d)hwc
        # output trans: n(d)hwc->nc(d)hw
        x = nchw_to_nhwc(x)
        w = nchw_to_nhwc(w)
        x = hbir.convtranspose(
            x,
            weight=w,
            stride=strides,
            pad=pads,
            dilation=dilations,
            groupNum=group,
            bias=b,
            illegalWeight=True,
        )
        x = nhwc_to_nchw(x)
        return x


ConvTranspose()


class EyeLike(Opset9):
    def __init__(self):
        super().__init__("EyeLike")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        input: mlir.Value,
        *,
        dtype: Optional[int] = None,
        k: int = 0,
    ):
        input_shape = adaptor.operands[0].type.shape
        if len(input_shape) != 2:
            raise ValueError(
                "Input tensor must be 2D. Received shape: {}".format(input_shape)
            )
        row = input_shape[0]
        col = input_shape[1]
        numpy_type_dict = {
            "i8": np.uint8,
            "i16": np.uint16,
            "i32": np.uint32,
            "si8": np.int8,
            "si16": np.int16,
            "si32": np.int32,
            "si64": np.int64,
            "f16": np.float16,
            "f32": np.float32,
            # "f64": np.float64, do not support f64
            # "bool": np.bool, do not support bool
        }
        onnx_type_to_numpy_type_dict = {
            1: np.float32,
            2: np.uint8,
            3: np.int8,
            4: np.uint16,
            5: np.int16,
            6: np.int32,
            7: np.int64,
            10: np.float16,
            11: np.double,
            12: np.uint32,
            13: np.uint64,
        }
        for key in numpy_type_dict:
            if key in str(input.type):
                x_base_type = numpy_type_dict[key]
        if dtype in onnx_type_to_numpy_type_dict:
            np_dtype = onnx_type_to_numpy_type_dict[dtype]
        else:
            np_dtype = x_base_type if x_base_type else np.float32

        output_matrix = np.eye(N=row, M=col, k=k, dtype=np_dtype)
        return hbir.constant(values=output_matrix, output_type=y)


EyeLike()


class Identity(Opset9):
    def __init__(self):
        super().__init__("Identity")

    def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value):
        return hbir.identity(x)


Identity()
