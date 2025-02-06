from typing import Any, List, Optional, Union
from hbdk4.compiler.frontend.adaptor import NodeAdaptor
from hbdk4.compiler.frontend.convertor import OpConvertor
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.ops import hbir
from hbdk4.compiler.ops.common import get_value_or_create_const
from hbdk4.compiler.frontend.common import nchw_to_nhwc, nhwc_to_nchw, get_unranked
from hbdk4.compiler.utils.cext import has_dynamic_dim

import torch
import math
import numpy as np


class AtenCvt(OpConvertor):
    def __init__(self, name: str):
        super().__init__(name, "aten", 0, True)


class TorchvisionCvt(OpConvertor):
    def __init__(self, name: str):
        super().__init__(name, "torchvision", 0, True)


class unimplemented(AtenCvt):
    def __init__(self, name: str, target_name: Optional[str] = None):
        self.name = name
        super().__init__(self.name)


unimplemented("ones")
unimplemented("zeros")

# aten::full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
unimplemented("full")

# aten::ScalarImplicit(Tensor a) -> Scalar
unimplemented("ScalarImplicit")

# aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
unimplemented("arange")

# aten::meshgrid(Tensor[] tensors) -> Tensor[]
unimplemented("meshgrid")

unimplemented("bitwise_not")

# aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
unimplemented("masked_fill")

# aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
unimplemented("copy_")

# aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
unimplemented("index")


class identity(AtenCvt):
    def __init__(self, name: str):
        super().__init__(name)

    def emit_mlir_op(self, adaptor: NodeAdaptor, otype: mlir.Type, x: Any, *args):
        return hbir.identity(x)


identity("contiguous")
identity("alias_copy")


class getitem(AtenCvt):
    def __init__(self, name):
        super().__init__(name)

    def emit_mlir_op(self, adaptor: NodeAdaptor, otype: mlir.Type, x: Any, key: Any):
        return [x[key]]


getitem("__getitem__")
getitem("getitem")


class size(AtenCvt):
    def __init__(self):
        super().__init__("size")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, otype: mlir.Type, x: mlir.Value, dim: int
    ):
        return [adaptor.operands[0].type.shape[dim]]


size()


class to(AtenCvt):
    def __init__(self):
        super().__init__("to")
        # aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
        # aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, otype: mlir.Type, x: mlir.Value, dtype, *others
    ):
        return hbir.cast_type(x, output_type=otype)


to()


def norm_pads(padding, output_padding=[0, 0]):
    if len(padding) == 1:
        return [
            padding[0],
            padding[0] - output_padding[0],
        ]
    elif len(padding) == 2:
        return [
            padding[0],
            padding[1],
            padding[0] - output_padding[0],
            padding[1] - output_padding[1],
        ]
    else:
        if len(output_padding) == 2:
            output_padding = [0, 0, 0]
        return [
            padding[0],
            padding[1],
            padding[2],
            padding[0] - output_padding[0],
            padding[1] - output_padding[1],
            padding[2] - output_padding[2],
        ]


class convolution(AtenCvt):
    def __init__(self, name):
        super().__init__(name)
        # aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        w: mlir.Value,
        b: Optional[mlir.Value],
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        transposed: bool,
        output_padding: List[int],
        groups: int,
        benchmark: bool = False,
        deterministic: bool = True,
        cudnn_enabled: bool = False,
        allow_tf32: bool = False,
    ):
        kernel_dim = len(w.type.shape)
        x = nchw_to_nhwc(x)
        w = nchw_to_nhwc(w)
        if transposed:
            if (kernel_dim != 3) and (kernel_dim != 4) and (kernel_dim != 5):
                raise ValueError(
                    f"Only support Conv1dTranspose, Conv2dTranspose and Conv2dTranspose Operator, got kernel_dim={kernel_dim}"
                )
            x = hbir.convtranspose(
                x,
                w,
                stride,
                norm_pads(padding, output_padding),
                dilation,
                groups,
                bias=b,
                illegalWeight=True,
            )
        else:
            x = hbir.conv2d(
                x,
                w,
                stride,
                norm_pads(padding, output_padding),
                dilation,
                groups,
                bias=b,
            )

        return nhwc_to_nchw(x, otype)


convolution("_convolution")
convolution("convolution")


class linear(AtenCvt):
    def __init__(self):
        super().__init__("linear")
        # aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        w: mlir.Value,
        b: Optional[mlir.Value],
    ):
        return hbir.linear(x, w, bias=b, output_type=otype)


linear()


class matmul(AtenCvt):
    def __init__(self, variant):
        super().__init__(variant)
        # aten::matmul(Tensor self, Tensor other) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        lhs: mlir.Value,
        rhs: mlir.Value,
    ):
        lhs_shape = np.array(adaptor.operands[0].type.shape)
        rhs_shape = np.array(adaptor.operands[0].type.shape)
        lhs_rank = len(lhs_shape)
        rhs_rank = len(rhs_shape)
        if lhs_rank == 1 and rhs_rank == 1:
            # the output of 1D matmul is a scalar, the out_shape is none, so add reshape
            out_shape = adaptor.results[0].type.shape
            return hbir.reshape(hbir.matmul(lhs, rhs), out_shape, output_type=otype)
        else:
            return hbir.matmul(lhs, rhs, output_type=otype)


matmul("matmul")
matmul("bmm")
matmul("mm")


class addmm(AtenCvt):
    def __init__(self, variant):
        super().__init__(variant)
        # aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        # aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        # beta * self + batch1 @ batch2

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        mask: mlir.Value,
        batch1: mlir.Value,
        batch2: mlir.Value,
        beta: float,
        alpha: float,
    ):
        assert adaptor.operands[0].is_constant and adaptor.operands[0].value.sum() == 0
        assert beta == 1
        assert alpha == 1
        return hbir.matmul(batch1, batch2)


addmm("baddbmm")
addmm("addmm")


class batch_norm(AtenCvt):
    def __init__(self, name):
        super().__init__(name)
        # aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? mean, Tensor? var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
        # aten::_native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        w: Optional[mlir.Value],
        b: Optional[mlir.Value],
        m: mlir.Value,
        v: mlir.Value,
        trainning: bool,
        momentum: float,
        eps: float,
        cudnn_enabled: bool,
    ):
        if adaptor.operands[0].type.rank == 2:
            return hbir.batchnorm(x, m, v, eps, weight=w, bias=b)
        x = nchw_to_nhwc(x)
        x = hbir.batchnorm(x, m, v, eps, weight=w, bias=b)
        return nhwc_to_nchw(x, otype)


batch_norm("batch_norm")


class layer_norm(AtenCvt):
    def __init__(self):
        super().__init__("layer_norm")
        # aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        normalized_shape: List[int],
        w: Optional[mlir.Value],
        b: Optional[mlir.Value],
        eps: float,
        cudnn_enable: bool,
    ):
        # layernorm subgraph:
        #    x -> reduce_mean -> sub -> mul -> reduce_mean -> normalize
        itype = adaptor.operands[0].type
        dims = []
        for i, shape in enumerate(normalized_shape):
            dim = itype.rank - len(normalized_shape) + i
            if itype.shape[dim] == shape:
                dims.append(dim)

        if w is not None:  # align weight rank to targeted rank
            wtype = adaptor.operands[2].type
            expansion = [1] * (itype.rank - wtype.rank)
            w = hbir.reshape(w, [*expansion, *wtype.shape])

        if b is not None:  # align bias rank to targeted rank
            btype = adaptor.operands[3].type
            expansion = [1] * (itype.rank - btype.rank)
            b = hbir.reshape(b, [*expansion, *btype.shape])

        return hbir.layernorm(x, dims, eps, weight=w, bias=b, output_type=otype)


layer_norm()


class instance_norm(AtenCvt):
    def __init__(self):
        super().__init__("instance_norm")
        # aten::instance_norm(Tensor input, Tensor? weight=None, Tensor? bias=None, Tensor? running_mean=None, Tensor? running_var=None, bool use_input_stats=True, float momentum=0.1, float eps=1e-05, bool cudnn_enabled=True) -> (Tensor)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        w: Optional[mlir.Value],
        b: Optional[mlir.Value],
        running_mean: Optional[mlir.Value],
        running_var: Optional[mlir.Value],
        use_input_stats: bool,
        momentum: float,
        eps: float,
        cudnn_enable: bool,
    ):
        itype = adaptor.operands[0].type
        dims = list(range(2, itype.rank))  # input is nchw, h & w axis is 2 and 3
        return hbir.layernorm(x, dims, eps)


instance_norm()


class max_pool1d(AtenCvt):
    def __init__(self):
        super().__init__("max_pool1d")
        # aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        k: List[int],
        s: List[int],
        pad: List[int],
        dilate: List[int],
        ceil_mode: bool,
    ):
        x = nchw_to_nhwc(x)
        new_pad = [pad[0], pad[0]]
        x = hbir.max_pool(x, k, s, new_pad, dilate, ceil_mode)
        return nhwc_to_nchw(x, otype)


max_pool1d()


class max_pool2d(AtenCvt):
    def __init__(self):
        super().__init__("max_pool2d")
        # aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        k: List[int],
        s: List[int],
        pad: List[int],
        dilate: List[int],
        ceil_mode: bool,
    ):
        x = nchw_to_nhwc(x)
        pad = norm_pads(pad)
        x = hbir.max_pool(x, k, s, pad, dilate, ceil_mode)
        return nhwc_to_nchw(x, otype)


max_pool2d()


class max_pool3d(AtenCvt):
    def __init__(self):
        super().__init__("max_pool3d")
        # aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        k: List[int],
        s: List[int],
        pad: List[int],
        dilate: List[int],
        ceil_mode: bool,
    ):
        x = nchw_to_nhwc(x)
        pad = norm_pads(pad)
        x = hbir.max_pool(x, k, s, pad, dilate, ceil_mode)
        return nhwc_to_nchw(x, otype)


max_pool3d()


class max_pool2d_with_indices(AtenCvt):
    def __init__(self):
        super().__init__("max_pool2d_with_indices")
        # aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        k: List[int],
        s: List[int],
        pad: List[int],
        dilate: List[int],
        ceil_mode: bool,
    ):
        x = nchw_to_nhwc(x)
        pad = norm_pads(pad)
        x, i = hbir.max_pool_with_indices(
            x,
            k,
            s,
            pad,
            dilate,
            ceil_mode,
            output_type=get_unranked(otype[0]),
            indices_type=get_unranked(otype[1]),
        )
        return [[nhwc_to_nchw(x, otype[0]), nhwc_to_nchw(i, otype[1])]]


max_pool2d_with_indices()


class avg_pool1d(AtenCvt):
    def __init__(self):
        super().__init__("avg_pool1d")
        # aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        k: List[int],
        s: List[int],
        pad: List[int],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override=None,
    ):
        if divisor_override is not None:
            raise ValueError("unsupported divisor_override")

        if count_include_pad is False:
            if pad != [0]:
                raise ValueError("unsupported count_include_pad")
        dilation = [1]
        x = nchw_to_nhwc(x)
        new_pad = [pad[0], pad[0]]
        x = hbir.avg_pool(x, k, s, new_pad, dilation, ceilMode=ceil_mode)
        return nhwc_to_nchw(x, otype)


avg_pool1d()


class avg_pool2d(AtenCvt):
    def __init__(self):
        super().__init__("avg_pool2d")
        # aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        k: List[int],
        s: List[int],
        pad: List[int],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: int,
    ):
        if divisor_override is not None:
            raise ValueError("unsupported divisor_override")

        if count_include_pad is False:
            if pad != [0, 0]:
                raise ValueError("unsupported count_include_pad")

        dilation = [1, 1]
        x = nchw_to_nhwc(x)
        pad = norm_pads(pad)
        x = hbir.avg_pool(x, k, s, pad, dilation, ceil_mode)
        return nhwc_to_nchw(x, otype)


avg_pool2d()


# class adaptive_avg_pool2d(AtenCvt):
#     def __init__(self):
#         super().__init__("adaptive_avg_pool2d")

#     def emit_mlir_op(
#         self,
#         adaptor: NodeAdaptor,
#         otype: mlir.Type,
#         x: mlir.Value,
#         output: List[int],
#     ):
#         if output != [1, 1]:
#             raise ValueError("unsupported output size")

#         shape = mlir.ShapedType(x.type).shape
#         kernel = [shape[-2], shape[-1]]
#         x = nchw_to_nhwc(x)
#         x = hbir.avg_pool(x, kernel=kernel)
#         return nhwc_to_nchw(x, otype=otype)


# adaptive_avg_pool2d()
class adaptive_avg_pool2d(AtenCvt):
    def __init__(self):
        super().__init__("adaptive_avg_pool2d")

    # aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        output_size: List[int],
    ):
        if output_size != [1, 1]:
            raise ValueError("unsupported output size")
        x = nchw_to_nhwc(x)
        x = hbir.reduce_mean(x, [-3, -2], True)
        return nhwc_to_nchw(x, otype)


adaptive_avg_pool2d()


class adaptive_avg_pool1d(AtenCvt):
    def __init__(self):
        super().__init__("adaptive_avg_pool1d")

    # aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        output_size: List[int],
    ):
        if output_size != [1]:
            raise ValueError("unsupported output size")
        x = nchw_to_nhwc(x)
        x = hbir.reduce_mean(x, [-2], True)
        return nhwc_to_nchw(x, otype)


adaptive_avg_pool1d()


class adaptive_max_pool2d(AtenCvt):
    def __init__(self):
        super().__init__("adaptive_max_pool2d")

    # aten::adaptive_max_pool2d(Tensor self, int[1] output_size) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        output_size: List[int],
    ):
        if output_size != [1, 1]:
            raise ValueError("unsupported output size")
        x = hbir.reduce_max(input=x, dims=[-2, -1], keepDim=True, output_type=otype[0])
        return x


adaptive_max_pool2d()


class reduce(AtenCvt):
    def __init__(self, name):
        self.name = name
        super().__init__(self.name)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        dim: Optional[List[int]] = None,
        keep_dim: bool = False,
        dtype=None,
    ):
        if isinstance(dim, int):
            dim = [dim]
        # dim can be None or an empty list. In this case, we need to reduce all dimensions
        if (dim is None) or (len(dim) == 0):
            dim = list(range(mlir.ShapedType(x.type).rank))
        return getattr(hbir, "reduce_" + self.name)(x, dim, keep_dim, output_type=otype)


reduce("mean")
reduce("sum")
reduce("all")


class min_and_max(AtenCvt):
    def __init__(self, name: str):
        assert name in ["min", "max"], "only min and max are valid name"
        self.name = name
        super().__init__(self.name)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: Union[mlir.Type, List[mlir.Type]],
        x: mlir.Value,
        dim: Optional[int] = None,
        keep_dim: bool = False,
    ):
        if dim is None:
            # If dim is None, torch.min = hbir.reduce_min
            assert not keep_dim, "keep_dim should be False when dim is not provided"
            assert isinstance(
                otype, mlir.Type
            ), f"torch.{self.name} should have single output when dim is not provided"
            dim = list(range(mlir.ShapedType(x.type).rank))
            return getattr(hbir, f"reduce_{self.name}")(
                x, dim, keep_dim, output_type=otype
            )
        else:
            # If dim is provided, torch.{self.name} = hbir.reduce_{self.name} + hbir.reduce_arg{self.name}
            assert isinstance(otype, list) and (
                len(otype) == 2
            ), "torch.min should have two outputs when dim is provided"
            return [
                getattr(hbir, f"reduce_{self.name}")(
                    x, [dim], keep_dim, output_type=otype[0]
                ),
                getattr(hbir, f"reduce_arg{self.name}")(
                    x, [dim], keep_dim, output_type=otype[1]
                ),
            ]


min_and_max("min")
min_and_max("max")


class argmax_and_argmin(AtenCvt):
    def __init__(self, name: str):
        assert name in ["argmax", "argmin"], "only argmax and argmin are valid names"
        self.name = name
        super().__init__(self.name)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        dim: Optional[int] = None,
        keep_dim: bool = False,
    ):
        if dim is None:
            # If dim is not provided, this is a all reduce and we should first flatten the operand and perform reduce_argmax on it
            from math import prod

            original_shape = mlir.ShapedType(x.type).shape
            flattened_x = hbir.reshape(x, [prod(original_shape)])
            return getattr(hbir, f"reduce_{self.name}")(
                flattened_x, [0], False, output_type=otype
            )
        else:
            return getattr(hbir, f"reduce_{self.name}")(
                x, [dim], keep_dim, output_type=otype
            )


argmax_and_argmin("argmax")
argmax_and_argmin("argmin")


class softmax(AtenCvt):
    def __init__(self):
        super().__init__("softmax")
        # aten::softmax(Tensor self, int dim, int? dtype) -> Tensor

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, otype: mlir.Type, x: mlir.Value, dim: int, dtype
    ):
        return hbir.softmax(x, dim, output_type=otype)


softmax()


class log_softmax(AtenCvt):
    def __init__(self):
        super().__init__("log_softmax")
        # aten::log_softmax(Tensor self, int dim, int? dtype) -> Tensor

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, otype: mlir.Type, x: mlir.Value, dim: int, dtype
    ):
        return hbir.log_softmax(x, dim, output_type=otype)


log_softmax()


class binary_with_alpha(AtenCvt):
    def __init__(self, name: str, reverse):
        self.name = name
        self.reverse = reverse
        super().__init__(self.name)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        lhs: mlir.Value,
        rhs: mlir.Value,
        alpha: float,
    ):
        if alpha != 1:
            raise ValueError("alpha must be 1")

        if self.reverse:
            return getattr(hbir, self.name)(rhs, lhs, output_type=otype)

        else:
            return getattr(hbir, self.name)(lhs, rhs, output_type=otype)


binary_with_alpha(
    "add", False
)  # aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
binary_with_alpha(
    "sub", False
)  # aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
binary_with_alpha(
    "rsub", True
)  # aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor


class binary(AtenCvt):
    def __init__(self, name: str, reverse):
        self.name = name
        self.reverse = reverse
        super().__init__(self.name)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        lhs: mlir.Value,
        rhs: mlir.Value,
    ):
        if self.reverse:
            return getattr(hbir, self.name)(rhs, lhs, output_type=otype)

        else:
            return getattr(hbir, self.name)(lhs, rhs, output_type=otype)


binary("mul", False)  # aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
binary("div", False)  # aten::div.Tensor(Tensor self, Tensor other) -> Tensor
binary("pow", False)  # aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
binary("rdiv", True)  # aten::rdiv.Tensor(Tensor self, Tensor other) -> Tensor
binary("floor_divide", True)  # aten::floor_divide(Tensor self, Tensor other) -> Tensor
binary("logical_and", False)  # aten::logical_and(Tensor self, Tensor other) -> Tensor
binary("logical_or", False)  # aten::logical_or(Tensor self, Tensor other) -> Tensor
binary("logical_xor", False)  # aten::logical_xor(Tensor self, Tensor other) -> Tensor
binary("bitwise_or", False)  # aten::bitwise_or(Tensor self, Tensor other) -> Tensor
binary("bitwise_and", False)  # aten::bitwise_and(Tensor self, Tensor other) -> Tensor


class fmod(AtenCvt):
    def __init__(self):
        super().__init__("fmod")
        # aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        lhs: mlir.Value,
        rhs: mlir.Value,
    ):
        return hbir.mod(lhs, rhs, sameSignAsDividend=True, output_type=otype)


fmod()


class reminder(AtenCvt):
    def __init__(self):
        super().__init__("remainder")
        # aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        lhs: mlir.Value,
        rhs: mlir.Value,
    ):
        return hbir.mod(lhs, rhs, sameSignAsDividend=False, output_type=otype)


reminder()


class topk(AtenCvt):
    def __init__(self):
        super().__init__("topk")
        # aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: List[mlir.Type],
        x: mlir.Value,
        k: int,
        dim: int,
        largest: bool,
        sorted: bool,
    ):
        assert len(otype) == 2
        return hbir.topk(
            x, k, dim, largest, sorted, values_type=otype[0], indices_type=otype[1]
        )


topk()


class stack(AtenCvt):
    def __init__(self):
        super().__init__("stack")
        # aten::stack(Tensor[] tensors, int dim=0) -> Tensor

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, otype: mlir.Type, x: List[mlir.Value], dim: int
    ):
        return hbir.stack(x, dim, output_type=otype)


stack()


class sort(AtenCvt):
    def __init__(self):
        super().__init__("sort")
        # aten::sort(Tensor self, bool stable=False, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: List[mlir.Type],
        x: mlir.Value,
        stable: Optional[bool] = False,
        dim: Optional[int] = -1,
        descending: Optional[bool] = False,
    ):
        if isinstance(dim, bool):
            raise ValueError(
                "Due to the impact of the torch interface, you must explicitly pass in the 'stable' parameter."
            )
        return hbir.sort(
            x,
            dim=dim,
            descending=descending,
            stable=stable,
            values_type=otype[0],
            indices_type=otype[1],
        )


sort()


class cat(AtenCvt):
    def __init__(self):
        super().__init__("cat")

    # aten::cat(Tensor[] tensors, int dim=0) -> Tensor

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, otype: mlir.Type, x: List[mlir.Value], dim: int
    ):
        return hbir.concat(x, dim, output_type=otype)


cat()


class unbind(AtenCvt):
    def __init__(self):
        super().__init__("unbind")
        # aten::unbind.int(Tensor(a) self, int dim=0)

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, otypes: List[mlir.Type], x: mlir.Value, dim: int
    ):
        return [
            [hbir.select(x, dim, i) for i in range(adaptor.operands[0].type.shape[dim])]
        ]


unbind()


class select(AtenCvt):
    def __init__(self):
        super().__init__("select")
        # aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        dim: int,
        index: int,
    ):
        return hbir.select(x, dim, index, output_type=otype)


select()


class permute(AtenCvt):
    def __init__(self):
        super().__init__("permute")
        # aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, otype: mlir.Type, x: mlir.Value, dims: List[int]
    ):
        return hbir.transpose(x, dims, output_type=otype)


permute()


class transpose(AtenCvt):
    def __init__(self):
        super().__init__("transpose")

    # aten::transpose(Tensor self, int dim0, int dim1) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        dim0: int,
        dim1: int,
    ):
        dims = [i for i in range(adaptor.operands[0].type.rank)]
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return hbir.transpose(x, dims, output_type=otype)


transpose()


class flatten(AtenCvt):
    def __init__(self):
        super().__init__("flatten")

    # aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        start_dim: int,
        end_dim: int,
    ):
        return hbir.reshape(x, adaptor.results[0].type.shape, output_type=otype)


flatten()


class reshape(AtenCvt):
    def __init__(self):
        super().__init__("reshape")
        # aten::reshape(Tensor(a) self, int[] shape) -> Tensor(a)

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, otype: mlir.Type, x: mlir.Value, shape: List[int]
    ):
        return hbir.reshape(x, adaptor.results[0].type.shape, output_type=otype)


reshape()


class view(AtenCvt):
    def __init__(self):
        super().__init__("view")
        # aten::view(Tensor self, int[] size) -> Tensor

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, otype: mlir.Type, x: mlir.Value, shape: List[int]
    ):
        return hbir.reshape(x, adaptor.results[0].type.shape, output_type=otype)


view()


class unsqueeze(AtenCvt):
    def __init__(self):
        super().__init__("unsqueeze")

    # aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, otype: mlir.Type, x: mlir.Value, dim: int
    ):
        return hbir.reshape(x, adaptor.results[0].type.shape, output_type=otype)


unsqueeze()


class squeeze(AtenCvt):
    def __init__(self):
        super().__init__("squeeze")
        # aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        dim: Optional[List[int]] = None,
    ):
        return hbir.reshape(x, adaptor.results[0].type.shape, output_type=otype)


squeeze()


class roll(AtenCvt):
    def __init__(self):
        super().__init__("roll")
        # aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        shifts: List[int],
        dims: List[int],
    ):
        return hbir.roll(x, shifts, dims, output_type=otype)


roll()


def genSliceInput(
    is_dynamic_slice,
    hbir_input,
    torch_input,
    shape,
    dim,
    input_has_dynamic_dim,
    default_value,
):
    if torch_input is not None:
        if isinstance(torch_input, mlir.Value):
            is_dynamic_slice = True
            hbir_input = hbir.reshape(torch_input, [1])
        else:
            # If the input is dynamic or the input is already a tensor, a tensor must be constructed
            if input_has_dynamic_dim or is_dynamic_slice:
                is_dynamic_slice = True
                hbir_input = get_value_or_create_const(np.array([torch_input]))
            else:
                hbir_input[dim] = torch_input
    elif input_has_dynamic_dim:
        is_dynamic_slice = True
        hbir_input = get_value_or_create_const(np.array([default_value]))
    return hbir_input, is_dynamic_slice


class slice(AtenCvt):
    def __init__(self):
        super().__init__("slice")
        # aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        dim: int = 0,
        start: Optional[Union[int, mlir.Value]] = None,
        end: Optional[Union[int, mlir.Value]] = None,
        step: Union[int, mlir.Value] = 1,
    ):
        rank = adaptor.operands[0].type.rank
        shape = adaptor.operands[0].type.shape

        begin = [0] * rank
        limit = shape
        stride = [1] * rank

        # pick dim
        input_has_dynamic_dim = has_dynamic_dim(x)
        is_dynamic_slice = False
        # prepare begin
        begin, is_dynamic_slice = genSliceInput(
            is_dynamic_slice, begin, start, shape, dim, input_has_dynamic_dim, 0
        )
        # prepare end
        limit, is_dynamic_slice = genSliceInput(
            is_dynamic_slice, limit, end, shape, dim, input_has_dynamic_dim, shape[dim]
        )
        # prepare step
        stride, is_dynamic_slice = genSliceInput(
            is_dynamic_slice, stride, step, shape, dim, input_has_dynamic_dim, 1
        )
        if is_dynamic_slice:
            axes = get_value_or_create_const(np.array([dim]))
            return hbir.dynamic_slice(x, begin, limit, axes, stride)
        else:
            return hbir.slice(x, begin, limit, stride, output_type=otype)


slice()


class IntImplicit(AtenCvt):
    def __init__(self):
        super().__init__("IntImplicit")
        # aten::IntImplicit.Tensor(Tensor(a) self) -> Tensor(a)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
    ):
        return x


IntImplicit()


class Int(AtenCvt):
    def __init__(self):
        super().__init__("Int")
        # aten::IntImplicit.Tensor(Tensor(a) self) -> Tensor(a)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
    ):
        return x


Int()


class flip(AtenCvt):
    def __init__(self):
        super().__init__("flip")
        # aten::flip(Tensor self, int[] dims) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        dims: List[int],
    ):
        return hbir.flip(x, dims)


flip()


class split(AtenCvt):
    def __init__(self, variant):
        super().__init__(variant)
        # aten::split(Tensor(a) self, int[] split_sizes, int dim=0) -> Tensor(a)[]
        # aten::split_with_sizes(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> Tensor(a)[]
        # aten::split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otypes: List[mlir.Type],
        x: mlir.Value,
        split_sizes: List[int],
        dim: int,
    ):
        rank = adaptor.operands[0].type.rank
        shape = adaptor.operands[0].type.shape

        if isinstance(split_sizes, int):
            split_sizes = [split_sizes]

        if len(split_sizes) == 1:
            split_sizes = split_sizes * (shape[dim] // split_sizes[0])

        accumulated_begin = 0
        results = []
        for size in split_sizes:
            begin = [0] * rank
            begin[dim] = accumulated_begin

            accumulated_begin += size

            end = shape
            end[dim] = accumulated_begin

            step = [1] * rank

            results.append(hbir.slice(x, begin, end, step))

        return [results]


split("split")
split("split_with_sizes")
split("split.Tensor")


class chunk(AtenCvt):
    def __init__(self):
        super().__init__("chunk")
        # aten::chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otypes: List[mlir.Type],
        x: mlir.Value,
        chunks: int,
        dim: int,
    ):
        rank = adaptor.operands[0].type.rank
        shape = adaptor.operands[0].type.shape

        total_size = shape[dim]
        each_chunk = int(math.floor(total_size / chunks))

        accumulated_begin = 0
        results = []
        while accumulated_begin < total_size:
            begin = [0] * rank
            begin[dim] = accumulated_begin

            accumulated_begin += each_chunk
            end = shape
            end[dim] = accumulated_begin

            step = [1] * rank

            results.append(hbir.slice(x, begin, end, step))

        return [results]


chunk()


class upsample_nearest(AtenCvt):
    def __init__(self):
        super().__init__("upsample_nearest2d")
        # aten::upsample_nearest2d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
        # aten::upsample_nearest2d(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        output_size: List[int],
        scales: Union[None, float, List[float]],
        scale_w: Union[None, float] = None,
    ):
        input_shape = np.array(adaptor.operands[0].type.shape)
        if scales is None:
            assert output_size is not None
            ratio = [
                float(output_size[-2]) / input_shape[-2],
                float(output_size[-1]) / input_shape[-1],
            ]
        else:
            if scale_w is None:
                ratio = scales
            else:
                assert isinstance(scales, float)
                ratio = [scales, scale_w]

        step = [1.0 / ratio[0], 1.0 / ratio[1]]

        length_resized = np.array(
            [input_shape[-2] * ratio[0], input_shape[-1] * ratio[1]]
        )
        initial_offset = 0.5 / np.array(ratio) - 0.5
        for index, resized_num in enumerate(length_resized):
            if resized_num <= 1:
                step[index] = 0.0
                initial_offset[index] = 0.0

        # round_prefer_floor
        initial_offset -= np.array([np.finfo(np.float32).eps, np.finfo(np.float32).eps])

        x = nchw_to_nhwc(x)
        x = hbir.resize2d(
            x,
            step,
            mode="nearest",
            ratio=ratio,
            initialOffset=initial_offset,
            expansionMode="border",
        )
        return nhwc_to_nchw(x, otype)


upsample_nearest()


class upsample_bilinear(AtenCvt):
    def __init__(self):
        super().__init__("upsample_bilinear2d")

    # aten::upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
    # aten::upsample_bilinear2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        output_size: List[int],
        align_corners: bool,
        scales: Union[None, float, List[float]],
        scale_w: Union[None, float] = None,
    ):
        input_shape = adaptor.operands[0].type.shape
        if scales is None:
            ratio = [
                float(output_size[-2]) / input_shape[-2],
                float(output_size[-1]) / input_shape[-1],
            ]
        else:
            if scale_w is None:
                ratio = scales
            else:
                assert isinstance(scales, float)
                ratio = [scales, scale_w]

        step = [0.0, 0.0]
        offset = [0.0, 0.0]

        shape = input_shape[2:]
        length_resized = np.array([shape[0] * ratio[0], shape[1] * ratio[1]])
        for index, resized_num in enumerate(length_resized):
            if resized_num > 1:
                if align_corners:
                    step[index] = (shape[index] - 1) / float(resized_num - 1)
                else:
                    step[index] = 1.0 / ratio[index]
            else:
                step[index] = 0.0

            if align_corners:
                offset[index] = 0.0
            else:
                offset[index] = 0.5 / ratio[index] - 0.5

        # 1. When there is a ratio parameter, if the value of ratio is negative, you need to correct the value of initialOffset;
        # 2. When there is a size parameter, if the value of step is negative, you need to correct the value of initialOffset;
        rank = len(input_shape)
        numOfResizeAxis = 2
        for i in range(numOfResizeAxis):
            axis = rank - numOfResizeAxis - 1 + i
            if ratio[i] < 0:
                offset[i] = float(input_shape[axis])

        x = nchw_to_nhwc(x)
        x = hbir.resize2d(
            x,
            step,
            mode="bilinear",
            ratio=ratio,
            initialOffset=offset,
            expansionMode="border",
        )
        return nhwc_to_nchw(x, otype)


upsample_bilinear()


class upsample_bicubic(AtenCvt):
    def __init__(self):
        super().__init__("upsample_bicubic2d")

    # aten::upsample_bicubic2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
    # aten::upsample_bicubic2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        output_size: List[int],
        align_corners: bool,
        scales: Union[None, float, List[float]],
        scale_w: Union[None, float] = None,
    ):
        input_shape = adaptor.operands[0].type.shape
        if scales is None:
            ratio = [
                float(output_size[-2]) / input_shape[-2],
                float(output_size[-1]) / input_shape[-1],
            ]
        else:
            if scale_w is None:
                ratio = scales
            else:
                assert isinstance(scales, float)
                ratio = [scales, scale_w]

        step = [0.0, 0.0]
        offset = [0.0, 0.0]

        shape = input_shape[2:]
        length_resized = np.array([shape[0] * ratio[0], shape[1] * ratio[1]])
        for index, resized_num in enumerate(length_resized):
            if resized_num > 1:
                if align_corners:
                    step[index] = (shape[index] - 1) / float(resized_num - 1)
                else:
                    step[index] = 1.0 / ratio[index]
            else:
                step[index] = 0.0

            if align_corners:
                offset[index] = 0.0
            else:
                offset[index] = 0.5 / ratio[index] - 0.5

        # 1. When there is a ratio parameter, if the value of ratio is negative, you need to correct the value of initialOffset;
        # 2. When there is a size parameter, if the value of step is negative, you need to correct the value of initialOffset;
        rank = len(input_shape)
        numOfResizeAxis = 2
        for i in range(numOfResizeAxis):
            axis = rank - numOfResizeAxis - 1 + i
            if ratio[i] < 0:
                offset[i] = float(input_shape[axis])

        x = nchw_to_nhwc(x)
        x = hbir.resize2d(
            x,
            step,
            mode="bicubic",
            ratio=ratio,
            initialOffset=offset,
            expansionMode="border",
        )
        return nhwc_to_nchw(x, otype)


upsample_bicubic()


class grid_sampler(AtenCvt):
    def __init__(self):
        super().__init__("grid_sampler")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        grid: mlir.Value,
        mode: int,
        pad: int,
        align_corners: bool,
    ):
        mode = ["bilinear", "nearest", "cubic"][mode]
        pad = ["constant", "border"][pad]
        x = nchw_to_nhwc(x)
        x = hbir.grid_sample(x, grid, mode, pad, align_corners, padValue=0)
        return nhwc_to_nchw(x, otype)


grid_sampler()


class pad(AtenCvt):
    def __init__(self):
        super().__init__("pad")
        # aten::pad(Tensor self, int[] pad, str mode="constant", float? value=None) -> (Tensor)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        pad: List[int],
        mode: str,
        value: float,
    ):
        if value is None:
            value = 0.0
        pad_rank = len(pad) // 2
        unpad_rank = adaptor.operands[0].type.rank - pad_rank
        unpad_values = [0] * unpad_rank

        begin = [*unpad_values, *pad[0::2][::-1]]
        end = [*unpad_values, *pad[1::2][::-1]]

        return hbir.pad(x, begin, end, mode, padValue=value)


pad()


class constant_pad_nd(AtenCvt):
    def __init__(self):
        super().__init__("constant_pad_nd")
        # aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        pad: List[int],
        value: float,
    ):
        if value is None:
            value = 0.0
        pad_rank = len(pad) // 2
        unpad_rank = adaptor.operands[0].type.rank - pad_rank
        unpad_values = [0] * unpad_rank

        begin = [*unpad_values, *pad[0::2][::-1]]
        end = [*unpad_values, *pad[1::2][::-1]]

        return hbir.pad(x, begin, end, "constant", padValue=value)


constant_pad_nd()


class unary(AtenCvt):
    def __init__(self, name: str, target_name: Optional[str] = None):
        self.name = name
        self.target_name = target_name
        super().__init__(self.name)

    def emit_mlir_op(self, adaptor: NodeAdaptor, otype: mlir.Type, x: mlir.Value):
        if self.target_name is not None:
            return getattr(hbir, self.target_name)(x, output_type=otype)

        return getattr(hbir, self.name)(x, output_type=otype)


unary("relu")
unary("sigmoid")
unary("mish")
unary("silu", "swish")
unary("sin")
unary("cos")
unary("tan")
unary("sinh")
unary("cosh")
unary("tanh")
unary("asin")
unary("acos")
unary("atan")
unary("asinh")
unary("acosh")
unary("atanh")
unary("sqrt")
unary("rsqrt")
unary("exp")
unary("log")
unary("erf")
unary("abs")
unary("floor")
unary("logical_not")
unary("reciprocal")
unary("sign")
unary("ceil")


class relu6(AtenCvt):
    def __init__(self):
        super().__init__("relu6")

    def emit_mlir_op(self, adaptor: NodeAdaptor, otype: mlir.Type, x: mlir.Value):
        return hbir.clip(x, 0.0, 6.0, output_type=otype)


relu6()


class hardtanh(AtenCvt):
    def __init__(self):
        super().__init__("hardtanh")

    # aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        min: Union[int, float],
        max: Union[int, float],
    ):
        return hbir.clip(x, min, max, output_type=otype)


hardtanh()


class gelu(AtenCvt):
    def __init__(self):
        super().__init__("gelu")
        # aten::gelu(Tensor self, *, str approximate="none") -> (Tensor)

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, otype: mlir.Type, x: mlir.Value, approximate="none"
    ):
        return hbir.gelu(x, approximate=approximate, output_type=otype)


gelu()


class leaky_relu(AtenCvt):
    def __init__(self):
        super().__init__("leaky_relu")
        # aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, otype: mlir.Type, x: mlir.Value, slop: float
    ):
        return hbir.leaky_relu(x, slop, output_type=otype)


leaky_relu()


class prelu(AtenCvt):
    def __init__(self):
        super().__init__("prelu")
        # aten::prelu(input: Tensor, weight: Tensor) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        input: mlir.Value,
        weight: mlir.Value,
    ):
        # prelu is unary op, can broadcast weight to avoid insert transpose op
        slope_shape = adaptor.operands[1].type.shape
        new_slope_shape = [slope_shape[0]]
        input_rank = len(adaptor.operands[0].type.shape)
        while (input_rank > 2) and ((input_rank - 2) != 0):
            new_slope_shape.append(1)
            input_rank = input_rank - 1

        new_slope = hbir.reshape(weight, new_slope_shape)
        return hbir.prelu(input, new_slope)


prelu()


class elu(AtenCvt):
    def __init__(self):
        super().__init__("elu")
        # aten::elu(Tensor self, Scalar alpha=1.0, Scalar scale=1.0, Scalar input_scale=1.0) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        alpha=1.0,
        scale=1.0,
        input_scale=1.0,
    ):
        return hbir.elu(x, alpha, output_type=otype)


elu()


class softplus(AtenCvt):
    def __init__(self):
        super().__init__("softplus")
        # aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        beta=1.0,
        threshold=20.0,
    ):
        return hbir.softplus(x, float(beta), float(threshold), output_type=otype)


softplus()


def get_torch_dtype(v: int):
    return [
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.complex32,
        torch.complex64,
        torch.complex128,
        torch.bool,
        torch.qint8,
        torch.quint8,
        torch.qint32,
        torch.bfloat16,
    ][v]


class zeros_like(AtenCvt):
    def __init__(self):
        super().__init__("zeros_like")
        # aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        dtype,
        layout,
        device,
        pin_memory: bool,
        memory_format,
    ):
        return [
            torch.zeros(adaptor.operands[0].type.shape, dtype=get_torch_dtype(dtype))
        ]


zeros_like()


class ones_like(AtenCvt):
    def __init__(self):
        super().__init__("ones_like")
        # aten::ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        dtype,
        layout,
        device,
        pin_memory: bool,
        memory_format,
    ):
        return [
            torch.ones(adaptor.operands[0].type.shape, dtype=get_torch_dtype(dtype))
        ]


ones_like()


class embedding(AtenCvt):
    def __init__(self):
        super().__init__("embedding")
        # aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> (Tensor)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        weight: mlir.Value,
        indices: mlir.Value,
        padding_index: int,
        scale_grad_by_freq: bool,
        sparse: bool,
    ):
        if sparse is True:
            raise ValueError("sparse embedding not supported")
        return hbir.embedding(
            indices, weight, padding_index, scale_grad_by_freq, sparse
        )


embedding()


class where(AtenCvt):
    def __init__(self):
        super().__init__("where")
        # aten::where(Tensor condition, Tensor x, Tensor y) -> (Tensor)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        condition: mlir.Value,
        lhs: mlir.Value,
        rhs: mlir.Value,
    ):
        return hbir.where(condition, lhs, rhs, output_type=otype)


where()


class repeat(AtenCvt):
    def __init__(self):
        super().__init__("repeat")
        # aten::repeat(Tensor self, SymInt[] repeats) -> Tensor

    def emit_mlir_op(
        self, adaptor, otype: mlir.Type, x: mlir.Value, repeats: List[int]
    ):
        return hbir.tile(x, repeats, output_type=otype)


repeat()


class tile(AtenCvt):
    def __init__(self):
        super().__init__("tile")
        # aten::tile(Tensor self, SymInt[] repeats) -> Tensor

    def emit_mlir_op(
        self, adaptor, otype: mlir.Type, x: mlir.Value, repeats: List[int]
    ):
        return hbir.tile(x, repeats, output_type=otype)


tile()


class expand(AtenCvt):
    def __init__(self):
        super().__init__("expand")
        # aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)

    def emit_mlir_op(
        self, adaptor, otype: mlir.Type, x: mlir.Value, size: List[int], implicit: bool
    ):
        irank = adaptor.operands[0].type.rank
        orank = len(size)
        assert orank >= irank

        shape = adaptor.operands[0].type.shape
        if irank < orank:  # align rank
            shape = [1] * (orank - irank) + shape
            x = hbir.reshape(x, shape)

        repeats = np.array(size) // np.array(shape)
        repeats[repeats == -1] = 1
        return hbir.tile(x, repeats, output_type=otype)


expand()


class gather(AtenCvt):
    def __init__(self):
        super().__init__("gather")
        # aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor

    def emit_mlir_op(
        self,
        adaptor,
        otype: mlir.Type,
        x: mlir.Value,
        dim: int,
        index: mlir.Value,
        sparse_grad: bool,
    ):
        return hbir.gather_elements(x, index, dim)


gather()


class masked_select(AtenCvt):
    def __init__(self):
        super().__init__("masked_select")
        # aten::masked_select(input: Tensor, mask: Tensor, *, out: Optional[Tensor]=None) -> Tensor

    def emit_mlir_op(self, adaptor, otype: mlir.Type, x: mlir.Value, mask: mlir.Value):
        return hbir.masked_select(x, mask)


masked_select()


class minmum(AtenCvt):
    def __init__(self):
        super().__init__("minimum")
        # aten::minimum(Tensor self, Tensor other) -> Tensor

    def emit_mlir_op(self, adaptor, otype: mlir.Type, x: mlir.Value, other: mlir.Value):
        return hbir.min(x, other)


minmum()


class maximum(AtenCvt):
    def __init__(self):
        super().__init__("maximum")
        # aten::maximum(Tensor self, Tensor other) -> Tensor

    def emit_mlir_op(self, adaptor, otype: mlir.Type, x: mlir.Value, other: mlir.Value):
        return hbir.max(x, other)


maximum()


class clamp(AtenCvt):
    def __init__(self):
        super().__init__("clamp")
        # aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor

    def emit_mlir_op(self, adaptor, otype: mlir.Type, x: mlir.Value, min_v, max_v):
        if min_v is None:
            return hbir.min(x, max_v, output_type=otype)
        elif max_v is None:
            return hbir.max(x, min_v, output_type=otype)
        return hbir.clip(x, min=min_v, max=max_v, output_type=otype)


clamp()


class lt(AtenCvt):
    def __init__(self):
        super().__init__("lt")
        # aten::lt.Scalar(Tensor self, Scalar other) -> Tensor

    def emit_mlir_op(self, adaptor, otype: mlir.Type, x: mlir.Value, other):
        if isinstance(other, (int, float)):
            shape = adaptor.operands[0].type.shape
            t = np.full(shape, other)
            c = get_value_or_create_const(t)
            return hbir.less(x, c, output_type=otype)
        else:
            return hbir.less(x, other, output_type=otype)


lt()


class le(AtenCvt):
    def __init__(self):
        super().__init__("le")
        # aten::le.Scalar(Tensor self, Scalar other) -> Tensor

    def emit_mlir_op(self, adaptor, otype: mlir.Type, x: mlir.Value, other):
        if isinstance(other, (int, float)):
            shape = adaptor.operands[0].type.shape
            t = np.full(shape, other)
            c = get_value_or_create_const(t)
            return hbir.less_equal(x, c, output_type=otype)
        else:
            return hbir.less_equal(x, other, output_type=otype)


le()


class gt(AtenCvt):
    def __init__(self):
        super().__init__("gt")
        # aten::gt.Scalar(Tensor self, Scalar other) -> Tensor

    def emit_mlir_op(self, adaptor, otype: mlir.Type, x: mlir.Value, other):
        if isinstance(other, (int, float)):
            shape = adaptor.operands[0].type.shape
            t = np.full(shape, other)
            c = get_value_or_create_const(t)
            return hbir.greater(x, c, output_type=otype)
        else:
            return hbir.greater(x, other, output_type=otype)


gt()


class ge(AtenCvt):
    def __init__(self):
        super().__init__("ge")
        # aten::ge.Scalar(Tensor self, Scalar other) -> Tensor

    def emit_mlir_op(self, adaptor, otype: mlir.Type, x: mlir.Value, other):
        if isinstance(other, (int, float)):
            shape = adaptor.operands[0].type.shape
            t = np.full(shape, other)
            c = get_value_or_create_const(t)
            return hbir.greater_equal(x, c, output_type=otype)
        else:
            return hbir.greater_equal(x, other, output_type=otype)


ge()


class NonZero(AtenCvt):
    def __init__(self):
        super().__init__("nonzero")
        # aten::nonzero(Tensor self)->Tensor

    def emit_mlir_op(self, adaptor: NodeAdaptor, otype: mlir.Type, x: mlir.Value):
        y = hbir.nonzero(x)
        return y


NonZero()


class Scatter(AtenCvt):
    def __init__(self):
        super().__init__("scatter")
        # aten::scatter(Tensor data, Tensor indices, Tensor updates, int axis, string reduce) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        data: mlir.Value,
        dim: int,
        indices: mlir.Value,
        updates: mlir.Value,
        reduce="none",
    ):
        return hbir.scatter_elements(data, indices, updates, dim, reduce)


Scatter()


class slice_scatter(AtenCvt):
    def __init__(self):
        super().__init__("slice_scatter")
        # aten::slice_scatter(Tensor data, Tensor src, int dim, int start, int end, int step) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        data: mlir.Value,
        src: mlir.Value,
        dim: int,
        start: int,
        end: int,
        step: int,
    ):
        return hbir.slice_scatter(data, src, dim, start, end, step)


slice_scatter()


class scatter_add(AtenCvt):
    def __init__(self):
        super().__init__("scatter_add")
        # aten::scatter_add(Tensor data, int dim, Tensor indices, Tensor updates) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        data: mlir.Value,
        dim: int,
        indices: mlir.Value,
        updates: mlir.Value,
    ):
        return hbir.scatter_elements(data, indices, updates, dim, "add")


scatter_add()


class scatter_reduce(AtenCvt):
    def __init__(self):
        super().__init__("scatter_reduce")
        # aten::scatter_reduce(Tensor data, int axis, Tensor indices, Tensor updates, string reduce, bool include_self) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        data: mlir.Value,
        dim: int,
        indices: mlir.Value,
        updates: mlir.Value,
        reduce="none",
        include_self: bool = True,
    ):
        # torch and onnx corresponding relationship:
        #     reduce_mode = {  # convert torch string name to onnx string name
        #     "mean": "none",  # 'mean' doesn't support in ONNX 1.14 definition
        #     "sum": "add",
        #     "prod": "mul",
        #     "amin": "min",
        #     "amax": "max",
        # }
        if reduce == "mean":
            return hbir.scatter_elements(data, indices, updates, dim, "mean")
        elif reduce == "sum":
            return hbir.scatter_elements(data, indices, updates, dim, "add")
        elif reduce == "prod":
            return hbir.scatter_elements(data, indices, updates, dim, "mul")
        elif reduce == "amin":
            return hbir.scatter_elements(data, indices, updates, dim, "min")
        elif reduce == "amax":
            return hbir.scatter_elements(data, indices, updates, dim, "max")
        else:
            print(
                "Reduce argument must be either sum, prod, mean, amax or amin, got ",
                reduce,
            )


scatter_reduce()


class Neg(AtenCvt):
    def __init__(self):
        super().__init__("neg")
        # aten::neg(Tensor self) -> (Tensor)

    def emit_mlir_op(self, adaptor: NodeAdaptor, otype: mlir.Type, x: mlir.Value):
        return hbir.neg(x)


Neg()


class Equal(AtenCvt):
    def __init__(self):
        super().__init__("eq")
        # aten::eq(Tensor self, Tensor other) -> Tensor

    def emit_mlir_op(self, adaptor, otype: mlir.Type, x: mlir.Value, other):
        if isinstance(other, (int, float)):
            shape = adaptor.operands[0].type.shape
            t = np.full(shape, other)
            c = get_value_or_create_const(t)
            return hbir.equal(x, c, output_type=otype)
        else:
            return hbir.equal(x, other, output_type=otype)


Equal()


class Triu(AtenCvt):
    def __init__(self):
        super().__init__("triu")
        # aten::triu(input, diagonal=0, *, out=None) -> Tensor

    def emit_mlir_op(self, adaptor, otype: mlir.Type, x: mlir.Value, diagonal: int = 0):
        shape = adaptor.operands[0].type.shape
        matrix = np.triu(np.ones(shape), k=diagonal)
        return hbir.mul(x, matrix)


Triu()


class Tril(AtenCvt):
    def __init__(self):
        super().__init__("tril")
        # aten::tril(input, diagonal=0, *, out=None) -> Tensor

    def emit_mlir_op(self, adaptor, otype: mlir.Type, x: mlir.Value, diagonal: int = 0):
        shape = adaptor.operands[0].type.shape
        matrix = np.tril(np.ones(shape), k=diagonal)
        return hbir.mul(x, matrix)


Tril()


class IndexSelect(AtenCvt):
    def __init__(self):
        super().__init__("index_select")
        # aten::index_select(input, dim, index, *, out=None) -> Tensor

    def emit_mlir_op(
        self, adaptor, otype: mlir.Type, x: mlir.Value, dim: int, index: mlir.Value
    ):
        return hbir.index(x, index, dim)


IndexSelect()


class CumSum(AtenCvt):
    def __init__(self):
        super().__init__("cumsum")
        # aten::cumsum(input, dim, *, dtype=None, out=None) -> Tensor

    def emit_mlir_op(
        self, adaptor, otype: mlir.Type, x: mlir.Value, dim: int, dtype: None
    ):
        return hbir.cumsum(x, dim)


CumSum()


class NanToNum(AtenCvt):
    def __init__(self):
        super().__init__("nan_to_num")
        # aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        nan: Optional[float] = 0.0,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
    ):
        return hbir.nan_to_num(
            x, nan=nan, posinf=posinf, neginf=neginf, output_type=otype
        )


NanToNum()


class Round(AtenCvt):
    def __init__(self):
        super().__init__("round")
        # aten::round.decimals(Tensor self, *, int decimals) -> Tensor

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        decimals=0,
    ):
        return hbir.round(x, decimals, output_type=otype)


Round()


class DeformConv(TorchvisionCvt):
    def __init__(self):
        # torchvision::deform_conv2d(Tensor, Tensor, Tensor, Optional[Tensor], Tuple[int, int], Tuple[int, int], Tuple[int, int]) -> Tensor
        super().__init__("deform_conv2d")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        w: mlir.Value,
        offset: mlir.Value,
        mask: Optional[mlir.Value] = None,
        b: Optional[mlir.Value] = None,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        dil_h=1,
        dil_w=1,
        group=1,
        offset_group=1,
        use_mask=True,
    ):
        # input trans: nchw->nhwc
        # output trans: nhwc->nchw
        mask_shape = offset.type.shape
        mask_shape[1] = mask_shape[1] // 2
        if use_mask is False:
            mask = get_value_or_create_const(np.ones(mask_shape))
        x = nchw_to_nhwc(x)
        w = nchw_to_nhwc(w)
        offset = nchw_to_nhwc(offset)
        strides = (stride_h, stride_w)
        mask = nchw_to_nhwc(mask)
        pads = (pad_h, pad_w, pad_h, pad_w)
        dilations = (dil_h, dil_w)
        x = hbir.deform_conv2d(
            x,
            w,
            offset,
            mask,
            strides,
            pads,
            dilations,
            group,
            offset_group,
            use_mask,
            bias=b,
        )
        return nhwc_to_nchw(x, otype)


DeformConv()


class NMS(TorchvisionCvt):
    def __init__(self):
        # torchvision::nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor
        super().__init__("nms")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        boxes: mlir.Value,
        scores: mlir.Value,
        iou_threshold=0.0,
    ):
        boxes_num = adaptor.operands[0].type.shape[0]
        boxes_shape = np.concatenate(([1], np.array(adaptor.operands[0].type.shape)))
        boxes = hbir.reshape(boxes, boxes_shape)

        scores_shape = np.concatenate(
            ([1, 1], np.array(adaptor.operands[1].type.shape))
        )
        scores = hbir.reshape(scores, scores_shape)

        nms_out = hbir.nms(
            boxes=boxes,
            scores=scores,
            mode="xyxy",
            iouThreshold=float(iou_threshold),
            scoreThreshold=float(0),
            maxOutputBoxesPerClass=boxes_num,
        )
        return hbir.select(nms_out, 1, 2)


NMS()
