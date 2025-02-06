import logging
import math
from distutils.version import LooseVersion
from typing import List, Optional, Tuple, Union

import torch
from hbdk4.compiler import ir
from hbdk4.compiler.ops import hbir
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torch.nn.modules.utils import _pair, _single, _triple
from torch.types import Number, Sequence, _bool, _int, _size

# from torchvision import ops
from horizon_plugin_pytorch._torchvision_wrapper import ops
from .export_hbir import (
    Exporter,
    FuncConverterBase,
    JitTensor,
    ModuleConverterBase,
)
from .utils import (
    LayoutConverter,
    check_inplace,
    check_training,
    get_hbdk4_version,
    get_hbir_tensor_type,
)

logger = logging.getLogger(__name__)

__all__ = []

const_ops = [
    torch.zeros_like,
    torch.ones_like,
    torch.full_like,
    torch.rand_like,
    Tensor.new_zeros,
    Tensor.new_ones,
    Tensor.new_empty,
    Tensor.new_full,
]
transparent_ops = [
    Tensor.contiguous,
    Tensor.detach,
    Tensor.clone,
]


def norm_pads(padding):
    return [padding[0], padding[1], padding[0], padding[1]]


# A place to hold ops that supported by Plugin but not has hbir
@JitTensor.register_converter(
    torch.repeat_interleave,
    Tensor.repeat_interleave,
    Tensor.nonzero,
)
class UnsupportedConverter(FuncConverterBase):
    def convert(self, *args, **kwargs):
        raise RuntimeError("unsupported by hbir")


@JitTensor.register_converter(Tensor.new_tensor)
class NewTensorConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output: Tensor, *args, **kwargs):
        assert not isinstance(
            args[1], Tensor
        ), "new_tensor arg must be array_like constant values, like [1., 2.]."
        hbir_output = JitTensor.gather_hbir(output)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(*const_ops)
class ConstOpConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output: Tensor, *args, **kwargs):
        hbir_output = JitTensor.gather_hbir(output)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(*transparent_ops)
class TransparentOpConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, input: ir.Value, *args, **kwargs):
        return input


@JitTensor.register_converter(Tensor.to, Tensor.type)
class ToConverter(FuncConverterBase):
    @classmethod
    def to_dtype(cls, input, dtype, input_shape):
        return hbir.cast_type(
            input, output_type=get_hbir_tensor_type(dtype, input_shape)
        )

    @classmethod
    def to_device(cls, input, device):
        return input

    @classmethod
    def convert(cls, output, input, *args, **kwargs):
        hbir_input = JitTensor.gather_hbir(input)
        input_base = JitTensor.get_base(input)

        if (
            output.as_subclass(Tensor).dtype
            != input_base.as_subclass(Tensor).dtype
        ):
            hbir_input = hbir.cast_type(
                hbir_input,
                output_type=get_hbir_tensor_type(
                    output.as_subclass(Tensor).dtype, list(output.shape)
                ),
            )

        return JitTensor.attach_hbir_to_tensor(output, hbir_input)


@JitTensor.register_converter(Tensor.cpu)
class ToCpuConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, input: ir.Value):
        return ToConverter.convert(output, input, torch.device("cpu"))


@JitTensor.register_converter(Tensor.cuda)
class ToCudaConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, input: ir.Value):
        return ToConverter.convert(output, input, torch.device("cuda"))


cast_func_mapping = {
    Tensor.float: torch.float,
    Tensor.double: torch.double,
    Tensor.half: torch.half,
    Tensor.bfloat16: torch.bfloat16,
    Tensor.char: torch.int8,
    Tensor.short: torch.short,
    Tensor.int: torch.int,
    Tensor.long: torch.long,
    Tensor.bool: torch.bool,
}


def get_cast_converter(dtype):
    class CastConverter(FuncConverterBase):
        @classmethod
        def convert(cls, output, input: ir.Value):
            return ToConverter.convert(output, input, dtype)

    return CastConverter


def register_cast_converter():
    for func, dtype in cast_func_mapping.items():
        JitTensor.register_converter(func)(get_cast_converter(dtype))


register_cast_converter()


def register_direct_mapping():
    direct_mapping = {
        (torch.abs, Tensor.abs): hbir.abs,
        (torch.acos, Tensor.acos, torch.arccos, Tensor.arccos): hbir.acos,
        (torch.acosh, Tensor.acosh, torch.arccosh, Tensor.arccosh): hbir.acosh,
        (torch.asin, Tensor.asin, torch.arcsin, Tensor.arcsin): hbir.asin,
        (torch.asinh, Tensor.asinh, torch.arcsinh, Tensor.arcsinh): hbir.asinh,
        (torch.atan, Tensor.atan, torch.arctan, Tensor.arctan): hbir.atan,
        (torch.atanh, Tensor.atanh, torch.arctanh, Tensor.arctanh): hbir.atanh,
        (torch.ceil, Tensor.ceil): hbir.ceil,
        (torch.cos, Tensor.cos): hbir.cos,
        (torch.cosh, Tensor.cosh): hbir.cosh,
        (torch.erf, Tensor.erf): hbir.erf,
        (torch.exp, Tensor.exp): hbir.exp,
        (torch.floor, Tensor.floor): hbir.floor,
        (torch.log, Tensor.log): hbir.log,
        (torch.maximum, Tensor.maximum): hbir.max,
        (torch.minimum, Tensor.minimum): hbir.min,
        (torch.neg, torch.negative, Tensor.negative, Tensor.neg): hbir.neg,
        (torch.reciprocal, Tensor.reciprocal): hbir.reciprocal,
        (torch.rsqrt, Tensor.rsqrt): hbir.rsqrt,
        (torch.sigmoid, Tensor.sigmoid): hbir.sigmoid,
        (torch.sin, Tensor.sin): hbir.sin,
        (torch.sinh, Tensor.sinh): hbir.sinh,
        (torch.sqrt, Tensor.sqrt): hbir.sqrt,
        (torch.tan, Tensor.tan): hbir.tan,
        (torch.tanh, Tensor.tanh): hbir.tanh,
        (torch.sign, Tensor.sign): hbir.sign,
    }

    def do(torch_funcs, hbir_func):
        # def converter in a func to hold things in local scope
        class DirectMappingConverter(FuncConverterBase):
            with_output_type = True

            @classmethod
            def convert_with_hbir(cls, output_type, *args, **kwargs):
                return hbir_func(*args, **kwargs, output_type=output_type)

        JitTensor.register_converter(*torch_funcs)(DirectMappingConverter)

    for torch_funcs, hbir_func in direct_mapping.items():
        do(torch_funcs, hbir_func)


register_direct_mapping()


def register_compare():
    compare_op_mapping = {
        (torch.eq, Tensor.eq): hbir.equal,
        (torch.less, torch.lt, Tensor.less, Tensor.lt): hbir.less,
        (
            torch.less_equal,
            torch.le,
            Tensor.less_equal,
            Tensor.le,
        ): hbir.less_equal,
        (torch.greater, torch.gt, Tensor.greater, Tensor.gt): hbir.greater,
        (
            torch.greater_equal,
            torch.ge,
            Tensor.greater_equal,
            Tensor.ge,
        ): hbir.greater_equal,
        (
            torch.not_equal,
            torch.ne,
            Tensor.not_equal,
            Tensor.ne,
        ): hbir.not_equal,
    }

    def do(torch_funcs, hbir_func):
        # def converter in a func to hold things in local scope
        class CompareConverter(FuncConverterBase):
            @classmethod
            def convert(cls, output, *args, **kwargs):
                hbir_args = JitTensor.gather_hbir(args)
                hbir_kwargs = JitTensor.gather_hbir(kwargs)
                hbir_output = hbir_func(
                    *hbir_args,
                    **hbir_kwargs,
                    output_type=get_hbir_tensor_type(torch.bool, output.shape),
                )

                return JitTensor.attach_hbir_to_tensor(output, hbir_output)

        JitTensor.register_converter(*torch_funcs)(CompareConverter)

    for torch_funcs, hbir_func in compare_op_mapping.items():
        do(torch_funcs, hbir_func)


register_compare()


@JitTensor.register_converter(Tensor.copy_)
class TensorCopyConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        self: JitTensor,
        src: JitTensor,
        non_blocking: bool = False,
    ):
        self.hbir_node = src.hbir_node
        if self.subclass_from is not None:
            self.subclass_from.hbir_node = src.hbir_node

        return output


@JitTensor.register_converter(F.adaptive_avg_pool1d)
class AdaptiveAvgPool1dConverter(FuncConverterBase):
    @classmethod
    def get_single_dim(cls, input_size, output_size):
        if input_size % output_size != 0:
            raise ValueError(
                "Input size must be divided equally by output size, "
                "but receive input_size {} and output_size {}".format(
                    input_size, output_size
                )
            )
        stride = math.floor(input_size / output_size)
        kernel_size = input_size - (output_size - 1) * stride
        return stride, kernel_size

    @classmethod
    def get_params(cls, input, output_size):
        stridew, kernelw = cls.get_single_dim(input.size(-1), output_size)
        return (kernelw,), (stridew,)

    @classmethod
    def convert(cls, output: Tensor, input, output_size):
        kernel, stride = cls.get_params(input, output_size)
        rank = output.ndim

        hbir_input = JitTensor.gather_hbir(input)

        layout_converter = LayoutConverter(force_2d=len(stride) == 2)

        if rank == 2:
            hbir_input = hbir.transpose(hbir_input, (1, 0))
        else:
            hbir_input = layout_converter.nchw_to_nhwc(hbir_input)
        hbir_output = hbir.avg_pool(
            hbir_input, kernel, stride, [0, 0] * len(stride), [1] * len(stride)
        )
        if rank == 2:
            hbir_output = hbir.transpose(hbir_output, (1, 0))
        else:
            hbir_output = layout_converter.nhwc_to_nchw(hbir_output)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(F.adaptive_avg_pool2d)
class AdaptiveAvgPool2dConverter(AdaptiveAvgPool1dConverter):
    @classmethod
    def get_params(cls, input, output_size):
        output_size = _pair(output_size)
        stridew, kernelw = cls.get_single_dim(
            input.size(-1), output_size[1] or input.size(-1)
        )
        strideh, kernelh = cls.get_single_dim(
            input.size(-2), output_size[0] or input.size(-2)
        )

        return (kernelh, kernelw), (strideh, stridew)


@JitTensor.register_converter(
    F.adaptive_max_pool1d, F.adaptive_max_pool1d_with_indices
)
class AdaptiveMaxPool1dConverter(AdaptiveAvgPool1dConverter):
    @classmethod
    def convert_adaptive_max_pool(
        cls, input, output_size, return_indices=False
    ):
        if return_indices:
            msg = "adaptive_max_pool does not support `return_indices` = True"
            logger.error(msg)
            raise ValueError(msg)

        kernel, stride = cls.get_params(input, output_size)

        hbir_input = JitTensor.gather_hbir(input)

        layout_converter = LayoutConverter(force_2d=len(kernel) == 2)

        hbir_input = layout_converter.nchw_to_nhwc(hbir_input)
        hbir_output = hbir.max_pool(
            hbir_input, kernel, stride, [0, 0] * len(kernel), [1] * len(kernel)
        )
        hbir_output = layout_converter.nhwc_to_nchw(hbir_output)

        return hbir_output

    @classmethod
    def convert(cls, output, *args, **kwargs):
        hbir_output = cls.convert_adaptive_max_pool(*args, **kwargs)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(
    F.adaptive_max_pool2d, F.adaptive_max_pool2d_with_indices
)
class AdaptiveMaxPool2dConverter(AdaptiveMaxPool1dConverter):
    @classmethod
    def get_params(cls, input, output_size):
        output_size = _pair(output_size)
        stridew, kernelw = cls.get_single_dim(
            input.size(-1), output_size[1] or input.size(-1)
        )
        strideh, kernelh = cls.get_single_dim(
            input.size(-2), output_size[0] or input.size(-2)
        )

        return (kernelh, kernelw), (strideh, stridew)


@JitTensor.register_converter(torch.all, Tensor.all)
class ReduceAllConverter(FuncConverterBase):
    @classmethod
    def convert_all_reduce_all(cls, input, output_type):
        hbir_input = JitTensor.gather_hbir(input)
        return hbir.reduce_all(
            input=hbir_input,
            dims=range(input.ndim),
            keepDim=False,
            output_type=output_type,
        )

    @classmethod
    def convert_reduce_all(cls, input, dim, keepdim=False, output_type=None):
        hbir_input = JitTensor.gather_hbir(input)
        if isinstance(dim, int):
            dim = [dim]
        return hbir.reduce_all(
            input=hbir_input,
            dims=dim,
            keepDim=keepdim,
            output_type=output_type,
        )

    @classmethod
    def convert(cls, output: Tensor, input: Tensor, *args, **kwargs):
        output_type = get_hbir_tensor_type(output.as_subclass(Tensor).dtype)
        if len(args) + len(kwargs) == 0:
            hbir_output = cls.convert_all_reduce_all(input, output_type)
        else:
            hbir_output = cls.convert_reduce_all(
                input,
                *args,
                **kwargs,
                output_type=output_type,
            )
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.add, Tensor.add)
class AddConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls, input: ir.Value, other: ir.Value, *, alpha: Optional[Number] = 1
    ):
        if alpha != 1:
            other = hbir.mul(other, alpha)

        return hbir.add(input, other)


@JitTensor.register_converter(Tensor.add_)
class AddInplacedConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        other: JitTensor,
        *,
        alpha: Optional[Number] = 1,
    ):
        input_hbir, other_hbir = JitTensor.gather_hbir((input, other))

        if alpha != 1:
            other_hbir = hbir.mul(other_hbir, alpha)

        hbir_output = hbir.add(input_hbir, other_hbir)

        JitTensor.attach_inplaced_output(input, hbir_output)

        return input


@JitTensor.register_converter(F.affine_grid)
class AffineGridConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        theta: ir.Value,
        size: List[int],
        align_corners: Optional[bool] = None,
    ):
        if len(size) == 4:
            N, _, H, W = size  # noqa: N806

            x = torch.linspace(-1, 1, W).unsqueeze(0).expand(H, W)
            y = torch.linspace(-1, 1, H).unsqueeze(-1).expand(H, W)
            ones = torch.full((H, W), 1.0)

            if not align_corners:
                x = x * ((W - 1) / W)
                y = y * ((H - 1) / H)

            base_grid = (
                torch.stack([x, y, ones], dim=-1)
                .unsqueeze(0)
                .expand(N, H, W, 3)
                .reshape(N, H * W, 3)
            )

            base_grid = JitTensor.gather_hbir(base_grid)

            # (N, 2, 3) -> (N, 3, 2)
            theta = hbir.transpose(theta, (0, 2, 1))

            grid = hbir.reshape(
                hbir.matmul(
                    base_grid,  # (N, H * W, 3)
                    theta,  # (N, 3, 2)
                ),
                (N, H, W, 2),
            )
        elif len(size) == 5:
            N, _, D, H, W = size  # noqa: N806

            x = (
                torch.linspace(-1, 1, W)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(D, H, W)
            )
            y = (
                torch.linspace(-1, 1, H)
                .unsqueeze(-1)
                .unsqueeze(0)
                .expand(D, H, W)
            )
            z = (
                torch.linspace(-1, 1, D)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(D, H, W)
            )

            ones = torch.full((D, H, W), 1.0)

            if not align_corners:
                x = x * ((W - 1) / W)
                y = y * ((H - 1) / H)
                z = z * ((D - 1) / D)

            base_grid = (
                torch.stack([x, y, z, ones], dim=-1)
                .unsqueeze(0)
                .expand(N, D, H, W, 4)
                .reshape(N, D * H * W, 4)
            )

            base_grid = JitTensor.gather_hbir(base_grid)

            # (N, 3, 4) -> (N, 4, 3)
            theta = hbir.transpose(theta, (0, 2, 1))

            grid = hbir.reshape(
                hbir.matmul(
                    base_grid,  # (N, D * H * W, 4)
                    theta,  # (N, 4, 3)
                ),
                (N, D, H, W, 3),
            )
        else:
            raise NotImplementedError(
                "affine_grid only supports 4D and 5D sizes, "
                "for 2D and 3D affine transforms, respectively. "
                "Got size {}.".format(size)
            )

        return grid


@JitTensor.register_converter(torch.argmax, Tensor.argmax)
class ArgmaxConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output: Tensor,
        input: JitTensor,
        dim: Optional[_int] = None,
        keepdim: _bool = False,
    ):
        hbir_input = JitTensor.gather_hbir(input)

        if dim is None:
            input_base = JitTensor.get_base(input)
            hbir_input = hbir.reshape(hbir_input, [input_base.numel()])
            dims = [0]
        else:
            dims = [dim]

        hbir_output = hbir.reduce_argmax(
            hbir_input,
            dims,
            keepDim=keepdim,
            output_type=ir.UnrankedTensorType.get(
                ir.IntegerType.get_signed(64)
            ),
        )
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.argmin, Tensor.argmin)
class ArgminConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output: Tensor,
        input: JitTensor,
        dim: Optional[_int] = None,
        keepdim: _bool = False,
    ):
        if dim is None:
            dims = list(range(input.ndim))
        else:
            dims = [dim]

        hbir_input = JitTensor.gather_hbir(input)
        hbir_output = hbir.reduce_argmin(
            hbir_input,
            dims,
            keepDim=keepdim,
            output_type=ir.UnrankedTensorType.get(
                ir.IntegerType.get_signed(64)
            ),
        )
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(F.avg_pool2d)
class AvgPool2dConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        kernel_size: Union[_int, _size],
        stride: Optional[Union[_int, _size]] = None,
        padding: Union[_int, _size] = 0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        kernel_size = _pair(kernel_size)
        stride = kernel_size if stride is None else _pair(stride)
        padding = _pair(padding)

        if count_include_pad is False and padding != [0, 0]:
            msg = (
                "AvgPool2d does not support `count_include_pad` = False"
                " when output padding is enabled"
            )
            logger.error(msg)
            raise ValueError(msg)

        if divisor_override is not None:
            msg = (
                "AvgPool2d only supports `divisor_override` = None, "
                "but receive {}".format(divisor_override)
            )
            logger.error(msg)
            raise ValueError(msg)

        layout_converter = LayoutConverter(force_2d=True)
        input = layout_converter.nchw_to_nhwc(input)
        output = hbir.avg_pool(
            input,
            kernel_size,
            stride,
            norm_pads(padding),
            (1, 1),
            ceilMode=ceil_mode,
        )
        output = layout_converter.nhwc_to_nchw(output)
        return output


@JitTensor.register_converter(F.batch_norm)
class BatchNormConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        running_mean: Optional[Tensor],
        running_var: Optional[Tensor],
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ):
        if running_mean is None or running_var is None:
            msg = "Batchnorm do not support get statistics at running time"
            logger.error(msg)
            raise ValueError(msg)

        check_training(training, "batchnorm")

        if ir.ShapedType(input.type).rank == 2:
            return hbir.batchnorm(
                input, running_mean, running_var, eps, weight=weight, bias=bias
            )

        layout_converter = LayoutConverter()

        input = layout_converter.nchw_to_nhwc(input)
        output = hbir.batchnorm(
            input, running_mean, running_var, eps, weight=weight, bias=bias
        )
        output = layout_converter.nhwc_to_nchw(output)
        return output


@JitTensor.register_converter(torch.cat, torch.concat, torch.concatenate)
class ConcatConverter(FuncConverterBase):
    with_output_type = True

    @classmethod
    def convert_with_hbir(cls, output_type, tensors, dim=0):
        return hbir.concat(tensors, dim, output_type=output_type)


@JitTensor.register_converter(
    torch.clamp, torch.clip, Tensor.clamp, Tensor.clip
)
class ClampConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output: Tensor, input: JitTensor, min=None, max=None):
        hbir_input, hbir_min, hbir_max = JitTensor.gather_hbir(
            (input, min, max)
        )

        output_type = get_hbir_tensor_type(output.as_subclass(Tensor).dtype)
        if isinstance(min, Tensor):
            hbir_input = hbir.max(
                hbir_input, hbir_min, output_type=output_type
            )
            min = None
        if isinstance(max, Tensor):
            hbir_input = hbir.min(
                hbir_input, hbir_max, output_type=output_type
            )
            max = None
        if min is not None or max is not None:
            hbir_input = hbir.clip(
                hbir_input,
                float(min) if min is not None else float("-inf"),
                float(max) if max is not None else float("inf"),
                output_type=output_type,
            )
        return JitTensor.attach_hbir_to_tensor(output, hbir_input)


@JitTensor.register_converter(torch.clamp_min, Tensor.clamp_min)
class ClampMinConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, input: JitTensor, min):
        return ClampConverter.convert(output, input, min, None)


@JitTensor.register_converter(torch.clamp_max, Tensor.clamp_max)
class ClampMaxConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, input: JitTensor, max):
        return ClampConverter.convert(output, input, None, max)


@JitTensor.register_converter(F.conv1d)
class Conv1dConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        weight: JitTensor,
        bias: Optional[ir.Value] = None,
        stride: Union[_int, _size] = 1,
        padding: Union[_int, _size, str] = 0,
        dilation: Union[_int, _size] = 1,
        groups: int = 1,
    ):
        stride = _single(stride) + (1,)
        padding = (
            padding if isinstance(padding, str) else _single(padding) + (0,)
        )
        dilation = _single(dilation) + (1,)

        if padding == "same":
            kernel_sizes = weight.shape[-1:] + (1,)
            total_padding = [
                d * (k - 1) for d, k in zip(dilation, kernel_sizes)
            ]
            left_padding = [p // 2 for p in total_padding]
            padding = (
                left_padding[0],
                left_padding[1],
                total_padding[0] - left_padding[0],
                total_padding[1] - left_padding[1],
            )
        elif padding == "valid":
            padding = (0, 0, 0, 0)
        else:
            padding = Conv2dConverter.norm_conv_pads(padding)

        return super().convert(
            output,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            tuple(input.shape),
            tuple(weight.shape),
            tuple(output.shape),
        )

    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        weight: ir.Value,
        bias: Optional[ir.Value] = None,
        stride: Union[_int, _size] = 1,
        padding: Union[_int, _size] = 0,
        dilation: Union[_int, _size] = 1,
        groups: int = 1,
        input_shape: Optional[tuple] = None,
        weight_shape: Optional[tuple] = None,
        output_shape: Optional[tuple] = None,
    ):
        weight = hbir.reshape(weight, weight_shape + (1,))
        output = hbir.reshape(input, input_shape + (1,))
        output = Conv2dConverter.convert_with_hbir(
            output,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )
        output = hbir.reshape(output, output_shape)
        return output


@JitTensor.register_converter(F.conv2d)
class Conv2dConverter(FuncConverterBase):
    @classmethod
    def norm_conv_pads(cls, padding, output_padding=(0, 0)):
        return [
            padding[0],
            padding[1],
            padding[0] - output_padding[0],
            padding[1] - output_padding[1],
        ]

    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        weight: JitTensor,
        bias: Optional[ir.Value] = None,
        stride: Union[_int, _size] = 1,
        padding: Union[_int, _size, str] = 0,
        dilation: Union[_int, _size] = 1,
        groups: int = 1,
    ):
        stride = _pair(stride)
        padding = padding if isinstance(padding, str) else _pair(padding)
        dilation = _pair(dilation)
        if padding == "same":
            kernel_sizes = weight.shape[-2:]
            total_padding = [
                d * (k - 1) for d, k in zip(dilation, kernel_sizes)
            ]
            left_padding = [p // 2 for p in total_padding]
            padding = (
                left_padding[0],
                left_padding[1],
                total_padding[0] - left_padding[0],
                total_padding[1] - left_padding[1],
            )
        elif padding == "valid":
            padding = (0, 0, 0, 0)
        else:
            padding = cls.norm_conv_pads(padding)

        return super().convert(
            output, input, weight, bias, stride, padding, dilation, groups
        )

    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        weight: ir.Value,
        bias: Optional[ir.Value] = None,
        stride: Union[_int, _size] = 1,
        padding: Union[_int, _size] = 0,
        dilation: Union[_int, _size] = 1,
        groups: int = 1,
    ):
        layout_converter = LayoutConverter()

        input = layout_converter.nchw_to_nhwc(input)
        weight = LayoutConverter().nchw_to_nhwc(weight)

        # hbir.conv2d is replaced by hbir.conv in newer hbdk4 version.
        if hasattr(hbir, "conv"):
            hbir_conv = hbir.conv
        else:
            hbir_conv = hbir.conv2d

        output = hbir_conv(
            input,
            weight,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        )
        output = layout_converter.nhwc_to_nchw(output)

        return output


@JitTensor.register_converter(F.conv3d)
class Conv3dConverter(FuncConverterBase):
    @classmethod
    def norm_conv_pads(cls, padding, output_padding=(0, 0, 0)):
        return [
            padding[0],
            padding[1],
            padding[2],
            padding[0] - output_padding[0],
            padding[1] - output_padding[1],
            padding[2] - output_padding[2],
        ]

    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        weight: ir.Value,
        bias: Optional[ir.Value] = None,
        stride: Union[_int, _size] = 1,
        padding: Union[_int, _size, str] = 0,
        dilation: Union[_int, _size] = 1,
        groups: int = 1,
    ):
        stride = _triple(stride)
        if isinstance(padding, str):
            if padding == "same":
                msg = "Conv3d do not support padding='same' now"
                logger.error(msg)
                raise ValueError(msg)
            else:
                padding = (0, 0, 0)
        else:
            padding = _triple(padding)
        dilation = _triple(dilation)

        layout_converter = LayoutConverter()

        input = layout_converter.nchw_to_nhwc(input)
        weight = LayoutConverter().nchw_to_nhwc(weight)

        # hbir.conv2d is replaced by hbir.conv in newer hbdk4 version.
        if hasattr(hbir, "conv"):
            hbir_conv = hbir.conv
        else:
            hbir_conv = hbir.conv2d

        output = hbir_conv(
            input,
            weight,
            stride,
            cls.norm_conv_pads(padding),
            dilation,
            groups,
            bias=bias,
        )
        output = layout_converter.nhwc_to_nchw(output)

        return output


@JitTensor.register_converter(F.conv_transpose2d)
class ConvTranspose2dConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        weight: ir.Value,
        bias: Optional[ir.Value] = None,
        stride: Union[_int, _size] = 1,
        padding: Union[_int, _size] = 0,
        output_padding: Union[_int, _size] = 0,
        groups: int = 1,
        dilation: Union[_int, _size] = 1,
    ):
        stride = _pair(stride)
        padding = _pair(padding)
        output_padding = _pair(output_padding)
        dilation = _pair(dilation)

        layout_converter = LayoutConverter()

        input = layout_converter.nchw_to_nhwc(input)
        weight = LayoutConverter().nchw_to_nhwc(weight)

        # hbir.conv2dtranspose is replaced by hbir.convtranspose in
        # newer hbdk4 version.
        if hasattr(hbir, "convtranspose"):
            hbir_conv_transpose = hbir.convtranspose
        else:
            hbir_conv_transpose = hbir.conv2dtranspose

        output = hbir_conv_transpose(
            input,
            weight,
            stride,
            Conv2dConverter.norm_conv_pads(padding, output_padding),
            dilation,
            groups,
            illegalWeight=True,
            bias=bias,
        )
        output = layout_converter.nhwc_to_nchw(output)

        return output


@JitTensor.register_converter(torch.cumsum, Tensor.cumsum)
class CumsumConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, input: ir.Value, dim: _int, **kwargs):
        return hbir.cumsum(input, dim)


# Because torchvision.ops.deform_conv2d do not handle torch_function,
# we need to register converter on module
@Exporter.register_converter(ops.DeformConv2d)
class DeformConv2dConverter(ModuleConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        mod: ops.DeformConv2d,
        input: ir.Value,
        offset: ir.Value,
        mask: Optional[ir.Value] = None,
    ):
        weight, bias = JitTensor.gather_hbir((mod.weight, mod.bias))
        layout_converter = LayoutConverter()

        input = layout_converter.nchw_to_nhwc(input)
        offset = LayoutConverter().nchw_to_nhwc(offset)
        if mask is not None:
            mask = LayoutConverter().nchw_to_nhwc(mask)
        weight = LayoutConverter().nchw_to_nhwc(weight)

        output = hbir.deform_conv2d(
            input,
            weight,
            offset,
            mask,
            mod.stride,
            mod.padding + mod.padding,
            mod.dilation,
            mod.groups,
            offset.type.shape[-1]
            // (2 * mod.kernel_size[0] * mod.kernel_size[1]),
            mask is not None,
            bias=bias,
        )

        output = layout_converter.nhwc_to_nchw(output)

        return output


@JitTensor.register_converter(
    torch.div,
    torch.divide,
    torch.true_divide,
    Tensor.div,
    Tensor.divide,
    Tensor.true_divide,
)
class DivConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        other: ir.Value,
        *,
        rounding_mode: Optional[str] = None,
    ):
        if rounding_mode is not None:
            msg = "Unsupported rounding_mode {}".format(rounding_mode)
            logger.error(msg)
            raise ValueError(msg)

        return hbir.div(input, other)


@JitTensor.register_converter(F.elu)
class EluConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, input: ir.Value, alpha: float, inplace: bool):
        check_inplace(inplace)
        return hbir.elu(input, alpha)


@JitTensor.register_converter(
    torch.floor_divide,
    Tensor.floor_divide,
    Tensor.__floordiv__,
)
class FloorDivConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        input: Union[int, JitTensor],
        other: Union[int, JitTensor],
    ):
        input_base, other_base = JitTensor.get_base((input, other))
        hbir_input, hbir_other = JitTensor.gather_hbir((input, other))

        msg = "floor_div is only supported when both input is interger"
        if isinstance(input_base, Tensor):
            if input_base.is_floating_point():
                raise ValueError(msg)
        elif int(input_base) != input_base:
            raise ValueError(msg)
        else:
            hbir_input = int(input_base)
        if isinstance(other_base, Tensor):
            if other_base.is_floating_point():
                raise ValueError(msg)
        elif int(other_base) != other_base:
            raise ValueError(msg)
        else:
            hbir_other = int(other_base)

        logger.warning(
            "floor_div is exported to trunc_div, please make sure"
            " the operands are positive"
        )

        hbir_output = hbir.div(hbir_input, hbir_other)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(F.dropout, F.dropout1d, F.dropout2d, F.dropout3d)
class DropoutOpConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        p: float = 0.5,
        training: bool = True,
        inplace: bool = False,
    ):
        check_inplace(inplace)
        check_training(training, "dropout")
        return input


@JitTensor.register_converter(torch.einsum)
class EinSumConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, equation: str, *operands: Tuple[ir.Value, ...]):
        return hbir.einsum(operands, equation)


@JitTensor.register_converter(F.embedding)
class EmbeddingConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output: Tensor,
        input: JitTensor,
        weight: Tensor,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ):
        if max_norm is not None:
            raise ValueError("max_norm is not supported")

        input_hbir, weight_hbir = JitTensor.gather_hbir((input, weight))

        input_hbir = hbir.reshape(
            input_hbir, [JitTensor.get_base(input).numel(), 1]
        )
        hbir_output = hbir.gather_nd(weight_hbir, input_hbir, 0)
        hbir_output = hbir.reshape(hbir_output, output.shape)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.empty_like)
class EmptyLikeConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        *,
        memory_format=None,
        dtype: Optional[torch.dtype] = None,
        layout=None,
        device=None,
        pin_memory: Optional[_bool] = False,
        requires_grad: Optional[_bool] = False,
    ):
        return hbir.empty_like(
            input, dtype=None if dtype is None else get_hbir_tensor_type(dtype)
        )


@JitTensor.register_converter(Tensor.expand, Tensor.expand_as)
class ExpandConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output: Tensor, input: JitTensor, *args, **kwargs):
        input_size = list(input.size())
        output_size = list(output.size())
        for _ in range(len(output_size) - len(input_size)):
            input_size.insert(0, 1)
        expand_size = [
            o if i == 1 else 1 for i, o in zip(input_size, output_size)
        ]

        hbir_input = JitTensor.gather_hbir(input)

        output_type = get_hbir_tensor_type(output.as_subclass(Tensor).dtype)
        hbir_output = hbir.reshape(
            hbir_input, input_size, output_type=output_type
        )
        hbir_output = hbir.tile(hbir_output, expand_size)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(
    torch.flip,
    Tensor.flip,
)
class FlipConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, input: ir.Value, *dim, dims=None):
        # Use dims to receive kwargs
        if dims is None:
            dims = dim
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = dims[0]
        return hbir.flip(input, dims)


@JitTensor.register_converter(torch.gather, Tensor.gather)
class GatherConverter(FuncConverterBase):
    with_output_type = True

    @classmethod
    def convert_with_hbir(
        cls,
        output_type,
        input: ir.Value,
        dim: _int,
        index: ir.Value,
        *,
        sparse_grad: _bool = False,
    ):
        # Rename hbir.gather to hbir.gather_elements in hbdk4 > 4.0.25
        if LooseVersion(get_hbdk4_version()) > LooseVersion("4.0.25"):
            return hbir.gather_elements(
                input, index, dim, output_type=output_type
            )
        else:
            return hbir.gather(input, index, dim, output_type=output_type)


@JitTensor.register_converter(F.gelu)
class GeLUConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, input: ir.Value, approximate="none"):
        if approximate != "none":
            msg = "GeLU only support approximate = 'none'"
            logger.error(msg)
            raise ValueError(msg)
        return hbir.gelu(input)


@JitTensor.register_converter(Tensor.__getitem__)
class GetItemConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output: Tensor,
        input: JitTensor,
        indices: Union[None, _int, slice, Tensor, List, Tuple],
    ):
        input_base = JitTensor.get_base(input)
        hbir_node = JitTensor.gather_hbir(input)

        output_type = get_hbir_tensor_type(output.as_subclass(Tensor).dtype)
        input_shape = list(input_base.shape)

        if not isinstance(indices, Sequence):
            indices = [indices]
        # a[0].shape = a[(0)].shape != a[[0]].shape
        elif isinstance(indices, list) and all(
            [isinstance(x, int) for x in indices]
        ):
            indices = [torch.tensor(indices).to(input_base.device)]

        # process none index
        has_none = any([i is None for i in indices])
        indices = [i for i in indices if i is not None]

        # process Ellipsis index
        has_ellipsis = any([i is Ellipsis for i in indices])
        if has_ellipsis:
            ellipsis_index = indices.index(Ellipsis)
            indices = (
                indices[:ellipsis_index]
                + [
                    slice(None, None, None),
                ]
                * (len(input_shape) - len(indices) + 1)
                + indices[ellipsis_index + 1 :]
            )

        # pad None index to input_shape len
        if len(indices) < len(input_shape):
            indices += [
                slice(None, None, None),
            ] * (len(input_shape) - len(indices))

        # replace list index to tensor index
        indices = [
            (
                torch.tensor(x).to(input_base.device)
                if isinstance(x, (list, tuple))
                else x
            )
            for x in indices
        ]

        # 1. process int index first
        int_index_i = []
        int_index_dims = []
        pop_dims = []
        for cur_dim, indice in enumerate(indices):
            if isinstance(indice, int):
                int_index_i.append(indice)
                int_index_dims.append(cur_dim - len(int_index_dims))
                pop_dims.append(cur_dim)

        for i, dim in zip(int_index_i, int_index_dims):
            hbir_node = hbir.select(hbir_node, dim, i, output_type=output_type)

        # 2. process tensor index
        #   1). only one contiguous index -> reshape + tensor index
        #   2). several split tensor index -> permute + reshape + tensor index
        input_shape = [
            input_shape[i]
            for i, indice in enumerate(indices)
            if not isinstance(indice, int)
        ]
        indices = [i for i in indices if not isinstance(i, int)]
        processed_indices = []

        tensor_index_i = []
        tensor_index_dims = []
        flatten_size = 1
        for cur_dim, indice in enumerate(indices):
            if isinstance(indice, Tensor):
                if JitTensor.get_base(indice).dtype == torch.bool:
                    if isinstance(indice, JitTensor):
                        raise ValueError(
                            "Boolean mask of indexing must be a constant"
                        )
                    tensor_index_i.append(
                        torch.nonzero(indice).transpose(1, 0).squeeze(0)
                    )
                else:
                    tensor_index_i.append(indice)
                tensor_index_dims.append(cur_dim)
                flatten_size *= input_shape[cur_dim]
            else:
                processed_indices.append(indice)

        # replace bool tensor index with long tensor
        if len(tensor_index_i) == 1:
            indices[tensor_index_dims[0]] = tensor_index_i[0]
        # do transpose or flatten when several tensor index
        elif len(tensor_index_i) > 1:
            # compute final tensor index
            hbir_indexes = JitTensor.gather_hbir(tensor_index_i)
            cur_len = 1
            for i, hbir_index in enumerate(reversed(hbir_indexes)):
                if i == 0:
                    hbir_index_node = hbir_index
                    tensor_index_node = tensor_index_i[-1]
                else:
                    cur_len *= input_shape[tensor_index_dims[-i]]
                    hbir_index_node = hbir.add(
                        hbir.mul(
                            hbir_index,
                            cur_len,
                        ),
                        hbir_index_node,
                    )
                    tensor_index_node = (
                        tensor_index_i[-1 - i] * cur_len + tensor_index_node
                    )
            tensor_index_node = JitTensor.attach_hbir_to_tensor(
                tensor_index_node, hbir_index_node
            )

            non_tensor_dims = [
                i for i in range(len(indices)) if i not in tensor_index_dims
            ]
            non_tensor_shape = [input_shape[i] for i in non_tensor_dims]

            # several split tensor index, do transpose first
            flatten_shape = None
            if not all(
                [
                    tensor_index_dims[i + 1] - tensor_index_dims[i] == 1
                    for i in range(len(tensor_index_dims) - 1)
                ]
            ):
                transpose_dims = tensor_index_dims + non_tensor_dims
                hbir_node = hbir.transpose(
                    hbir_node, transpose_dims, output_type=output_type
                )
                flatten_shape = [
                    flatten_size,
                ] + non_tensor_shape
                # insert tensor indice at first pos
                indices = [
                    tensor_index_node,
                ] + processed_indices
            else:
                non_tensor_shape.insert(tensor_index_dims[0], flatten_size)
                processed_indices.insert(
                    tensor_index_dims[0], tensor_index_node
                )
                flatten_shape = non_tensor_shape
                indices = processed_indices

            hbir_node = hbir.reshape(
                hbir_node, flatten_shape, output_type=output_type
            )
            input_shape = flatten_shape

        dims = len(input_shape)
        slice_begins = [0] * dims
        slice_ends = input_shape
        slice_steps = [1] * dims
        slice_enabled = False

        tensor_index = None
        tensor_index_dim = None
        for cur_dim, indice in enumerate(indices):
            if isinstance(indice, slice):
                slice_enabled = True
                (
                    slice_begins[cur_dim],
                    slice_ends[cur_dim],
                    slice_steps[cur_dim],
                ) = indice.indices(input_shape[cur_dim])
            elif isinstance(indice, Tensor):
                assert (
                    tensor_index is None and tensor_index_dim is None
                ), "Unexpected two tensor index!"
                tensor_index = indice
                tensor_index_dim = cur_dim
            else:
                msg = "Unsupported indice {}".format(indice)
                logger.error(msg)
                raise ValueError(msg)

        if slice_enabled:
            hbir_node = hbir.slice(
                hbir_node,
                slice_begins,
                slice_ends,
                slice_steps,
                output_type=output_type,
            )

        if tensor_index is not None:
            assert (
                JitTensor.get_base(tensor_index).dtype != torch.bool
            ), "Unexpected bool tensor index"
            hbir_node = hbir.index(
                hbir_node,
                JitTensor.gather_hbir(tensor_index),
                tensor_index_dim,
                output_type=output_type,
            )

        if has_none:
            hbir_node = hbir.reshape(
                hbir_node, list(output.shape), output_type=output_type
            )

        return JitTensor.attach_hbir_to_tensor(output, hbir_node)


# https://www.tensorflow.org/jvm/api_docs/java/org/tensorflow/op/core/ScatterNd
@JitTensor.register_converter(Tensor.__setitem__)
class SetItemConverter(FuncConverterBase):
    @classmethod
    def gen_index_tensor_from_indices(cls, indices, input_shape):
        if isinstance(indices, (list, tuple)):
            indices = list(indices)
            if Ellipsis in indices and indices[-1] is not Ellipsis:
                input_rank = len(input_shape)
                elps_index = indices.index(Ellipsis)
                indices = (
                    indices[:elps_index]
                    + ([slice(None)] * (input_rank + 1 - len(indices)))
                    + indices[elps_index + 1 :]
                )

            index_tensor = torch.stack(
                torch.meshgrid(
                    *[
                        cls.gen_index_tensor_from_indices(i, [l]).flatten()
                        for i, l in zip(
                            indices,
                            (
                                input_shape
                                if len(input_shape) == len(indices)
                                else input_shape * len(indices)
                            ),
                        )
                    ],
                    indexing="ij",
                ),
                dim=-1,
            )

        elif indices is None:
            index_tensor = torch.tensor(list(range(input_shape[0]))).unsqueeze(
                -1
            )
        elif isinstance(indices, int):
            index_tensor = torch.tensor([indices]).unsqueeze(-1)
        elif isinstance(indices, slice):
            if indices.start is None:
                start = 0
            elif indices.start < 0:
                start = input_shape[0] + indices.start
            else:
                start = indices.start

            if indices.stop is None:
                stop = input_shape[0]
            elif indices.stop < 0:
                stop = input_shape[0] + indices.stop
            else:
                stop = indices.stop

            index_tensor = torch.tensor(
                list(range(start, stop, indices.step or 1))
            ).unsqueeze(-1)
        elif isinstance(indices, Tensor):
            index_tensor = indices
        else:
            raise ValueError(
                "Unsupported indices type: {}".format(type(indices))
            )

        return index_tensor

    @classmethod
    def get_slice_scatter_args(cls, indices):
        ret = None
        ellipsis_idx = None

        if not isinstance(indices, (list, tuple)):
            indices = (indices,)

        for idx, i in enumerate(indices):
            if i is Ellipsis:
                ellipsis_idx = idx
            elif isinstance(i, slice):
                if i.start is None and i.stop is None and i.step is None:
                    continue
                elif ret is None:
                    ret = [idx, i.start, i.stop, i.step]
                else:
                    # has multi slices
                    return None
            else:
                # has indice type not in (Ellipsis, slice)
                return None

        if ellipsis_idx is None:
            return ret
        elif ellipsis_idx > ret[0]:
            # if Ellipsis at right side of slice, dim=ret[0]
            return ret
        else:
            # if Ellipsis at left side of slice, count from right side
            ret[0] = ret[0] - len(indices)
            return ret

    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        indices: Union[_int, slice, JitTensor, Tensor, List, Tuple, None],
        val: Union[JitTensor, Tensor, Number],
    ):
        hbir_input, hbir_indices, hbir_val = JitTensor.gather_hbir(
            (input, indices, val)
        )
        input_base, indices_base, val_base = JitTensor.get_base(
            (input, indices, val)
        )

        slice_scatter_args = cls.get_slice_scatter_args(hbir_indices)
        if slice_scatter_args is not None:
            slice_scatter_args = SliceScatterConverter.canonicalize_args(
                list(input_base.shape), *slice_scatter_args
            )
            if isinstance(val_base, (float, int)):
                val_shape = list(input_base.shape)
                dim, start, stop, step = slice_scatter_args
                val_shape[dim] = (stop - start) // step
                val = torch.full(
                    val_shape,
                    val_base,
                    dtype=input_base.as_subclass(Tensor).dtype,
                )
                hbir_val = JitTensor.gather_hbir(val)

            hbir_output = SliceScatterConverter.convert_with_hbir(
                hbir_input, hbir_val, *slice_scatter_args
            )
            JitTensor.attach_inplaced_output(input, hbir_output)
            return output

        if (
            isinstance(indices_base, Tensor)
            and indices_base.dtype != torch.bool
        ):
            # index_put
            hbir_indices = hbir.reshape(
                hbir_indices,
                list(indices_base.shape) + [1],
                output_type=get_hbir_tensor_type(
                    indices_base.as_subclass(Tensor).dtype
                ),
            )
            values_shape = (
                list(indices_base.shape) + list(input_base.shape)[1:]
            )

        else:
            if isinstance(indices_base, Tensor):
                # For boolean mask
                if isinstance(indices, JitTensor):
                    raise ValueError(
                        "indices bool Tensor must be constant in because "
                        "torch.nonzero is not supported by hbir"
                    )
                index_tensor = torch.nonzero(indices_base)
            else:
                index_tensor = cls.gen_index_tensor_from_indices(
                    indices, list(input_base.shape)
                )

            hbir_indices = JitTensor.gather_hbir(index_tensor)

            values_shape = (
                list(index_tensor.shape)[:-1]
                + list(input_base.shape)[index_tensor.size(-1) :]
            )

        if isinstance(val_base, (int, float)):
            val_base = torch.full(values_shape, val)
            hbir_val = JitTensor.gather_hbir(val_base)
        elif val_base.numel() == 1:
            hbir_val = hbir.reshape(
                hbir_val,
                [1 for _ in len(values_shape)],
                output_type=get_hbir_tensor_type(
                    val_base.as_subclass(Tensor).dtype
                ),
            )
        else:
            hbir_val = hbir.reshape(
                hbir_val,
                values_shape,
                output_type=get_hbir_tensor_type(
                    val_base.as_subclass(Tensor).dtype
                ),
            )

        hbir_output = hbir.scatter_nd(hbir_input, hbir_indices, hbir_val)

        JitTensor.attach_inplaced_output(input, hbir_output)

        return output


@JitTensor.register_converter(F.grid_sample)
class GridSampleConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        grid: ir.Value,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: Optional[bool] = None,
    ):
        if padding_mode == "zeros":
            padding_mode = "constant"
            pad_value = 0
        else:
            pad_value = None

        layout_converter = LayoutConverter()
        input = layout_converter.nchw_to_nhwc(input)
        output = hbir.grid_sample(
            input, grid, mode, padding_mode, align_corners, padValue=pad_value
        )
        output = layout_converter.nhwc_to_nchw(output)
        return output


@JitTensor.register_converter(F.hardsigmoid)
class HardsigmoidConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        inplace: bool = False,
    ):
        check_inplace(inplace)
        return hbir.clip(hbir.add(hbir.div(input, 6.0), 0.5), 0.0, 1.0)


@JitTensor.register_converter(F.hardtanh)
class HardtanhConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        min_val: float = -1.0,
        max_val: float = 1.0,
        inplace: bool = False,
    ):
        check_inplace(inplace)
        return hbir.clip(input, min_val, max_val)


@JitTensor.register_converter(torch.index_select, Tensor.index_select)
class IndexSelectConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        dim: int,
        index: ir.Value,
    ):
        return hbir.index(input, index, dim)


@JitTensor.register_converter(F.instance_norm)
class InstanceNormConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        running_mean: Optional[JitTensor] = None,
        running_var: Optional[JitTensor] = None,
        weight: Optional[JitTensor] = None,
        bias: Optional[JitTensor] = None,
        use_input_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ):
        if use_input_stats is False:
            msg = "InstanceNorm only supports `track_running_stats` = `False`"
            logger.error(msg)
            raise ValueError(msg)
        normalized_shape = output.shape[2:]  # HW for 2d
        normalized_ndim = len(normalized_shape)
        dims = list(range(-normalized_ndim, 0, 1))
        affine_shape = [1, -1] + [1] * len(normalized_shape)
        if weight is not None:
            weight = weight.reshape(affine_shape)
            bias = bias.reshape(affine_shape)
        hbir_input = JitTensor.gather_hbir(input)
        hbir_weight = JitTensor.gather_hbir(weight)
        hbir_bias = JitTensor.gather_hbir(bias)
        hbir_output = hbir.layernorm(
            hbir_input,
            dims,
            float(eps),
            weight=hbir_weight,
            bias=hbir_bias,
        )
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(F.normalize)
class NormalizeConverter(FuncConverterBase):
    # after hbdk4 support, directly use convert_with_hbir
    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        p: float = 2.0,
        dim: int = 1,
        eps: float = 1e-12,
        out: Optional[torch.Tensor] = None,
    ):
        hbir_input = JitTensor.gather_hbir(input)
        hbir_node = hbir.mul(hbir_input, hbir_input)
        hbir_node = hbir.reduce_sum(
            hbir_node,
            dims=[
                dim,
            ],
            keepDim=True,
        )
        hbir_node = hbir.sqrt(hbir_node)
        hbir_node = hbir.clip(hbir_node, min=eps, max=float("inf"))
        input_size = list(input.size())
        output_size = list(output.size())
        input_size[dim] = 1
        for _ in range(len(output_size) - len(input_size)):
            input_size.insert(0, 1)
        expand_size = [
            o if i == 1 else 1 for i, o in zip(input_size, output_size)
        ]
        output_type = get_hbir_tensor_type(output.as_subclass(Tensor).dtype)
        hbir_node = hbir.reshape(
            hbir_node, input_size, output_type=output_type
        )
        hbir_node = hbir.tile(hbir_node, expand_size)
        hbir_node = hbir.div(hbir_input, hbir_node)
        return JitTensor.attach_hbir_to_tensor(output, hbir_node)

    # hbdk4 do not support now
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        p: float = 2.0,
        dim: int = 1,
        eps: float = 1e-12,
        out: Optional[torch.Tensor] = None,
    ):
        return hbir.lp_normalize(input, p, dim, eps)


@JitTensor.register_converter(F.interpolate)
class InterpolateConverter(FuncConverterBase):
    @classmethod
    def step_fq(cls, step: Tuple[float, float]):
        return step

    @classmethod
    def convert_interpolate(
        cls,
        output: JitTensor,
        input: JitTensor,
        size: Optional[Union[int, List[int]]] = None,
        scale_factor=None,
        mode="nearest",
        align_corners=False,
        recompute_scale_factor=None,
        antialias=False,
    ):
        if antialias is not False:
            msg = "Interpolate only support antialias=False"
            logger.error(msg)
            raise ValueError(msg)

        input_h, input_w = float(input.size(-2)), float(input.size(-1))
        output_h, output_w = float(output.size(-2)), float(output.size(-1))

        if scale_factor is None or recompute_scale_factor is True:
            scale_factor = [output_h / input_h, output_w / input_w]
        else:
            scale_factor = _pair(scale_factor)

        if align_corners:
            step = [
                (input_h - 1) / (output_h - 1),
                (input_w - 1) / (output_w - 1),
            ]
            offset = [0.0, 0.0]
        else:
            step = [1 / scale_factor[0], 1 / scale_factor[1]]
            offset = [0.5 / scale_factor[0] - 0.5, 0.5 / scale_factor[1] - 0.5]

        hbir_input = JitTensor.gather_hbir(input)

        layout_converter = LayoutConverter()
        hbir_input = layout_converter.nchw_to_nhwc(hbir_input)
        hbir_output = hbir.resize2d(
            hbir_input,
            cls.step_fq(step),
            initialOffset=offset,
            mode=mode,
            expansionMode="border",
            # ratio=[output_h / input_h, output_w / input_w],
            size=[int(output_h), int(output_w)],
        )
        hbir_output = layout_converter.nhwc_to_nchw(hbir_output)
        return hbir_output

    @classmethod
    def convert(cls, output, *args, **kwargs):
        hbir_output = cls.convert_interpolate(output, *args, **kwargs)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(F.layer_norm)
class LayerNormConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output: Tensor,
        input: JitTensor,
        normalized_shape: List[int],
        weight: Optional[JitTensor] = None,
        bias: Optional[JitTensor] = None,
        eps: float = 1e-5,
    ):
        hbir_input, hbir_weight, hbir_bias = JitTensor.gather_hbir(
            (input, weight, bias)
        )
        input_base, weight_base = JitTensor.get_base((input, weight))

        if weight is not None and input_base.ndim > weight_base.ndim:
            weight_shape = [1] * (input_base.ndim - weight_base.ndim) + list(
                weight_base.shape
            )

            hbir_weight = hbir.reshape(hbir_weight, weight_shape)
            hbir_bias = hbir.reshape(hbir_bias, weight_shape)

        dims = list(range(-len(normalized_shape), 0, 1))
        hbir_output = hbir.layernorm(
            hbir_input, dims, float(eps), weight=hbir_weight, bias=hbir_bias
        )

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(F.leaky_relu)
class LeakyReluConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        negative_slope: float = 0.01,
        inplace: bool = False,
    ):
        check_inplace(inplace)
        return hbir.leaky_relu(input, negative_slope)


@JitTensor.register_converter(F.linear)
class LinearConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls, input: ir.Value, weight: ir.Value, bias: Optional[ir.Value] = None
    ):
        # weight layout is (Cout, Cin), which is same as F.linear
        # weight = hbir.transpose(weight, (1, 0))
        return hbir.linear(input, weight, bias=bias)


@JitTensor.register_converter(
    torch.logical_and, Tensor.logical_and, Tensor.__and__
)
class LogicalAndConverter(FuncConverterBase):
    hbir_operation = hbir.logical_and
    inplaced = False

    @classmethod
    def get_hbir_from_input(cls, input: JitTensor):
        hbir_input = JitTensor.gather_hbir(input)
        input_base = JitTensor.get_base(input).as_subclass(Tensor)

        if input_base.dtype is not torch.bool:
            hbir_zeros = JitTensor.gather_hbir(torch.zeros_like(input_base))
            hbir_input = hbir.not_equal(
                hbir_input,
                hbir_zeros,
                output_type=get_hbir_tensor_type(torch.bool),
            )
        return hbir_input

    @classmethod
    def convert(cls, output: Tensor, *args):
        hbir_output = cls.hbir_operation(
            *(cls.get_hbir_from_input(input) for input in args),
            output_type=get_hbir_tensor_type(torch.bool),
        )
        output_type = output.as_subclass(Tensor).dtype
        if output_type is not torch.bool:
            hbir_output = hbir.cast_type(
                hbir_output, output_type=get_hbir_tensor_type(output_type)
            )

        if cls.inplaced:
            JitTensor.attach_inplaced_output(args[0], hbir_output)
            return args[0]
        else:
            return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(Tensor.logical_and_)
class LogicalAndInplacedConverter(LogicalAndConverter):
    hbir_operation = hbir.logical_and
    inplaced = True


@JitTensor.register_converter(
    torch.logical_not, Tensor.logical_not, Tensor.__invert__
)
class LogicalNotConverter(LogicalAndConverter):
    hbir_operation = hbir.logical_not


@JitTensor.register_converter(Tensor.logical_not_)
class LogicalNotInplacedConverter(LogicalAndConverter):
    hbir_operation = hbir.logical_not
    inplaced = True


@JitTensor.register_converter(
    torch.logical_or, Tensor.logical_or, Tensor.__or__
)
class LogicalOrConverter(LogicalAndConverter):
    hbir_operation = hbir.logical_or


@JitTensor.register_converter(Tensor.logical_or_)
class LogicalOrInplacedConverter(LogicalAndConverter):
    hbir_operation = hbir.logical_or
    inplaced = True


@JitTensor.register_converter(F.log_softmax)
class FLogSoftmaxConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        dim: Optional[int] = None,
        _stacklevel: int = 3,
        dtype: Optional[torch.dtype] = None,
    ):
        hbir_input = JitTensor.gather_hbir(input)
        input_base = JitTensor.get_base(input)

        if dim is None:
            from torch.nn.functional import _get_softmax_dim

            dim = _get_softmax_dim("softmax", input_base.ndim, 3)
        if dtype is not None:
            dtype = get_hbir_tensor_type(dtype, list(output.shape))

        hbir_output = hbir.softmax(hbir_input, dim, output_type=dtype)
        hbir_output = hbir.log(hbir_output)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.masked_fill, Tensor.masked_fill)
class MaskedFillConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        mask: Union[Tensor, JitTensor],
        value: Union[Tensor, JitTensor, Number],
    ):
        if isinstance(value, (int, float)):
            value = torch.full_like(
                JitTensor.get_base(input).as_subclass(Tensor), value
            )

        input, mask, value = JitTensor.gather_hbir((input, mask, value))
        hbir_output = hbir.where(mask, value, input)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(Tensor.masked_fill_)
class InplacedMaskedFillConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        mask: Union[Tensor, JitTensor],
        value: Union[Tensor, JitTensor, Number],
    ):
        if isinstance(value, (int, float)):
            value = torch.full_like(
                JitTensor.get_base(input).as_subclass(Tensor), value
            )

        input_hbir, mask, value = JitTensor.gather_hbir((input, mask, value))
        hbir_output = hbir.where(mask, value, input_hbir)

        JitTensor.attach_inplaced_output(input, hbir_output)

        return input


@JitTensor.register_converter(torch.masked_scatter, Tensor.masked_scatter)
class MaskedScatterConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        mask: Tensor,
        source: JitTensor,
    ):
        if isinstance(mask, JitTensor):
            raise ValueError(
                "mask must be constant in because "
                "torch.nonzero is not supported by hbir"
            )

        hbir_input, _, hbir_val = JitTensor.gather_hbir((input, mask, source))
        input_base, mask_base, val_base = JitTensor.get_base(
            (input, mask, source)
        )

        index_tensor = torch.nonzero(mask_base)
        if index_tensor.size(0) == 0:
            return JitTensor.attach_hbir_to_tensor(output, hbir_input)

        hbir_indices = JitTensor.gather_hbir(index_tensor)

        values_shape = (
            list(index_tensor.shape)[:-1]
            + list(input_base.shape)[index_tensor.size(-1) :]
        )

        hbir_val_type = get_hbir_tensor_type(
            val_base.as_subclass(Tensor).dtype
        )
        needed_val_num = math.prod(values_shape)
        if val_base.numel() > needed_val_num:
            hbir_val = hbir.reshape(
                hbir_val, (val_base.numel(),), output_type=hbir_val_type
            )
            hbir_val = hbir.slice(
                hbir_val,
                (0,),
                (needed_val_num,),
                (1,),
                output_type=hbir_val_type,
            )

        hbir_val = hbir.reshape(
            hbir_val,
            values_shape,
            output_type=hbir_val_type,
        )

        hbir_output = hbir.scatter_nd(hbir_input, hbir_indices, hbir_val)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.matmul, Tensor.matmul)
class MatmulConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, input: JitTensor, other: JitTensor):
        input_base, other_base = JitTensor.get_base((input, other))
        if input_base.ndim < 2 or other_base.ndim < 2:
            msg = "Both inputs of matmul must have 2 dimentions at least"
            logger.error(msg)
            raise ValueError(msg)
        hbir_input, hbir_other = JitTensor.gather_hbir((input, other))
        hbir_output = hbir.matmul(hbir_input, hbir_other)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.max, Tensor.max)
class MaxConverter(FuncConverterBase):
    @classmethod
    def convert_allreduce_max(cls, input: ir.Value, input_dim, output_type):
        dims = list(range(input_dim))
        return hbir.reduce_max(input, dims, False, output_type=output_type)

    @classmethod
    def convert_reduce_max(
        cls, input: ir.Value, dim: int, keepdim=False, output_type=None
    ):
        value = hbir.reduce_max(input, [dim], keepdim, output_type=output_type)
        indice = hbir.reduce_argmax(
            input,
            [dim],
            keepdim,
            output_type=ir.UnrankedTensorType.get(
                ir.IntegerType.get_signed(64)
            ),
        )
        return (value, indice)

    @classmethod
    def convert(cls, output: Tensor, *args, **kwargs):
        hbir_args = JitTensor.gather_hbir(args)
        hbir_kwargs = JitTensor.gather_hbir(kwargs)

        if len(args) + len(kwargs) == 1:
            hbir_output = cls.convert_allreduce_max(
                *hbir_args,
                **hbir_kwargs,
                input_dim=JitTensor.get_base(args[0]).ndim,
                output_type=get_hbir_tensor_type(
                    output.as_subclass(Tensor).dtype
                ),
            )
        elif isinstance(output, (list, tuple)) and len(output) == 2:
            hbir_output = cls.convert_reduce_max(
                *hbir_args,
                **hbir_kwargs,
                output_type=get_hbir_tensor_type(
                    output[0].as_subclass(Tensor).dtype
                ),
            )
            output = tuple(output)
        else:
            converter = JitTensor._converters.get(torch.maximum)
            hbir_output = converter.convert_with_hbir(
                *hbir_args, **hbir_kwargs
            )
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.min, Tensor.min)
class MinConverter(FuncConverterBase):
    @classmethod
    def convert_allreduce_min(cls, input: ir.Value, input_dim, output_type):
        dims = list(range(input_dim))
        return hbir.reduce_min(input, dims, False, output_type=output_type)

    @classmethod
    def convert_reduce_min(
        cls, input: ir.Value, dim: int, keepdim=False, output_type=None
    ):
        value = hbir.reduce_min(input, [dim], keepdim, output_type=output_type)
        indice = hbir.reduce_argmin(
            input,
            [dim],
            keepdim,
            output_type=ir.UnrankedTensorType.get(
                ir.IntegerType.get_signed(64)
            ),
        )
        return (value, indice)

    @classmethod
    def convert(cls, output, *args, **kwargs):
        hbir_args = JitTensor.gather_hbir(args)
        hbir_kwargs = JitTensor.gather_hbir(kwargs)

        if len(args) + len(kwargs) == 1:
            hbir_output = cls.convert_allreduce_min(
                *hbir_args,
                **hbir_kwargs,
                input_dim=JitTensor.get_base(args[0]).ndim,
                output_type=get_hbir_tensor_type(
                    output.as_subclass(Tensor).dtype
                ),
            )
        elif isinstance(output, (list, tuple)) and len(output) == 2:
            hbir_output = cls.convert_reduce_min(
                *hbir_args,
                **hbir_kwargs,
                output_type=get_hbir_tensor_type(
                    output[0].as_subclass(Tensor).dtype
                ),
            )
            output = tuple(output)
        else:
            converter = JitTensor._converters.get(torch.minimum)
            hbir_output = converter.convert_with_hbir(
                *hbir_args, **hbir_kwargs
            )
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(F.max_pool1d, F.max_pool1d_with_indices)
class MaxPool1dConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output: Tensor,
        input: Tensor,
        kernel_size: Union[_int, _size],
        stride: Optional[Union[_int, _size]] = None,
        padding: Union[_int, _size] = 0,
        dilation: Union[_int, _size] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False,
    ):
        return super().convert(
            output,
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            return_indices,
            tuple(input.shape),
            tuple(output.shape),
        )

    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        kernel_size: Union[_int, _size],
        stride: Optional[Union[_int, _size]] = None,
        padding: Union[_int, _size] = 0,
        dilation: Union[_int, _size] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False,
        input_shape: Optional[tuple] = None,
        output_shape: Optional[tuple] = None,
    ):
        if dilation != 1:
            msg = "MaxPool1d only support `dilation` = 1"
            logger.error(msg)
            raise ValueError(msg)
        if return_indices:
            msg = "MaxPool1d does not support `return_indices` = True"
            logger.error(msg)
            raise ValueError(msg)

        kernel_size = _single(kernel_size) + (1,)
        if stride is None:
            stride = kernel_size
        else:
            stride = _single(stride) + (1,)
        padding = _single(padding) + (0,)
        dilation = _single(dilation) + (1,)
        output = hbir.reshape(input, input_shape + (1,))
        output = MaxPool2dConverter.convert_with_hbir(
            output,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            return_indices,
        )
        output = hbir.reshape(output, output_shape)
        return output


@JitTensor.register_converter(F.max_pool2d, F.max_pool2d_with_indices)
class MaxPool2dConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        kernel_size: Union[_int, _size],
        stride: Optional[Union[_int, _size]] = None,
        padding: Union[_int, _size] = 0,
        dilation: Union[_int, _size] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False,
    ):
        if dilation != 1 and dilation != (1, 1):
            msg = "MaxPool2d only support `dilation` = 1"
            logger.error(msg)
            raise ValueError(msg)
        if return_indices:
            msg = "MaxPool2d does not support `return_indices` = True"
            logger.error(msg)
            raise ValueError(msg)

        kernel_size = _pair(kernel_size)
        if stride is None:
            stride = kernel_size
        else:
            stride = _pair(stride)
        padding = norm_pads(_pair(padding))
        dilation = _pair(dilation)

        layout_converter = LayoutConverter()
        input = layout_converter.nchw_to_nhwc(input)
        output = hbir.max_pool(
            input, kernel_size, stride, padding, dilation, ceil_mode
        )
        output = layout_converter.nhwc_to_nchw(output)
        return output


@JitTensor.register_converter(torch.mean, Tensor.mean)
class MeanConverter(FuncConverterBase):
    @classmethod
    def convert_allreduce_mean(cls, input: ir.Value, input_dim):
        dims = list(range(input_dim))
        return hbir.reduce_mean(input, dims, False)

    @classmethod
    def convert_reduce_mean(cls, input: ir.Value, dim: int, keepdim=False):
        return hbir.reduce_mean(input, [dim], keepdim)

    @classmethod
    def convert(cls, output, *args, **kwargs):
        hbir_args = JitTensor.gather_hbir(args)
        hbir_kwargs = JitTensor.gather_hbir(kwargs)
        if len(args) + len(kwargs) == 1:
            hbir_output = cls.convert_allreduce_mean(
                *hbir_args,
                **hbir_kwargs,
                input_dim=JitTensor.get_base(args[0]).ndim,
            )
        else:
            hbir_output = cls.convert_reduce_mean(*hbir_args, **hbir_kwargs)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.mul, Tensor.mul)
class MulConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls, input: Union[ir.Value, Number], other: Union[ir.Value, Number]
    ):
        return hbir.mul(input, other)


@JitTensor.register_converter(Tensor.mul_)
class MulInplacedConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        input: Union[JitTensor, Number],
        other: Union[JitTensor, Number],
    ):
        input_hbir, other_hbir = JitTensor.gather_hbir((input, other))
        hbir_output = hbir.mul(input_hbir, other_hbir)

        JitTensor.attach_inplaced_output(input, hbir_output)

        return input


@JitTensor.register_converter(torch.norm, torch.linalg.norm)
class NormConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        p: Optional[Union[float, str]] = "fro",
        dim=None,
        keepdim=False,
        out=None,
        dtype=None,
    ):
        if p not in (2.0, "fro", None):
            raise ValueError("Only vector norm of order 2 is supported")
        if out is not None:
            raise ValueError("arg 'out' is not supported")
        if dtype is not None:
            raise ValueError("arg 'dtype' is not supported")

        output = hbir.mul(input, input)
        output = hbir.reduce_sum(
            output, None if dim is None else [dim], keepdim
        )
        output = hbir.sqrt(output)

        return output


@JitTensor.register_converter(F.pad)
class PadConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output: Tensor,
        input: JitTensor,
        pad: Sequence[int],
        mode: str = "constant",
        value: Optional[float] = None,
    ):
        if mode not in ["constant", "replicate"]:
            msg = f'Pad does not support padding mode "{mode}"'
            logger.error(msg)
            raise ValueError(msg)
        if mode == "replicate":
            mode = "border"
        if value is None:
            value = 0.0
        pad_rank = len(pad) // 2
        unpad_rank = JitTensor.get_base(input).ndim - pad_rank
        unpad_values = [0] * unpad_rank

        begin = [*unpad_values, *pad[0::2][::-1]]
        end = [*unpad_values, *pad[1::2][::-1]]

        hbir_input = JitTensor.gather_hbir(input)
        hbir_output = hbir.pad(
            hbir_input,
            begin,
            end,
            mode,
            padValue=value,
            output_type=get_hbir_tensor_type(output.as_subclass(Tensor).dtype),
        )
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.permute, Tensor.permute)
class PermuteConverter(FuncConverterBase):
    with_output_type = True

    @classmethod
    def convert_with_hbir(cls, output_type, input: ir.Value, *dims: _size):
        if isinstance(dims[0], (list, tuple)):
            if len(dims) != 1:
                msg = "Invalid dims arg for permute {}".format(dims)
                logger.error(msg)
                raise ValueError(msg)
            dims = dims[0]
        return hbir.transpose(input, dims, output_type=output_type)


@JitTensor.register_converter(F.pixel_shuffle)
class PixelShuffleConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, input: JitTensor, upscale_factor: _int):
        input_shape = tuple(JitTensor.get_base(input).shape)
        # (*, C, H, W) -> (*, C // r**2, r, r, H, W)
        inter_shape = (
            input_shape[:-3]
            + (
                input_shape[-3] // upscale_factor ** 2,
                upscale_factor,
                upscale_factor,
            )
            + input_shape[-2:]
        )
        input_hbir = JitTensor.gather_hbir(input)
        inter_hbir = hbir.reshape(input_hbir, inter_shape)
        # (*, C // r**2, r, r, H, W) -> (*, C // r**2, H, r, W, r)
        dims = tuple(range(len(inter_shape)))
        dims = dims[:-4] + (dims[-2], dims[-4], dims[-1], dims[-3])
        inter_hbir = hbir.transpose(inter_hbir, dims)
        # (*, C // r**2, H, r, W, r) -> (*, C // r**2, H * r, W * r)
        output_shape = input_shape[:-3] + (
            input_shape[-3] // upscale_factor ** 2,
            input_shape[-2] * upscale_factor,
            input_shape[-1] * upscale_factor,
        )
        output_hbir = hbir.reshape(inter_hbir, output_shape)
        return JitTensor.attach_hbir_to_tensor(output, output_hbir)


@JitTensor.register_converter(F.pixel_unshuffle)
class PixelUnShuffleConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, input: JitTensor, downscale_factor: _int):
        input_shape = tuple(JitTensor.get_base(input).shape)
        # (*, C, H, W) -> (*, C, H // r, r, W // r, r)
        inter_shape = input_shape[:-2] + (
            input_shape[-2] // downscale_factor,
            downscale_factor,
            input_shape[-1] // downscale_factor,
            downscale_factor,
        )
        input_hbir = JitTensor.gather_hbir(input)
        inter_hbir = hbir.reshape(input_hbir, inter_shape)
        # (*, C, H // r, r, W // r, r) -> (*, C, r, r, H // r, W // r)
        dims = tuple(range(len(inter_shape)))
        dims = dims[:-4] + (dims[-3], dims[-1], dims[-4], dims[-2])
        inter_hbir = hbir.transpose(inter_hbir, dims)
        # (*, C, r, r, H // r, W // r) -> (*, C * r * r, H // r, W // r)
        output_shape = input_shape[:-3] + (
            input_shape[-3] * downscale_factor ** 2,
            input_shape[-2] // downscale_factor,
            input_shape[-1] // downscale_factor,
        )
        output_hbir = hbir.reshape(inter_hbir, output_shape)
        return JitTensor.attach_hbir_to_tensor(output, output_hbir)


@JitTensor.register_converter(F.prelu)
class PreluConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, input: JitTensor, weight: JitTensor):
        hbir_input, hbir_weight = JitTensor.gather_hbir((input, weight))
        if JitTensor.get_base(weight).numel() > 1:
            need_transpose = True
        else:
            need_transpose = False

        if need_transpose:
            dims = list(range(JitTensor.get_base(input).ndim))
            dims[1], dims[-1] = dims[-1], dims[1]
            hbir_input = hbir.transpose(hbir_input, dims)

        hbir_output = hbir.prelu(hbir_input, hbir_weight)

        if need_transpose:
            hbir_output = hbir.transpose(hbir_output, dims)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.pow, Tensor.pow)
class PowConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, input: ir.Value, exponent):
        return hbir.pow(input, float(exponent))


@JitTensor.register_converter(torch.relu, Tensor.relu, F.relu)
class ReLUConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, input: ir.Value, inplace: bool = False):
        check_inplace(inplace)
        return hbir.relu(input)


@JitTensor.register_converter(F.relu6)
class ReLU6Converter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, input: ir.Value, inplace: bool = False):
        check_inplace(inplace)
        return hbir.clip(input, 0.0, 6.0)


@JitTensor.register_converter(torch.remainder, Tensor.remainder)
class RemainderConverter(FuncConverterBase):
    same_sign_as_dividend = False

    @classmethod
    def convert(
        cls,
        output: Tensor,
        input: Union[Number, Tensor],
        other: Union[Number, Tensor],
    ):
        if not isinstance(input, Tensor):
            input = torch.tensor(input, dtype=other.dtype)
        if not isinstance(other, Tensor):
            other = torch.tensor(other, dtype=input.dtype)

        input_hbir, other_hbir = JitTensor.gather_hbir((input, other))

        if hasattr(hbir, "mod"):
            output_hbir = hbir.mod(
                input_hbir,
                other_hbir,
                cls.same_sign_as_dividend,
                output_type=get_hbir_tensor_type(output.dtype),
            )
        elif cls.same_sign_as_dividend is False:
            output_hbir = hbir.rem(
                input_hbir,
                other_hbir,
                output_type=get_hbir_tensor_type(output.dtype),
            )
        else:
            raise ValueError("Please update hbdk to support export fmod")

        return JitTensor.attach_hbir_to_tensor(output, output_hbir)


@JitTensor.register_converter(torch.fmod, Tensor.fmod)
class FModConverter(RemainderConverter):
    same_sign_as_dividend = True


@JitTensor.register_converter(Tensor.repeat)
class RepeatConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, input: JitTensor, *repeats):
        hbir_input = JitTensor.gather_hbir(input)
        if isinstance(repeats[0], (list, tuple)):
            if len(repeats) != 1:
                msg = "Invalid dims arg for repeat {}".format(repeats)
                logger.error(msg)
                raise ValueError(msg)
            repeats = repeats[0]
        hbir_output = hbir.tile(
            hbir_input,
            repeats,
            output_type=get_hbir_tensor_type(
                output.as_subclass(Tensor).dtype,
            ),
        )
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(
    torch.reshape,
    Tensor.reshape,
    Tensor.view,
    torch.flatten,
    Tensor.flatten,
    torch.squeeze,
    Tensor.squeeze,
    torch.unsqueeze,
    Tensor.unsqueeze,
)
class ReshapeConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, input: JitTensor, *args, **kwargs):
        hbir_input = JitTensor.gather_hbir(input)
        hbir_output = hbir.reshape(
            hbir_input,
            list(output.shape),
            output_type=get_hbir_tensor_type(output.as_subclass(Tensor).dtype),
        )
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.roll, Tensor.roll)
class RollConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        shifts: Union[_int, _size],
        dims: Optional[Union[_int, _size]] = None,
    ):
        hbir_input = JitTensor.gather_hbir(input)
        input_base = JitTensor.get_base(input)
        input_dim = input_base.ndim

        if not isinstance(shifts, Sequence):
            shifts = [shifts]
        if not isinstance(dims, Sequence):
            dims = [dims]

        dims = [d if d >= 0 else input_dim + d for d in dims]

        act_shifts = []
        for s, d in zip(shifts, dims):
            act_shifts.append(s % input_base.size(d))

        hbir_output = hbir.roll(hbir_input, act_shifts, dims)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.scatter, Tensor.scatter)
class ScatterConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output: Tensor,
        input: JitTensor,
        dim: _int,
        index: Union[JitTensor, Tensor],
        src: Union[JitTensor, float, int],
    ):
        if isinstance(src, (int, float)):
            src = JitTensor.gather_hbir(
                torch.full_like(JitTensor.get_base(input), src)
            )

        input, index, src = JitTensor.gather_hbir((input, index, src))
        hbir_output = hbir.scatter_elements(input, index, src, dim)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.scatter_add, Tensor.scatter_add)
class ScatterAddConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output: Tensor,
        input: JitTensor,
        dim: _int,
        index: Union[JitTensor, Tensor],
        src: Union[JitTensor, float, int],
    ):
        if isinstance(src, (int, float)):
            src = JitTensor.gather_hbir(
                torch.full_like(JitTensor.get_base(input), src)
            )

        input, index, src = JitTensor.gather_hbir((input, index, src))
        hbir_output = hbir.scatter_elements(
            input,
            index,
            src,
            dim,
            "add",
            output_type=get_hbir_tensor_type(output.as_subclass(Tensor).dtype),
        )

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.scatter_reduce, Tensor.scatter_reduce)
class ScatterReduceConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output: Tensor,
        input: JitTensor,
        dim: _int,
        index: Union[JitTensor, Tensor],
        src: Union[JitTensor, float, int],
        reduce: str,
        *,
        include_self=True,
        out=None,
    ):
        if not include_self:
            raise ValueError(
                "include_self must be true in scatter_reduce export"
            )
        if out is not None:
            raise ValueError(
                "inplaced output is not supported in scatter_reduce export"
            )

        reduce_mapping = {
            "sum": "add",
            "prod": "mul",
            "amax": "max",
            "amin": "min",
            "mean": "mean",
        }

        reduce = reduce_mapping[reduce]

        # if isinstance(src, (int, float)):
        #     src = JitTensor.gather_hbir(
        #         torch.full_like(JitTensor.get_base(input), src)
        #     )

        input, index, src = JitTensor.gather_hbir((input, index, src))
        hbir_output = hbir.scatter_elements(input, index, src, dim, reduce)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(Tensor.scatter_reduce_)
class ScatterReduceInplacedConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output: Tensor,
        input: JitTensor,
        dim: _int,
        index: Union[JitTensor, Tensor],
        src: Union[JitTensor, float, int],
        reduce: str,
        *,
        include_self=True,
    ):
        if include_self:
            raise ValueError(
                "include_self is not supported in scatter_reduce export"
            )

        if isinstance(src, (int, float)):
            src = JitTensor.gather_hbir(
                torch.full_like(JitTensor.get_base(input), src)
            )

        input, index, src = JitTensor.gather_hbir((input, index, src))
        hbir_output = hbir.scatter_elements(input, index, src, dim, reduce)

        JitTensor.attach_inplaced_output(input, hbir_output)
        return input


@JitTensor.register_converter(F.silu)
class SiluConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, input: ir.Value, inplace: bool = False):
        check_inplace(inplace)
        return hbir.swish(input)


@JitTensor.register_converter(torch.slice_scatter, Tensor.slice_scatter)
class SliceScatterConverter(FuncConverterBase):
    @classmethod
    def canonicalize_args(cls, input_shape, dim, start, end, step):
        if dim < 0:
            dim = len(input_shape) + dim
        if start is None:
            start = 0
        if start < 0:
            start = input_shape[dim] + start
        if end is None:
            end = input_shape[dim]
        if end < 0:
            end = input_shape[dim] + end
        if step is None:
            step = 1

        return dim, start, end, step

    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        src: ir.Value,
        dim=0,
        start=None,
        end=None,
        step=1,
    ):
        dim, start, end, step = cls.canonicalize_args(
            input.type.shape, dim, start, end, step
        )
        return hbir.slice_scatter(input, src, dim, start, end, step)


@JitTensor.register_converter(torch.sort, Tensor.sort)
class SortConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        dim: _int = -1,
        descending: _bool = False,
        stable=False,
    ):
        return hbir.sort(
            input,
            dim,
            descending,
            stable,
            indices_type=ir.UnrankedTensorType.get(
                ir.IntegerType.get_signed(64)
            ),
        )


@JitTensor.register_converter(torch.argsort, Tensor.argsort)
class ArgSortConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, input, dim=-1, descending=False, stable=False):
        return hbir.sort(
            input,
            dim,
            descending,
            stable,
            indices_type=ir.UnrankedTensorType.get(
                ir.IntegerType.get_signed(64)
            ),
        )[1]


@JitTensor.register_converter(torch.masked_select, Tensor.masked_select)
class MaskedSelectConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, input, mask):
        hbir_input = JitTensor.gather_hbir(input)
        hbir_mask = JitTensor.gather_hbir(mask)
        pad_torch_output = input.as_subclass(Tensor).reshape(-1).clone()
        valid_num = torch.sum(mask).item()
        pad_torch_output[:valid_num] = output.as_subclass(Tensor)
        hbir_output = hbir.masked_select(hbir_input, hbir_mask)
        return JitTensor.attach_hbir_to_tensor(pad_torch_output, hbir_output)


# This converter is needed because hbir has no default value for `dim`.
@JitTensor.register_converter(torch.stack)
class StackConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, tensors: List[ir.Value], dim: int = 0):
        return hbir.stack(tensors, dim)


@JitTensor.register_converter(torch.split)
class TorchSplitConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        tensor: JitTensor,
        split_size_or_sections: Union[int, List[int]],
        dim: int = 0,
    ):
        hbir_input = JitTensor.gather_hbir(tensor)
        input_base = JitTensor.get_base(tensor)
        dims = input_base.ndim
        slice_begins = [0] * dims
        slice_ends = list(input_base.shape)
        slice_steps = [1] * dims
        split_len = slice_ends[dim]

        if isinstance(split_size_or_sections, int):
            slice_points = list(
                range(0, split_len + 1, split_size_or_sections)
            )
        else:
            slice_points = [0]
            for split_size in split_size_or_sections:
                slice_points.append(slice_points[-1] + split_size)
        if not slice_points[-1] == split_len:
            msg = "Invalid slice points {} for dim size {}.".format(
                slice_points, split_len
            )
            logger.error(msg)
            raise ValueError(msg)

        hbir_rets = []
        for i in range(len(slice_points) - 1):
            slice_begins[dim] = slice_points[i]
            slice_ends[dim] = slice_points[i + 1]
            hbir_rets.append(
                hbir.slice(
                    hbir_input,
                    slice_begins,
                    slice_ends,
                    slice_steps,
                    output_type=get_hbir_tensor_type(
                        input_base.as_subclass(Tensor).dtype
                    ),
                )
            )

        return JitTensor.attach_hbir_to_tensor(output, hbir_rets)


@JitTensor.register_converter(Tensor.split)
class TensorSplitConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        self: JitTensor,
        split_size: Union[int, List[int]],
        dim: int = 0,
    ):
        return TorchSplitConverter.convert(output, self, split_size, dim)


@JitTensor.register_converter(torch.chunk, Tensor.chunk)
class ChunkConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        chunks: int,
        dim: int = 0,
    ):
        size = JitTensor.get_base(input).size(dim)
        each_size = math.ceil(size / chunks)
        sections = [each_size] * (size // each_size)
        last_size = size % each_size
        if last_size != 0:
            sections.append(last_size)
        return TorchSplitConverter.convert(output, input, sections, dim)


@JitTensor.register_converter(F.softplus)
class SoftplusConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, input, beta=1, threshold=20):
        return hbir.softplus(input, float(beta), float(threshold))


@JitTensor.register_converter(F.softmax)
class FSoftmaxConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        dim: Optional[int] = None,
        _stacklevel: int = 3,
        dtype: Optional[torch.dtype] = None,
    ):
        hbir_input = JitTensor.gather_hbir(input)
        input_base = JitTensor.get_base(input)

        if dim is None:
            from torch.nn.functional import _get_softmax_dim

            dim = _get_softmax_dim("softmax", input_base.ndim, 3)
        if dtype is not None:
            dtype = get_hbir_tensor_type(dtype, list(output.shape))

        hbir_output = hbir.softmax(hbir_input, dim, output_type=dtype)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(Tensor.softmax, torch.softmax)
class SoftmaxConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        dim: int,
        dtype: Optional[torch.dtype] = None,
    ):
        return FSoftmaxConverter.convert(output, input, dim, dtype=dtype)


@JitTensor.register_converter(torch.sub, Tensor.sub)
class SubConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls, input: ir.Value, other: ir.Value, *, alpha: Optional[Number] = 1
    ):
        if alpha != 1:
            other = hbir.mul(other, alpha)

        return hbir.sub(input, other)


@JitTensor.register_converter(Tensor.sub_)
class SubInplacedConverter(SubConverter):
    @classmethod
    def convert(cls, output: Tensor, *args, **kwargs):
        hbir_args = JitTensor.gather_hbir(args)
        hbir_kwargs = JitTensor.gather_hbir(kwargs)
        hbir_output = cls.convert_with_hbir(*hbir_args, **hbir_kwargs)
        JitTensor.attach_inplaced_output(args[0], hbir_output)

        return args[0]


@JitTensor.register_converter(torch.sum, Tensor.sum)
class SumConverter(FuncConverterBase):
    @classmethod
    def convert_allreduce_sum(cls, input: ir.Value, input_dim):
        dims = list(range(input_dim))
        return hbir.reduce_sum(input, dims, False)

    @classmethod
    def convert_reduce_sum(cls, input: ir.Value, dim: int, keepdim=False):
        return hbir.reduce_sum(input, [dim], keepdim)

    @classmethod
    def convert(cls, output, *args, **kwargs):
        hbir_args = JitTensor.gather_hbir(args)
        hbir_kwargs = JitTensor.gather_hbir(kwargs)
        if len(args) + len(kwargs) == 1:
            hbir_output = cls.convert_allreduce_sum(
                *hbir_args,
                **hbir_kwargs,
                input_dim=JitTensor.get_base(args[0]).ndim,
            )
        else:
            hbir_output = cls.convert_reduce_sum(*hbir_args, **hbir_kwargs)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.tile, Tensor.tile)
class TileConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output: Tensor, input: JitTensor, *dims):
        if isinstance(dims[0], (list, tuple)):
            if len(dims) != 1:
                msg = "Invalid dims arg for repeat {}".format(dims)
                logger.error(msg)
                raise ValueError(msg)
            dims = dims[0]
        dims = list(dims)

        hbir_input = JitTensor.gather_hbir(input)
        input_base = JitTensor.get_base(input)

        output_type = get_hbir_tensor_type(output.as_subclass(Tensor).dtype)

        if len(dims) < input_base.ndim:
            dims = ([1] * (input_base.ndim - len(dims))) + dims

        if len(dims) > input_base.ndim:
            shape = ([1] * (len(dims) - input_base.ndim)) + list(
                input_base.shape
            )
            hbir_input = hbir.reshape(
                hbir_input, shape, output_type=output_type
            )

        hbir_output = hbir.tile(hbir_input, dims, output_type=output_type)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.topk, Tensor.topk)
class TopkConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        input: ir.Value,
        k: _int,
        dim: _int = -1,
        largest: _bool = True,
        sorted: _bool = True,
    ):
        return hbir.topk(
            input,
            k,
            dim,
            largest,
            sorted,
            indices_type=ir.UnrankedTensorType.get(
                ir.IntegerType.get_signed(64)
            ),
        )


@JitTensor.register_converter(torch.transpose, Tensor.transpose)
class TransposeConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls, output: JitTensor, input: JitTensor, dim0: _int, dim1: _int
    ):
        dims = list(range(JitTensor.get_base(input).ndim))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        hbir_input = JitTensor.gather_hbir(input)
        hbir_output = hbir.transpose(hbir_input, dims)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.t, Tensor.t)
class TConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output: JitTensor, input: JitTensor):
        ndim = JitTensor.get_base(input).ndim
        assert ndim <= 2
        hbir_input = JitTensor.gather_hbir(input)
        if ndim == 2:
            hbir_output = hbir.transpose(hbir_input, [1, 0])
        else:
            hbir_output = hbir_input
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


class TriluConverter(FuncConverterBase):
    torch_func = None

    @classmethod
    def convert(cls, output: Tensor, input: JitTensor, diagonal: _int = 0):
        input_base = JitTensor.get_base(input)
        mask = torch.ones_like(input_base, dtype=torch.bool)
        mask = cls.torch_func(mask, diagonal)

        hbir_input = JitTensor.gather_hbir(input)
        hbir_mask = JitTensor.gather_hbir(mask)

        hbir_output = hbir.where(hbir_mask, hbir_input, 0.0)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(torch.tril, Tensor.tril)
class TrilConverter(TriluConverter):
    torch_func = torch.tril


@JitTensor.register_converter(torch.triu, Tensor.triu)
class TriuConverter(TriluConverter):
    torch_func = torch.triu


@JitTensor.register_converter(torch.unbind, Tensor.unbind)
class UnbindConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output: List[Tensor],
        input: JitTensor,
        dim: int = 0,
    ):
        hbir_input = JitTensor.gather_hbir(input)
        input_base = JitTensor.get_base(input)

        hbir_rets = []
        for i in range(input_base.size(dim)):
            hbir_rets.append(hbir.index(hbir_input, i, dim))

        return JitTensor.attach_hbir_to_tensor(output, hbir_rets)


@JitTensor.register_converter(torch.where, Tensor.where)
class WhereConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        condition: ir.Value,
        input: ir.Value,
        other: ir.Value,
    ):
        return hbir.where(condition, input, other)
