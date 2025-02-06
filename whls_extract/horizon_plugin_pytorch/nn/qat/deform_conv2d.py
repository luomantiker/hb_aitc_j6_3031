from typing import List, Optional

import torch
from torch.overrides import handle_torch_function, has_torch_function

from horizon_plugin_pytorch._torchvision_wrapper import ops
from horizon_plugin_pytorch.dtype import QuantDType, qint16
from horizon_plugin_pytorch.nn import intrinsic
from horizon_plugin_pytorch.nn.qat.qat_meta import (
    init_input_preprocess,
    pre_process,
)
from horizon_plugin_pytorch.qtensor import QTensor
from . import compatible_ops
from .functional import scale_quanti

__all__ = [
    "DeformConv2d",
    "DeformConvReLU2d",
    "DeformConvReLU62d",
    "DeformConvAdd2d",
    "DeformConvAddReLU2d",
    "DeformConvAddReLU62d",
]


def _to_pow_quantized(
    input: QTensor,
    min_shift: int = None,
    max_shift: int = None,
    output_dtype: Optional[QuantDType] = None,
):
    if has_torch_function and input.__class__ is not QTensor:
        return handle_torch_function(
            _to_pow_quantized, input, input, min_shift, max_shift, output_dtype
        )

    if output_dtype is None:
        output_dtype = input.dtype

    scale = input.q_scale()

    # Do not inplaced modify scale from QTensor.q_scale baceuse it may be
    # a buffer under FakeQuantize
    scale = scale * (input.dtype.min / output_dtype.min)
    fp_shift = -torch.log2(scale)
    # if input is already shift quantized, try to use origin shift value
    shift = torch.where(
        (fp_shift - fp_shift.round()).abs() < 1e-5,
        fp_shift.round(),
        fp_shift.floor(),
    )
    if min_shift is not None or max_shift is not None:
        shift = torch.clamp(shift, min_shift, max_shift)
    scale = 1 / 2 ** shift

    ret = scale_quanti(
        input.as_subclass(torch.Tensor),
        scale,
        input.q_zero_point(),
        -1,
        output_dtype.min,
        output_dtype.max,
        True,
        False,
    )

    return QTensor(
        ret,
        scale,
        output_dtype,
        input.q_per_channel_axis(),
    )


QTensor.register_func_impl(_to_pow_quantized)(_to_pow_quantized)


def deform_conv2d_torch_function(*args, **kwargs):
    if has_torch_function(args[:2]):
        return handle_torch_function(
            deform_conv2d_torch_function,
            args[:2],
            *args,
            **kwargs,
        )
    return ops.deform_conv2d(*args, **kwargs)


class DeformConv2d(ops.DeformConv2d):

    _FLOAT_MODULE = ops.DeformConv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        qconfig=None,
    ):
        super(DeformConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        assert qconfig is not None, "qconfig must be provided for QAT module"
        self.qconfig = qconfig

        self.weight_fake_quant = self.qconfig.weight(channel_len=out_channels)
        self.activation_post_process = None
        if self.qconfig.activation is not None:
            self.activation_post_process = self.qconfig.activation(
                channel_len=out_channels
            )
        self.activation_pre_process = init_input_preprocess(self.qconfig)

    def _conv_forward(
        self, input: QTensor, offset: QTensor, mask: QTensor = None
    ):
        if isinstance(offset, QTensor):
            offset = _to_pow_quantized(offset, 0, 8, qint16)
        return deform_conv2d_torch_function(
            input.as_subclass(torch.Tensor),
            offset.as_subclass(torch.Tensor),
            self.weight_fake_quant(self.weight).as_subclass(torch.Tensor),
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=None if mask is None else mask.as_subclass(torch.Tensor),
        )

    def forward(self, input: QTensor, offset: QTensor, mask: QTensor = None):
        input, offset, mask = pre_process(
            self.activation_pre_process, input, offset, mask
        )
        out = self._conv_forward(input, offset, mask)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict.

        Args: `mod` a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert (
            mod.qconfig is not None
        ), "Input float module must have a valid qconfig"

        if type(mod) in (
            intrinsic.DeformConvAdd2d,
            intrinsic.DeformConvAddReLU2d,
            intrinsic.DeformConvAddReLU62d,
            intrinsic.DeformConvReLU2d,
            intrinsic.DeformConvReLU62d,
        ):
            deform_conv2d_mod = mod.deform_conv2d
        else:
            deform_conv2d_mod = mod

        qat_mod = cls(
            deform_conv2d_mod.in_channels,
            deform_conv2d_mod.out_channels,
            deform_conv2d_mod.kernel_size,
            stride=deform_conv2d_mod.stride,
            padding=deform_conv2d_mod.padding,
            dilation=deform_conv2d_mod.dilation,
            groups=deform_conv2d_mod.groups,
            bias=deform_conv2d_mod.bias is not None,
            qconfig=mod.qconfig,
        )
        qat_mod.weight.copy_(deform_conv2d_mod.weight)
        if deform_conv2d_mod.bias is not None:
            qat_mod.bias.copy_(deform_conv2d_mod.bias)

        if hasattr(qat_mod, "swap_inputs"):
            assert hasattr(mod, "_swap_inputs")
            qat_mod.swap_inputs(mod._swap_inputs)

        return qat_mod


class DeformConvReLU2d(DeformConv2d):

    _FLOAT_MODULE = intrinsic.DeformConvReLU2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        qconfig=None,
    ):
        super(DeformConvReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            qconfig,
        )

    def forward(self, input: QTensor, offset: QTensor, mask: QTensor = None):
        input, offset, mask = pre_process(
            self.activation_pre_process, input, offset, mask
        )
        out = self._conv_forward(input, offset, mask)
        out = compatible_ops.relu(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out


class DeformConvReLU62d(DeformConv2d):

    _FLOAT_MODULE = intrinsic.DeformConvReLU62d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        qconfig=None,
    ):
        super(DeformConvReLU62d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            qconfig,
        )

    def forward(self, input: QTensor, offset: QTensor, mask: QTensor = None):
        input, offset, mask = pre_process(
            self.activation_pre_process, input, offset, mask
        )
        out = self._conv_forward(input, offset, mask)
        out = compatible_ops.relu6(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out


class DeformConvAdd2d(DeformConv2d):

    _FLOAT_MODULE = intrinsic.DeformConvAdd2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        qconfig=None,
    ):
        super(DeformConvAdd2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            qconfig,
        )
        self._swap_inputs = False

    def swap_inputs(self, v=True):
        self._swap_inputs = v

    def __call__(self, x, y):
        if self._swap_inputs:
            x, y = y, x
        return super().__call__(x, y)

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1: List[QTensor], x2: QTensor):
        if self.activation_pre_process:
            x = pre_process(self.activation_pre_process, *x1, x2)
            x1 = x[:-1]
            x2 = x[-1]
        out = self._conv_forward(*x1)
        out = torch.add(out, x2.as_subclass(torch.Tensor))
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out


class DeformConvAddReLU2d(DeformConvAdd2d):

    _FLOAT_MODULE = intrinsic.DeformConvAddReLU2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        qconfig=None,
    ):
        super(DeformConvAddReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            qconfig,
        )

    def forward(self, x1: List[QTensor], x2: QTensor):
        if self.activation_pre_process:
            x = pre_process(self.activation_pre_process, *x1, x2)
            x1 = x[:-1]
            x2 = x[-1]
        out = self._conv_forward(*x1)
        out = torch.add(out, x2.as_subclass(torch.Tensor))
        out = compatible_ops.relu(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out


class DeformConvAddReLU62d(DeformConvAdd2d):

    _FLOAT_MODULE = intrinsic.DeformConvAddReLU62d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        qconfig=None,
    ):
        super(DeformConvAddReLU62d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            qconfig,
        )

    def forward(self, x1: List[QTensor], x2: QTensor):
        if self.activation_pre_process:
            x = pre_process(self.activation_pre_process, *x1, x2)
            x1 = x[:-1]
            x2 = x[-1]
        out = self._conv_forward(*x1)
        out = torch.add(out, x2.as_subclass(torch.Tensor))
        out = compatible_ops.relu6(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out
