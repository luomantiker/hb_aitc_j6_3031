import torch.nn.intrinsic as nni
from torch import Tensor, nn
from torch.nn import functional as F  # noqa N812
from torch.nn.common_types import _size_3_t

import horizon_plugin_pytorch as hz
from horizon_plugin_pytorch.nn import intrinsic
from horizon_plugin_pytorch.nn.qat.qat_meta import (
    init_input_preprocess,
    pre_process,
)
from horizon_plugin_pytorch.qat_mode import handle_relu6_trick
from horizon_plugin_pytorch.qtensor import QTensor
from . import compatible_ops

__all__ = [
    "Conv3d",
    "ConvReLU3d",
    "ConvAdd3d",
    "ConvAddReLU3d",
    "ConvReLU63d",
    "ConvAddReLU63d",
]


class Conv3d(nn.Conv3d):
    _FLOAT_MODULE = nn.Conv3d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        qconfig=None,
    ) -> None:
        super(Conv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight(channel_len=out_channels)
        self.activation_post_process = None
        if self.qconfig.activation is not None:
            self.activation_post_process = self.qconfig.activation(
                channel_len=out_channels
            )
        self.activation_pre_process = init_input_preprocess(self.qconfig)

    def _conv_forward(self, input: QTensor):
        return super(Conv3d, self)._conv_forward(
            input.as_subclass(Tensor),
            self.weight_fake_quant(self.weight).as_subclass(Tensor),
            self.bias,
        )

    def forward(self, input: QTensor) -> QTensor:
        input = pre_process(self.activation_pre_process, input)
        out = self._conv_forward(input)

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
        assert mod.qconfig, "Input float module must have a valid qconfig"

        if type(mod) == nni.ConvReLU3d:
            conv3d_mod = mod[0]
        elif (
            type(mod) == intrinsic.ConvAdd3d
            or type(mod) == intrinsic.ConvReLU63d
            or type(mod) == intrinsic.ConvAddReLU3d
            or type(mod) == intrinsic.ConvAddReLU63d
        ):
            conv3d_mod = mod.conv
        else:
            conv3d_mod = mod

        qat_conv3d = cls(
            conv3d_mod.in_channels,
            conv3d_mod.out_channels,
            conv3d_mod.kernel_size,
            stride=conv3d_mod.stride,
            padding=conv3d_mod.padding,
            dilation=conv3d_mod.dilation,
            groups=conv3d_mod.groups,
            bias=conv3d_mod.bias is not None,
            padding_mode=conv3d_mod.padding_mode,
            device=None,
            dtype=None,
            qconfig=mod.qconfig,
        )
        qat_conv3d.weight.copy_(conv3d_mod.weight)
        if qat_conv3d.bias is not None:
            qat_conv3d.bias.copy_(conv3d_mod.bias)

        if hasattr(qat_conv3d, "swap_inputs"):
            assert hasattr(mod, "_swap_inputs")
            qat_conv3d.swap_inputs(mod._swap_inputs)

        return qat_conv3d


@handle_relu6_trick
class ConvReLU3d(Conv3d):
    _FLOAT_MODULE = nni.ConvReLU3d
    _version = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        qconfig=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
            qconfig,
        )
        self.use_relu6 = False

    def forward(self, input: QTensor):
        input = pre_process(self.activation_pre_process, input)
        out = self._conv_forward(input)

        if self.activation_post_process is not None:

            out = hz.nn.qat.compatible_ops.relu(out, self.use_relu6)
            return self.activation_post_process(out)
        else:
            out = compatible_ops.relu(out)
            return out


class ConvReLU63d(Conv3d):
    _FLOAT_MODULE = intrinsic.ConvReLU63d

    def forward(self, input: QTensor):
        input = pre_process(self.activation_pre_process, input)
        out = self._conv_forward(input)
        out = compatible_ops.relu6(out)

        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out


class ConvAdd3d(Conv3d):
    _FLOAT_MODULE = intrinsic.ConvAdd3d

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_pre_process = init_input_preprocess(self.qconfig)
        self._swap_inputs = False

    def swap_inputs(self, v=True):
        self._swap_inputs = v

    def __call__(self, x, y):
        if self._swap_inputs:
            x, y = y, x
        return super().__call__(x, y)

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    def forward(self, input1: QTensor, input2: QTensor):
        input1, input2 = pre_process(
            self.activation_pre_process, input1, input2
        )
        out = self._conv_forward(input1)
        out = out + input2.as_subclass(Tensor)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out


@handle_relu6_trick
class ConvAddReLU3d(ConvAdd3d):
    _FLOAT_MODULE = intrinsic.ConvAddReLU3d
    _version = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        qconfig=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
            qconfig,
        )
        self.use_relu6 = False

    def forward(self, input1, input2):
        input1, input2 = pre_process(
            self.activation_pre_process, input1, input2
        )
        out = self._conv_forward(input1)
        out = out + input2.as_subclass(Tensor)
        if self.activation_post_process is not None:

            out = hz.nn.qat.compatible_ops.relu(out, self.use_relu6)
            return self.activation_post_process(out)
        else:
            out = compatible_ops.relu(out)
            return out


class ConvAddReLU63d(ConvAdd3d):
    _FLOAT_MODULE = intrinsic.ConvAddReLU63d

    def forward(self, input1, input2):
        input1, input2 = pre_process(
            self.activation_pre_process, input1, input2
        )
        out = self._conv_forward(input1)
        out = out + input2.as_subclass(Tensor)
        out = compatible_ops.relu6(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out
