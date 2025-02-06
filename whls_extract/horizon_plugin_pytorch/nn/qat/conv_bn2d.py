"""Fused conv2d+add+relu modules."""

import copy

import torch
from torch import nn

import horizon_plugin_pytorch as hz
from horizon_plugin_pytorch.march import March, get_march
from horizon_plugin_pytorch.nn import intrinsic
from horizon_plugin_pytorch.nn.batchnorm import BatchNorm2d
from horizon_plugin_pytorch.qat_mode import (
    QATMode,
    get_qat_mode,
    handle_relu6_trick,
)
from horizon_plugin_pytorch.qtensor import QTensor
from . import compatible_ops
from .qat_meta import BiasHook, init_input_preprocess, pre_process

__all__ = [
    "ConvBN2d",
    "ConvBNReLU2d",
    "ConvBNAdd2d",
    "ConvBNAddReLU2d",
    "ConvBNReLU62d",
    "ConvBNAddReLU62d",
]


class ConvBN2d(nn.Conv2d):
    r"""Refine this docstring in the future.

    A ConvBN2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.
    """

    _FLOAT_MODULE = intrinsic.ConvBN2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        if self.qconfig.activation is not None:
            self.activation_post_process = self.qconfig.activation()
        else:
            self.activation_post_process = None
        if self.qconfig.weight is None:
            raise ValueError("qconfig must include weight")
        self.weight_fake_quant = self.qconfig.weight(
            channel_len=self.out_channels
        )
        if get_march() == March.BERNOULLI:
            self.register_buffer(
                "bias_scale",
                torch.ones(self.out_channels, dtype=torch.float32),
            )

            if get_qat_mode() == QATMode.WithBNReverseFold:
                raise ValueError(
                    "QATMode.WithBNReverseFold mode is "
                    "not supported for {}".format(March.BERNOULLI)
                )
        self.activation_pre_process = init_input_preprocess(self.qconfig)

    def _fake_quant_bias(self, input_scale):
        BiasHook.set_input_scale(input_scale)
        with BiasHook.enable():
            return BiasHook.process(self, self.bias)

    def _fuse_bn_weight(self):
        assert self.bn.running_var is not None
        running_std = torch.rsqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight * running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        fused_weight = self.weight * scale_factor.reshape(weight_shape)
        return fused_weight, scale_factor

    def _forward_with_fold_and_reverse_bn(self, input):
        fused_weight, scale_factor = self._fuse_bn_weight()
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(fused_weight)
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
        else:
            zero_bias = torch.zeros(
                self.out_channels,
                device=scaled_weight.as_subclass(torch.Tensor).device,
            )
        conv = self._conv_forward(
            input.dequantize(),
            scaled_weight.as_subclass(torch.Tensor),
            zero_bias,
        )
        conv_orig = conv / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            conv_orig = (
                conv_orig
                + self._fake_quant_bias(
                    input.q_scale() if isinstance(input, QTensor) else None,
                ).reshape(bias_shape)
            )
        # adapt for torch.cuda.amp.autocast
        conv_bn_out = self.bn(conv_orig.to(conv.dtype))
        return conv_bn_out

    def _get_weight_for_fake_quant(self):
        # for calibration
        if get_qat_mode() == QATMode.WithBNReverseFold and isinstance(
            self.bn, nn.modules.batchnorm._BatchNorm
        ):
            return self._fuse_bn_weight()[0]
        else:
            return self.weight

    def _conv_bn_forward(self, input):
        if get_qat_mode() == QATMode.WithBNReverseFold and isinstance(
            self.bn, nn.modules.batchnorm._BatchNorm
        ):
            out = self._forward_with_fold_and_reverse_bn(input)
        else:
            out = self._conv_forward(
                input.as_subclass(torch.Tensor),
                self.weight_fake_quant(self.weight).as_subclass(torch.Tensor),
                self._fake_quant_bias(
                    input.q_scale() if isinstance(input, QTensor) else None,
                ),
            )
            out = self.bn(out)

        return out

    def forward(self, input):
        input = pre_process(self.activation_pre_process, input)
        out = self._conv_bn_forward(input)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out

    def fuse_norm(self, inplace=False):
        if inplace:
            (self.weight, self.bias) = nn.utils.fuse_conv_bn_weights(
                self.weight,
                self.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.bn.weight,
                self.bn.bias,
            )
            self.bn = nn.Identity()
            return self
        else:
            fused_conv = copy.deepcopy(self)
            (
                fused_conv.weight,
                fused_conv.bias,
            ) = nn.utils.fuse_conv_bn_weights(
                self.weight,
                self.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.bn.weight,
                self.bn.bias,
            )
            fused_conv.bn = nn.Identity()
            return fused_conv

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

        conv = mod.conv
        qat_conv_bn = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            qconfig=mod.qconfig,
        )
        with torch.no_grad():
            qat_conv_bn.weight = conv.weight
            qat_conv_bn.bias = conv.bias

            from horizon_plugin_pytorch.nn import ChannelScale2d

            nn.SyncBatchNorm
            if isinstance(mod.bn, ChannelScale2d):
                qat_conv_bn.bn = copy.deepcopy(mod.bn)
            else:
                # Use our bn to support horizon_plugin_pytorch.utils.checkpoint
                qat_conv_bn.bn = BatchNorm2d(
                    mod.bn.num_features,
                    mod.bn.eps,
                    mod.bn.momentum,
                    mod.bn.affine,
                    mod.bn.track_running_stats,
                    mod.bn.weight.device,
                    mod.bn.weight.dtype,
                )
                qat_conv_bn.bn.weight = mod.bn.weight
                qat_conv_bn.bn.bias = mod.bn.bias
                qat_conv_bn.bn.running_mean = mod.bn.running_mean
                qat_conv_bn.bn.running_var = mod.bn.running_var
                qat_conv_bn.bn.num_batches_tracked = mod.bn.num_batches_tracked

        if hasattr(qat_conv_bn, "swap_inputs"):
            assert hasattr(mod, "_swap_inputs")
            qat_conv_bn.swap_inputs(mod._swap_inputs)

        return qat_conv_bn


@handle_relu6_trick
class ConvBNReLU2d(ConvBN2d):
    r"""Refine this docstring in the future.

    A ConvBNReLU2d module is a fused module of Conv2d and BatchNorm2d and
    ReLU, attached with FakeQuantize modules for weight for quantization
    aware training.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE = intrinsic.ConvBNReLU2d
    _version = 2

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvBNReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )
        self.use_relu6 = False

    def forward(self, input):
        input = pre_process(self.activation_pre_process, input)
        out = self._conv_bn_forward(input)
        if self.activation_post_process is not None:
            out = hz.nn.qat.compatible_ops.relu(out, self.use_relu6)
            return self.activation_post_process(out)
        else:
            out = compatible_ops.relu(out)
            return out


class ConvBNReLU62d(ConvBN2d):
    r"""Refine this docstring in the future.

    A ConvReLU62d module is a fused module of Conv2d and BatchNorm2d and ReLU,
    attached with FakeQuantize modules for weight for quantization aware
    training.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE = intrinsic.ConvBNReLU62d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvBNReLU62d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )

    def forward(self, input):
        input = pre_process(self.activation_pre_process, input)
        out = self._conv_bn_forward(input)
        out = compatible_ops.relu6(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out


class ConvBNAdd2d(ConvBN2d):
    _FLOAT_MODULE = intrinsic.ConvBNAdd2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvBNAdd2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )
        self._swap_inputs = False

    def swap_inputs(self, v=True):
        self._swap_inputs = v

    def __call__(self, x, y):
        if self._swap_inputs:
            x, y = y, x
        return super().__call__(x, y)

    def add(self, input1, input2):
        input1, input2 = pre_process(
            self.activation_pre_process, input1, input2
        )
        return self.__call__(input1, input2)

    def forward(self, input1, input2):
        input1, input2 = pre_process(
            self.activation_pre_process, input1, input2
        )
        out = self._conv_bn_forward(input1)
        out = out + input2.as_subclass(torch.Tensor)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out


@handle_relu6_trick
class ConvBNAddReLU2d(ConvBNAdd2d):
    _FLOAT_MODULE = intrinsic.ConvBNAddReLU2d
    _version = 2

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvBNAddReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )
        self.use_relu6 = False

    def forward(self, input1, input2):
        input1, input2 = pre_process(
            self.activation_pre_process, input1, input2
        )
        out = self._conv_bn_forward(input1)
        out = out + input2.as_subclass(torch.Tensor)
        if self.activation_post_process is not None:
            out = hz.nn.qat.compatible_ops.relu(out, self.use_relu6)
            return self.activation_post_process(out)
        else:
            out = compatible_ops.relu(out)
            return out


class ConvBNAddReLU62d(ConvBNAdd2d):
    _FLOAT_MODULE = intrinsic.ConvBNAddReLU62d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvBNAddReLU62d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )

    def forward(self, input1, input2):
        input1, input2 = pre_process(
            self.activation_pre_process, input1, input2
        )
        out = self._conv_bn_forward(input1)
        out = out + input2.as_subclass(torch.Tensor)
        out = compatible_ops.relu6(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out
