"""Fused conv2d+add+relu modules."""
import torch
import torch.nn.functional as F  # noqa N812
from torch import nn
from torch.autograd import Function

import horizon_plugin_pytorch as hz
from horizon_plugin_pytorch.march import March, get_march
from horizon_plugin_pytorch.nn import intrinsic
from horizon_plugin_pytorch.nn.qat.qat_meta import (
    init_input_preprocess,
    pre_process,
)
from horizon_plugin_pytorch.qat_mode import handle_relu6_trick
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.onnx_helper import (
    _set_is_in_onnx_export_false,
)
from . import compatible_ops
from .functional import scale_quanti

__all__ = [
    "ConvTranspose2d",
    "ConvTransposeReLU2d",
    "ConvTransposeAdd2d",
    "ConvTransposeAddReLU2d",
    "ConvTransposeReLU62d",
    "ConvTransposeAddReLU62d",
]


class _FakeQuantWeight(Function):
    @staticmethod
    def forward(ctx, weight, groups, weight_fake_quant):
        wsize = weight.size()
        qat_weight = weight.reshape(
            (
                groups,
                wsize[0] // groups,
                wsize[1],
                wsize[2],
                wsize[3],
            )
        )
        qat_weight = qat_weight.transpose(dim0=1, dim1=2)
        qat_weight = qat_weight.reshape(
            (
                wsize[1] * groups,
                wsize[0] // groups,
                wsize[2],
                wsize[3],
            )
        )
        qat_weight = weight_fake_quant(qat_weight).as_subclass(torch.Tensor)

        wsize = weight.size()
        qat_weight = qat_weight.reshape(
            (
                groups,
                wsize[1],
                wsize[0] // groups,
                wsize[2],
                wsize[3],
            )
        )
        qat_weight = qat_weight.transpose(dim0=1, dim1=2)
        qat_weight = qat_weight.reshape(wsize)

        return qat_weight

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

    @staticmethod
    def symbolic(g, weight, groups, weight_fake_quant):
        scale = g.op(
            "Constant",
            value_t=weight_fake_quant.scale,
        )
        zero_point = g.op(
            "Constant",
            value_t=weight_fake_quant.zero_point.char(),
        )

        return g.op(
            "horizon::HzDequantize",
            g.op(
                "horizon::HzQuantize",
                weight,
                scale,
                zero_point,
                bits_i=8,
                axis_i=1,
            ),
            scale,
            zero_point,
            axis_i=1,
        )


class ConvTranspose2d(nn.ConvTranspose2d):
    r"""Refine this docstring in the future.

    A ConvTrnaspose2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.ConvTranspose2d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv2d#
    torch.nn.ConvTranspose2d
    for documentation.

    Similar to `torch.nn.ConvTranspose2d`, with FakeQuantize modules
    initialized to default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE = nn.ConvTranspose2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for \
                ConvTranspose2d"
            )
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight(channel_len=out_channels)
        self.activation_post_process = None
        if self.qconfig.activation is not None:
            self.activation_post_process = self.qconfig.activation(
                channel_len=out_channels
            )

        if get_march() == March.BERNOULLI:
            self.register_buffer(
                "bias_scale", torch.ones(out_channels, dtype=torch.float32)
            )
        self.activation_pre_process = init_input_preprocess(self.qconfig)

    def _get_weight_for_fake_quant(self, fused_weight=None):
        return self._convert_weight()

    def _fake_quant_bias(self, input_scale, weight_scale):
        if get_march() != March.BERNOULLI or self.bias is None:
            return self.bias
        else:
            _, e = torch.ops.horizon.frexp(torch.clamp(self.bias, -128, 127))
            bias_shift = e - 7
            if input_scale is not None and weight_scale is not None:
                input_shift = torch.log2(input_scale).to(torch.int32)
                weight_shift = torch.log2(weight_scale).to(torch.int32)
                bias_shift.clamp_(
                    input_shift + weight_shift - 23,
                    input_shift + weight_shift + 23,
                )
            self.bias_scale.copy_(2.0 ** bias_shift)
            return scale_quanti(
                self.bias,
                self.bias_scale,
                torch.zeros(len(self.bias_scale), dtype=torch.int64).to(
                    self.bias.device
                ),
                0,
                -128,
                127,
                True,
                False,
            )

    def _convert_weight(self):
        wsize = self.weight.size()
        qat_weight = self.weight.reshape(
            (
                self.groups,
                wsize[0] // self.groups,
                wsize[1],
                wsize[2],
                wsize[3],
            )
        )
        qat_weight = qat_weight.transpose(dim0=1, dim1=2)
        qat_weight = qat_weight.reshape(
            (
                wsize[1] * self.groups,
                wsize[0] // self.groups,
                wsize[2],
                wsize[3],
            )
        )
        return qat_weight

    def _restore_weight(self, qat_weight):
        wsize = self.weight.size()
        qat_weight = qat_weight.reshape(
            (
                self.groups,
                wsize[1],
                wsize[0] // self.groups,
                wsize[2],
                wsize[3],
            )
        )
        qat_weight = qat_weight.transpose(dim0=1, dim1=2)
        qat_weight = qat_weight.reshape(wsize)
        return qat_weight

    def _fake_quant_weight(self):
        # remove reshape and transpose in symbolic function. otherwise, the
        # weight scale of convtranspose in hybrid onnx model cannot be found
        # by horizon nn. The backward isn't implemented, only used in exporting
        # onnx.
        if torch.onnx.is_in_onnx_export():
            with _set_is_in_onnx_export_false():
                fake_quant_weight = _FakeQuantWeight.apply(
                    self.weight,
                    self.groups,
                    self.weight_fake_quant,
                )
            return fake_quant_weight

        qat_weight = self.weight_fake_quant(self._convert_weight())
        return self._restore_weight(qat_weight.as_subclass(torch.Tensor))

    def _forward_func(self, output_padding):
        def _func(*input):
            qat_weight = self._fake_quant_weight()
            out = F.conv_transpose2d(
                input[0].as_subclass(torch.Tensor),
                qat_weight,
                self._fake_quant_bias(
                    input[0].q_scale()
                    if isinstance(input[0], QTensor)
                    else None,
                    self.weight_fake_quant.scale,
                ),
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
            if self.activation_post_process is not None:
                return self.activation_post_process(out)
            else:
                return out

        return _func

    def forward(self, input, output_size=None):
        input = pre_process(self.activation_pre_process, input)
        output_padding = self._output_padding(
            input.as_subclass(torch.Tensor),
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            self.dilation,
        )
        return self._forward_func(output_padding)(input)

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

        if (
            type(mod) == intrinsic.ConvTransposeReLU2d
            or type(mod) == intrinsic.ConvTransposeAdd2d
            or type(mod) == intrinsic.ConvTransposeAddReLU2d
            or type(mod) == intrinsic.ConvTransposeReLU62d
            or type(mod) == intrinsic.ConvTransposeAddReLU62d
        ):
            conv_transpose2d_mod = mod.conv_transpose2d
        else:
            conv_transpose2d_mod = mod

        qat_deconv = cls(
            conv_transpose2d_mod.in_channels,
            conv_transpose2d_mod.out_channels,
            conv_transpose2d_mod.kernel_size,
            stride=conv_transpose2d_mod.stride,
            padding=conv_transpose2d_mod.padding,
            output_padding=conv_transpose2d_mod.output_padding,
            dilation=conv_transpose2d_mod.dilation,
            groups=conv_transpose2d_mod.groups,
            bias=conv_transpose2d_mod.bias is not None,
            padding_mode=conv_transpose2d_mod.padding_mode,
            qconfig=mod.qconfig,
        )
        qat_deconv.weight = conv_transpose2d_mod.weight
        qat_deconv.bias = conv_transpose2d_mod.bias

        if hasattr(qat_deconv, "swap_inputs"):
            assert hasattr(mod, "_swap_inputs")
            qat_deconv.swap_inputs(mod._swap_inputs)

        return qat_deconv


@handle_relu6_trick
class ConvTransposeReLU2d(ConvTranspose2d):
    r"""Refine this docstring in the future.

    A ConvTransposeReLU2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.


    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE = intrinsic.ConvTransposeReLU2d
    _version = 2

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvTransposeReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )
        self.use_relu6 = False

    def _forward_func(self, output_padding):
        def _func(*input):
            qat_weight = self._fake_quant_weight()
            out = F.conv_transpose2d(
                input[0].as_subclass(torch.Tensor),
                qat_weight,
                self._fake_quant_bias(
                    input[0].q_scale()
                    if isinstance(input[0], QTensor)
                    else None,
                    self.weight_fake_quant.scale,
                ),
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )

            if self.activation_post_process is not None:
                out = hz.nn.qat.compatible_ops.relu(out, self.use_relu6)
                return self.activation_post_process(out)
            else:
                out = compatible_ops.relu(out)
                return out

        return _func


class ConvTransposeReLU62d(ConvTranspose2d):
    r"""Refine this docstring in the future.

    A ConvTransposeReLU62d module attached with FakeQuantize modules for
    weight, used for quantization aware training.


    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE = intrinsic.ConvTransposeReLU62d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvTransposeReLU62d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )

    def _forward_func(self, output_padding):
        def _func(*input):
            qat_weight = self._fake_quant_weight()
            out = F.conv_transpose2d(
                input[0].as_subclass(torch.Tensor),
                qat_weight,
                self._fake_quant_bias(
                    input[0].q_scale()
                    if isinstance(input[0], QTensor)
                    else None,
                    self.weight_fake_quant.scale,
                ),
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
            out = compatible_ops.relu6(out)
            if self.activation_post_process is not None:
                return self.activation_post_process(out)
            else:
                return out

        return _func


class ConvTransposeAdd2d(ConvTranspose2d):
    r"""Refine this docstring in the future.

    A ConvTransposeAdd2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.


    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE = intrinsic.ConvTransposeAdd2d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvTransposeAdd2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
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
        return self.__call__(input1, input2)

    def _forward_func(self, output_padding):
        def _func(*input):
            qat_weight = self._fake_quant_weight()
            out = F.conv_transpose2d(
                input[0].as_subclass(torch.Tensor),
                qat_weight,
                self._fake_quant_bias(
                    input[0].q_scale()
                    if isinstance(input[0], QTensor)
                    else None,
                    self.weight_fake_quant.scale,
                ),
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
            out = out + input[1]
            if self.activation_post_process is not None:
                return self.activation_post_process(out)
            else:
                return out

        return _func

    def forward(self, input1, input2, output_size=None):
        # When fused, Floatfunctional is replaced by XxxAdd and is called by
        # self.add.add, so we need def add and forward here at the same time.
        input1, input2 = pre_process(
            self.activation_pre_process, input1, input2
        )
        output_padding = self._output_padding(
            input1.as_subclass(torch.Tensor),
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            self.dilation,
        )
        return self._forward_func(output_padding)(
            input1, input2.as_subclass(torch.Tensor)
        )


@handle_relu6_trick
class ConvTransposeAddReLU2d(ConvTransposeAdd2d):
    r"""Refine this docstring in the future.

    A ConvTransposeAddReLU2d module attached with FakeQuantize modules for
    weight, used for quantization aware training.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE = intrinsic.ConvTransposeAddReLU2d
    _version = 2

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvTransposeAddReLU2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )
        self.use_relu6 = False

    def add(self, input1, input2):
        input1, input2 = pre_process(
            self.activation_pre_process, input1, input2
        )
        return self.__call__(input1, input2)

    def _forward_func(self, output_padding):
        def _func(*input):
            qat_weight = self._fake_quant_weight()
            out = F.conv_transpose2d(
                input[0].as_subclass(torch.Tensor),
                qat_weight,
                self._fake_quant_bias(
                    input[0].q_scale()
                    if isinstance(input[0], QTensor)
                    else None,
                    self.weight_fake_quant.scale,
                ),
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
            out = out + input[1]

            if self.activation_post_process is not None:
                out = hz.nn.qat.compatible_ops.relu(out, self.use_relu6)
                return self.activation_post_process(out)
            else:
                out = compatible_ops.relu(out)
                return out

        return _func


class ConvTransposeAddReLU62d(ConvTransposeAdd2d):
    r"""Refine this docstring in the future.

    A ConvTransposeAddReLU62d module attached with FakeQuantize modules for
    weight, used for quantization aware training.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE = intrinsic.ConvTransposeAddReLU62d

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super(ConvTransposeAddReLU62d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
        )

    def add(self, input1, input2):
        return self.__call__(input1, input2)

    def _forward_func(self, output_padding):
        def _func(*input):
            qat_weight = self._fake_quant_weight()
            out = F.conv_transpose2d(
                input[0].as_subclass(torch.Tensor),
                qat_weight,
                self._fake_quant_bias(
                    input[0].q_scale()
                    if isinstance(input[0], QTensor)
                    else None,
                    self.weight_fake_quant.scale,
                ),
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
            out = out + input[1]
            out = compatible_ops.relu6(out)
            if self.activation_post_process is not None:
                return self.activation_post_process(out)
            else:
                return out

        return _func
