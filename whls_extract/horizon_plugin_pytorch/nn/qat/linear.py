import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.autograd import Function

from horizon_plugin_pytorch.march import March, get_march
from horizon_plugin_pytorch.nn import intrinsic
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.load_state_dict_helper import get_version
from horizon_plugin_pytorch.utils.onnx_helper import (
    _set_is_in_onnx_export_false,
)

__all__ = [
    "Linear",
    "LinearReLU",
    "LinearReLU6",
    "LinearAdd",
    "LinearAddReLU",
    "LinearAddReLU6",
]


def _fake_quant_weight(weight, out_features, in_features, weight_fake_quant):
    return weight_fake_quant(
        weight.reshape(out_features, in_features, 1, 1)
    ).reshape(weight.shape)


class _FakeQuantWeight(Function):
    @staticmethod
    def forward(ctx, weight, out_features, in_features, weight_fake_quant):
        return _fake_quant_weight(
            weight, out_features, in_features, weight_fake_quant
        )

    @staticmethod
    def symbolic(g, weight, out_features, in_features, weight_fake_quant):
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
                axis_i=0,
            ),
            scale,
            zero_point,
            axis_i=0,
        )


class Linear(nn.Linear):
    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features, out_features, bias, qconfig=None):
        super(Linear, self).__init__(in_features, out_features, bias)
        assert get_march() in [
            March.BAYES,
            March.BAYES_E,
            March.NASH,
            March.NASH_E,
            March.NASH_M,
            March.NASH_P,
            March.META,
        ]
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight(channel_len=out_features)
        self.activation_post_process = None
        if self.qconfig.activation is not None:
            self.activation_post_process = self.qconfig.activation()

    def _get_weight_for_fake_quant(self):
        return self.weight.reshape(self.out_features, self.in_features, 1, 1)

    def _linear(self, input):
        if torch.onnx.is_in_onnx_export():
            with _set_is_in_onnx_export_false():
                fake_quant_weight = _FakeQuantWeight.apply(
                    self.weight,
                    self.out_features,
                    self.in_features,
                    self.weight_fake_quant,
                ).as_subclass(torch.Tensor)
        else:
            fake_quant_weight = _fake_quant_weight(
                self.weight,
                self.out_features,
                self.in_features,
                self.weight_fake_quant,
            ).as_subclass(torch.Tensor)

        out = torch.nn.functional.linear(
            input.as_subclass(torch.Tensor),
            fake_quant_weight,
            self.bias,
        )
        return out

    def forward(self, input: QTensor) -> QTensor:
        out = self._linear(input)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        return out

    @classmethod
    def from_float(cls, mod):
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
            type(mod) == intrinsic.LinearAdd
            or type(mod) == intrinsic.LinearReLU
            or type(mod) == intrinsic.LinearReLU6
            or type(mod) == intrinsic.LinearAddReLU
            or type(mod) == intrinsic.LinearAddReLU6
        ):
            linear_mod = mod.linear
        else:
            linear_mod = mod

        qat_linear = cls(
            linear_mod.in_features,
            linear_mod.out_features,
            linear_mod.bias is not None,
            qconfig=mod.qconfig,
        )
        qat_linear.weight = linear_mod.weight
        qat_linear.bias = linear_mod.bias

        if hasattr(qat_linear, "swap_inputs"):
            assert hasattr(mod, "_swap_inputs")
            qat_linear.swap_inputs(mod._swap_inputs)

        return qat_linear


class LinearReLU(Linear):
    _FLOAT_MODULE = intrinsic.LinearReLU
    _version = 2

    def __init__(self, in_features, out_features, bias, qconfig=None):
        super(LinearReLU, self).__init__(
            in_features, out_features, bias, qconfig
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if get_version(self, prefix, local_metadata) == 1:
            linear_prefix = prefix + "0."
            for key in list(state_dict.keys()):
                if key.startswith(linear_prefix):
                    value = state_dict[key]
                    suffix = key.split(linear_prefix)[-1]
                    state_dict.pop(key)
                    state_dict[prefix + suffix] = value

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, input: QTensor) -> QTensor:
        out = self._linear(input)
        out = F.relu(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out


class LinearReLU6(Linear):
    _FLOAT_MODULE = intrinsic.LinearReLU6

    def __init__(self, in_features, out_features, bias, qconfig=None):
        super(LinearReLU6, self).__init__(
            in_features, out_features, bias, qconfig
        )

    def forward(self, input: QTensor) -> QTensor:
        out = self._linear(input)
        out = F.relu6(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out


class LinearAdd(Linear):
    _FLOAT_MODULE = intrinsic.LinearAdd

    def __init__(self, in_features, out_features, bias, qconfig=None):
        super(LinearAdd, self).__init__(
            in_features, out_features, bias, qconfig
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

    def forward(self, input1: QTensor, input2: QTensor) -> QTensor:
        out = self._linear(input1)
        out = out + input2.as_subclass(torch.Tensor)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out


class LinearAddReLU(LinearAdd):
    _FLOAT_MODULE = intrinsic.LinearAddReLU

    def __init__(self, in_features, out_features, bias, qconfig=None):
        super(LinearAddReLU, self).__init__(
            in_features, out_features, bias, qconfig
        )

    def forward(self, input1: QTensor, input2: QTensor) -> QTensor:
        out = self._linear(input1)
        out = out + input2.as_subclass(torch.Tensor)
        out = F.relu(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out


class LinearAddReLU6(LinearAdd):
    _FLOAT_MODULE = intrinsic.LinearAddReLU6

    def __init__(self, in_features, out_features, bias, qconfig=None):
        super(LinearAddReLU6, self).__init__(
            in_features, out_features, bias, qconfig
        )

    def forward(self, input1: QTensor, input2: QTensor) -> QTensor:
        out = self._linear(input1)
        out = out + input2.as_subclass(torch.Tensor)
        out = F.relu6(out)
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        else:
            return out
