import torch
from torch import nn

import horizon_plugin_pytorch as hz
from horizon_plugin_pytorch.utils.load_state_dict_helper import (
    replace_mod_name,
)
from ..div import Div as float_div  # noqa: N813
from .functional_modules import FloatFunctional
from .qat_meta import init_input_preprocess, is_float, pre_process
from .reciprocal import Reciprocal
from .segment_lut import SegmentLUT


class Div(nn.Module):
    """Qat version of div module."""

    _FLOAT_MODULE = float_div

    def __init__(self, qconfig=None):
        super(Div, self).__init__()
        assert qconfig is not None, "qconfig must be provided"
        self.qconfig = qconfig
        assert (
            self.qconfig.activation is not None
        ), "qconfig activation must be provided"
        self.activation_pre_process = init_input_preprocess(qconfig)
        self.activation_post_process = self.qconfig.activation()

    def forward(self, x, y):
        if isinstance(y, (float, int)):
            r = torch.mul(x.as_subclass(torch.Tensor), 1 / y)
        else:
            x, y = pre_process(self.activation_pre_process, x, y)
            r = torch.div(
                x.as_subclass(torch.Tensor),
                y.as_subclass(torch.Tensor),
            )
        return self.activation_post_process(r)

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
        assert mod.qconfig, "Input float module must have a valid qconfig"

        if not is_float(mod.qconfig.activation()) and SegmentLUT.activated():
            return SegmentLUTDiv.from_float(mod)

        qconfig = mod.qconfig
        qat_div = cls(qconfig=qconfig)
        return qat_div


class SegmentLUTDiv(nn.Module):
    _FLOAT_MODULE = float_div

    def __init__(self, reciprocal_max_value, qconfig):
        super(SegmentLUTDiv, self).__init__()
        self.qconfig = qconfig

        int16_qconfig = (
            hz.quantization.qconfig.promote_int8_activation_to_int16(qconfig)
        )

        self.reciprocal = Reciprocal(
            max_value=reciprocal_max_value, qconfig=int16_qconfig
        )
        self.mul = FloatFunctional(qconfig=qconfig)

    def forward(self, input, other, rounding_mode=None):
        if rounding_mode is not None:
            raise ValueError(
                "Unsupported rounding_mode {}".format(rounding_mode)
            )

        if isinstance(other, torch.Tensor):
            rec = self.reciprocal(other)
        else:
            rec = 1 / other
        return self.mul.mul(input, rec)

    @replace_mod_name("reciprocal", "reciprocal.reciprocal")
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
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_div = cls(
            reciprocal_max_value=mod.reciprocal_max_value, qconfig=qconfig
        )
        return qat_div
