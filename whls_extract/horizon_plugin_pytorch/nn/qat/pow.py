import torch
from torch import nn

from horizon_plugin_pytorch.nn.qat.qat_meta import (
    init_input_preprocess,
    is_float,
    pre_process,
)
from horizon_plugin_pytorch.qtensor import QTensor
from ..pow import Pow as FloatPow
from .functional_modules import FloatFunctional
from .segment_lut import SegmentLUT


class Pow(nn.Module):
    r"""qat version of pow."""

    _FLOAT_MODULE = FloatPow

    def __init__(self, exponent=None, qconfig=None):
        super(Pow, self).__init__()
        self.exponent = exponent
        self.qconfig = qconfig
        self.is_float = is_float(qconfig.activation())
        if self.is_float:
            self.activation_post_process = self.qconfig.activation()
            self.input_pre_process = init_input_preprocess(self.qconfig)
            return

        if self.exponent is not None:
            self.init_submod(self.exponent)

    def _float16_forward(self, x):
        x = pre_process(self.input_pre_process, x)
        out = torch.pow(x, self.exponent)
        out = self.activation_post_process(out)
        return out

    def init_submod(self, exponent, device=None):
        if exponent == 2:
            self.mul = FloatFunctional(qconfig=self.qconfig)
            if device is not None:
                self.mul.to(device)
        else:
            self.pow = SegmentLUT(
                lambda x: torch.pow(x, self.exponent),
                False,
                None,
                qconfig=self.qconfig,
            )
            if device is not None:
                self.pow.to(device)

    def forward(self, input: QTensor, exponent=None):
        if self.is_float:
            return self._float16_forward(input)
        if exponent is None:
            assert (
                self.exponent is not None
            ), "exponent must be provided either in __init__ or forward"
            exponent = self.exponent
        else:
            if not isinstance(exponent, (int, float)):
                assert (
                    isinstance(exponent, torch.Tensor)
                    and exponent.numel() == 1
                ), "Only support power which exponent is scalar"
            if self.exponent is None:
                self.exponent = exponent
                self.init_submod(self.exponent, input.device)
            else:
                assert self.exponent == exponent, (
                    f"This Pow is only used for exponent {self.exponent}, "
                    f"but get {exponent}"
                )

        if self.exponent == 2:
            return self.mul.mul(input, input)
        else:
            return self.pow(input)

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

        qat_mod = cls(mod.exponent, qconfig=mod.qconfig)
        return qat_mod
