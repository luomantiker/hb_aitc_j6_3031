import torch
from torch import nn

from horizon_plugin_pytorch.dtype import qint8
from ..floor import Floor as FloatFloor
from .functional_modules import FloatFunctional
from .segment_lut import SegmentLUT


class Floor(nn.Module):
    """qat version of Floor module."""

    _FLOAT_MODULE = FloatFloor

    def __init__(self, qconfig):
        super(Floor, self).__init__()
        self.qconfig = qconfig
        assert qconfig is not None, "qconfig must be provided"
        assert (
            self.qconfig.activation is not None
        ), "qconfig activation must be provided"

        if self.qconfig.activation().dtype == qint8:
            self.floor = SegmentLUT(
                torch.floor,
                False,
                None,
                input_range=None,
                auto_divide_strategy="evenly",
                qconfig=qconfig,
            )
        else:
            self.floor = FloatFunctional(qconfig=qconfig)

    def forward(self, input: torch.Tensor):
        # float
        if isinstance(self.floor, SegmentLUT) or (
            hasattr(self.floor, "_QAT_MODULE")
            and self.floor._QAT_MODULE is SegmentLUT
        ):
            assert input.dtype == qint8, (
                "nn.Floor only support same input/output dtype, but get input,"
                "{} while output qint8. Try nn.quantized.FloatFunctional.floor"
                " instead.".format(input.dtype)
            )
            return self.floor(input)

        return self.floor.floor(input)

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
        qat_floor = cls(qconfig=qconfig)
        return qat_floor
