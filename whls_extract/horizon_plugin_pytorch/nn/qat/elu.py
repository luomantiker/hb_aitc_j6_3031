import torch
from torch import nn
from torch.quantization import QConfig

from .segment_lut import SegmentLUT


class ELU(nn.ELU):
    """Qat implementation of nn.ELU.

    Args:
        alpha (float) : the α value for the ELU formulation. Default is 1.0

        inplace (bool) : can optionally do the operation in-place.
            Default is False

    """

    _FLOAT_MODULE = nn.ELU

    def __init__(self, alpha=1.0, inplace=False, qconfig=None):
        super(ELU, self).__init__(alpha, inplace)
        self.qconfig = qconfig
        assert (
            self.qconfig.activation is not None
        ), "qconfig activation must be provided"
        self.elu = SegmentLUT(
            lambda x: torch.nn.functional.elu(x, self.alpha, self.inplace),
            False,
            qconfig=qconfig,
        )

    def forward(self, input, *args, **kwargs):
        return self.elu(input)

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
        assert isinstance(
            mod.qconfig, QConfig
        ), "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_elu = cls(mod.alpha, mod.inplace, qconfig)
        return qat_elu
