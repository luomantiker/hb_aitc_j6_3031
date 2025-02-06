import torch
from torch import nn
from torch.quantization import QConfig

from .segment_lut import SegmentLUT


class Softplus(nn.Softplus):
    """QAT module of torch.nn.Softplus.

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.

    Args:
        beta (int) : the β value for the Softplus formulation. Default is 1.
        threshold (int) : values above this revert to a linear function.
            Default is 20.

    """

    _FLOAT_MODULE = nn.Softplus

    def __init__(self, beta=1, threshold=20, qconfig=None):
        super(Softplus, self).__init__(beta, threshold)
        self.qconfig = qconfig
        assert (
            self.qconfig.activation is not None
        ), "qconfig activation must be provided"
        self.softplus = SegmentLUT(
            lambda x: torch.nn.functional.softplus(
                x, self.beta, self.threshold
            ),
            False,
            qconfig=qconfig,
        )

    def forward(self, input):
        return self.softplus(input)

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
        qat_softplus = cls(mod.beta, mod.threshold, qconfig)
        return qat_softplus
