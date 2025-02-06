import torch
from torch import nn
from torch.quantization import QConfig

from .segment_lut import SegmentLUT


class HardSigmoid(nn.Hardsigmoid):
    r"""QAT module of torch.nn.Hardsigmoid.

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    Args:
        inplace: can optionally do the operation in-place.
        Default is `False`.

    """

    _FLOAT_MODULE = nn.Hardsigmoid

    def __init__(self, inplace=False, qconfig=None):
        super(HardSigmoid, self).__init__(inplace)
        self.qconfig = qconfig
        assert (
            self.qconfig.activation is not None
        ), "qconfig activation must be provided"
        self.hardsigmoid = SegmentLUT(
            lambda x: torch.nn.functional.hardsigmoid(x, self.inplace),
            False,
            input_range=(-3, 3),
            gradients=(0.0, 0.0),
            qconfig=qconfig,
        )

    def forward(self, input):
        return self.hardsigmoid(input)

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
        qat_hardsigmoid = cls(mod.inplace, qconfig)
        return qat_hardsigmoid
