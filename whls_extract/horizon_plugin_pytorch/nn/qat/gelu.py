from functools import partial

from torch import nn
from torch.nn import functional as F  # noqa: N812

from .segment_lut import SegmentLUT


class GELU(nn.GELU):
    """Apply GELU operation to input array."""

    _FLOAT_MODULE = nn.GELU

    def __init__(self, approximate="none", qconfig=None):
        super(GELU, self).__init__()
        self.approximate = approximate
        self.simulated_func = partial(F.gelu, approximate=self.approximate)
        self.lut = SegmentLUT(
            self.simulated_func,
            False,
            [-5, -3, -2, -1, -0.75, 0, 5.5, float("inf")],
            qconfig=qconfig,
        )

    def forward(self, input):
        return self.lut(input)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict.

        Args:
            'mod' a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qat_mod = cls(
            approximate=getattr(mod, "approximate", "none"),
            qconfig=mod.qconfig,
        )
        return qat_mod
