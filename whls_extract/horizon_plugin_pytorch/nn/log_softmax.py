from typing import Optional

from torch import nn

from horizon_plugin_pytorch.utils._swap_horizon_float_nn import (
    replace_torch_nn_module,
)
from .log import HardLog


@replace_torch_nn_module(nn.LogSoftmax)
class LogSoftmax(nn.LogSoftmax):
    _FLOAT_MODULE = nn.LogSoftmax

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__(dim)
        self.softmax = nn.Softmax(dim)
        self.log = HardLog()

    def forward(self, input, dim=None, _stacklevel=3, dtype=None):
        if dim is not None:
            self.softmax.dim = dim
        return self.log(self.softmax(input))

    @classmethod
    def from_torch(cls, mod: nn.LogSoftmax):
        new_mod = cls(mod.dim)
        if hasattr(mod, "qconfig"):
            new_mod.qconfig = mod.qconfig
        return new_mod
