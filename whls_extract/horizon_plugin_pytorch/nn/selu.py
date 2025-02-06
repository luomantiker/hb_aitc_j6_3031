import torch
from torch import nn

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op
from horizon_plugin_pytorch.utils._swap_horizon_float_nn import (
    replace_torch_nn_module,
)
from .segment_lut import SegmentLUT


@replace_torch_nn_module(nn.SELU)
@replace_torch_op("selu", is_nn_op=True)
class SELU(torch.nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.func = SegmentLUT(torch.selu, False)
        if inplace:
            raise ValueError("inplace is not supported in selu")

    def forward(self, x):
        return self.func(x)

    @classmethod
    def from_torch(cls, mod):
        return cls()
