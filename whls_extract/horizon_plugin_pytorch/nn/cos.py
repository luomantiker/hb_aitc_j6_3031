import torch

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op
from .segment_lut import SegmentLUT


@replace_torch_op("cos")
class Cos(torch.nn.Module):
    """Module implementation of torch.cos."""

    def __init__(self):
        super(Cos, self).__init__()

        self.cos = SegmentLUT(torch.cos, False, None, None, "curvature")

    def forward(self, input):
        return self.cos(input)
