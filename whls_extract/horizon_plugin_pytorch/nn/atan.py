import torch

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op
from .segment_lut import SegmentLUT


@replace_torch_op("atan")
class Atan(torch.nn.Module):
    """Module implementation of torch.atan."""

    def __init__(self):
        super(Atan, self).__init__()

        self.atan = SegmentLUT(torch.atan, True, None, None, "curvature")

    def forward(self, input):
        return self.atan(input)
