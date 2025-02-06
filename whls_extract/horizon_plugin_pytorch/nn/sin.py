import torch

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op
from .segment_lut import SegmentLUT


@replace_torch_op("sin")
class Sin(torch.nn.Module):
    """Module implementation of torch.sin."""

    def __init__(self):
        super(Sin, self).__init__()

        self.sin = SegmentLUT(torch.sin, True, None, None, "curvature")

    def forward(self, input):
        return self.sin(input)
