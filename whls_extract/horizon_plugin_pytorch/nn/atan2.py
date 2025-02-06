import math

import torch

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op
from .div import Div
from .segment_lut import SegmentLUT
from .where import Where


def x_neg_atan2(input):
    return torch.where(
        input < 0,
        torch.arctan(input) + math.pi,
        torch.arctan(input) - math.pi,
    )


@replace_torch_op("atan2")
class Atan2(torch.nn.Module):
    """Module implementation of torch.atan2."""

    def __init__(self, input_range=50) -> None:
        super().__init__()
        self.div = Div()
        self.arctan = SegmentLUT(
            torch.arctan,
            is_centrosymmetric=True,
            input_range=[-input_range, input_range],
            auto_divide_strategy="curvature",
        )
        self.arctan2 = SegmentLUT(
            x_neg_atan2,
            is_centrosymmetric=True,
            input_range=[-input_range, input_range],
            auto_divide_strategy="curvature",
        )
        self.where = Where()

    def forward(self, input, other, out=None):
        nan_mask = torch.logical_and(input == 0, other == 0)
        x = self.div(input, other)
        x = torch.masked_fill(x, nan_mask, 0)
        atan_mask = other >= 0
        return self.where(atan_mask, self.arctan(x), self.arctan2(x))
