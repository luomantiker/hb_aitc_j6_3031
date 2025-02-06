import torch

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op
from horizon_plugin_pytorch.qtensor import QTensor
from .segment_lut import SegmentLUT


@replace_torch_op("sqrt")
class Sqrt(torch.nn.Module):
    """Module implementation of torch.sqrt."""

    def __init__(self):
        super(Sqrt, self).__init__()

        self.sqrt = SegmentLUT(torch.sqrt, True, None, None, "curvature")

    def forward(self, input):
        # if qat, input is clamped with q_scale so that no 0 input
        # is taken which could yield nan gradient in the backward
        if isinstance(input, QTensor) and not input.is_quantized:
            input = QTensor(
                input.as_subclass(torch.Tensor).clamp_min(input.q_scale()),
                input.q_scale(),
                input.dtype,
                input.q_per_channel_axis(),
            )
        return self.sqrt(input)
