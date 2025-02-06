import torch

from horizon_plugin_pytorch.fx import fx_helper
from .functional import bgr_to_yuv444


@fx_helper.wrap(skip_compile=True)
class BgrToYuv444(torch.nn.Module):
    """
    Convert image color format from bgr to yuv444.

    Args:
        channel_reversal: Color channel order,
            set to True when used on RGB input. Defaults to False.
    """

    def __init__(self, channel_reversal: bool = False):
        super(BgrToYuv444, self).__init__()
        assert isinstance(
            channel_reversal, bool
        ), "param 'channel_reversal' must be bool"
        self.channel_reversal = channel_reversal

    def forward(self, input: torch.Tensor):
        """Forward pass of BgrToYuv444."""
        return bgr_to_yuv444(input, self.channel_reversal)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "channel_reversal=" + str(self.channel_reversal)
        tmpstr += ")"
        return tmpstr
