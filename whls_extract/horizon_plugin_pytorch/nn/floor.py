import torch

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op


@replace_torch_op("floor")
class Floor(torch.nn.Module):
    """Module implementation of torch.floor."""

    def __init__(self):
        super(Floor, self).__init__()

    def forward(self, input):
        return torch.floor(input)
