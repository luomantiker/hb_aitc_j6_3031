import torch

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op


@replace_torch_op("ceil")
class Ceil(torch.nn.Module):
    """Module implementation of torch.ceil."""

    def __init__(self):
        super(Ceil, self).__init__()

    def forward(self, input):
        return torch.ceil(input)
