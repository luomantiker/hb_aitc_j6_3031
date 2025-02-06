import torch

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op


@replace_torch_op("exp")
class Exp(torch.nn.Module):
    """Module implementation of torch.exp."""

    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, input):
        return torch.exp(input)
