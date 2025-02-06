import torch

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op


@replace_torch_op("einsum", True)
class EinSum(torch.nn.Module):
    """Module implementation of torch.einsum."""

    def __init__(self, equation):
        super(EinSum, self).__init__()
        self.equation = equation

    def forward(self, *inputs):
        return torch.einsum(self.equation, *inputs)
