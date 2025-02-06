import torch

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op


@replace_torch_op("cumsum")
class CumSum(torch.nn.Module):
    """Module implementation of torch.cos."""

    def forward(self, input, dim, *, dtype=None, out=None):
        return torch.cumsum(input, dim, dtype=dtype, out=out)
