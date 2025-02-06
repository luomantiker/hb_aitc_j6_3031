import torch

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op


@replace_torch_op("reciprocal")
class Reciprocal(torch.nn.Module):
    """Module implementation of torch.reciprocal."""

    def __init__(self, max_value=None):
        super(Reciprocal, self).__init__()
        self.max_value = None if max_value is None else abs(max_value)

    def forward(self, input):
        ret = torch.reciprocal(input)
        if self.max_value is not None:
            ret = ret.clamp(-self.max_value, self.max_value)
        return ret
