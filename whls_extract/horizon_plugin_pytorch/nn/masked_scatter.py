import torch

from horizon_plugin_pytorch.fx import fx_helper


@fx_helper.replace_torch_op("masked_scatter")
class MaskedScatter(torch.nn.Module):
    def forward(self, input, mask, source):
        return torch.masked_scatter(input, mask, source)
