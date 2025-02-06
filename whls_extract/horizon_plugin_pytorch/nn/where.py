import torch

from horizon_plugin_pytorch.fx import fx_helper


@fx_helper.replace_torch_op("where")
class Where(torch.nn.Module):
    def forward(self, condition, input, other):
        return torch.where(condition, input, other)
