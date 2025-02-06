import torch

from horizon_plugin_pytorch.fx import fx_helper


@fx_helper.replace_torch_op("fmod")
class FMod(torch.nn.Module):
    def forward(self, input, other):
        return torch.fmod(input, other)
