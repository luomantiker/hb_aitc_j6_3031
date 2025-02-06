import torch

from horizon_plugin_pytorch.fx import fx_helper


@fx_helper.replace_torch_op("mod")  # replace built-in func 'mod'
@fx_helper.replace_torch_op("remainder")
class Remainder(torch.nn.Module):
    def forward(self, input, other):
        return torch.remainder(input, other)
