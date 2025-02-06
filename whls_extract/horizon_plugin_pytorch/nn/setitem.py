import torch

from horizon_plugin_pytorch.fx import fx_helper


@fx_helper.replace_torch_op("__setitem__")
class SetItem(torch.nn.Module):
    def forward(self, tensor, indices, val):
        torch.Tensor.__setitem__(tensor, indices, val)
        return tensor
