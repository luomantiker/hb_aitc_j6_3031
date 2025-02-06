import torch

from horizon_plugin_pytorch.fx import fx_helper


@fx_helper.replace_torch_op("scatter")
class Scatter(torch.nn.Module):
    def forward(self, input, dim, index, src):
        return torch.scatter(input, dim, index, src)


@fx_helper.replace_torch_op("scatter_add")
class ScatterAdd(torch.nn.Module):
    def forward(self, input, dim, index, src):
        return torch.scatter_add(input, dim, index, src)


@fx_helper.replace_torch_op("scatter_reduce")
class ScatterReduce(torch.nn.Module):
    def forward(
        self, input, dim, index, src, reduce, *, include_self=True, out=None
    ):
        return torch.scatter_reduce(
            input, dim, index, src, reduce, include_self=include_self, out=out
        )
