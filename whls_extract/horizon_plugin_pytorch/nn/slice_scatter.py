import torch

from horizon_plugin_pytorch.fx import fx_helper


@fx_helper.replace_torch_op("slice_scatter", is_nn_op=True)
class SliceScatter(torch.nn.Module):
    def __init__(self, dim=0, start=None, end=None, step=1):
        super().__init__()
        self.dim = dim
        self.start = start
        self.end = end
        self.step = step

    def forward(self, input, src):
        return torch.slice_scatter(
            input, src, self.dim, self.start, self.end, self.step
        )
