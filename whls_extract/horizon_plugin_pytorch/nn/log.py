import torch
from torch import autograd

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op
from .segment_lut import SegmentLUT


def _hard_log(x):
    return torch.clamp(torch.log(x), min=-10)


class HardLogFunction(autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ret = _hard_log(i)
        ctx.save_for_backward(i, ret)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        i, ret = ctx.saved_tensors
        # use torch.tensor to avoid torch.where scalar type error in torch 1.10
        return grad_output * torch.where(
            ret > -10, 1 / i, torch.tensor(0.0, device=i.device)
        )


def hard_log(x):
    return (
        _hard_log(x)
        if torch._C._get_tracing_state()
        else HardLogFunction.apply(x)
    )


@replace_torch_op("log")
class HardLog(torch.nn.Module):
    """Module implementation of torch.log."""

    def __init__(self):
        super(HardLog, self).__init__()

        self.log = SegmentLUT(
            hard_log,
            True,
            None,
            None,
            "curvature",
            inverse_func=torch.exp,
        )

    def forward(self, input):
        return self.log(input)
