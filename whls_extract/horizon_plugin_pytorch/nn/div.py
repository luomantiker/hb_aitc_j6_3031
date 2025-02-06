import torch

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op


@replace_torch_op("truediv")
@replace_torch_op("div")
class Div(torch.nn.Module):
    def __init__(self, reciprocal_max_value=None) -> None:
        super().__init__()
        self.reciprocal_max_value = reciprocal_max_value

    def forward(self, input, other, rounding_mode=None):
        if rounding_mode is not None:
            raise ValueError(
                "Unsupported rounding_mode {}".format(rounding_mode)
            )

        return torch.div(input, other)
