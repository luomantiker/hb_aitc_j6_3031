import torch

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op


@replace_torch_op("pow", is_nn_op=True)
class Pow(torch.nn.Module):
    """Module implementation of torch.pow."""

    def __init__(self, exponent=None):
        super(Pow, self).__init__()
        self.exponent = exponent

    def forward(self, data, exponent=None):
        if exponent is None:
            assert (
                self.exponent is not None
            ), "exponent must be provided either in __init__ or forward"
            exponent = self.exponent
        else:
            if not isinstance(exponent, (int, float)):
                assert (
                    isinstance(exponent, torch.Tensor)
                    and exponent.numel() == 1
                ), "Only support power which exponent is scalar"
            if self.exponent is None:
                self.exponent = exponent
            else:
                assert self.exponent == exponent, (
                    f"This Pow is only used for exponent {self.exponent}, "
                    f"but get {exponent}"
                )

        return torch.pow(data, exponent)
