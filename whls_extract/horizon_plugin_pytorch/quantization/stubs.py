from torch import nn


class QuantStub(nn.Module):
    r"""Refine this docstring in the future.

    Same as torch.nn.QuantStub, with an additional param to
    specify a fixed scale.

    Args:
        scale (float, optional): Pass a number to use as fixed scale.
            If set to None, scale will be computed by observer during forward.
            Defaults to None.
        zero_point (int, optional): Pass a number to use as fixed zero_point.
            Defaults to None.
        qconfig (optional): Quantization configuration for the tensor, if
            qconfig is not provided, we will get qconfig from parent modules.
            Defaults to None.
    """

    def __init__(
        self, scale: float = None, zero_point: int = None, qconfig=None
    ):
        super(QuantStub, self).__init__()
        if scale is not None and zero_point is None:
            zero_point = 0

        self.scale = scale
        self.zero_point = zero_point
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x):
        return x
