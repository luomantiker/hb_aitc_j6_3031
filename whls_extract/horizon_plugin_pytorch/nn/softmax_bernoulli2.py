from typing import Optional

import torch.nn.functional as F  # noqa F812
from torch import nn

from horizon_plugin_pytorch.utils.typeguard import typechecked


class SoftmaxBernoulli2(nn.Module):
    """SoftmaxBernoulli2 is designed to run on Bernoulli2.

    This operator is considered hacky and should not been used by most users.

    The calculation logic of this operator is as follows roughly:

    .. code-block:: python

        y = exp(x - x.max(dim)) / sum(exp(x - x.max(dim)), dim)

    The output of this operator is float type and cannot be fed into other
    quantized operators.

    In the FLOAT phase, users can set qconfig to this operator as usual.
    However, there are some peculiarities in QAT and QUANTIZED inference
    phases. Please read the following carefully.

    In the QAT phase, the operator only applies fake quantization to exp(x),
    then computes the division in the float domain and returns the
    unfakequantized(float) result directly. This operator will ignore the
    `qconfig` set by users or propagated from the parent module. However, to
    integrate this into the workflow of converting QAT models to QUANTIZED
    models, a reasonable `qconfig` is needed.

    In the QUANTIZED inference phase, the operator retrieves the result of
    exp(x) from a lookup table and computes the division in the float domain.

    When `max_value_only` is set to True, the maximum value of softmax  along
    `dim` will be returned, which is equal to max(softmax(x, dim), dim). We
    combine softmax and max in this op because the hbdk compiler requires it to
    optimize performance without the effort of graph analysis. This argument is
    only intended for this specific purpose and should not be used in other
    cases.

    Args:
        dim: The dimension along which Softmax will be computed. only supports
                dim=1.
        max_value_only: If True, return the max value along `dim`, if
                False, equal to normal softmax. Refer to the above for more
                information.
    """

    @typechecked
    def __init__(
        self, dim: int = None, max_value_only: Optional[bool] = False
    ):
        super(SoftmaxBernoulli2, self).__init__()
        assert (
            dim == 1
        ), f"SoftmaxBernoulli2 only support dim=1, {dim} is given"
        self.dim = dim
        self.max_value_only = max_value_only

    def forward(self, x):
        out = F.softmax(x, dim=self.dim)
        if self.max_value_only:
            return out.max(dim=self.dim, keepdim=True)[0]
        return out
