from typing import Optional, Union

import torch
from torch import nn

from horizon_plugin_pytorch.fx.fx_helper import replace_torch_op
from horizon_plugin_pytorch.utils.load_state_dict_helper import (
    replace_mod_name,
)
from .div import Div
from .sqrt import Sqrt


@replace_torch_op("norm", is_nn_op=True)
class Norm(nn.Module):
    _version = 2

    def __init__(
        self,
        p: Optional[Union[float, str]] = "fro",
        dim=None,
        keepdim: bool = False,
        out=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.p = p
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype

        if p not in (2.0, "fro"):
            raise ValueError("Only frobenius norm is supported")

        from horizon_plugin_pytorch.nn.quantized import FloatFunctional

        self.mul = FloatFunctional()
        self.sum = FloatFunctional()
        self.sqrt = Sqrt()

    def forward(self, input):
        x = self.mul.mul(input, input)
        x = self.sum.sum(x, self.dim, self.keepdim)
        x = self.sqrt(x)
        return x

    @replace_mod_name("sqrt", "sqrt.sqrt", lambda v: v < 2)
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def propagate_qconfig(self, qconfig):
        from horizon_plugin_pytorch.quantization.qconfig import (
            promote_int8_activation_to_int16,
        )

        int16_qconfig = promote_int8_activation_to_int16(qconfig)
        self.mul.qconfig = int16_qconfig
        self.sum.qconfig = int16_qconfig
        self.sqrt.qconfig = qconfig


@replace_torch_op("linalg_norm", is_nn_op=True)
class LinalgNorm(Norm):
    def __init__(
        self, ord=None, dim=None, keepdim=False, *, out=None, dtype=None
    ) -> None:
        self.out = out
        if ord not in (2, None):
            raise ValueError("Only vector norm of order 2 is supported")
        if not isinstance(dim, int):
            raise ValueError("Arg dim must be a int")
        if ord is None:
            ord = 2
        return super().__init__(
            p=ord, dim=dim, keepdim=keepdim, out=out, dtype=dtype
        )


@replace_torch_op("normalize", is_nn_op=True)
class Normalize(nn.Module):
    def __init__(
        self,
        p: float = 2.0,
        dim: int = 1,
        eps: float = 1e-12,
        out: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps
        self.norm = Norm(self.p, self.dim, keepdim=True)
        self.div = Div()

    def forward(self, input):
        return self.div(
            input, self.norm(input).clamp(min=self.eps).expand_as(input)
        )
