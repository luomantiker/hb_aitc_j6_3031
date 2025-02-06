import logging
import warnings
from typing import List, Tuple, Union

import torch
import torch.nn.quantized as nnq

from horizon_plugin_pytorch.dtype import qinfo
from horizon_plugin_pytorch.fx import fx_helper
from horizon_plugin_pytorch.march import March, get_march
from horizon_plugin_pytorch.nn.qat import FloatFunctional as QATFloatFunctional
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils._swap_horizon_float_nn import (
    replace_torch_nn_module,
)
from horizon_plugin_pytorch.utils.load_state_dict_helper import get_version
from horizon_plugin_pytorch.utils.model_helper import call_with_hooks
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .div import Div
from .exp import Exp
from .functional import (
    add,
    cat,
    ceil,
    floor,
    matmul,
    mean,
    mul,
    requantize,
    sub,
    sum,
)

logger = logging.getLogger(__name__)


@fx_helper.wrap()
@replace_torch_nn_module(nnq.FloatFunctional)
class FloatFunctional(torch.nn.Module):
    r"""Apply float functionals."""

    _version = 2

    def __init__(self):
        super(FloatFunctional, self).__init__()
        self.default_keep_dim = False

    def forward(self, x):
        raise RuntimeError(
            "FloatFunctional is not intended to use the "
            + "'forward'. Please use the underlying operation"
        )

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
        if get_version(self, prefix, local_metadata) < 2:
            self.default_keep_dim = True
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @call_with_hooks
    def add(self, x, y):
        return torch.add(x, y)

    @call_with_hooks
    def add_scalar(
        self, x: torch.Tensor, y: Union[int, float]
    ) -> torch.Tensor:
        return torch.add(x, y)

    @call_with_hooks
    def sub(self, x, y):
        return torch.sub(x, y)

    def __rsub__(self, tensor: torch.Tensor, scalar: Union[int, float]):
        return self.sub(scalar, tensor)

    @call_with_hooks
    def concatenate(self, x, dim=0):
        return torch.cat(x, dim)

    @call_with_hooks
    def concat(self, x, dim=0):
        return torch.cat(x, dim)

    @call_with_hooks
    def cat(self, x, dim=0):
        return torch.cat(x, dim=dim)

    @call_with_hooks
    def stack(self, x, dim=0):
        return torch.stack(x, dim=dim)

    @call_with_hooks
    def matmul(self, x, y, x_trans=False, y_trans=False):
        r = torch.matmul(
            torch.transpose(x, -1, -2) if x_trans else x,
            torch.transpose(y, -1, -2) if y_trans else y,
        )
        return r

    @call_with_hooks
    def mul(self, x, y):
        return torch.mul(x, y)

    @call_with_hooks
    def mul_scalar(
        self, x: torch.Tensor, y: Union[int, float]
    ) -> torch.Tensor:
        return torch.mul(x, y)

    @call_with_hooks
    def div(self, x, y):
        warnings.warn(
            "horizon_plugin_pytorch.nn.quantized.Floatfunctional.div "
            + "will be deprecated. Please use module "
            + "horizon_plugin_pytorch.nn.Div!"
        )
        return torch.div(x, y)

    @call_with_hooks
    def sum(self, x, dim=None, keepdim=False):
        return torch.sum(x, dim, keepdim)

    @call_with_hooks
    def exp(self, x):
        warnings.warn(
            "horizon_plugin_pytorch.nn.quantized.Floatfunctional.exp "
            + "will be deprecated. Please use module "
            + "horizon_plugin_pytorch.nn.Exp!",
        )
        return torch.exp(x)

    @call_with_hooks
    def mean(self, x, dim=None, keepdim=None):
        if keepdim is None:
            keepdim = self.default_keep_dim
        if self.default_keep_dim:
            logger.warning(
                "The default value of keepdim argumant in Floatfunctional.mean"
                " has been changed from True to False, please manually specify"
                "it to keep the old behaviour",
                extra={"call_times_context": ("message")},
            )

        return torch.mean(x, dim, keepdim)

    @call_with_hooks
    def maximum(self, x, y):
        return torch.maximum(x, y)

    @call_with_hooks
    def minimum(self, x, y):
        return torch.minimum(x, y)

    @call_with_hooks
    def ceil(self, x):
        return torch.ceil(x)

    @call_with_hooks
    def floor(self, x):
        return torch.floor(x)

    @classmethod
    def from_torch(cls, mod: nnq.FloatFunctional):
        new_mod = cls()
        if hasattr(mod, "qconfig"):
            new_mod.qconfig = mod.qconfig
        return new_mod


@fx_helper.wrap()
class QFunctional(torch.nn.Module):
    r"""Quantized version."""

    _version = 2
    _QAT_MODULE = QATFloatFunctional

    def __init__(self, out_dtype):
        super(QFunctional, self).__init__()
        # register scale
        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))
        self.out_dtype = out_dtype
        self._div = Div(self.scale)
        self._exp = Exp(self.scale, out_dtype)
        self.default_keep_dim = False

    def forward(self, x):
        raise RuntimeError(
            "QFunctional is not intended to use the "
            + "'forward'. Please use the underlying operation"
        )

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
        if get_version(self, prefix, local_metadata) < 2:
            self.default_keep_dim = True
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @call_with_hooks
    @typechecked
    def add(
        self, x: Union[int, float, QTensor], y: Union[int, float, QTensor]
    ) -> QTensor:
        if isinstance(y, (int, float)):
            return self._add_scalar(x, y)
        elif isinstance(x, (int, float)):
            return self._add_scalar(y, x)

        r = add(
            x.int_repr(),
            y.int_repr(),
            x.q_scale(),
            y.q_scale(),
            x.q_zero_point(),
            y.q_zero_point(),
            x.dtype,
            y.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(torch.long),
            self.out_dtype,
        )
        return QTensor(
            r,
            self.scale,
            dtype=self.out_dtype,
            per_channel_axis=-1 if self.scale.numel() == 1 else 1,
        )

    @call_with_hooks
    @typechecked
    def add_scalar(self, x: QTensor, y: Union[int, float]) -> QTensor:
        return self._add_scalar(x, y)

    @typechecked
    def _add_scalar(self, x: QTensor, y: Union[int, float]) -> QTensor:
        if y == 0:
            r = x.int_repr()
        else:
            other_data = y / abs(y)
            other_scale = abs(y)

            # guarantee result precision after requantization
            if x.dtype == "qint16" and get_march() != March.BERNOULLI2:
                other_data = other_data * 32767
                other_scale = other_scale / 32767
            r = add(
                x.int_repr(),
                torch.tensor([other_data]).to(x.as_subclass(torch.Tensor)),
                x.q_scale(),
                torch.tensor([other_scale], dtype=torch.float32).to(x.device),
                x.q_zero_point(),
                x.q_zero_point(),
                x.dtype,
                x.dtype,
                self.scale,
                torch.zeros_like(self.scale).to(torch.long),
                self.out_dtype,
            )
        return QTensor(
            r,
            self.scale,
            dtype=self.out_dtype,
            per_channel_axis=-1 if self.scale.numel() == 1 else 1,
        )

    @call_with_hooks
    @typechecked
    def sub(self, x: QTensor, y: QTensor) -> QTensor:
        r = sub(
            x.int_repr(),
            y.int_repr(),
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            y.q_scale(),
            y.q_zero_point(),
            y.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(torch.long),
            self.out_dtype,
        )
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    @typechecked
    def concatenate(
        self, x: Union[Tuple[QTensor, ...], List[QTensor]], dim: int = 0
    ):
        return self._cat(x, dim)

    @call_with_hooks
    @typechecked
    def concat(
        self, x: Union[Tuple[QTensor, ...], List[QTensor]], dim: int = 0
    ):
        return self._cat(x, dim)

    @call_with_hooks
    @typechecked
    def cat(
        self, x: Union[Tuple[QTensor, ...], List[QTensor]], dim: int = 0
    ) -> QTensor:
        return self._cat(x, dim)

    def _cat(
        self, x: Union[Tuple[QTensor, ...], List[QTensor]], dim: int = 0
    ) -> QTensor:
        r = cat(
            [qt.int_repr() for qt in x],
            dim,
            [qt.q_scale() for qt in x],
            [qt.q_zero_point() for qt in x],
            [qt.dtype for qt in x],
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    @typechecked
    def matmul(
        self, x: QTensor, y: QTensor, x_trans=False, y_trans=False
    ) -> QTensor:
        r = matmul(
            x.int_repr(),
            y.int_repr(),
            x_trans,
            y_trans,
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            y.q_scale(),
            y.q_zero_point(),
            y.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(torch.long),
            self.out_dtype,
        )
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    @typechecked
    def mul(
        self,
        x: Union[QTensor, torch.Tensor],
        y: Union[int, float, QTensor, torch.Tensor],
    ) -> QTensor:
        if isinstance(y, (int, float)):
            return self._mul_scalar(x, y)
        elif isinstance(x, (int, float)):
            return self._mul_scalar(y, x)

        # hbdk need the dtype be "qbool", and must provide scale & zero_point
        # so we construct a qtensor for convenience
        if x.dtype == torch.bool:
            x = QTensor(x, torch.tensor([1], device=x.device), "qbool")
        if y.dtype == torch.bool:
            y = QTensor(y, torch.tensor([1], device=y.device), "qbool")

        r = mul(
            x.int_repr(),
            y.int_repr(),
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            y.q_scale(),
            y.q_zero_point(),
            y.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(torch.long),
            self.out_dtype,
        )
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    @typechecked
    def mul_scalar(self, x: QTensor, y: Union[int, float]) -> QTensor:
        return self._mul_scalar(x, y)

    @typechecked
    def _mul_scalar(self, x: QTensor, y: Union[int, float]) -> QTensor:
        scalar = 0 if y == 0 else y / abs(y)
        # if 0 or 1, directly return QTensor, avoid extra 'mul' on HW
        if scalar == 0:
            odtype = qinfo(self.out_dtype)._storage_type
            r = torch.zeros_like(x.as_subclass(torch.Tensor)).to(odtype)
        elif scalar == 1:
            r = x.int_repr()
            if x.dtype != self.out_dtype:
                r = requantize(
                    r,
                    x.q_scale() * abs(y),
                    x.q_zero_point(),
                    x.dtype,
                    self.scale,
                    x.q_zero_point(),
                    self.out_dtype,
                )
        else:
            r = mul(
                x.int_repr(),
                torch.tensor([scalar]).to(x.as_subclass(torch.Tensor)),
                x.q_scale(),
                x.q_zero_point(),
                x.dtype,
                torch.tensor([abs(y)], dtype=torch.float32).to(x.device),
                x.q_zero_point(),
                x.dtype,
                self.scale,
                x.q_zero_point(),
                self.out_dtype,
            )
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    @typechecked
    def div(self, x: QTensor, y: QTensor) -> QTensor:
        return self._div(x, y)

    @call_with_hooks
    @typechecked
    def sum(
        self, x: QTensor, dim: Union[int, None], keepdim: bool = False
    ) -> QTensor:
        r = sum(
            x.int_repr(),
            dim,
            keepdim,
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    @typechecked
    def exp(self, x: QTensor) -> QTensor:
        return self._exp(x)

    @call_with_hooks
    @typechecked
    def mean(self, x: QTensor, dim: int, keepdim: bool = None) -> QTensor:
        if keepdim is None:
            keepdim = self.default_keep_dim
        if self.default_keep_dim:
            logger.warning(
                "The default value of keepdim argumant in Floatfunctional.mean"
                " has been changed from True to False, please manually specify"
                "it to keep the old behaviour",
                extra={"call_times_context": ("message")},
            )

        r = mean(
            x.int_repr(),
            dim,
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        if not keepdim:
            r = r.squeeze(dim)
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    @typechecked
    def maximum(self, x: QTensor, y: QTensor) -> QTensor:
        x = requantize(
            x.int_repr(),
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        y = requantize(
            y.int_repr(),
            y.q_scale(),
            y.q_zero_point(),
            y.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        r = torch.maximum(x, y)
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    @typechecked
    def minimum(self, x: QTensor, y: QTensor) -> QTensor:
        x = requantize(
            x.int_repr(),
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        y = requantize(
            y.int_repr(),
            y.q_scale(),
            y.q_zero_point(),
            y.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        r = torch.minimum(x, y)
        return QTensor(r, self.scale, dtype=self.out_dtype)

    @call_with_hooks
    @typechecked
    def ceil(self, x: QTensor) -> QTensor:
        r = ceil(
            x.int_repr(),
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        return QTensor(
            r,
            self.scale,
            dtype=self.out_dtype,
        )

    @call_with_hooks
    @typechecked
    def floor(self, x: QTensor) -> QTensor:
        r = floor(
            x.int_repr(),
            x.q_scale(),
            x.q_zero_point(),
            x.dtype,
            self.scale,
            torch.zeros_like(self.scale).to(dtype=torch.long),
            self.out_dtype,
        )
        return QTensor(
            r,
            self.scale,
            dtype=self.out_dtype,
        )

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict.

        Args: `mod` a float module
        """
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        out_dtype = (
            mod.activation_post_process.dtype
            if mod.activation_post_process is not None
            else "qint32"
        )
        func = cls(out_dtype)
        func.scale.resize_as_(mod.activation_post_process.scale)
        func.scale.copy_(mod.activation_post_process.scale)
        func.default_keep_dim = mod.default_keep_dim
        return func
