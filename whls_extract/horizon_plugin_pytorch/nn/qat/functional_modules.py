import logging
from typing import List, Tuple, Union

import torch
import torch.nn.quantized as nnq
from torch import Tensor
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
)

from horizon_plugin_pytorch.fx import fx_helper
from horizon_plugin_pytorch.nn.qat.qat_meta import (
    init_input_preprocess,
    is_float,
    pre_process,
)
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.load_state_dict_helper import get_version
from horizon_plugin_pytorch.utils.model_helper import call_with_hooks
from horizon_plugin_pytorch.utils.typeguard import typechecked

logger = logging.getLogger(__name__)


# stubs for exporting hbir
def _add_scalar_stub(x, scalar):
    if has_torch_function_unary(x):
        return handle_torch_function(_add_scalar_stub, (x,), x, scalar)

    return torch.add(x, scalar)


QTensor.register_func_impl(_add_scalar_stub)(
    QTensor.DispatchMode.CALL_AS_TENSOR
)


def _sub_stub(x, y):
    if has_torch_function((x, y)):
        return handle_torch_function(_sub_stub, (x, y), x, y)

    return torch.sub(x, y)


QTensor.register_func_impl(_sub_stub)(QTensor.DispatchMode.CALL_AS_TENSOR)


@fx_helper.wrap()
class FloatFunctional(torch.nn.Module):
    r"""Apply qat functionals."""

    _version = 2
    _FLOAT_MODULE = (nnq.FloatFunctional,)

    def __init__(self, qconfig=None):
        super(FloatFunctional, self).__init__()
        assert qconfig, "qconfig must be provided for QAT FloatFunctional"
        self.qconfig = qconfig
        self.activation_pre_process = init_input_preprocess(qconfig)
        if self.qconfig.activation is not None:
            self.activation_post_process = qconfig.activation()
        else:
            self.activation_post_process = None
        self.default_keep_dim = False
        self.input_pre_process = init_input_preprocess(qconfig)

    def forward(self, *args, **kwargs):
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
    @typechecked
    def add(
        self, x: Union[Tensor, int, float], y: Union[int, float, Tensor]
    ) -> Tensor:
        if isinstance(y, (int, float)):
            x = pre_process(self.activation_pre_process, x)
            return self._add_scalar(x, y)
        elif isinstance(x, (int, float)):
            x = pre_process(self.activation_pre_process, y)
            return self._add_scalar(y, x)

        x, y = pre_process(self.activation_pre_process, x, y)
        r = torch.add(
            x.as_subclass(torch.Tensor),
            y.as_subclass(torch.Tensor),
        )
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    @typechecked
    def add_scalar(self, x: Tensor, y: Union[int, float]) -> Tensor:
        return self._add_scalar(x, y)

    @typechecked
    def _add_scalar(self, x: Tensor, y: Union[int, float]) -> Tensor:
        if y == 0:
            return self._identity_with_dtype_check(x)
        else:
            r = _add_scalar_stub(x, y)
            r = self.activation_post_process(r)
            return r

    @call_with_hooks
    @typechecked
    def sub(
        self, x: Union[Tensor, int, float], y: Union[Tensor, int, float]
    ) -> Tensor:
        if isinstance(x, (int, float)) and x == 0:
            pre_process(self.activation_pre_process, y)
            return self._mul_scalar(y, -1)
        if isinstance(y, (int, float)) and y == 0:
            pre_process(self.activation_pre_process, x)
            return self._identity_with_dtype_check(x)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            raise ValueError(
                "Operation between scalar is not supported by FloatFunctional"
            )

        x, y = pre_process(self.activation_pre_process, x, y)
        # _sub_stub is used in export to add fq on scalar
        # so skip it for float input
        if isinstance(x, QTensor) or isinstance(y, QTensor):
            r = _sub_stub(x, y)
        else:
            r = torch.sub(x, y)
        r = self.activation_post_process(r)
        return r

    def __rsub__(self, tensor: Tensor, scalar: Union[int, float]):
        return self.sub(scalar, tensor)

    @call_with_hooks
    @typechecked
    def concatenate(
        self,
        x: Union[Tuple[Tensor, ...], List[Tensor]],
        dim: int = 0,
    ):
        return self._cat(x, dim)

    @call_with_hooks
    @typechecked
    def concat(
        self,
        x: Union[Tuple[Tensor, ...], List[Tensor]],
        dim: int = 0,
    ):
        return self._cat(x, dim)

    @call_with_hooks
    @typechecked
    def cat(
        self,
        x: Union[
            Tuple[Tensor, ...],
            List[Tensor],
        ],
        dim: int = 0,
    ):
        return self._cat(x, dim)

    def _cat(
        self,
        x: Union[Tuple[Tensor, ...], List[Tensor]],
        dim: int = 0,
    ) -> Tensor:
        if is_float(self.activation_post_process):
            x = [pre_process(self.activation_pre_process, t) for t in x]
            out = torch.cat(x, dim)
            return self.activation_post_process(out)

        same_scale = True
        for rest in x:
            if (
                rest.q_scale() != x[0].q_scale()
                or rest.dtype != self.activation_post_process.dtype
            ):
                same_scale = False
                break

        r = torch.cat(
            [qt.as_subclass(torch.Tensor) for qt in x],
            dim=dim,
        )

        # if activation_post_process is FakeQuantize, only disable fakequant when qat # noqa: E501
        if same_scale and getattr(
            self.activation_post_process, "_fake_quant_enabled", True
        ):
            self.activation_post_process.disable_observer()
            self.activation_post_process.set_qparams(x[0].q_scale())
        else:
            self.activation_post_process.enable_observer()

        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    @typechecked
    def stack(
        self,
        x: Union[Tuple[Tensor, ...], List[Tensor]],
        dim: int = 0,
    ) -> Tensor:
        if self.activation_post_process.dtype in (
            torch.float16,
            torch.float32,
        ):
            r = torch.stack(
                [qt.as_subclass(torch.Tensor) for qt in x],
                dim=dim,
            )
            r = self.activation_post_process(r)
            return r
        same_scale = True
        for rest in x:
            if (
                rest.q_scale() != x[0].q_scale()
                or rest.dtype != self.activation_post_process.dtype
            ):
                same_scale = False
                break

        r = torch.stack(
            [qt.as_subclass(torch.Tensor) for qt in x],
            dim=dim,
        )

        # if activation_post_process is FakeQuantize, only disable fakequant when qat # noqa: E501
        if same_scale and getattr(
            self.activation_post_process, "_fake_quant_enabled", True
        ):
            self.activation_post_process.disable_observer()
            self.activation_post_process.set_qparams(x[0].q_scale())
        else:
            self.activation_post_process.enable_observer()

        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    @typechecked
    def matmul(
        self,
        x: Union[Tensor, QTensor],
        y: Union[Tensor, QTensor],
        x_trans=False,
        y_trans=False,
    ) -> Union[Tensor, QTensor]:
        x, y = pre_process(self.input_pre_process, x, y)
        x = x.as_subclass(torch.Tensor)
        y = y.as_subclass(torch.Tensor)

        r = torch.matmul(
            torch.transpose(x, -1, -2) if x_trans else x,
            torch.transpose(y, -1, -2) if y_trans else y,
        )
        if self.activation_post_process is not None:
            r = self.activation_post_process(r)
        return r

    @call_with_hooks
    @typechecked
    def mul(
        self,
        x: Union[torch.Tensor, int, float],
        y: Union[torch.Tensor, int, float],
    ) -> Tensor:
        if isinstance(y, (int, float)):
            x = pre_process(self.activation_pre_process, x)
            return self._mul_scalar(x, y)
        elif isinstance(x, (int, float)):
            y = pre_process(self.activation_pre_process, y)
            return self._mul_scalar(y, x)
        x, y = pre_process(self.activation_pre_process, x, y)
        r = torch.mul(
            x.as_subclass(torch.Tensor),
            y.as_subclass(torch.Tensor),
        )
        if torch.bool in (x.dtype, y.dtype):
            if self.activation_post_process._observer_enabled:
                self.activation_post_process.disable_fake_quant()
                self.activation_post_process.disable_observer()
            if x.dtype == torch.bool:
                oscale = y.q_scale()
                zero_point = y.q_zero_point()
                input_dtype = y.dtype
            else:
                oscale = x.q_scale()
                zero_point = x.q_zero_point()
                input_dtype = x.dtype
            if input_dtype != self.activation_post_process.dtype:
                old_dtype = self.activation_post_process.dtype
                self.activation_post_process.reset_dtype(input_dtype, False)
                logger.warning(
                    f"{self.__class__.__name__} output dtype {old_dtype} will "
                    f"be changed to {input_dtype}.",
                    extra={"call_times_context": ("message")},
                )
            self.activation_post_process.set_qparams(oscale, zero_point)
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    @typechecked
    def mul_scalar(self, x: Tensor, y: Union[int, float]) -> Tensor:
        return self._mul_scalar(x, y)

    @typechecked
    def _mul_scalar(self, x: Tensor, y: Union[int, float]) -> Tensor:
        x = pre_process(self.activation_pre_process, x)
        r = torch.mul(x.as_subclass(torch.Tensor), y)
        if is_float(self.activation_post_process):
            r = self.activation_post_process(r)
            return r
        return self._identity_with_dtype_check(
            QTensor(
                r,
                (x.q_scale() * abs(y)).clamp_min(
                    torch.finfo(torch.float32).eps
                ),
                x.dtype,
            )
        )

    @call_with_hooks
    @typechecked
    def div(self, x: Tensor, y: Tensor) -> Tensor:
        logger.warning(
            "horizon_plugin_pytorch.nn.quantized.Floatfunctional.div "
            "will be deprecated. Please use module "
            "horizon_plugin_pytorch.nn.Div!",
            extra={"call_times_context": ("message")},
        )
        r = torch.div(
            x.as_subclass(torch.Tensor),
            y.as_subclass(torch.Tensor),
        )
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    @typechecked
    def sum(
        self,
        x: Tensor,
        dim: Union[int, None, Tuple] = None,
        keepdim: bool = False,
    ) -> Union[Tensor, QTensor]:
        x = pre_process(self.input_pre_process, x)
        r = torch.sum(x.as_subclass(torch.Tensor), dim, keepdim)
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    @typechecked
    def exp(self, x: Tensor) -> Tensor:
        logger.warning(
            "horizon_plugin_pytorch.nn.quantized.Floatfunctional.exp "
            "will be deprecated. Please use module "
            "horizon_plugin_pytorch.nn.Exp!",
            extra={"call_times_context": ("message")},
        )
        r = torch.exp(x.as_subclass(torch.Tensor))
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    @typechecked
    def mean(
        self,
        x: QTensor,
        dim: Union[int, None] = None,
        keepdim: bool = None,
    ) -> QTensor:
        if keepdim is None:
            keepdim = self.default_keep_dim
        if self.default_keep_dim:
            logger.warning(
                "The default value of keepdim argumant in Floatfunctional.mean"
                " has been changed from True to False, please manually specify"
                "it to keep the old behaviour",
                extra={"call_times_context": ("message")},
            )

        r = torch.mean(x.as_subclass(torch.Tensor), dim, keepdim)
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    @typechecked
    def maximum(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.maximum(
            x.as_subclass(torch.Tensor), y.as_subclass(torch.Tensor)
        )
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    @typechecked
    def minimum(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.minimum(
            x.as_subclass(torch.Tensor),
            y.as_subclass(torch.Tensor),
        )
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    @typechecked
    def ceil(self, x: Tensor) -> Tensor:
        assert x.q_scale().numel() == 1, "ceil only support per-tensor scale."
        r = torch.ceil(x.as_subclass(torch.Tensor))
        r = self.activation_post_process(r)
        return r

    @call_with_hooks
    @typechecked
    def floor(self, x: Tensor) -> Tensor:
        if isinstance(x, QTensor):
            assert (
                x.q_scale().numel() == 1
            ), "floor only support per-tensor scale."
        x = pre_process(self.activation_pre_process, x)
        r = torch.floor(x.as_subclass(torch.Tensor))
        r = self.activation_post_process(r)
        return r

    def _identity_with_dtype_check(self, x: Tensor) -> Tensor:
        if x.dtype.is_floating_point:
            return self.activation_post_process(x)
        if x.dtype == self.activation_post_process.dtype:
            self.activation_post_process.disable_observer()
            self.activation_post_process.set_qparams(x.q_scale())
        else:
            self.activation_post_process.enable_observer()
        return self.activation_post_process(x.as_subclass(torch.Tensor))

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict.

        Args: `mod` a float module
        """
        from horizon_plugin_pytorch.nn import quantized

        assert type(mod) in cls._FLOAT_MODULE or isinstance(
            mod, quantized.FloatFunctional
        ), (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + (modc.__name__ for modc in cls._FLOAT_MODULE)
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qat_func = cls(mod.qconfig)
        if hasattr(mod, "default_keep_dim"):
            qat_func.default_keep_dim = mod.default_keep_dim
        if hasattr(mod, "_last_called_method_name"):
            qat_func._last_called_method_name = mod._last_called_method_name

        return qat_func
