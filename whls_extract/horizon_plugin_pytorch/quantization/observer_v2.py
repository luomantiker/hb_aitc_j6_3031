import logging
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter
from functools import partial
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist

from horizon_plugin_pytorch.dtype import (
    QuantDType,
    get_horizon_quant_dtype,
    qinfo,
    qint8,
)
from horizon_plugin_pytorch.march import get_march
from horizon_plugin_pytorch.quantization.misc import pow_quantization
from horizon_plugin_pytorch.utils.global_quant_round_mode import QuantRoundMode
from horizon_plugin_pytorch.utils.typeguard import typechecked

logger = logging.getLogger(__name__)


__all__ = [
    "load_observer_params",
    "MinMaxObserver",
    "FixedScaleObserver",
    "ClipObserver",
    "PercentileObserver",
    "ClipStdObserver",
    "MSEObserver",
]


# horizon_plugin_pytorch.nn.qat.functional.scale_quanti can be disabled by
# GlobalFakeQuantSwitch, so re-define it here to always enable fake quant
def scale_quanti(
    x,
    scale,
    zero_point,
    ch_axis,
    quant_min,
    quant_max,
):
    march = get_march()
    round_mode = QuantRoundMode.get()
    return torch.ops.horizon.scale_quanti(
        x,
        scale,
        zero_point,
        ch_axis,
        quant_min,
        quant_max,
        False,
        False,
        False,
        round_mode,
        march,
    )


@typechecked
def load_observer_params(
    from_model: torch.nn.Module,
    to_model: torch.nn.Module,
    verbose: bool = False,
):
    r"""Load observer parameters.

    When observers of the prepared model is changed, this function is needed
    to load observer parameters. e.g. calibration model -> qat model

    Args:
        from_model: from model.
        to_model: to model.
        verbose: Show unexpect_key and miss_key info.
    """
    miss_key, unexpect_key = to_model.load_state_dict(
        from_model.state_dict(),
        strict=False,
    )
    miss_key_str = " ".join(miss_key)
    unexpect_key_str = " ".join(unexpect_key)
    if len(miss_key) > 0 or len(unexpect_key) > 0:
        logger.warning("Please check if qconfig is correct.")
        logger.warning(
            f"Keys in from_model: {len(from_model.state_dict())}\n"
            f"Keys in to_model: {len(to_model.state_dict())}\n"
            f"Missing key: {len(miss_key)}\n"
            f"Unexpected key: {len(unexpect_key)}\n"
        )
        if verbose:
            logger.warning(
                f"miss_key: {miss_key_str}\n"
                f"unexpect_key: {unexpect_key_str}"
            )
    else:
        logger.info("Load observer parameters successfully!")


def _with_args(cls_or_self, **kwargs):
    r"""Wrap kwargs for creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.

    Example::

        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """

    class _PartialWrapper(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, *args, **keywords):
            return self.p(*args, **keywords)

        def __repr__(self):
            return self.p.__repr__()

        with_args = _with_args

    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r


@torch.jit.script
def _compute_scale_symmetric(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    quant_min: int,
    quant_max: int,
    eps: torch.Tensor,
    pow_quantization: bool,
) -> torch.Tensor:
    scale = (
        torch.max(-min_val, max_val)
        .clamp_min(0)
        .div(float(quant_max - quant_min) / 2)
        .clamp_min(eps)
    )
    if pow_quantization:
        scale = 1 / 2 ** (torch.floor((-1) * torch.log2(scale)).clamp(1, 14))
    return scale


@torch.jit.script
def _compute_moving_average(
    old_min: torch.Tensor,
    old_max: torch.Tensor,
    current_min: torch.Tensor,
    current_max: torch.Tensor,
    averaging_constant: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    min_val = old_min + averaging_constant * (current_min - old_min)
    max_val = old_max + averaging_constant * (current_max - old_max)
    return min_val, max_val


class ObserverBase(torch.nn.Module, metaclass=ABCMeta):
    r"""Base observer Module.

    Any observer implementation should derive from this class.

    Concrete observers should follow the same API. In forward, they will update
    the statistics of the observed Tensor. And they should provide a
    `calculate_qparams` function that computes the quantization parameters
    given the collected statistics.

    Args:
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis.
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        quant_min: Min quantization value. Will follow dtype if unspecified.
        quant_max: Max quantization value. Will follow dtype if unspecified.
        is_sync_quantize: If sync statistics when training with multiple
            devices.
        factory_kwargs: kwargs which are passed to factory functions for
            min_val and max_val.
    """

    _version = 3

    eps: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

    @typechecked
    def __init__(
        self,
        averaging_constant: float = 0.01,
        ch_axis: int = -1,
        dtype: Union[torch.dtype, QuantDType] = qint8,
        qscheme: torch.qscheme = torch.per_tensor_symmetric,
        quant_min: int = None,
        quant_max: int = None,
        is_sync_quantize: bool = False,
        factory_kwargs: Dict = None,
    ):
        super(ObserverBase, self).__init__()

        if qscheme == torch.per_channel_symmetric:
            assert (
                ch_axis >= 0
            ), "ch_axis should be non-negative when using per_channel_symmetric qcsheme"  # noqa: E501
        else:
            assert (
                ch_axis < 0
            ), "ch_axis should be negative when using per_tensor_symmetric qcsheme"  # noqa: E501
        dtype = get_horizon_quant_dtype(dtype)
        assert qscheme in (
            torch.per_tensor_symmetric,
            torch.per_channel_symmetric,
        ), (
            "only support per_tensor_symmetric and per_channel_symmetric "
            "qscheme"
        )

        self.averaging_constant = averaging_constant
        self.ch_axis = ch_axis
        self.dtype = dtype
        self.qscheme = qscheme

        self._set_quant_min_max(self.dtype, quant_min, quant_max)

        self.is_sync_quantize = is_sync_quantize
        self.pow_quantization = pow_quantization()

        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer(
            "eps",
            torch.tensor([torch.finfo(torch.float32).eps], **factory_kwargs),
        )
        self.register_buffer("min_val", torch.tensor([], **factory_kwargs))
        self.register_buffer("max_val", torch.tensor([], **factory_kwargs))

    def _set_quant_min_max(
        self,
        dtype,
        quant_min=None,
        quant_max=None,
    ):
        if (quant_min is not None) and (quant_max is not None):
            assert quant_min < quant_max, (
                "qmin must be strictly less than qmax for user-specified "
                "quantization range."
            )
            assert (
                quant_min <= 0 <= quant_max
            ), "Used-specified quantization range must include 0."
            assert qinfo(dtype).min <= quant_min, "quant_min out of bound"
            assert quant_max <= qinfo(dtype).max, "quant_max out of bound"
            self.quant_min, self.quant_max = quant_min, quant_max
        else:
            self.quant_min, self.quant_max = (
                qinfo(self.dtype).min,
                qinfo(self.dtype).max,
            )

    def reset_dtype(self, dtype):
        dtype = get_horizon_quant_dtype(dtype)
        if dtype == self.dtype:
            return
        self.dtype = dtype
        self._set_quant_min_max(self.dtype)

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
        # buffers has been renamed from min/max_vals to min/max_val
        buffer_name_mapping = {"min_vals": "min_val", "max_vals": "max_val"}
        for old_name in buffer_name_mapping:
            k = prefix + old_name
            if k in state_dict:
                v = state_dict.pop(k)
                state_dict[prefix + buffer_name_mapping[old_name]] = v

        eps_key = prefix + "eps"
        if eps_key not in state_dict:
            # eps was moved to a buffer in version 2
            eps = torch.tensor([torch.finfo(torch.float32).eps])
            state_dict[eps_key] = eps

        local_state = ["min_val", "max_val"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                # if ndim=0, make it ndim=1
                state_dict[key] = state_dict[key].reshape(-1)

                val = state_dict[key]

                # Custom handling to allow loading min_val or max_val
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == "min_val" and hasattr(self, "min_val"):
                    self.min_val.resize_(val.shape)
                elif hasattr(self, "max_val"):
                    self.max_val.resize_(val.shape)

        super(ObserverBase, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _load_from_state_dict_script(
        self,
        state_dict: Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        self._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculate the quantization parameters.

        Returns:
            scales: Scales tensor of shape (#channels,)
            zero_points: Zero points tensor of shape (#channels,)
        """
        if self.min_val.numel() == 0 or self.max_val.numel() == 0:
            warnings.warn(
                "Must run observer before calling calculate_qparams. "
                "Returning default scale and zero point. "
                "This is an expected behavior if you use KLObserver "
                "and set 1 < update_interval <= total steps. ",
            )
            return torch.tensor(
                [1.0], device=self.min_val.device
            ), torch.tensor([0], device=self.min_val.device)

        if not torch.jit.is_scripting():
            if self.is_sync_quantize and dist.is_initialized():
                dist.all_reduce(self.min_val, op=dist.ReduceOp.MIN)
                dist.all_reduce(self.max_val, op=dist.ReduceOp.MAX)

        scale = _compute_scale_symmetric(
            self.min_val,
            self.max_val,
            self.quant_min,
            self.quant_max,
            self.eps,
            self.pow_quantization,
        )

        return scale, None

    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)

    @abstractmethod
    def forward(self, x):
        pass

    with_args = classmethod(_with_args)


class MinMaxObserver(ObserverBase):
    r"""Min max observer.

    This observer computes the quantization parameters based on minimums and
    maximums of the incoming tensors. The module records the moving average
    minimum and maximum of incoming tensors, and uses this statistic to compute
    the quantization parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis.
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        quant_min: Min quantization value. Will follow dtype if unspecified.
        quant_max: Max quantization value. Will follow dtype if unspecified.
        is_sync_quantize: If sync statistics when training with multiple
            devices.
        factory_kwargs: kwargs which are passed to factory functions for
            min_val and max_val.
    """

    @typechecked
    def __init__(
        self,
        averaging_constant: float = 0.01,
        ch_axis: int = -1,
        dtype: Union[torch.dtype, QuantDType] = qint8,
        qscheme: torch.qscheme = torch.per_tensor_symmetric,
        quant_min: int = None,
        quant_max: int = None,
        is_sync_quantize: bool = False,
        factory_kwargs: Dict = None,
    ) -> None:
        super(MinMaxObserver, self).__init__(
            averaging_constant=averaging_constant,
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )

    def forward(self, x_orig):
        r"""Record the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.detach().to(self.min_val.dtype)

        if self.qscheme == torch.per_tensor_symmetric:
            min_val_cur, max_val_cur = torch.aminmax(x)
            min_val_cur = min_val_cur.reshape(-1)
            max_val_cur = max_val_cur.reshape(-1)
        else:
            if self.ch_axis > 0:
                # swap perchannel dim
                x = x.transpose(0, self.ch_axis)
            min_val_cur, max_val_cur = torch.aminmax(
                x.flatten(start_dim=1), dim=1
            )
        if self.min_val.numel() == 0 or self.max_val.numel() == 0:
            self.min_val, self.max_val = min_val_cur, max_val_cur
        else:
            (self.min_val, self.max_val,) = _compute_moving_average(
                self.min_val,
                self.max_val,
                min_val_cur,
                max_val_cur,
                self.averaging_constant,
            )

        return x_orig


class FixedScaleObserver(ObserverBase):
    r"""Fixed scale observer.

    This observer always return a fixed scale and zero_point regardless of
    input data.

    Args:
        scale: Fixed scale value.
        zero_point: Fixed zero_point value.
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis.
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        quant_min: Min quantization value. Will follow dtype if unspecified.
        quant_max: Max quantization value. Will follow dtype if unspecified.
        is_sync_quantize: If sync statistics when training with multiple
            devices.
        factory_kwargs: kwargs which are passed to factory functions for
            min_val and max_val.
    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    @typechecked
    def __init__(
        self,
        scale: float = 1,
        zero_point: float = 0,
        averaging_constant: float = 0,
        ch_axis: int = -1,
        dtype: Union[torch.dtype, QuantDType] = qint8,
        qscheme: torch.qscheme = torch.per_tensor_symmetric,
        quant_min: int = None,
        quant_max: int = None,
        is_sync_quantize: bool = False,
        factory_kwargs: Dict = None,
    ):
        assert scale > 0, "scale must bigger than 0"
        super(FixedScaleObserver, self).__init__(
            averaging_constant=averaging_constant,
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )

        del self.min_val
        del self.max_val
        self.register_buffer(
            "scale", torch.tensor(scale, dtype=torch.float).reshape(-1)
        )
        self.register_buffer(
            "zero_point",
            torch.tensor(zero_point, dtype=torch.long).reshape(-1),
        )

    def forward(self, x_orig):
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculate the quantization parameters."""
        return self.scale, self.zero_point

    @torch.jit.export
    def extra_repr(self):
        return "scale={}, zero_point={}".format(self.scale, self.zero_point)


class ClipObserver(ObserverBase):
    r"""Clip observer.

    This observer uses the tensor min/max statistics to compute the
    quantization parameters. The module records the running minimum and
    maximum of incoming tensors, if the runing minimum is greater the
    designated min value, the statistical minimum result is runing minimum,
    otherwise is the designated min value.And if the running minumum is less
    than the designated xmax, the statistical maxmum is running maxmum,
    otherwise is the designated max value. And uses this statistic to compute
    the quantization parameters.

    Args:
        xmin: Lower bound of statistical minimum
        xmax: Upper bound of statistical maximum
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis.
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        quant_min: Min quantization value. Will follow dtype if unspecified.
        quant_max: Max quantization value. Will follow dtype if unspecified.
        is_sync_quantize: If sync statistics when training with multiple
            devices.
        factory_kwargs: kwargs which are passed to factory functions for
            min_val and max_val.
    """

    setted_min: torch.Tensor
    setted_max: torch.Tensor

    @typechecked
    def __init__(
        self,
        xmin: float = -1.0,
        xmax: float = 1.0,
        averaging_constant: float = 1.0,
        ch_axis: int = -1,
        dtype: Union[torch.dtype, QuantDType] = qint8,
        qscheme: torch.qscheme = torch.per_tensor_symmetric,
        quant_min: int = None,
        quant_max: int = None,
        is_sync_quantize: bool = False,
        factory_kwargs: Dict = None,
    ):
        assert xmin <= xmax, "xmin must less than or equal to xmax."
        super(ClipObserver, self).__init__(
            averaging_constant=averaging_constant,
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )

        self.setted_min = xmin
        self.setted_max = xmax

    def forward(self, x_orig):
        r"""Record and clip the running minimum and maximum of ``x``."""
        x = x_orig.detach().to(self.min_val.dtype)

        if self.qscheme == torch.per_tensor_symmetric:
            min_val_cur, max_val_cur = torch.aminmax(x)
            min_val_cur = min_val_cur.reshape(-1)
            max_val_cur = max_val_cur.reshape(-1)
        else:
            if self.ch_axis > 0:
                x = x.transpose(0, self.ch_axis)
            min_val_cur, max_val_cur = torch.aminmax(
                x.flatten(start_dim=1), dim=1
            )
        if self.min_val.numel() == 0 or self.max_val.numel() == 0:
            self.min_val, self.max_val = min_val_cur, max_val_cur
        else:
            (self.min_val, self.max_val,) = _compute_moving_average(
                self.min_val,
                self.max_val,
                min_val_cur,
                max_val_cur,
                self.averaging_constant,
            )
        self.min_val.clamp_min_(self.setted_min)
        self.max_val.clamp_max_(self.setted_max)

        return x_orig

    def _load_from_state_dict(
        self,
        state_dict: Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        loaded_min = state_dict.get(prefix + "setted_min", None)
        loaded_max = state_dict.get(prefix + "setted_max", None)
        if loaded_min is not None:
            if self.setted_min != loaded_min:
                raise ValueError(
                    f"loaded setted_min {loaded_min} is conflict with current "
                    f"setted_min {self.setted_min}!"
                )
            state_dict.pop(prefix + "setted_min")
        if loaded_max is not None:
            if self.setted_max != loaded_max:
                raise ValueError(
                    f"loaded setted_max {loaded_max} is conflict with current "
                    f"setted_max {self.setted_max}!"
                )
            state_dict.pop(prefix + "setted_max")

        super(ClipObserver, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class PercentileObserver(ObserverBase):
    """Percentile observer.

    Percentile observer based on histogram. Histogram is calculated online
    and won't be saved. The minimum and maximum are moving averaged to compute
    the quantization parameters.

    Args:
        percentile: Index percentile of histrogram
        bins: Number of histograms bins.
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis.
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        quant_min: Min quantization value. Will follow dtype if unspecified.
        quant_max: Max quantization value. Will follow dtype if unspecified.
        is_sync_quantize: If sync statistics when training with multiple
            devices.
        factory_kwargs: kwargs which are passed to factory functions for
            min_val and max_val.
    """

    @typechecked
    def __init__(
        self,
        percentile: float = 99.99,
        bins: int = 2048,
        averaging_constant: float = 0.01,
        ch_axis: int = -1,
        dtype: Union[torch.dtype, QuantDType] = qint8,
        qscheme: torch.qscheme = torch.per_tensor_symmetric,
        quant_min: int = None,
        quant_max: int = None,
        is_sync_quantize: bool = False,
        factory_kwargs: Dict = None,
    ):
        super(PercentileObserver, self).__init__(
            averaging_constant=averaging_constant,
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )
        assert 0 <= percentile <= 100, "Percentile must be in range [0, 100]."
        assert (
            qscheme == torch.per_tensor_symmetric
        ), "Percentile observer only support per_tensor_symmetric qscheme."

        self.threshold = percentile / 100
        self.bins = bins

    def get_min_max(self, x_orig):
        x = x_orig.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        max_hist_range = torch.max(-min_val_cur, max_val_cur)
        hist = torch.histc(
            torch.abs(x), bins=self.bins, min=0.0, max=max_hist_range
        )
        i = torch.searchsorted(
            torch.cumsum(hist, dim=0), self.threshold * x.numel()
        )
        clip_value = (i + 0.5) * (max_hist_range / self.bins)
        min_val_cur = max(min_val_cur, -clip_value).reshape(-1)
        max_val_cur = min(max_val_cur, clip_value).reshape(-1)
        return min_val_cur, max_val_cur

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig

        min_val_cur, max_val_cur = self.get_min_max(x_orig)

        if self.min_val.numel() == 0 or self.max_val.numel() == 0:
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            (self.min_val, self.max_val,) = _compute_moving_average(
                self.min_val,
                self.max_val,
                min_val_cur,
                max_val_cur,
                self.averaging_constant,
            )
        return x_orig


class ClipStdObserver(ObserverBase):
    """Clip std observer.

    This observer computes the quantization parameters based on minimums and
    maximums of the incoming tensors. if the minimum or maximum exceeds
    std_scale times the standard deviation, it will be clipped to std_scale
    times the standard deviation.

    Args:
        std_scale: The scale for standard deviation.
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis.
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        quant_min: Min quantization value. Will follow dtype if unspecified.
        quant_max: Max quantization value. Will follow dtype if unspecified.
        is_sync_quantize: If sync statistics when training with multiple
            devices.
        factory_kwargs: kwargs which are passed to factory functions for
            min_val and max_val.
    """

    @typechecked
    def __init__(
        self,
        std_scale: float = 3.0,
        averaging_constant: float = 0.01,
        ch_axis: int = -1,
        dtype: Union[torch.dtype, QuantDType] = qint8,
        qscheme: torch.qscheme = torch.per_tensor_symmetric,
        quant_min: int = None,
        quant_max: int = None,
        is_sync_quantize: bool = False,
        factory_kwargs: Dict = None,
    ):
        super(ClipStdObserver, self).__init__(
            averaging_constant=averaging_constant,
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )
        assert std_scale > 0, "std_scale should be greated than 0."
        assert (
            qscheme == torch.per_tensor_symmetric
        ), "ClipStdObserver only support per_tensor_symmetric qscheme."
        self.std_scale = std_scale

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        mean = x.mean()
        std = x.std()
        min_val_cur = torch.maximum(min_val_cur, mean - self.std_scale * std)
        max_val_cur = torch.minimum(max_val_cur, mean + self.std_scale * std)

        if self.min_val.numel() == 0 or self.max_val.numel() == 0:
            self.min_val = min_val_cur.reshape(-1)
            self.max_val = max_val_cur.reshape(-1)
        else:
            (self.min_val, self.max_val,) = _compute_moving_average(
                self.min_val,
                self.max_val,
                min_val_cur.reshape(-1),
                max_val_cur.reshape(-1),
                self.averaging_constant,
            )
        return x_orig


class NormObserver(ObserverBase):
    """Norm observer.

    This observer computes the quantization parameters based on the norm of
    the incoming tensors.

    Args:
        norm: Norm of input tensor for computing scale.
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis.
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        quant_min: Min quantization value. Will follow dtype if unspecified.
        quant_max: Max quantization value. Will follow dtype if unspecified.
        is_sync_quantize: If sync statistics when training with multiple
            devices.
        factory_kwargs: kwargs which are passed to factory functions for
            min_val and max_val.
    """

    @typechecked
    def __init__(
        self,
        norm: float = 2.0,
        averaging_constant: float = 0.01,
        ch_axis: int = -1,
        dtype: Union[torch.dtype, QuantDType] = qint8,
        qscheme: torch.qscheme = torch.per_tensor_symmetric,
        quant_min: int = None,
        quant_max: int = None,
        is_sync_quantize: bool = False,
        factory_kwargs: Dict = None,
    ):
        super(NormObserver, self).__init__(
            averaging_constant=averaging_constant,
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )
        assert norm > 0, "norm should be greated than 0."
        assert (
            qscheme == torch.per_tensor_symmetric
        ), "Norm observer only support per_tensor_symmetric qscheme."
        self.norm = norm

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        max_val_cur = torch.linalg.norm(x.abs(), ord=self.norm, dim=0)
        min_val_cur = -max_val_cur

        if self.min_val.numel() == 0 or self.max_val.numel() == 0:
            self.min_val = min_val_cur.reshape(-1)
            self.max_val = max_val_cur.reshape(-1)
        else:
            (self.min_val, self.max_val,) = _compute_moving_average(
                self.min_val,
                self.max_val,
                min_val_cur.reshape(-1),
                max_val_cur.reshape(-1),
                self.averaging_constant,
            )
        return x_orig


@torch.jit.script
def l2_loss(pred, tgt):
    # input shape (N) or (C, N), output shape ([]) or (C)
    return torch.linalg.norm(pred - tgt, ord=2, dim=-1)


def _mse(
    x: torch.Tensor,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    quant_min: int,
    quant_max: int,
    eps: torch.Tensor,
    pow_quantization: bool,
    num_iter: int,
    stride: int,
):
    mses = []
    for i in range(0, num_iter, stride):
        new_min = x_min * (1.0 - (i * 0.01))
        new_max = x_max * (1.0 - (i * 0.01))
        scale = _compute_scale_symmetric(
            new_min,
            new_max,
            quant_min,
            quant_max,
            eps,
            pow_quantization,
        )

        x_fakequant = scale_quanti(
            x,
            scale,
            torch.zeros_like(scale),
            0 if scale.numel() > 1 else -1,
            quant_min,
            quant_max,
        )

        mse = l2_loss(x_fakequant, x).reshape(-1)
        mses.append(mse)
    # mses is shape [num_iter] or [num_iter, num_channel], argmin should apply on num_iter # noqa: E501
    mses = torch.stack(mses[::-1], dim=0).to(x.device)
    last_argmin = mses.size(0) - 1 - torch.argmin(mses, dim=0)
    best_min = x_min * (1.0 - (last_argmin * 0.01))
    best_max = x_max * (1.0 - (last_argmin * 0.01))
    return best_min, best_max


@torch.jit.script
def _broadcast_scale_quant_half_to_even(
    x: torch.Tensor,
    scale: torch.Tensor,
    quant_min: int,
    quant_max: int,
):
    return torch.round(x / scale).clamp(quant_min, quant_max) * scale


@torch.jit.script
def _broadcast_scale_quant_bpu_round(
    x: torch.Tensor,
    scale: torch.Tensor,
    quant_min: int,
    quant_max: int,
):
    return torch.floor((x / scale) + 0.5).clamp(quant_min, quant_max) * scale


class MSEObserver(ObserverBase):
    r"""MSE observer.

    Observer module for computing the quantization parameters based on the
    Mean Square Error (MSE) between the original tensor and the quantized one.

    This observer linear searches the quantization scales that minimize MSE.

    Args:
        stride: Searching stride. Larger value gives smaller search space,
            which means less computing time but possibly poorer accuracy.
            Default is 1. Suggests no greater than 20.
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis.
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        quant_min: Min quantization value. Will follow dtype if unspecified.
        quant_max: Max quantization value. Will follow dtype if unspecified.
        is_sync_quantize: If sync statistics when training with multiple
            devices.
        factory_kwargs: kwargs which are passed to factory functions for
            min_val and max_val.
    """

    minmax_scale: torch.Tensor
    _parallel_compute: bool = False

    @typechecked
    def __init__(
        self,
        stride: int = 1,
        averaging_constant: float = 0.01,
        ch_axis: int = -1,
        dtype: Union[torch.dtype, QuantDType] = qint8,
        qscheme: torch.qscheme = torch.per_tensor_symmetric,
        quant_min: int = None,
        quant_max: int = None,
        is_sync_quantize: bool = False,
        factory_kwargs: Dict = None,
    ):
        super(MSEObserver, self).__init__(
            averaging_constant=averaging_constant,
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )
        self.stride = stride
        self.num_iter = 95
        self.register_buffer(
            "minmax_scale",
            1.0
            - (
                torch.arange(0, self.num_iter, self.stride).reshape(-1, 1)
                * 0.01
            ),
            persistent=False,
        )

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.detach().to(self.min_val.dtype)

        if self.qscheme == torch.per_tensor_symmetric:
            x = x.flatten()
            min_val_cur, max_val_cur = torch.aminmax(x)
            x = x.reshape(1, -1)
        else:
            if self.ch_axis > 0:
                # swap perchannel dim
                x = x.transpose(0, self.ch_axis)
            # [channel_len, -1]
            x = x.flatten(start_dim=1)
            min_val_cur, max_val_cur = torch.aminmax(x, dim=1)

        if self._parallel_compute:
            # [num_iter, channel_len]
            new_min = min_val_cur * self.minmax_scale
            new_max = max_val_cur * self.minmax_scale

            # [num_iter, channel_len, 1]
            scale = _compute_scale_symmetric(
                new_min,
                new_max,
                self.quant_min,
                self.quant_max,
                self.eps,
                self.pow_quantization,
            ).unsqueeze(-1)

            # [num_iter, channel_len, -1]
            if QuantRoundMode.get() == QuantRoundMode.HALF_TO_EVEN:
                x_fakequant = _broadcast_scale_quant_half_to_even(
                    x, scale, self.quant_min, self.quant_max
                )
            elif QuantRoundMode.get() == QuantRoundMode.BPU_ROUND:
                x_fakequant = _broadcast_scale_quant_bpu_round(
                    x, scale, self.quant_min, self.quant_max
                )
            else:
                raise ValueError(
                    "Unexpected RoundMode {}".format(QuantRoundMode.get())
                )
            # [num_iter]
            mses = l2_loss(x_fakequant.flatten(1, -1), x.flatten()).flatten()

            best_index = torch.argmin(mses, dim=0)
            min_val_cur = new_min[best_index]
            max_val_cur = new_max[best_index]
        else:
            min_val_cur, max_val_cur = _mse(
                x,
                min_val_cur.reshape(-1),
                max_val_cur.reshape(-1),
                self.quant_min,
                self.quant_max,
                self.eps,
                self.pow_quantization,
                95,  # num_iter
                self.stride,
            )

        if self.min_val.numel() == 0 or self.max_val.numel() == 0:
            self.min_val = min_val_cur.reshape(-1)
            self.max_val = max_val_cur.reshape(-1)
        else:
            (self.min_val, self.max_val,) = _compute_moving_average(
                self.min_val,
                self.max_val,
                min_val_cur.reshape(-1),
                max_val_cur.reshape(-1),
                self.averaging_constant,
            )
        return x_orig


def _norm_hist(hist):
    sums = torch.sum(hist)
    return hist / sums if sums != 0 else hist


def _entropy_torch(pk, qk=None, axis=0):
    """Compute entropy.

    This is the pytorch implementation of `scipy.stats.entropy`,
        and it is as close as possible to the original implementation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html

    Args:
        pk (torch.Tensor): Defines the (discrete) distribution. pk[i] is
            the (possibly unnormalized) probability of event i.
        qk (torch.Tensor): Sequence against which the relative
            entropy is computed. Should be in the same format as pk.
        axis (int): The axis along which the entropy is calculated.
            Default is 0.

    Returns:
        torch.Tensor: The calculated entropy.
    """

    pk = _norm_hist(pk)

    if qk is None:
        # Note: `torch.special,entr()` requires torch.__version__>=1.9.0
        vec = torch.special.entr(pk)
    else:
        if qk.shape != pk.shape:
            raise ValueError("qk and pk must have same shape.")
        qk = _norm_hist(qk) + 1e-10
        vec = -(qk) * (torch.special.entr(pk / qk))
    ret = torch.sum(vec, dim=axis)
    return ret


def _compute_amax_entropy(
    calib_hist,
    calib_bin_edges,
    quantization_nbins,
    stride=1,
    start_bin=128,
    device=None,
):
    """Compute amax that minimizes KL-Divergence of the collected histogram."""  # noqa: E501

    # If calibrator hasn't collected any data, return none
    if calib_bin_edges.numel() == 0 and calib_hist.numel() == 0:
        return None

    if device is None:
        device = calib_bin_edges.device

    bins = calib_hist
    bins[0] = bins[1]
    bins_numpy = bins.cpu().numpy()

    divergences = []

    starting = start_bin
    stop = len(bins)

    new_density_counts = np.zeros(quantization_nbins, dtype=np.float64)

    for i in range(starting, stop + 1, stride):
        new_density_counts.fill(0)
        space = np.linspace(0, i, quantization_nbins + 1)

        digitized_space = np.digitize(range(i), space) - 1

        digitized_space[bins_numpy[:i] == 0] = -1

        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                new_density_counts[digitized] += bins_numpy[idx]

        counter = Counter(digitized_space)
        for key, val in counter.items():
            if key != -1:
                new_density_counts[key] = new_density_counts[key] / val

        new_density = np.zeros(i, dtype=np.float64)
        for idx, digitized in enumerate(digitized_space):
            if digitized.item() != -1:
                new_density[idx] = new_density_counts[digitized]

        reference_density = bins[:i].clone().detach()
        reference_density[-1] += torch.sum(bins[i:])

        ent = _entropy_torch(
            pk=reference_density.to(torch.float64),
            qk=torch.from_numpy(new_density).to(device),
        )
        if ent.isnan():
            raise ValueError("Fatal error, kl is nan!")
        divergences.append(ent)

    divergences = torch.as_tensor(divergences, device=device)
    smooth_step = 9
    smoothed_divergences = torch.as_tensor(
        [
            torch.sum(divergences[i : i + smooth_step])
            for i in range(len(divergences) - smooth_step - 1)
        ][::-1],
        device=device,
    )
    last_argmin = (
        len(smoothed_divergences) - 1 - torch.argmin(smoothed_divergences)
    )
    calib_amax = calib_bin_edges[
        last_argmin * stride + starting + smooth_step // 2
    ]

    logger.debug(
        f"Absolute amax is {calib_bin_edges[-1]}, KL output amax is {calib_amax}, which is {last_argmin * stride + starting}/{len(calib_bin_edges)-1}."  # noqa: E501
    )

    return calib_amax


class KLObserver(ObserverBase):
    """KL observer.

    KL observer based on histogram. Histogram is calculated online
    and won't be saved.

    Args:
        bins: Number of histograms bins.
        update_interval: Interval of computing KL entropy and update min/max.
            KLObserver will constantly collect histograms of activations,
            but only perform KL calculation when update_interval is satisfied.
            if it is set to 1, KL entropy will be computed every forward step.
            Larger interval guarantees less time and does no harm to
            calibration accuracy. Set it to the total calibration steps can
            achieve best performance. update_interval must be no greater than
            total calibration steps, otherwise no min/max will be computed.
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis.
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        quant_min: Min quantization value. Will follow dtype if unspecified.
        quant_max: Max quantization value. Will follow dtype if unspecified.
        is_sync_quantize: If sync statistics when training with multiple
            devices.
        factory_kwargs: kwargs which are passed to factory functions for
            min_val and max_val.
    """

    @typechecked
    def __init__(
        self,
        bins: int = 512,
        update_interval: int = 1,
        averaging_constant: float = 0.01,
        ch_axis: int = -1,
        dtype: Union[torch.dtype, QuantDType] = qint8,
        qscheme: torch.qscheme = torch.per_tensor_symmetric,
        quant_min: int = None,
        quant_max: int = None,
        is_sync_quantize: bool = False,
        factory_kwargs: Dict = None,
    ):
        super(KLObserver, self).__init__(
            averaging_constant=averaging_constant,
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )
        assert (
            qscheme == torch.per_tensor_symmetric
        ), "KL observer only support per_tensor_symmetric qscheme."

        assert bins >= 256, "`bins` must be greater than 256!"
        if update_interval == 1:
            logger.warning(
                "`update_interval` for KL observer is 1. "
                "Recommend to set it equal to the "
                "number of calibration steps for best performance, "
                "which does no harm to accuracy. In this way the observer "
                "will constantly collect histograms of activations "
                "and only perform KL calculation at the last step.",
                extra={"call_times_context": ("message")},
            )
        self.orig_bins = bins
        self.bins = bins
        self.update_interval = update_interval
        self.cur_step = 0
        self.register_buffer(
            "_calib_bin_edges", torch.tensor([]), persistent=False
        )
        self.register_buffer("_calib_hist", torch.tensor([]), persistent=False)

    def forward(self, x_orig):
        self._device = x_orig.device
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach().float()
        x = x.abs()
        x_max = x.max()

        if x_max == 0:
            logger.warning(
                "The inputs of this layer are all 0,"
                "check you input and model",
                extra={"call_times_context": ("message")},
            )
        if (
            self._calib_bin_edges.numel() == 0
            and self._calib_hist.numel() == 0
        ):
            self._calib_hist = torch.histc(
                x,
                bins=self.bins,
                min=0.0,
                max=x_max,
            )
            self._calib_bin_edges = torch.linspace(
                0, x_max, self.bins + 1, device=self._device
            )
        else:
            if x_max > self._calib_bin_edges[-1]:
                width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                self.bins = int((x_max / width).ceil().item())
                self._calib_bin_edges = torch.arange(
                    0, x_max + width, width, device=self._device
                )

            hist = torch.histc(
                x,
                bins=self.bins,
                min=0,
                max=self._calib_bin_edges[-1],
            )
            hist[: self._calib_hist.numel()] += self._calib_hist
            self._calib_hist = hist

        self.cur_step += 1

        if self.cur_step == self.update_interval:
            assert (
                self._calib_hist.numel() > 0
                and self._calib_bin_edges.numel() > 0
            ), "no hist is collected!"
            clip_value = _compute_amax_entropy(
                self._calib_hist, self._calib_bin_edges, -qinfo(self.dtype).min
            )

            if self.min_val.numel() == 0 or self.max_val.numel() == 0:
                self.min_val = (-clip_value).reshape(-1)
                self.max_val = clip_value.reshape(-1)
            else:
                (self.min_val, self.max_val,) = _compute_moving_average(
                    self.min_val,
                    self.max_val,
                    (-clip_value).reshape(-1),
                    clip_value.reshape(-1),
                    self.averaging_constant,
                )

            # reinitialize states and buffers
            self.cur_step = 0
            self._calib_hist = torch.tensor([], device=self._calib_hist.device)
            self._calib_bin_edges = torch.tensor(
                [], device=self._calib_bin_edges.device
            )
            self.bins = self.orig_bins

        return x_orig


class MixObserver(ObserverBase):
    """Mix observer.

    This observer computes the quantization parameters based on multiple
    calibration methods and selects the quantization parameters with the
    smallest quantization error.

    Args:
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis.
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        quant_min: Min quantization value. Will follow dtype if unspecified.
        quant_max: Max quantization value. Will follow dtype if unspecified.
        is_sync_quantize: If sync statistics when training with multiple
            devices.
        factory_kwargs: kwargs which are passed to factory functions for
            min_val and max_val.
    """

    @typechecked
    def __init__(
        self,
        averaging_constant: float = 0.01,
        ch_axis: int = -1,
        dtype: Union[torch.dtype, QuantDType] = qint8,
        qscheme: torch.qscheme = torch.per_tensor_symmetric,
        quant_min: int = None,
        quant_max: int = None,
        is_sync_quantize: bool = False,
        factory_kwargs: Dict = None,
    ):
        super(MixObserver, self).__init__(
            averaging_constant=averaging_constant,
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )
        assert (
            qscheme == torch.per_tensor_symmetric
        ), "Mix observer only support per_tensor_symmetric qscheme."

    def get_min_max_of_percentile_observer(self, x):
        min_val_cur, max_val_cur = torch.aminmax(x)
        max_hist_range = torch.max(-min_val_cur, max_val_cur)
        hist = torch.histc(
            torch.abs(x), bins=2048, min=0.0, max=max_hist_range
        )
        indexs = torch.searchsorted(
            torch.cumsum(hist, dim=0),
            torch.tensor(
                [
                    0.999,
                    0.9995,
                    0.9999,
                    0.99993,
                    0.99995,
                    0.99997,
                    0.99999,
                    0.999995,
                    0.999999,
                ],
                device=hist.device,
            )
            * x.numel(),
        )
        indexs = torch.unique(indexs)
        clip_values = (indexs + 0.5) * (max_hist_range / 2048)
        min_val_curs = [max(min_val_cur, -i).reshape(-1) for i in clip_values]
        max_val_curs = [min(max_val_cur, i).reshape(-1) for i in clip_values]
        return min_val_curs, max_val_curs

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach().to(self.min_val.dtype)
        min_val_curs, max_val_curs = self.get_min_max_of_percentile_observer(x)

        best_similarity = -float("inf")

        for min_val_cur, max_val_cur in zip(min_val_curs, max_val_curs):
            scale = _compute_scale_symmetric(
                min_val_cur,
                max_val_cur,
                self.quant_min,
                self.quant_max,
                self.eps,
                self.pow_quantization,
            )

            x_fake_quantized = scale_quanti(
                x,
                scale,
                torch.zeros_like(scale),
                self.ch_axis,
                self.quant_min,
                self.quant_max,
            )

            similarity = -l2_loss(x.flatten(), x_fake_quantized.flatten())
            if similarity > best_similarity:
                best_similarity = similarity
                best_min, best_max = min_val_cur, max_val_cur

        if self.min_val.numel() == 0 or self.max_val.numel() == 0:
            self.min_val = best_min
            self.max_val = best_max
        else:
            (self.min_val, self.max_val,) = _compute_moving_average(
                self.min_val,
                self.max_val,
                best_min,
                best_max,
                self.averaging_constant,
            )
        return x_orig
