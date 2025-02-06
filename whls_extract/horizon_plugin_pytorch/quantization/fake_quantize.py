"""same as torch.quantization.FakeQuantize."""

import logging
import re
from enum import Enum
from typing import Sequence, Union

import torch
from torch.quantization.fake_quantize import FakeQuantizeBase, _is_per_channel
from torch.quantization.observer import NoopObserver, _with_args

from horizon_plugin_pytorch.dtype import qinfo
from horizon_plugin_pytorch.fx import fx_helper
from horizon_plugin_pytorch.march import March, get_march
from horizon_plugin_pytorch.nn.qat.functional import scale_quanti
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.checkpoint import CheckpointState
from horizon_plugin_pytorch.utils.global_quant_round_mode import QuantRoundMode
from horizon_plugin_pytorch.utils.load_state_dict_helper import get_version
from horizon_plugin_pytorch.utils.quant_switch import GlobalFakeQuantSwitch
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .misc import pow_quantization, set_qparam
from .observer import CalibObserver, FixedScaleObserver
from .observer import MinMaxObserver as MinMaxObserverV1
from .observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from .observer_v2 import FixedScaleObserver as FixedScaleObserverv2
from .observer_v2 import MinMaxObserver

logger = logging.getLogger(__name__)
saturate_grad = True


def _is_affine(qscheme: "torch.qscheme") -> bool:
    return qscheme in [torch.per_tensor_affine, torch.per_channel_affine]


@fx_helper.wrap()
class FakeQuantize(FakeQuantizeBase):
    r"""Simulate the quantize and dequantize operations in training time.

    The output of this module is given by

    fake_quant_x = clamp(round(x / scale), quant_min, quant_max) * scale



    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating
      point maps to

    * :attr:`quant_min` specifies the minimum allowable quantized value.

    * :attr:`quant_max` specifies the maximum allowable quantized value.

    * :attr:`fake_quant_enabled` controls the application of fake quantization
      on tensors, note that statistics can still be updated.

    * :attr:`observer_enabled` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with
      fake-quantization, the allowable values is qint8 and qint16. The values
      of quant_min and quant_max should be chosen to be consistent with the
      dtype


    Args:
        observer: Module for observing statistics on input
            tensors and calculating scale and zero-point.
        saturate: Whether zero out the grad for value out of quanti range.
        in_place: Whether use in place fake quantize.
        compat_mask: Whether pack the bool mask into bitfield
            when saturate = True.
        channel_len: Size of data at channel dim.
        fast_training: Whether use fast training mode. If True, computing scale
            and fake quantization will be done in one step.
        observer_kwargs: Arguments for the observer module

    Attributes:
        observer: User provided module that collects statistics on the input
            tensor and provides a method to calculate scale and zero-point.

    """

    scale: torch.Tensor
    zero_point: torch.Tensor
    _version = 2

    @typechecked
    def __init__(
        self,
        observer: type(torch.nn.Module) = MovingAverageMinMaxObserver,
        saturate: bool = None,
        in_place: bool = False,
        compat_mask: bool = True,
        channel_len: int = 1,
        fast_training=True,
        **observer_kwargs,
    ):
        self.fast_training = fast_training
        assert (
            channel_len >= 1
        ), "channel_len should greater than or equal to 1"

        super(FakeQuantize, self).__init__()
        # use flags rather than buffer to avoid cuda to cpu copy and
        # speed up forward
        self._fake_quant_enabled = True
        self._observer_enabled = True

        self.activation_post_process = observer(
            **observer_kwargs,
        )
        # get quant minmax from observer where they are properly configured
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.is_per_channel = _is_per_channel(self.qscheme)
        self.is_affine = _is_affine(self.qscheme)
        self.ch_axis = (
            self.activation_post_process.ch_axis if self.is_per_channel else -1
        )

        if self.is_per_channel:
            scale_len = channel_len
        else:
            scale_len = 1

        self.register_buffer(
            "scale", torch.ones(scale_len, dtype=torch.float32)
        )
        self.register_buffer(
            "zero_point", torch.zeros(scale_len, dtype=torch.long)
        )

        if observer in (FixedScaleObserver, FixedScaleObserverv2):
            fixed_scale, fixed_zero_point = self.calculate_qparams()
            self.set_qparams(fixed_scale, fixed_zero_point)

        self.march = get_march()

        if saturate is not None:
            self.saturate = saturate
        else:
            self.saturate = saturate_grad
        self.in_place = in_place
        self.compat_mask = compat_mask

    def get_dtype(self):
        return self.dtype

    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    @torch.jit.export
    def enable_fake_quant(self, enabled: bool = True) -> None:
        self.fake_quant_enabled[0] = 1 if enabled else 0
        self._fake_quant_enabled = enabled

    @torch.jit.export
    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled[0] = 1 if enabled else 0
        self._observer_enabled = enabled

    def reset_dtype(self, dtype=None, update_scale=True):
        if dtype is None:
            dtype = self.dtype
        self.activation_post_process.reset_dtype(dtype)
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.dtype = self.activation_post_process.dtype
        if update_scale:
            _scale, _zero_point = self.calculate_qparams()
            self.scale = _scale
            # self.zero_point = _zero_point

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if get_version(self, prefix, local_metadata) < 2:
            QuantRoundMode.set(QuantRoundMode.BPU_ROUND)

        v = state_dict.get(prefix + "observer_enabled", None)
        if v is not None:
            self._observer_enabled = v[0].item() == 1  # use item to get a bool

        v = state_dict.get(prefix + "fake_quant_enabled", None)
        if v is not None:
            self._fake_quant_enabled = v[0].item() == 1

        super(FakeQuantize, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def set_qparams(
        self,
        scale: Union[torch.Tensor, Sequence, float],
        zero_point: Union[torch.Tensor, Sequence, int] = None,
    ):
        """Set qparams, default symmetric."""
        set_qparam(scale, self.scale, "scale")
        if zero_point is not None:
            set_qparam(zero_point, self.zero_point, "zero_point")
        else:
            self.zero_point.copy_(torch.zeros_like(self.zero_point))

    def _fast_train_forward(self, x):
        logger.warning(
            "fast training is experimental",
            extra={"call_times_context": ("message")},
        )
        self.activation_post_process(x.detach())

        def get_minmax():
            if (
                self.activation_post_process.min_val.numel() == 0
                or self.activation_post_process.max_val.numel() == 0
            ):
                logger.warning(
                    "Must run observer before calling calculate_qparams. "
                    "Returning default scale and zero point. "
                    "This is an expected behavior if you use KLObserver "
                    "and set 1 < update_interval <= total steps. ",
                    extra={"call_times_context": ("message")},
                )
                return torch.tensor(
                    [-128.0],
                    device=self.activation_post_process.min_val.device,
                ), torch.tensor(
                    [127.0],
                    device=self.activation_post_process.min_val.device,
                )
            else:
                return (
                    self.activation_post_process.min_val,
                    self.activation_post_process.max_val,
                )

        min_val, max_val = get_minmax()
        x, scale, mask = torch.ops.horizon.fake_quantize_minmax(
            x,
            min_val.to(x.dtype),
            max_val.to(x.dtype),
            self.ch_axis,
            self.quant_min,
            self.quant_max,
            self.saturate,
            QuantRoundMode.get(),
        )
        self.scale = scale.detach()

        return QTensor(
            data=x,
            scale=self.scale,
            dtype=self.dtype,
            per_channel_axis=self.ch_axis,
        )

    def forward(self, x):
        if all(
            (
                x.dtype
                == torch.float32,  # gradient with fp16 and bfp16 is unstable # noqa: E501
                self.fast_training,
                self.training,
                self._observer_enabled,
                self._fake_quant_enabled,
                GlobalFakeQuantSwitch.state(),
                get_march() != March.BERNOULLI,  # shift quant is not supported
                not CheckpointState.supress_update(),
                not self.in_place,
                isinstance(
                    self.activation_post_process,
                    (
                        MinMaxObserver,
                        MinMaxObserverV1,
                    ),
                ),
            )
        ):
            return self._fast_train_forward(x)
        else:
            pass
        # only update scale when training
        # supress update in CheckpointFunction.backward
        if (
            self.training
            and self._observer_enabled
            and not CheckpointState.supress_update()
        ):
            self.activation_post_process(x.detach())
            _scale, _zero_point = self.calculate_qparams()
            self.scale = _scale
            # if _zero_point is not None:
            #     self.zero_point = _zero_point

        if self._fake_quant_enabled and GlobalFakeQuantSwitch.state():
            if self.training:
                x = scale_quanti(
                    x,
                    self.scale,
                    self.zero_point,
                    self.ch_axis,
                    self.quant_min,
                    self.quant_max,
                    self.saturate,
                    self.in_place,
                    self.compat_mask,
                )
            else:
                x = torch.ops.horizon.scale_quanti_opt(
                    x,
                    self.scale,
                    self.quant_min,
                    self.quant_max,
                    self.ch_axis,
                    QuantRoundMode.get(),
                )

        # return qtensor type
        return QTensor(
            data=x,
            scale=self.scale,
            dtype=self.dtype,
            per_channel_axis=self.ch_axis,
        )

    with_args = classmethod(_with_args)

    @torch.jit.export
    def extra_repr(self):
        return f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, quant_min={self.quant_min}, quant_max={self.quant_max}, dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}, scale={self.scale}, zero_point={self.zero_point}, saturate={self.saturate}"  # noqa: E501


class CalibFakeQuantize(FakeQuantizeBase):
    def __init__(
        self,
        channel_len=1,
        dtype="qint8",
        ch_axis=-1,
        observer=CalibObserver,
        **observer_kwargs,
    ):
        assert dtype in (
            "qint8",
            "qint16",
        ), f"unsupported dtype: {dtype}"
        super(CalibFakeQuantize, self).__init__()
        self._observer_enabled = True
        self.activation_post_process = observer(**observer_kwargs)
        if observer in (NoopObserver,):
            self.scale_len = channel_len
        else:
            self.scale_len = 1
        self.register_buffer(
            "scale", torch.ones(self.scale_len, dtype=torch.float32)
        )
        self.register_buffer(
            "zero_point", torch.zeros(self.scale_len, dtype=torch.long)
        )
        self.channel_len = channel_len
        self.dtype = dtype
        self.ch_axis = ch_axis

    def get_dtype(self):
        return self.dtype

    @torch.jit.export
    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled[0] = 1 if enabled else 0
        self._observer_enabled = enabled

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        v = state_dict.get(prefix + "observer_enabled", None)
        if v is not None:
            self._observer_enabled = v[0].item() == 1  # use item to get a bool

        super(CalibFakeQuantize, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        if isinstance(self.activation_post_process, FixedScaleObserver):
            (
                _scale,
                _zero_point,
            ) = self.activation_post_process.calculate_qparams()
            self.set_qparams(_scale, _zero_point)
        elif self._observer_enabled and self.scale_len == 1:
            self.activation_post_process(x.detach())
            fmax = torch.max(torch.abs(x))
            scale = 2 * fmax / (qinfo(self.dtype).max - qinfo(self.dtype).min)
            if pow_quantization():
                min_valid_shift = 1
                max_valid_shift = 14
                shift = torch.floor((-1) * torch.log2(scale))
                shift = torch.clamp(shift, min_valid_shift, max_valid_shift)
                scale = 1 / 2 ** shift
            self.scale.copy_(scale.detach())
        else:
            pass
        return QTensor(
            data=x,
            scale=self.scale,
            dtype=self.dtype,
            per_channel_axis=self.ch_axis,
        )

    def calculate_qparams(self, **kwargs):
        pass

    def set_qparams(
        self,
        scale,
        zero_point=None,
    ):
        """Set qparams, default symmetric."""
        set_qparam(scale, self.scale, "scale")
        if zero_point is not None:
            set_qparam(zero_point, self.zero_point, "zero_point")
        else:
            self.zero_point.copy_(torch.zeros_like(self.zero_point))

    with_args = classmethod(_with_args)


default_8bit_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    dtype="qint8",
)

per_channel_8bit_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    dtype="qint8",
    ch_axis=1,
)

default_weight_8bit_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    dtype="qint8",
    ch_axis=0,
)

default_4bit_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    dtype="qint4",
)
default_uint4_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    dtype="quint4",
)

default_weight_4bit_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    dtype="qint4",
    ch_axis=0,
)
default_16bit_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    dtype="qint16",
)
default_weight_16bit_fake_quant = FakeQuantize.with_args(
    observer=MovingAveragePerChannelMinMaxObserver,
    dtype="qint16",
    ch_axis=0,
)
default_calib_fake_quant = CalibFakeQuantize.with_args(
    observer=CalibObserver,
    num_bits=8,
    axis=None,
    unsigned=False,
    num_bins=2048,
    skip_zeros=False,
    method="percentile",
    percentile=99.99,
)

default_weight_calib_fake_quant = CalibFakeQuantize.with_args(
    observer=NoopObserver,
    ch_axis=0,
)


def _is_fake_quant_script_module(mod):
    """Refine this docstring.

    Returns true if given mod is an instance of FakeQuantize script module.
    """
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        # qualified name looks like
        # '__torch__.torch.quantization.fake_quantize.___torch_mangle_2.FakeQuantize' # noqa
        suffix = mod._c.qualified_name.split(".", 1)[1]
        name = re.sub(r"\.___torch_mangle_\d+", "", suffix)
        return (
            name
            == "horizon_plugin_pytorch.quantization.fake_quantize.FakeQuantize"
        )  # noqa: F401
    return False


def disable_fake_quant(mod):
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.disable_fake_quant()


def enable_fake_quant(mod):
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.enable_fake_quant()


def disable_observer(mod):
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.disable_observer()


def enable_observer(mod):
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.enable_observer()


class FakeQuantState(Enum):
    """Defines the working mode of FakeQuantize."""

    # qat mode: fake quantization is enabled. scales will be updated.
    # must work with model.train()
    QAT = "qat"

    # calibration mode: fake quantization is disabled. only updates scales.
    # must work with model.eval()
    CALIBRATION = "calibration"

    # validation mode: fake quantization is enabled. scales will be fixed.
    # must work with model.eval()
    VALIDATION = "validation"

    # float mode: developer ONLY
    # fake quantization is disabled. scales will be fixed.
    _FLOAT = "float"


@typechecked
def set_fake_quantize(model: torch.nn.Module, mode: FakeQuantState):
    r"""Set the state of fake quantize.

    Args:
        model: prepared model.
        mode: Mode of fake quantize.
    """
    if mode in (FakeQuantState.QAT, FakeQuantState.VALIDATION):
        GlobalFakeQuantSwitch.enable()
    else:
        GlobalFakeQuantSwitch.disable()

    observers = set()
    for _, mod in model.named_modules():
        if isinstance(mod, FakeQuantizeBase):
            observers.add(mod.activation_post_process)

    for _, mod in model.named_modules():
        if mod in observers:
            continue
        if mode == FakeQuantState.QAT:
            assert (
                mod.training
            ), "Call model.train() before set fake quant to QAT mode."
            enable_fake_quant(mod)
        elif mode == FakeQuantState.CALIBRATION:
            assert (
                not mod.training
            ), "Call model.eval() before set fake quant to CALIBRATION mode."
            disable_fake_quant(mod)
            if isinstance(
                mod, FakeQuantizeBase
            ) or _is_fake_quant_script_module(mod):
                mod.train()
        elif mode == FakeQuantState.VALIDATION:
            assert (
                not mod.training
            ), "Call model.eval() before set fake quant to VALIDATION mode."
            # observer won't work in eval mode
            enable_fake_quant(mod)
        elif mode == FakeQuantState._FLOAT:
            disable_fake_quant(mod)
            disable_observer(mod)


def update_scale_by_qtype(model: torch.nn.Module):
    for _, m in model.named_modules():
        if (
            isinstance(m, FakeQuantize)
            and m._observer_enabled
            and not isinstance(
                m.activation_post_process,
                (FixedScaleObserver, FixedScaleObserverv2),
            )
        ):
            m.reset_dtype()
