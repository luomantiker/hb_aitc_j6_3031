from typing import Sequence, Union

import torch
from torch.nn.parameter import Parameter
from torch.quantization.fake_quantize import _is_per_channel
from torch.quantization.observer import _with_args

from horizon_plugin_pytorch.dtype import qinfo
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .fake_quantize import _is_affine
from .misc import set_qparam
from .observer import FixedScaleObserver, MovingAverageMinMaxObserver


class PACTFakeQuantize(torch.quantization.FakeQuantizeBase):
    r"""Refine this docstring in the future.

    This is an extension of the FakeQuantize module in fake_quantize.py
    which support learning of the alpha,which is an activation
    clipping parameter to  find the right quantization scale.
    When using symmetric quantization ,scale can be calculated by

    scale = alpha / (float(quant_max - quant_min) / 2)

    Args:
        observer: Module for observing statistics on input tensors and
            calculating scale and zero-point.
        alpha: An activation clipping parameter
        channel_len: Size of data at channel dim,default is 1
        observer_kwargs: Arguments for the observer module
    """

    @typechecked
    def __init__(
        self,
        observer: type(torch.nn.Module),
        alpha: float = 6.0,
        channel_len: int = 1,
        **observer_kwargs,
    ):
        super(PACTFakeQuantize, self).__init__()
        self._fake_quant_enabled = True
        self._observer_enabled = True
        self.activation_post_process = observer(**observer_kwargs)
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        if observer == FixedScaleObserver:
            fixed_scale, fixed_zero_point = self.calculate_qparams()
            self.register_buffer("scale", fixed_scale)
            self.register_buffer("zero_point", fixed_zero_point)
        else:
            self.register_buffer(
                "scale", torch.tensor([1.0], dtype=torch.float)
            )
            self.register_buffer(
                "zero_point", torch.tensor([0], dtype=torch.long)
            )
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.is_per_channel = _is_per_channel(self.qscheme)
        self.is_affine = _is_affine(self.qscheme)
        self.ch_axis = (
            self.activation_post_process.ch_axis if self.is_per_channel else -1
        )

        self.is_symmetric_quant = True
        self.alpha = Parameter(torch.tensor([alpha]))
        if self.qscheme not in (
            torch.per_tensor_symmetric,
            torch.per_channel_symmetric,
        ):
            self.is_symmetric_quant = False
            self.n_alpha = Parameter(torch.tensor([-alpha]))

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

        v = state_dict.get(prefix + "fake_quant_enabled", None)
        if v is not None:
            self._fake_quant_enabled = v[0].item() == 1

        super(PACTFakeQuantize, self)._load_from_state_dict(
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
        """Set qparams."""
        set_qparam(scale, self.scale, "scale")
        if zero_point is not None:
            set_qparam(zero_point, self.zero_point, "zero_point")
        else:
            self.zero_point.copy_(torch.zeros_like(self.zero_point))

    def forward(self, x):
        if self._observer_enabled and self.training:
            self.activation_post_process(x.detach())
            x = torch.where(x > self.alpha, self.alpha, x)
            self.activation_post_process.max_val.data.fill_(self.alpha.data[0])
            if x.min() < 0:
                if self.is_symmetric_quant:
                    x = torch.where(x < -self.alpha, -self.alpha, x)
                    self.activation_post_process.min_val.data.fill_(
                        -self.alpha[0].data
                    )
                else:
                    x = torch.where(x < self.n_alpha, self.n_alpha, x)
                    self.activation_post_process.min_val.data.fill_(
                        self.n_alpha[0].data
                    )
            else:
                self.activation_post_process.min_val.data.fill_(0.0)

            (
                _scale,
                _zero_point,
            ) = self.activation_post_process.calculate_qparams()
            assert self.scale.shape == _scale.shape, (
                "mismatched shape when update scale {} vs {}".format(
                    self.scale.shape, _scale.shape
                )
                + ". Please set or check channel_len param in qconfig"
            )
            self.scale.copy_(_scale)
            if _zero_point is not None:
                assert self.zero_point.shape == _zero_point.shape, (
                    "mismatched shape when update zero_point {} vs {}".format(
                        self.zero_point.shape, _zero_point.shape
                    )
                    + ". Please set or check channel_len param in qconfig"
                )
                self.zero_point.copy_(_zero_point)

        if self._fake_quant_enabled:
            x = torch.fake_quantize_per_tensor_affine(
                x,
                self.scale.item(),
                self.zero_point.item(),
                self.quant_min,
                self.quant_max,
            )

        # return qtensor type
        return QTensor(
            data=x,
            scale=self.scale,
            dtype=self.dtype,
        )

    with_args = classmethod(_with_args)


default_8bit_pact_quant = PACTFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=qinfo("qint8").min,
    quant_max=qinfo("qint8").max,
    dtype="qint8",
    qscheme=torch.per_tensor_symmetric,
)

default_4bit_pact_quant = PACTFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=qinfo("qint4").min,
    quant_max=qinfo("qint4").max,
    dtype="qint4",
    qscheme=torch.per_tensor_symmetric,
)

default_uint4_pact_quant = PACTFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=qinfo("quint4").min,
    quant_max=qinfo("quint4").max,
    dtype="quint4",
    qscheme=torch.per_tensor_symmetric,
)

default_16bit_pact_quant = PACTFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=qinfo("qint16").min,
    quant_max=qinfo("qint16").max,
    dtype="qint16",
    qscheme=torch.per_tensor_symmetric,
)
