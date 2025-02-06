import torch
from torch.nn.parameter import Parameter

from horizon_plugin_pytorch.fx import fx_helper
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.misc import copy_module_attrs
from .fake_quantize import FakeQuantize
from .observer_v2 import MinMaxObserver


@fx_helper.wrap()
class AdaRoundFakeQuantize(FakeQuantize):
    """Adaround FakerQuantize based on the FakeQuantize.

    Serve as weight quantizer, only works for Conv and Linear.

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
        orig_fake_quant: The original FakeQuantize instance, all of its
            attributes will be copied.
        weight_tensor: The weight tensor of Conv/Linear to initialize the
        adaround parameters.
        observer: Module for observing statistics on input
            tensors and calculating scale and zero-point.
        saturate: Whether zero out the grad for value out of quanti range.
        in_place: Whether use in place fake quantize.
        compat_mask: Whether pack the bool mask into bitfield
            when saturate = True.
        channel_len: Size of data at channel dim.
        observer_kwargs: Arguments for the observer module

    Attributes:
        observer: User provided module that collects statistics on the input
            tensor and provides a method to calculate scale and zero-point.
    """

    def __init__(
        self,
        orig_fake_quant: FakeQuantize,
        weight_tensor: torch.Tensor,
        observer: type(torch.nn.Module) = MinMaxObserver,
        saturate: bool = None,
        in_place: bool = False,
        compat_mask: bool = True,
        channel_len: int = 1,
        **observer_kwargs,
    ):
        super().__init__(
            observer,
            saturate,
            in_place,
            compat_mask,
            channel_len,
            **observer_kwargs,
        )
        copy_module_attrs(orig_fake_quant, self)
        self.gamma, self.zeta = -0.1, 1.1
        weight_tensor = weight_tensor.detach()
        if self.ch_axis != -1:
            self.new_shape = [1] * len(weight_tensor.shape)
            self.new_shape[self.ch_axis] = -1
            scale = self.scale.reshape(self.new_shape)
        else:
            scale = self.scale
        weight_tensor_floor = torch.floor(weight_tensor / scale)
        rest = (
            weight_tensor / scale
        ) - weight_tensor_floor  # rest of rounding [0, 1)
        alpha = -torch.log(
            (self.zeta - self.gamma) / (rest - self.gamma) - 1
        )  # => sigmoid(alpha) = rest
        self.alpha = Parameter(alpha)

    def rectified_sigmoid(self) -> torch.Tensor:
        """Generate rounding mask."""
        return (
            (self.zeta - self.gamma) * torch.sigmoid(self.alpha) + self.gamma
        ).clamp(0, 1)

    def adaround_forward(self, x, hard_value=False):
        if self.ch_axis != -1:
            scale = self.scale.reshape(self.new_shape)
            zero_point = self.zero_point.reshape(self.new_shape)
        else:
            scale = self.scale
            zero_point = self.zero_point

        x = torch.floor(x / scale)
        if hard_value:
            x += (self.alpha >= 0).float()
        else:
            x += self.rectified_sigmoid()
        x += zero_point
        x = torch.clamp(x, self.quant_min, self.quant_max)
        x = (x - zero_point) * scale
        return x

    def get_hard_value(self, x):
        return self.adaround_forward(x, hard_value=True)

    def forward(self, x):
        if self._fake_quant_enabled:
            x = self.adaround_forward(x.reshape_as(self.alpha))

        return QTensor(
            data=x,
            scale=self.scale,
            dtype=self.dtype,
            per_channel_axis=self.ch_axis,
        )
