# Quant interpolate is supported by QTensor (F.interpolate), do patch on torch
# modules to handle the loading of old state_dict.

from torch import Tensor, nn
from torch.jit.annotations import BroadcastingList2, Optional
from torch.nn import functional as F  # noqa: N812
from torch.nn.modules.utils import _pair

from horizon_plugin_pytorch.march import is_nash_series
from horizon_plugin_pytorch.nn import interpolate as float_interpolate
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.load_state_dict_helper import (
    load_state_dict_ignore_act,
)
from ..interpolate import Interpolate

_fake_quant_step = False


def use_step_fake_quantized_impl(v: bool):
    """Whether use the old fasion qat interpolate impl.

    The old fasion impl is inconsistent with torch impl, so you should
    only use it if you have a trained qat model produced by
    horizon_plugin_pytorch<=2.4.8 and want to preserve its precision.
    """
    assert isinstance(v, bool)
    global _fake_quant_step
    _fake_quant_step = v


def _patch_torch_modules():
    """Patch some Qtensor ops to be compatible with old ckpt."""
    if Interpolate._load_from_state_dict is load_state_dict_ignore_act:
        return
    Interpolate._load_from_state_dict = load_state_dict_ignore_act
    nn.Upsample._load_from_state_dict = load_state_dict_ignore_act
    nn.UpsamplingNearest2d._load_from_state_dict = load_state_dict_ignore_act
    nn.UpsamplingBilinear2d._load_from_state_dict = load_state_dict_ignore_act


_patch_torch_modules()


@QTensor.register_func_impl(F.interpolate)
def _qtensor_interpolate(
    input,
    size: Optional[BroadcastingList2[int]] = None,
    scale_factor: Optional[BroadcastingList2[float]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: Optional[bool] = None,
):
    from horizon_plugin_pytorch.nn.qat.functional import (
        interpolate as qat_interploate,
    )
    from horizon_plugin_pytorch.nn.qat.functional import scale_quanti
    from horizon_plugin_pytorch.nn.quantized.functional import (
        interpolate as quantized_interploate,
    )

    if input.dim() == 4:
        size = _pair(size) if size else None
        scale_factor = _pair(scale_factor) if scale_factor else None

    if not input.is_quantized:
        if is_nash_series() and not _fake_quant_step:
            ret = F.interpolate(
                input.as_subclass(Tensor),
                size,
                scale_factor,
                mode,
                align_corners,
                recompute_scale_factor,
                antialias=antialias,
            )
        else:
            ret = qat_interploate(
                input.as_subclass(Tensor),
                size,
                scale_factor,
                mode,
                align_corners,
                recompute_scale_factor,
                input.q_scale(),
                input.q_zero_point(),
                input.dtype,
                False,
                antialias=antialias,
            )

        ret = scale_quanti(
            ret,
            input.q_scale(),
            input.q_zero_point(),
            input.q_per_channel_axis(),
            input.dtype.min,
            input.dtype.max,
            True,
            False,
        )
    else:
        ret = quantized_interploate(
            input.as_subclass(Tensor),
            size,
            scale_factor,
            mode,
            align_corners,
            recompute_scale_factor,
            input.q_scale(),
            input.q_zero_point(),
            input.dtype,
        )

    return QTensor(
        ret, input.q_scale(), input.dtype, input.q_per_channel_axis()
    )


QTensor.patch_torch_func(
    F.interpolate, float_interpolate.autocasted_interpolate_outer
)
