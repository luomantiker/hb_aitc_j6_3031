from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.march import is_nash_series
from horizon_plugin_pytorch.nn import grid_sample as float_grid_sample
from horizon_plugin_pytorch.qtensor import QTensor
from .. import functional as hF  # noqa: N812
from .functional import grid_sample, scale_quanti


@QTensor.register_func_impl(F.grid_sample)
def _qtensor_grid_sample(
    input,
    grid,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
):

    if is_nash_series():
        grid_type = qint16
        input_h, input_w = input.shape[-2:]

        if align_corners:
            grid_scale_x = (
                1 if input_w == 1 else (2 / (input_w - 1) + 1)
            ) / grid_type.max
            grid_scale_y = (
                1 if input_h == 1 else (2 / (input_h - 1) + 1)
            ) / grid_type.max
        else:
            grid_scale_x = (2 / input_w + 1) / grid_type.max
            grid_scale_y = (2 / input_h + 1) / grid_type.max
        grid_scale = torch.tensor([grid_scale_x, grid_scale_y]).to(
            input.device
        )

        grid = scale_quanti(
            grid.as_subclass(Tensor),
            grid_scale,
            torch.zeros_like(grid_scale, dtype=torch.long),
            3,
            grid_type.min,
            grid_type.max,
            True,
            False,
        )
        ret = F.grid_sample(
            input.as_subclass(Tensor),
            grid,
            mode,
            padding_mode,
            align_corners,
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

        return QTensor(
            ret, input.q_scale(), input.dtype, input.q_per_channel_axis()
        )

    assert isinstance(grid, QTensor)
    if input.is_quantized:
        from horizon_plugin_pytorch.nn.quantized.functional import (
            grid_sample_norm_grid as quantized_grid_sample,
        )

        ret = quantized_grid_sample(
            input.as_subclass(Tensor),
            grid.as_subclass(Tensor),
            mode,
            padding_mode,
            align_corners,
            grid.q_scale(),
            grid.q_zero_point(),
            grid.dtype,
        )
    else:
        from horizon_plugin_pytorch.nn.qat.functional import (
            grid_sample_norm_grid as qat_grid_sample,
        )

        if torch.onnx.is_in_onnx_export():
            ret = F.grid_sample(
                input.as_subclass(Tensor),
                grid.as_subclass(Tensor),
                mode,
                padding_mode,
                align_corners,
            )
        else:
            ret = qat_grid_sample(
                input.as_subclass(Tensor),
                grid.as_subclass(Tensor),
                mode,
                padding_mode,
                align_corners,
                input.q_scale(),
                input.q_zero_point(),
                input.dtype,
            )

    return QTensor(
        ret, input.q_scale(), input.dtype, input.q_per_channel_axis()
    )


QTensor.patch_torch_func(
    torch.nn.functional.grid_sample,
    float_grid_sample.autocasted_grid_sample_outer,
)


@QTensor.register_func_impl(hF.warp)
def warp(x: QTensor, grid: QTensor, mode="bilinear", padding_mode="zeros"):
    if not x.is_quantized:
        if is_nash_series():
            ret = hF.warp(
                x.as_subclass(Tensor),
                grid.as_subclass(Tensor),
                mode,
                padding_mode,
            )
        else:
            ret = grid_sample(
                x.as_subclass(torch.Tensor),
                grid.as_subclass(torch.Tensor),
                mode,
                padding_mode,
                True,
                x.q_scale(),
                x.q_zero_point(),
                x.dtype,
            )
        ret = scale_quanti(
            ret,
            x.q_scale(),
            x.q_zero_point(),
            x.q_per_channel_axis(),
            x.dtype.min,
            x.dtype.max,
            True,
            False,
        )
        return QTensor(ret, x.q_scale(), x.dtype, x.q_per_channel_axis())
    else:
        from .functional import grid_sample as quantized_grid_sample

        r = quantized_grid_sample(
            x.int_repr(),
            grid.int_repr(),
            mode,
            padding_mode,
            True,
            grid.q_scale(),
            grid.q_zero_point(),
            grid.dtype,
        )
        return QTensor(r, x.q_scale(), x.dtype)
