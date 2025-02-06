import logging
import math
from collections.abc import Sequence
from typing import Optional

import horizon_plugin_pytorch.nn as hnn
import torch.nn as nn
from horizon_plugin_pytorch.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    QuantStub,
    qinfo,
)
from torch.quantization import QConfig

from hat.registry import OBJECT_REGISTRY

__all__ = ["WarpingModule"]

logger = logging.getLogger(__file__)


@OBJECT_REGISTRY.register
class WarpingModule(nn.Module):
    """Module of warping the input img.

    Args:
        max_side_length: max side length of input and output.
        uv_map_desc: description of uv_map.
        mode (str, optional): Interpolation mode to calculate output values.
            Only "bilinear" and "nearest" supported now.
            Defaults to "bilinear".
        padding_mode (str, optional): Padding mode for outside grid values.
            Only "zeros" and "border" is supported now.
            Defaults to "border".
    """

    def __init__(
        self,
        max_side_length: Optional[int] = None,
        uv_map_desc: Optional[nn.Module] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ):
        super(WarpingModule, self).__init__()
        self.grid_sample = hnn.GridSample(mode=mode, padding_mode=padding_mode)

        if max_side_length is not None:
            coord_bit_num = math.ceil(math.log(max_side_length + 1, 2))
            coord_shift = 15 - coord_bit_num
            coord_shift = max(min(coord_shift, 8), 0)
            grid_quant_scale = 1.0 / (1 << coord_shift)
        else:
            grid_quant_scale = None
        self.grid_quant_stub = QuantStub(scale=grid_quant_scale)
        self.warning = True
        self.uv_map_desc = uv_map_desc

    def forward(self, x, uv_map):
        if uv_map is None:
            if self.warning:
                logger.warning("Input uv_map is None, skip warping!!!")
                self.warning = False
            return x
        if isinstance(uv_map, Sequence):
            assert len(uv_map) == 1, "Not support Sequence > 1"
            uv_map = uv_map[0]
        if self.uv_map_desc is not None:
            uv_map = self.uv_map_desc(uv_map)
        uv_map = self.grid_quant_stub(uv_map)
        return self.grid_sample(x, uv_map)

    def set_qconfig(self):
        self.grid_quant_stub.qconfig = QConfig(
            activation=FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=qinfo("qint16").min,
                quant_max=qinfo("qint16").max,
                dtype="qint16",
                saturate=True,
            ),
            weight=None,
        )
