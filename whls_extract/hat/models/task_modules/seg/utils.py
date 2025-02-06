# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Tuple, Union

import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.models.weight_init import normal_init
from hat.registry import OBJECT_REGISTRY

__all__ = ["CoordConv"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class CoordConv(nn.Module):
    """Coordinate Conv more detail ref to https://arxiv.org/pdf/1807.03247.pdf.

    Args:
        ref to torch.nn.Conv2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super(CoordConv, self).__init__()
        self.mod = torch.nn.quantized.FloatFunctional()
        self.conv = nn.Conv2d(
            in_channels + 2,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.weight_init()
        self.dequant = DeQuantStub()
        self.quant = QuantStub(1.0 / 128)

    def forward(self, feats):
        for i in range(len(feats)):
            feat = feats[i]
            batch_size, w, h, dev = (
                feat.shape[0],
                feat.shape[-2],
                feat.shape[-1],
                feat.device,
            )
            w = torch.linspace(-1, 1, w, device=dev)
            h = torch.linspace(-1, 1, h, device=dev)
            y, x = torch.meshgrid(w, h)
            y = y.expand([batch_size, 1, -1, -1])
            x = x.expand([batch_size, 1, -1, -1])
            y = self.quant(y)
            x = self.quant(x)
            feat = self.mod.cat([feat, x, y], dim=1)
            feat = self.conv(feat)
            feats[i] = feat
        return feats

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
