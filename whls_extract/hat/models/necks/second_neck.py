from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.nn.quantized import FloatFunctional

from hat.models.base_modules.conv_module import (
    ConvModule2d,
    ConvTransposeModule2d,
)
from hat.registry import OBJECT_REGISTRY

__all__ = ["SECONDNeck"]


@OBJECT_REGISTRY.register
class SECONDNeck(nn.Module):
    """Second FPN modules.

    Implements the network structure of PointPillars:
    <https://arxiv.org/abs/1812.05784>

    Although the structure is called backbone in the original paper, we
    follow the publicly available code structure and use it as a neck
    module.

    Adapted from GitHub second.pytorch:
    <https://github.com/traveller59/second.pytorch>

    Args:
        in_feature_channel: number of input feature channels.
        down_layer_nums: number of layers for each down-sample stage.
        down_layer_strides: stride for each down-sampling stage.
        down_layer_channels: number of filters for each down-sample stage.
        up_layer_strides: stride for each up-sample stage.
        up_layer_channels: number of filters for each up-sampling stage.
        bn_kwargs: batch norm kwargs.
        use_relu6: whether to use relu6.
        quantize: whether to quantize the module.
        quant_scale: init scale for Quantstub.
    """

    def __init__(
        self,
        in_feature_channel: int,
        down_layer_nums: List[int],
        down_layer_strides: List[int],
        down_layer_channels: List[int],
        up_layer_strides: List[int],
        up_layer_channels: List[int],
        bn_kwargs: Optional[Dict] = None,
        use_relu6: bool = False,
        quantize: bool = False,
        quant_scale: float = 1 / 128.0,
    ):
        super(SECONDNeck, self).__init__()

        if bn_kwargs is None:
            bn_kwargs = {"eps": 1e-3, "momentum": 0.01}
        self.bn_kwargs = bn_kwargs
        self.use_relu6 = use_relu6

        if quantize:
            self.quant = QuantStub(scale=quant_scale)

        down_blocks, deblocks = [], []
        in_channels = [in_feature_channel, *down_layer_channels[:-1]]

        for i, layer_num in enumerate(down_layer_nums):

            mid_channels = down_layer_channels[i]

            block = self._make_down_block(
                in_channels=in_channels[i],
                out_channels=mid_channels,
                kernel_size=3,
                stride=down_layer_strides[i],
                num_blocks=layer_num,
            )
            down_blocks.append(block)

            deblock = self._make_up_block(
                in_channels=mid_channels,
                out_channels=up_layer_channels[i],
                stride=up_layer_strides[i],
            )
            deblocks.append(deblock)

        self.blocks = nn.ModuleList(down_blocks)
        self.deblocks = nn.ModuleList(deblocks)

        self.cat = FloatFunctional()

    def forward(self, x: torch.Tensor, quant: bool = False):
        if quant:
            x = self.quant(x)
        # down sample
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(x)

        # up sample
        for i in range(len(ups)):
            ups[i] = self.deblocks[i](ups[i])

        out = self.cat.cat(ups, dim=1)
        return out

    def _make_down_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        num_blocks: int,
        **kwargs,
    ):

        block = nn.Sequential(
            nn.Identity(),
            ConvModule2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                bias=False,
                padding=1,
                norm_layer=nn.BatchNorm2d(out_channels, **self.bn_kwargs),
                act_layer=nn.ReLU6(inplace=True)
                if self.use_relu6
                else nn.ReLU(inplace=True),
            ),
        )

        for idx in range(num_blocks):
            block.add_module(
                str(idx + 2),
                ConvModule2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    norm_layer=nn.BatchNorm2d(out_channels, **self.bn_kwargs),
                    act_layer=nn.ReLU6(inplace=True)
                    if self.use_relu6
                    else nn.ReLU(inplace=True),
                ),
            )

        return block

    def _make_up_block(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        **kwargs,
    ):

        if stride > 1:
            deblock = ConvTransposeModule2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=stride,
                stride=stride,
                bias=False,
                norm_layer=nn.BatchNorm2d(out_channels, **self.bn_kwargs),
                act_layer=nn.ReLU6(inplace=True)
                if self.use_relu6
                else nn.ReLU(inplace=True),
            )
        else:
            stride = np.round(1 / stride).astype(np.int64)
            deblock = ConvModule2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=stride,
                stride=stride,
                bias=False,
                norm_layer=nn.BatchNorm2d(out_channels, **self.bn_kwargs),
                act_layer=nn.ReLU6(inplace=True)
                if self.use_relu6
                else nn.ReLU(inplace=True),
            )
        return deblock

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, (ConvModule2d, ConvTransposeModule2d)):
                m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
