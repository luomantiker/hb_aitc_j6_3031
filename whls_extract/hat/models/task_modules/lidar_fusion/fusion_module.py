import torch
import torch.nn as nn

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["TinySEBlock", "BevFuseModule"]


class TinySEBlock(nn.Module):
    """Block similar to SEBlock but only with one layer of conv2d.

    Args:
        in_channels:  The number of input channels.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        return x * self.att(x)


@OBJECT_REGISTRY.register
class BevFuseModule(nn.Module):
    """BevFuseModule fuses features using convolutions and SE block.

    Args:
        input_c: The number of input channels.
        fuse_c: The number of channels after fusion.
    """

    def __init__(self, input_c: int, fuse_c: int):
        super().__init__()
        self.reduce_conv = ConvModule2d(
            input_c,
            fuse_c,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=nn.BatchNorm2d(fuse_c, eps=1e-3, momentum=0.01),
            act_layer=nn.ReLU(inplace=True),
        )
        self.conv2 = ConvModule2d(
            fuse_c,
            fuse_c,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_layer=nn.BatchNorm2d(fuse_c, eps=1e-3, momentum=0.01),
            act_layer=nn.ReLU(inplace=True),
        )
        self.seblock = TinySEBlock(fuse_c)

    def forward(self, x: torch.Tensor):
        x = self.reduce_conv(x)
        x = self.conv2(x)
        pts_feats = self.seblock(x)
        return pts_feats
