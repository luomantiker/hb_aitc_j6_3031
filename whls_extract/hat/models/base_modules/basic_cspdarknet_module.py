from typing import Tuple

import torch
import torch.nn as nn

from .conv_module import ConvModule2d
from .separable_conv_module import SeparableConvModule2d

__all__ = [
    "Bottleneck",
    "CSPLayer",
    "Focus",
    "SPPBottleneck",
]


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class Bottleneck(nn.Module):
    """Standard bottleneck used in CSPLayer.

    Args:
        in_channels: The input channels of this Module.
        out_channels: The output channels of this Module.
        shortcut: Whether to add skip connection to the out. Only works
            when in_channels == out_channels.
        expansion: Expand ratio of the hidden channel.
        depthwise: Whether to use depthwise separable convolution in
            this Module.
        act: Activation layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
        act: str = "silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvModule2d(
            in_channels,
            hidden_channels,
            1,
            stride=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(hidden_channels),
            act_layer=get_activation(act, inplace=True),
        )
        if depthwise:
            self.conv2 = SeparableConvModule2d(
                hidden_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                bias=False,
                dw_norm_layer=nn.BatchNorm2d(out_channels),
                dw_act_layer=get_activation(act, inplace=True),
            )
        else:
            self.conv2 = ConvModule2d(
                hidden_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(out_channels),
                act_layer=get_activation(act, inplace=True),
            )
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

    def fuse_model(self):
        self.conv1.fuse_model()
        self.conv2.fuse_model()


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.

    Args:
        in_channels: The input channels of the CSP layer.
        out_channels: The output channels of the CSP layer.
        n: Number of blocks.
        shortcut: Whether to add skip connection in blocks.
        expansion: Ratio to adjust the number of channels of the hidden layer.
        depthwise: Whether to use depthwise separable convolution in blocks.
        act: Activation layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: float = False,
        act: str = "silu",
    ):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvModule2d(
            in_channels,
            hidden_channels,
            1,
            stride=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(hidden_channels),
            act_layer=get_activation(act, inplace=True),
        )
        self.conv2 = ConvModule2d(
            in_channels,
            hidden_channels,
            1,
            stride=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(hidden_channels),
            act_layer=get_activation(act, inplace=True),
        )
        self.conv3 = ConvModule2d(
            2 * hidden_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(out_channels),
            act_layer=get_activation(act, inplace=True),
        )
        module_list = [
            Bottleneck(
                hidden_channels,
                hidden_channels,
                shortcut,
                1.0,
                depthwise,
                act=act,
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

    def fuse_model(self):
        self.conv1.fuse_model()
        self.conv2.fuse_model()
        self.conv3.fuse_model()
        for module in self.m:
            module.fuse_model()


class Focus(nn.Module):
    """Focus width and height information into channel space.

    Args:
        in_channels: The input channels of this Module.
        out_channels: The output channels of this Module.
        ksize: The kernel size of the convolution.
        stride: The stride of the convolution.
        act: Activation layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int = 1,
        stride: int = 1,
        act: str = "silu",
    ):
        super().__init__()
        self.conv = ConvModule2d(
            in_channels * 4,
            out_channels,
            ksize,
            stride,
            padding=(ksize - 1) // 2,
            bias=False,
            norm_layer=nn.BatchNorm2d(out_channels),
            act_layer=get_activation(act, inplace=True),
        )

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

    def fuse_model(self):
        self.conv.fuse_model()


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Args:
        in_channels: The input channels of this Module.
        out_channels: The output channels of this Module.
        kernel_sizes: Sequential of kernel sizes of pooling layers.
        activation: Activation layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Tuple[int, ...] = (5, 9, 13),
        activation: str = "silu",
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvModule2d(
            in_channels,
            hidden_channels,
            1,
            stride=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(hidden_channels),
            act_layer=get_activation(activation, inplace=True),
        )
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule2d(
            conv2_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            norm_layer=nn.BatchNorm2d(out_channels),
            act_layer=get_activation(activation, inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

    def fuse_model(self):
        self.conv1.fuse_model()
        self.conv2.fuse_model()
