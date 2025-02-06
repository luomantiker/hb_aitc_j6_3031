import torch.nn as nn

from .conv_module import ConvModule2d
from .separable_conv_module import SeparableGroupConvModule2d


class VargConvNetBlock(nn.Module):
    """
    A basic block for vargconvnet.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Stride of basic block.
        bn_kwargs: Dict for BN layer.
        kernel_size: Kernel size of basic block.
        factor: Factor for channels expansion.
        padding: Padding of basic block.
        bias: Whether to use bias in module.
        groups: Same as nn.Conv2d.
        dw_with_relu: Whether to use relu in dw conv.
        pw_with_relu: Whether to use relu in pw conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        bn_kwargs: dict,
        kernel_size: int = 3,
        factor: float = 1.0,
        padding: int = 1,
        bias: bool = True,
        groups: int = 8,
        dw_with_relu: bool = True,
        pw_with_relu: bool = True,
    ):
        super(VargConvNetBlock, self).__init__()
        self.factor = factor
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(2),
                ConvModule2d(
                    in_channels,
                    out_channels,
                    1,
                    bias=bias,
                    padding=0,
                    stride=1,
                    norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                ),
            )
        else:
            self.downsample = None

        self.body = nn.Sequential(
            SeparableGroupConvModule2d(
                in_channels,
                out_channels,
                kernel_size,
                bias=bias,
                padding=padding,
                factor=factor,
                groups=groups // 2 if stride == 2 and groups > 1 else groups,
                stride=stride,
                dw_norm_layer=nn.BatchNorm2d(
                    int(in_channels * factor), **bn_kwargs
                ),
                dw_act_layer=nn.ReLU(inplace=True) if dw_with_relu else None,
                pw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            ),
            SeparableGroupConvModule2d(
                out_channels,
                out_channels,
                kernel_size=3,
                bias=bias,
                padding=1,
                factor=factor,
                groups=groups,
                stride=1,
                dw_norm_layer=nn.BatchNorm2d(
                    int(out_channels * factor), **bn_kwargs
                ),
                dw_act_layer=nn.ReLU(inplace=True) if dw_with_relu else None,
                pw_norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            ),
        )

        self.shortcut_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        out = self.body(x)
        out = self.shortcut_add.add(out, shortcut)
        return out
