import copy
from typing import List, Optional

import torch
import torch.nn as nn

from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["ChannelMapperNeck"]


@OBJECT_REGISTRY.register
class ChannelMapperNeck(nn.Module):
    """Map the backbone features to the same channels feature with convs.

    Args:
        in_channels: Each specifying the number of input channels
            for a layer.
        out_indices: The indices at which output channels are to be mapped.
        out_channel: An integer specifying the number of output channels
            after mapping.
        extra_convs: The number of extra convolutional layers to be added,
            extra convs is default configured with stride=2, kernel=3.
        kernel_size: The size of the conv kernel.
        stride: The stride for convolution.
        bias: If bias be used for convolution.
        groups: The number of groups for convolution.
        dilation: The dilation for convolution.
        norm_layer: The normalization layer for convolution.
        activation: The activation function for convolution.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_indices: List[int],
        out_channel: int,
        extra_convs: int = 0,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None,
        activation: Optional[nn.Module] = None,
    ):
        super(ChannelMapperNeck, self).__init__()
        self.out_indices = out_indices

        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                ConvModule2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=bias,
                    groups=groups,
                    dilation=dilation,
                    norm_layer=copy.deepcopy(norm_layer),
                    act_layer=copy.deepcopy(activation),
                )
            )

        self.extra_convs = nn.ModuleList()
        in_channel = in_channels[-1]
        for _ in range(extra_convs):
            self.extra_convs.append(
                ConvModule2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=bias,
                    groups=groups,
                    dilation=dilation,
                    norm_layer=copy.deepcopy(norm_layer),
                    act_layer=copy.deepcopy(activation),
                )
            )
            in_channel = out_channel
        self.init_weights()

    def init_weights(self):
        for _, neck_layer in self.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

    def forward(self, inputs: List[torch.Tensor]):
        """Forward function for ChannelMapperNeck.

        Args:
            inputs: The backbone feature list.
        """
        outs = []
        for i, indix in enumerate(self.out_indices):
            outs.append(self.convs[i](inputs[indix]))

        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[-1]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return outs
