from typing import List

import torch.nn as nn

from hat.models.base_modules.conv_module import ConvTransposeModule2d
from hat.registry import OBJECT_REGISTRY

__all__ = ["DeconvDecoder"]


@OBJECT_REGISTRY.register
class DeconvDecoder(nn.Module):
    """Deconder Head consists of multi deconv layers.

    Args:
        input_index: The stage index of the pre backbone outputs.
        in_channels: Number of input channels of the feature output
                     from backbone.
        out_channels: Number of out channels of the DeconvDecoder.
        num_conv_layers: Number of convolutional layers for decoder.
        num_deconv_filters: List of the number of filters for deconv layers
        num_deconv_kernels: List of the kernel sizes for deconv layers.
        final_conv_kernel: Kernel size of the final convolutional layer.
    """

    def __init__(
        self,
        input_index,
        in_channels: int,
        out_channels: int,
        num_conv_layers,
        num_deconv_filters: List[int],
        num_deconv_kernels: List[int],
        final_conv_kernel: int,
    ):
        super(DeconvDecoder, self).__init__()
        self.BN_MOMENTUM = 0.1
        self.input_index = input_index
        self.in_channels = in_channels

        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(
            num_conv_layers, num_deconv_filters, num_deconv_kernels
        )
        self.final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=out_channels,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=1 if final_conv_kernel == 3 else 0,
        )

    def forward(self, x):
        x = x[self.input_index]
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(
            num_filters
        ), "ERROR: num_deconv_layers is different len(num_deconv_filters)"
        assert num_layers == len(
            num_kernels
        ), "ERROR: num_deconv_layers is different len(num_deconv_filters)"

        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            padding, output_padding = self._get_deconv_cfg(kernel, i)

            planes = num_filters[i]
            conv_layer = ConvTransposeModule2d(
                in_channels=self.in_channels,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias,
                norm_layer=nn.BatchNorm2d(planes, momentum=self.BN_MOMENTUM),
                act_layer=nn.ReLU(inplace=True),
            )
            layers.append(conv_layer)
            self.in_channels = planes

        return nn.Sequential(*layers)
