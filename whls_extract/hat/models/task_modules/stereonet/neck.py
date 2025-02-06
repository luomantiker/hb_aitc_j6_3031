# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List

import horizon_plugin_pytorch as hpp
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor

from hat.models.base_modules.basic_resnet_module import BasicResBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.weight_init import kaiming_init
from hat.registry import OBJECT_REGISTRY

__all__ = ["StereoNetNeck"]


@OBJECT_REGISTRY.register
class StereoNetNeck(nn.Module):
    """
    A extra features module of stereonet.

    Args:
        out_channels: Channels for each block.
        use_bn: Whether to use BN in module.
        bn_kwargs: Dict for BN layer.
        bias: Whether to use bias in module.
        act_type: Activation layer.
    """

    def __init__(
        self,
        out_channels: List,
        use_bn: bool = True,
        bn_kwargs: Dict = None,
        bias: bool = False,
        act_type: nn.Module = None,
    ):
        super(StereoNetNeck, self).__init__()

        self.bias = bias
        self.bn_kwargs = bn_kwargs
        self.quant = QuantStub(scale=1.0 / 128.0)
        self.use_bn = use_bn
        self.firstconv = self._make_firstconv(out_channels[0], act_type)
        self.layer1 = self._make_layer(
            BasicResBlock, out_channels[0], out_channels[1], 3, 2
        )
        self.layer2 = self._make_layer(
            BasicResBlock, out_channels[1], out_channels[2], 16, 2
        )
        self.layer3 = self._make_layer(
            BasicResBlock, out_channels[2], out_channels[3], 3, 2
        )
        self.layer4 = self._make_layer(
            BasicResBlock, out_channels[3], out_channels[4], 3, 1
        )
        self.lastconv = self._make_lastconv(
            (out_channels[3] + out_channels[4]), out_channels[5], act_type
        )
        self.l_cat = hpp.nn.quantized.FloatFunctional()
        self.r_cat = hpp.nn.quantized.FloatFunctional()
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of stereonet module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(
                    m,
                    mode="fan_in",
                    nonlinearity="relu",
                    bias=0,
                    distribution="normal",
                )

    def _make_firstconv(
        self, out_channels: int, act_type: nn.Module
    ) -> nn.Sequential:
        """Make the firstconv module.

        Args:
            out_channels: The output channels of firstconv.
            act_type: The activation module.

        """

        layers = []
        layers.append(
            ConvModule2d(
                in_channels=3,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=self.bias,
                norm_layer=nn.BatchNorm2d(out_channels, **self.bn_kwargs)
                if self.use_bn
                else None,
                act_layer=act_type,
            ),
        )
        for _ in range(2):
            layers.append(
                ConvModule2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=self.bias,
                    norm_layer=nn.BatchNorm2d(out_channels, **self.bn_kwargs)
                    if self.use_bn
                    else None,
                    act_layer=act_type,
                ),
            )
        return nn.Sequential(*layers)

    def _make_lastconv(
        self,
        in_channels: int,
        out_channels: int,
        act_type: nn.Module,
    ) -> nn.Sequential:
        """Make the lastconv module.

        Args:
            in_channels: The input channels of lastconv.
            out_channels: The output channels of lastconv.
            act_type: The activation module.

        """
        layers = []
        layers.append(
            ConvModule2d(
                in_channels=in_channels,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.bias,
                norm_layer=nn.BatchNorm2d(128, **self.bn_kwargs)
                if self.use_bn
                else None,
                act_layer=act_type,
            )
        )
        layers.append(
            ConvModule2d(
                in_channels=128,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=self.bias,
                norm_layer=None,
                act_layer=None,
            )
        )

        return nn.Sequential(*layers)

    def _make_layer(
        self,
        block: nn.Module,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Make hidden layer.

        Args:
            block: The base module of block.
            in_channels: The input channels of blocks.
            out_channels: The output channels of blocks.
            blocks: Num of blocks.
            act_type: The activation module.
            stride: The stride of blocks.

        """
        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bias=self.bias,
                bn_kwargs=self.bn_kwargs,
            ),
        )
        for _ in range(1, blocks):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    stride=1,
                    bias=self.bias,
                    bn_kwargs=self.bn_kwargs,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, imgs: Tensor) -> List[Tensor]:
        """Perform the forward pass of the model.

        Args:
            imgs: The inputs images.

        Returns:
            gwc_feature_left: The gwc features of left image.
            gwc_feature_right: The gwc features of right image.
            concat_feature_left: The concat features of left image.
            concat_feature_right: The concat features of right image.
            imgl: The left image.
        """

        imgs = self.quant(imgs)
        imgl = imgs[:, :3, ...]
        imgr = imgs[:, 3:, ...]

        xl = self.firstconv(imgl)
        xl = self.layer1(xl)
        l2 = self.layer2(xl)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        xr = self.firstconv(imgr)
        xr = self.layer1(xr)
        r2 = self.layer2(xr)
        r3 = self.layer3(r2)
        r4 = self.layer4(r3)

        gwc_feature_left = self.l_cat.cat((l3, l4), dim=1)

        gwc_feature_right = self.r_cat.cat((r3, r4), dim=1)

        concat_feature_left = self.lastconv(gwc_feature_left)
        concat_feature_right = self.lastconv(gwc_feature_right)

        return (
            gwc_feature_left,
            gwc_feature_right,
            concat_feature_left,
            concat_feature_right,
            imgl,
        )

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""

        modules = [
            self.firstconv,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.lastconv,
        ]
        for m in modules:
            for m_ in m:
                if hasattr(m_, "fuse_model"):
                    m_.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
