# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Dict

import horizon_plugin_pytorch.nn as hnn
from torch import Tensor, nn

from hat.models.base_modules.basic_vargnet_module import BasicVarGBlock
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "BevEncoder",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class BevEncoder(nn.Module):
    """The basic encoder structure of bev.

    Args:
        backbone: Backbone module.
        neck: Neck module.
    """

    def __init__(self, backbone: nn.Module, neck: nn.Module = None):
        super(BevEncoder, self).__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, feat: Tensor, meta: Dict) -> Tensor:
        """Perform the forward pass through the model's backbone and neck.

        Args:
            feat: The input feature.
            meta: The meta information.

        Returns:
            feat: The output feature after passing
                  through the backbone and neck.
        """

        feat = self.backbone(feat)
        if self.neck is not None:
            feat = self.neck(feat)
        return feat

    def fuse_model(self) -> None:
        """Perform model fusion on the `backbone` and `neck` modules."""

        modules = [self.backbone]
        if self.neck is not None:
            modules.append(self.neck)

        for module in modules:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration (qconfig)."""

        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        modules = [self.backbone]
        if self.neck is not None:
            modules.append(self.neck)

        for module in modules:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


class BevBackbone(nn.Module):
    """The basic Backbone of bev feat.

    Args:
        in_chanels: Num of channels for input.
        feat_chanels: Num of channels for  feat.
        out_chanels: Num of channels for output.
        bn_kwargs: Dict for BN layer.
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        out_channels: int,
        bn_kwargs: Dict = None,
    ):
        super(BevBackbone, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.bn_kwargs = bn_kwargs
        self.convs = self._make_layers()

    def forward(self, feat: Tensor) -> Tensor:
        """Perform the forward pass through the convolutional layers of the model.

        Args:
            feat: The input feature tensor.

        Returns:
            The output tensor after passing through the convolutional layers.
        """

        return self.convs(feat)

    def fuse_model(self) -> None:
        """Perform model fusion on the convolutional layers."""

        for mod in self.convs:
            if hasattr(mod, "fuse_model"):
                mod.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

    def _make_layers(self) -> nn.Module:
        """
        Create a sequence of convolutional layers.

        Returns:
            A sequential module containing the convolutional layers.
        """

        convs = []
        c_in = self.in_channels
        for c_f in self.feat_channels:
            convs.append(self._make_conv(c_in, c_f))
            convs.append(
                hnn.Interpolate(scale_factor=2, recompute_scale_factor=True)
            )
            c_in = c_f
        convs.append(self._make_conv(c_in, self.out_channels))
        return nn.Sequential(*convs)


@OBJECT_REGISTRY.register
class VargBevBackbone(BevBackbone):
    """The bev Backbone using varg block."""

    def __init__(self, **kwargs):
        super(VargBevBackbone, self).__init__(**kwargs)

    def _make_conv(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Create a single convolutional block (BasicVarGBlock).

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.

        Returns:
            An instance of the BasicVarGBlock convolutional block.
        """

        return BasicVarGBlock(
            in_channels=in_channels,
            mid_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            bias=True,
            bn_kwargs=self.bn_kwargs,
            merge_branch=False,
        )
