# Copyright (c) Horizon Robotics. All rights reserved.

from torch import nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.weight_init import kaiming_init, normal_init
from hat.registry import OBJECT_REGISTRY

__all__ = ["GaNetHead"]


@OBJECT_REGISTRY.register
class GaNetHead(nn.Module):
    """
    A basic head module of ganet.

    Args:
        in_channel: Number of channel in the input feature map.
    """

    def __init__(self, in_channel: int):
        super(GaNetHead, self).__init__()

        self.keypts_head = self._make_head_module(
            in_channel=in_channel,
            out_channel=1,
            final_kernel_size=1,
        )
        self.offset_head = self._make_head_module(
            in_channel=in_channel,
            out_channel=2,
            final_kernel_size=1,
        )
        self.reg_head = self._make_head_module(
            in_channel=in_channel,
            out_channel=2,
            final_kernel_size=1,
        )
        self.dequant = DeQuantStub()

        self._init_weights()

    def _init_head_layer_weights(
        self, head_layer, bias_1=0.0, bias_2=0.0, std=0.01
    ):
        """Initialize the weights of headlayer."""

        for i, module in enumerate(head_layer):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    if i == 0:
                        kaiming_init(
                            m,
                            mode="fan_out",
                            nonlinearity="relu",
                            bias=bias_1,
                            distribution="normal",
                        )
                    else:
                        normal_init(m, std=std, bias=bias_2)

    def _init_weights(self):
        """Initialize the weights of GaNetHead module."""

        self._init_head_layer_weights(self.keypts_head, 0.0, -2.19)
        self._init_head_layer_weights(self.offset_head)
        self._init_head_layer_weights(self.reg_head)

    def _make_head_module(self, in_channel, out_channel, final_kernel_size):

        layers = nn.Sequential(
            ConvModule2d(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                norm_layer=None,
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=final_kernel_size,
                stride=1,
                padding=final_kernel_size // 2,
                bias=True,
                norm_layer=None,
                act_layer=None,
            ),
        )
        return layers

    def forward(self, x):

        kpts_hm = self.keypts_head(x)
        pts_offset = self.offset_head(x)
        int_offset = self.reg_head(x)

        kpts_hm = self.dequant(kpts_hm)
        pts_offset = self.dequant(pts_offset)
        int_offset = self.dequant(int_offset)

        return kpts_hm, pts_offset, int_offset

    def fuse_model(self):
        for module in [self.keypts_head, self.offset_head, self.reg_head]:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.keypts_head[
            -1
        ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
        self.offset_head[
            -1
        ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
        self.reg_head[
            -1
        ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
