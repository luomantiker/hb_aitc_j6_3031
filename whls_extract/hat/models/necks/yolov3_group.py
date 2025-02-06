# Copyright (c) Horizon Robotics. All rights reserved.

import horizon_plugin_pytorch as horizon
import torch.nn as nn

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.separable_conv_module import (
    SeparableConvModule2d,
    SeparableGroupConvModule2d,
)
from hat.models.weight_init import xavier_init
from hat.registry import OBJECT_REGISTRY

__all__ = ["YoloGroupNeck"]


class InputModule(nn.Sequential):
    """
    Input module of yolov3 neck.

    Args:
        input_channels: Input channels of this module.
        output_channels: Output channels of this module.
        bn_kwargs: Dict for BN layers.
        bias: Whether to use bias in this module.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        bn_kwargs: dict,
        bias: bool = True,
        head_group: bool = True,
    ):
        conv_list = []
        for i in range(2):
            conv_list.append(
                ConvModule2d(
                    input_channels if i == 0 else output_channels * 2,
                    output_channels,
                    1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(output_channels, **bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                )
            )
            if head_group:
                conv_list.append(
                    SeparableGroupConvModule2d(
                        in_channels=output_channels,
                        out_channels=output_channels * 2,
                        kernel_size=5,
                        padding=2,
                        stride=1,
                        bias=bias,
                        dw_norm_layer=nn.BatchNorm2d(
                            output_channels, **bn_kwargs
                        ),
                        dw_act_layer=nn.ReLU(inplace=True),
                        pw_norm_layer=nn.BatchNorm2d(
                            output_channels * 2, **bn_kwargs
                        ),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                )
            else:
                conv_list.append(
                    SeparableConvModule2d(
                        in_channels=output_channels,
                        out_channels=output_channels * 2,
                        kernel_size=5,
                        padding=2,
                        stride=1,
                        bias=bias,
                        dw_norm_layer=nn.BatchNorm2d(
                            output_channels, **bn_kwargs
                        ),
                        dw_act_layer=nn.ReLU(inplace=True),
                        pw_norm_layer=nn.BatchNorm2d(
                            output_channels * 2, **bn_kwargs
                        ),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                )
        conv_list.append(
            ConvModule2d(
                output_channels * 2,
                output_channels,
                1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(output_channels, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            )
        )
        super(InputModule, self).__init__(*conv_list)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def fuse_model(self):
        for i in range(5):
            getattr(self, "%d" % (i)).fuse_model()


@OBJECT_REGISTRY.register
class YoloGroupNeck(nn.Module):
    """
    Necks module of yolov3.

    Args:
        backbone_idx: Index of backbone output for necks.
        in_channels_list: List of input channels.
        out_channels_list: List of output channels.
        bn_kwargs: Config dict for BN layer.
        bias: Whether to use bias in module.
    """

    def __init__(
        self,
        backbone_idx: list,
        in_channels_list: list,
        out_channels_list: list,
        bn_kwargs: dict,
        bias: bool = True,
        head_group: bool = True,
    ):
        super(YoloGroupNeck, self).__init__()
        assert len(backbone_idx) == len(in_channels_list)
        assert len(in_channels_list) == len(out_channels_list)
        self.backbone_idx = backbone_idx

        for i, channels in enumerate(out_channels_list):
            if i > 0:
                in_channels_list[i] += out_channels_list[i - 1]

            self.add_module(
                "module%d" % (i),
                InputModule(
                    in_channels_list[i],
                    channels,
                    bn_kwargs=bn_kwargs,
                    bias=bias,
                    head_group=head_group,
                ),
            )
            if head_group:
                self.add_module(
                    "conv%d" % (i),
                    SeparableGroupConvModule2d(
                        channels,
                        channels * 2,
                        3,
                        padding=1,
                        stride=1,
                        bias=bias,
                        dw_norm_layer=nn.BatchNorm2d(channels, **bn_kwargs),
                        dw_act_layer=nn.ReLU(inplace=True),
                        pw_norm_layer=nn.BatchNorm2d(
                            channels * 2, **bn_kwargs
                        ),
                        pw_act_layer=nn.ReLU(inplace=True),
                    ),
                )
            else:
                self.add_module(
                    "conv%d" % (i),
                    SeparableConvModule2d(
                        channels,
                        channels * 2,
                        3,
                        padding=1,
                        stride=1,
                        bias=bias,
                        dw_norm_layer=nn.BatchNorm2d(channels, **bn_kwargs),
                        dw_act_layer=nn.ReLU(inplace=True),
                        pw_norm_layer=nn.BatchNorm2d(
                            channels * 2, **bn_kwargs
                        ),
                        pw_act_layer=nn.ReLU(inplace=True),
                    ),
                )

            if i < len(self.backbone_idx) - 1:
                self.add_module(
                    "concat%d" % (i), nn.quantized.FloatFunctional()
                )
                self.add_module(
                    "resize%d" % (i),
                    horizon.nn.Interpolate(
                        scale_factor=2, recompute_scale_factor=True
                    ),
                )
                self.add_module(
                    "trans_conv%d" % (i),
                    ConvModule2d(
                        channels,
                        channels,
                        1,
                        bias=bias,
                        norm_layer=nn.BatchNorm2d(channels, **bn_kwargs),
                        act_layer=nn.ReLU(inplace=True),
                    ),
                )
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x):
        outputs = []
        input = x[self.backbone_idx[0]]
        for i, _ in enumerate(self.backbone_idx):
            out = getattr(self, "module%d" % (i))(input)
            outputs.append(getattr(self, "conv%d" % (i))(out))
            if i < len(self.backbone_idx) - 1:
                out = getattr(self, "trans_conv%d" % (i))(out)
                out = getattr(self, "resize%d" % (i))(out)
                input = getattr(self, "concat%d" % (i)).cat(
                    (out, x[self.backbone_idx[i + 1]]), dim=1
                )
        return outputs

    def fuse_model(self):
        for i in range(len(self.backbone_idx)):
            getattr(self, "module%d" % (i)).fuse_model()
            getattr(self, "conv%d" % (i)).fuse_model()
            if i < len(self.backbone_idx) - 1:
                getattr(self, "trans_conv%d" % (i)).fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
