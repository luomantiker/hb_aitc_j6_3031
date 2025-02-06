# Copyright (c) Horizon Robotics. All rights reserved.

import torch.nn as nn
from horizon_plugin_pytorch.dtype import qint16
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.weight_init import xavier_init
from hat.registry import OBJECT_REGISTRY

__all__ = ["YOLOV3Head"]


@OBJECT_REGISTRY.register
class YOLOV3Head(nn.Module):
    """
    Heads module of yolov3.

    shared convs -> conv head (include all objs)

    Args:
        in_channels_list: List of input channels.
        feature_idx: Index of feature for head.
        num_classes: Num classes of detection object.
        anchors: Anchors for all feature maps.
        bn_kwargs: Config dict for BN layer.
        bias: Whether to use bias in module.
    """

    def __init__(
        self,
        in_channels_list: list,
        feature_idx: list,
        num_classes: int,
        anchors: list,
        bn_kwargs: dict,
        bias: bool = True,
        reverse_feature: bool = True,
        int16_output: bool = True,
        dequant_output: bool = True,
    ):
        super(YOLOV3Head, self).__init__()
        assert len(in_channels_list) == len(feature_idx)
        assert len(feature_idx) == len(anchors)
        self.num_classes = num_classes
        self.anchors = anchors
        self.reverse = reverse_feature
        self.int16_output = int16_output
        self.dequant_output = dequant_output

        self.feature_idx = feature_idx
        for i, anchor, in_channels in zip(
            range(len(anchors)), anchors, in_channels_list
        ):
            num_anchor = len(anchor)
            self.add_module(
                "head%d" % (i),
                ConvModule2d(
                    in_channels,
                    num_anchor * (num_classes + 5),
                    1,
                    bias=bias,
                ),
            )
            self.add_module("dequant%d" % (i), DeQuantStub())
        self.init_weight()

    def init_weight(self):
        for i, _idx in enumerate(self.feature_idx):
            for m in getattr(self, "head%d" % (i)):
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution="uniform")

    def forward(self, x):
        output = []
        for i, idx in zip(range(len(self.feature_idx)), self.feature_idx):
            out = getattr(self, "head%d" % (i))(x[idx])
            if self.dequant_output:
                out = getattr(self, "dequant%d" % (i))(out)
            output.append(out)
        if self.reverse:
            output.reverse()
        return output

    def fuse_model(self):
        for i in range(len(self.feature_idx)):
            getattr(self, "head%d" % (i)).fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        if self.int16_output:
            self.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
        else:
            self.qconfig = qconfig_manager.get_default_qat_out_qconfig()
