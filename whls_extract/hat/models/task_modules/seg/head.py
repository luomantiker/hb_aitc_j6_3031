# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Sequence

import horizon_plugin_pytorch as horizon
import horizon_plugin_pytorch.nn as hnn
import torch
import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.separable_conv_module import SeparableConvModule2d
from hat.models.weight_init import normal_init
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, multi_apply

__all__ = ["SegHead"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class SegHead(nn.Module):
    """Head Module for segmentation task.

    Args:
        num_classes (int): Number of classes.
        in_strides (list[int]): The strides corresponding to the inputs of
            seg_head, the inputs usually come from backbone or neck.
        out_strides (list[int]): List of output strides.
        stride2channels (dict): A stride to channel dict.
        feat_channels (int or list[int]):
            Number of hidden channels (of each output stride).
        stride_loss_weights (list[int]): loss weight of each stride.
        stacked_convs (int): Number of stacking convs of head.
        padding_to_alighed_dim (tuple): padding before argmax.
        argmax_output (bool): Whether conduct argmax on output. Default: False
        dequant_output (bool): Whether to dequant output. Default: True
        int8_output (bool): If True, output int8, otherwise output int32.
            Default: True
        upscale (bool): If True, stride{x}'s feature map
            is upsampled by 2x, then the upsampled feature is adding
            supervisory signal. Default is False.
        upscale_stride (int): Specify which stride's feature need to
            be upsampled when upscale is True.
        interpolation (str): Interpolation method of image scaling, candidate
            value is ['nearest', 'bilinear'].
        output_with_bn (bool): Whether add bn layer to the output conv.
        bn_kwargs (dict): Extra keyword arguments for bn layers.
        upsample_output_scale (int): Output upsample scale, only used in
            qat model, default is None.
        with_relu6 (bool): Whether replace relu with relu6.
    """

    def __init__(
        self,
        num_classes,
        in_strides,
        out_strides,
        stride2channels,
        feat_channels=256,
        stride_loss_weights=None,
        stacked_convs=1,
        padding_to_alighed_dim=None,
        argmax_output=False,
        dequant_output=True,
        int8_output=True,
        upscale=False,
        upscale_stride=4,
        interpolation="bilinear",
        output_with_bn=False,
        bn_kwargs=None,
        upsample_output_scale=None,
        output_conf=False,
        only_export_first=False,
        with_relu6=False,
    ):
        super(SegHead, self).__init__()
        if argmax_output:
            assert (
                not dequant_output
            ), "argmax output tensor, no need to dequant"
        self.in_strides = sorted(_as_list(in_strides))
        self.out_strides = sorted(_as_list(out_strides))
        assert max(self.out_strides) <= max(
            self.in_strides
        ), "max(out_strides) should <= max(in_strides)"
        if min(self.out_strides) in self.in_strides:
            self.feat_start_index = self.in_strides.index(
                min(self.out_strides)
            )
        else:
            # extra upsampling exist
            assert upscale, "upscale should be True"
            self.feat_start_index = 0
        if max(self.out_strides) in self.in_strides:
            self.feat_end_index = (
                self.in_strides.index(max(self.out_strides)) + 1
            )
        else:
            # extra upsampling exist
            assert upscale, "upscale should be True"
            self.feat_end_index = 1
        self.stride2channels = stride2channels
        self.in_channels = [self.stride2channels[i] for i in self.in_strides]
        if upscale:
            assert (
                in_strides[0] == upscale_stride
            ), "upscale_stride must be the first in_stride"
            self.in_channels.insert(
                0, self.stride2channels[self.in_strides[0]]
            )
        if isinstance(feat_channels, int):
            self.feat_channels = [feat_channels] * len(self.out_strides)
        else:
            assert len(feat_channels) == len(self.out_strides)
            self.feat_channels = feat_channels
        self.num_classes = num_classes
        if stride_loss_weights is None:
            self.stride_loss_weights = [1.0] * len(self.out_strides)
        else:
            self.stride_loss_weights = stride_loss_weights
            assert len(self.stride_loss_weights) == len(self.out_strides)
        self.stacked_convs = stacked_convs
        self.padding_to_alighed_dim = padding_to_alighed_dim
        self.argmax_output = argmax_output
        self.dequant_output = dequant_output
        self.int8_output = int8_output
        self.upscale = upscale

        assert interpolation in ["nearest", "bilinear"], (
            "Currently interpolation only supports ['nearest', 'bilinear'], "
            f"but got {interpolation}."
        )
        self.interpolation = interpolation
        if self.interpolation == "nearest":
            self.align_corners = None
        else:
            self.align_corners = False

        self.output_with_bn = output_with_bn
        self.bn_kwargs = bn_kwargs or {}
        self.output_conf = output_conf
        self.with_relu6 = with_relu6

        self.upsample_output_scale = upsample_output_scale
        if upsample_output_scale:
            self.resize = hnn.Interpolate(
                scale_factor=self.upsample_output_scale,
                align_corners=None,
                recompute_scale_factor=True,
            )

        self.only_export_first = only_export_first
        self.dequant = DeQuantStub()
        self.out_strides_prefix = ["stride" + str(i) for i in self.out_strides]
        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_seg_convs()
        self._init_predictor()
        self.upsample = horizon.nn.Interpolate(
            scale_factor=2,
            align_corners=self.align_corners,
            recompute_scale_factor=True,
            mode=self.interpolation,
        )

    def _init_seg_convs(self):
        self.seg_convs = nn.ModuleDict()
        for ii in range(len(self.out_strides)):
            seg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = (
                    self.in_channels[ii + self.feat_start_index]
                    if i == 0
                    else self.feat_channels[ii]
                )
                seg_convs.append(
                    SeparableConvModule2d(
                        chn,
                        self.feat_channels[ii],
                        3,
                        padding=1,
                        pw_norm_layer=nn.BatchNorm2d(
                            self.feat_channels[ii], **self.bn_kwargs
                        ),
                        pw_act_layer=nn.ReLU(inplace=True)
                        if not self.with_relu6
                        else nn.ReLU6(inplace=True),
                    )
                )
            self.seg_convs[self.out_strides_prefix[ii]] = seg_convs

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.output_convs = nn.ModuleDict()
        for i in range(len(self.out_strides)):
            if self.output_with_bn:
                output_conv = ConvModule2d(
                    self.feat_channels[i],
                    self.num_classes,
                    1,
                    norm_layer=nn.BatchNorm2d(
                        self.num_classes, **self.bn_kwargs
                    ),
                )
            else:
                output_conv = nn.Conv2d(
                    self.feat_channels[i], self.num_classes, 1
                )
            self.output_convs[self.out_strides_prefix[i]] = output_conv

    def _init_weights(self):
        """Initialize weights of the head."""
        for m in self.seg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.output_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)

    def forward_single(self, x, stride_index=0):
        """Forward features of a single scale level.

        Args:
            x (Tensor): feature maps of the specified stride.
            stride_index (int): stride index of input feature map.

        Returns:
            tuple: seg predictions of input feature maps.
        """
        seg_feat = x
        seg_convs = self.seg_convs[self.out_strides_prefix[stride_index]]
        conv_seg = self.output_convs[self.out_strides_prefix[stride_index]]
        for seg_layer in seg_convs:
            seg_feat = seg_layer(seg_feat)
        seg_pred = conv_seg(seg_feat)

        if self.upsample_output_scale:
            seg_pred = self.resize(seg_pred)
        if self.argmax_output:
            if self.padding_to_alighed_dim is not None:
                seg_pred = torch.nn.functional.pad(
                    seg_pred, self.padding_to_alighed_dim
                )
            if self.output_conf:
                seg_conf, seg_cls = seg_pred.max(dim=1, keepdim=True)
                seg_pred = torch.cat(
                    (seg_cls, seg_conf.as_subclass(torch.Tensor)), dim=1
                )
            else:
                seg_cls = seg_pred.argmax(dim=1, keepdim=True)
                seg_pred = seg_cls
        if self.dequant_output:
            seg_pred = self.dequant(seg_pred)
        return (seg_pred,)

    def forward(self, feats):
        feats = _as_list(feats[self.feat_start_index : self.feat_end_index])
        if self.upscale:
            upsampled_feature = self.upsample(feats[0])
            feats.insert(0, upsampled_feature)
        outs = multi_apply(
            self.forward_single, feats, range(len(self.out_strides))
        )
        assert (
            isinstance(outs, Sequence)
            and len(outs)
            == 1  # does not mean that limit len(out_strides)==1 # noqa
            and isinstance(outs[0], Sequence)
        )
        outs = outs[0]
        if self.only_export_first is True:
            outs = outs[0]
        return outs

    def fuse_model(self):
        for m in self.seg_convs.modules():
            if isinstance(m, SeparableConvModule2d) and hasattr(
                m, "fuse_model"
            ):  # noqa
                m.fuse_model()

        for m in self.output_convs.modules():
            if isinstance(m, ConvModule2d) and hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        if not self.int8_output:
            getattr(
                self.output_convs, "0"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()
