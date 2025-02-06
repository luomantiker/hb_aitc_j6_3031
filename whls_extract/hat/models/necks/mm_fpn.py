# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import Interpolate

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.weight_init import normal_init
from hat.registry import OBJECT_REGISTRY

__all__ = ["MMFPN"]


@OBJECT_REGISTRY.register
class MMFPN(nn.Module):
    def __init__(
        self,
        in_strides: List[int],
        in_channels: List[int],
        out_strides: List[int],
        fix_out_channel: Optional[int],
        bn_kwargs: Optional[Dict] = None,
    ):
        """FPN neck.

        Args:
            in_strides (list): strides of each input feature map
            in_channels (list): channels of each input feature map,
                the length of in_channels should be equal to in_strides
            out_strides (list): strides of each output feature map,
                should be a subset of in_strides, and continuous (any
                subsequence of 2, 4, 8, 16, 32, 64 ...). The largest
                stride in in_strides and out_strides should be equal
            out_channels (list): channels of each output feature maps
                the length of out_channels should be equal to out_strides
            fix_out_channel (:obj:`int`, optional): if set, there will be
                a 1x1 conv following each output feature map so that each
                final output has fix_out_channel channels
            bn_kwargs (dict): Dict for Bn layer. No Bn layer if
                bn_kwargs=None
        """

        super(MMFPN, self).__init__()
        self._valid_strides = [2, 4, 8, 16, 32, 64, 128, 256]
        self.bn_kwargs = bn_kwargs
        # in_strides check
        assert len(in_strides) == len(in_channels)
        for stride_i in in_strides:
            assert stride_i in self._valid_strides

        min_idx = self._valid_strides.index(in_strides[0])
        max_idx = self._valid_strides.index(in_strides[-1])

        assert tuple(in_strides) == tuple(
            self._valid_strides[min_idx : max_idx + 1]
        ), "Input strides must be continuous and in ascending order"
        self.in_strides = in_strides

        min_idx = self._valid_strides.index(out_strides[0])
        max_idx = self._valid_strides.index(out_strides[-1])

        assert tuple(out_strides) == tuple(
            self._valid_strides[min_idx : max_idx + 1]
        ), "Output strides must be continuous"

        assert all(
            [stride in in_strides for stride in out_strides]
        ), "all stride of output stride must be in input stride"

        assert (
            out_strides[-1] == in_strides[-1]
        ), "The largest stride in in_strides and out_strides should be equal"

        self.out_strides = out_strides

        self.src_min_stride_idx = self.in_strides.index(self.out_strides[0])

        # init modules
        self.conv_extract = nn.ModuleList()
        self.conv_add = nn.ModuleList()
        self.upscale = nn.ModuleList()
        self.fpn_conv = nn.ModuleList()

        in_channels = in_channels[self.src_min_stride_idx :]
        for idx in range(len(out_strides)):
            if idx == 0:
                self.conv_extract.append(
                    ConvModule2d(
                        in_channels=in_channels[idx],
                        out_channels=fix_out_channel,
                        kernel_size=1,
                        padding=0,
                        stride=1,
                        bias=True,
                        norm_layer=None
                        if bn_kwargs is None
                        else nn.BatchNorm2d(fix_out_channel, **bn_kwargs),
                    )
                )
            else:
                self.upscale.append(
                    Interpolate(
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False,
                        recompute_scale_factor=True,
                    )
                )

                self.conv_extract.append(
                    ConvModule2d(
                        in_channels=in_channels[idx],
                        out_channels=fix_out_channel,
                        kernel_size=1,
                        padding=0,
                        stride=1,
                        bias=True,
                        norm_layer=None
                        if bn_kwargs is None
                        else nn.BatchNorm2d(fix_out_channel, **bn_kwargs),
                    )
                )
            self.conv_add.append(nn.quantized.FloatFunctional())

            self.fpn_conv.append(
                ConvModule2d(
                    in_channels=fix_out_channel,
                    out_channels=fix_out_channel,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=True,
                    norm_layer=None
                    if bn_kwargs is None
                    else nn.BatchNorm2d(fix_out_channel, **bn_kwargs),
                )
            )

    def _init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01, bias=0)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(features) == len(self.in_strides)

        # slice features
        in_features = features[self.src_min_stride_idx :]
        strides = self.in_strides[self.src_min_stride_idx :]

        length = len(strides)
        fpn_fuse = []

        for idx, _ in enumerate(strides):
            fpn_fuse.append(self.conv_extract[idx](in_features[idx]))

        length = len(strides)
        for idx in range(length - 1, 0, -1):
            cur_feat = self.upscale[idx - 1](fpn_fuse[idx])
            fpn_fuse[idx - 1] = self.conv_add[length - idx - 1].add(
                fpn_fuse[idx - 1], cur_feat
            )

        for idx in range(length):
            fpn_fuse[idx] = self.fpn_conv[idx](fpn_fuse[idx])
        return fpn_fuse

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
