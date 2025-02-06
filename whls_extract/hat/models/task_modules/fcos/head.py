# Copyright (c) Horizon Robotics. All rights reserved.

import os
from collections import OrderedDict
from typing import Sequence

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.quantization import get_default_calib_qconfig
from torch.quantization import DeQuantStub

from hat.models.base_modules.activation import Scale
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.separable_conv_module import SeparableConvModule2d
from hat.models.weight_init import bias_init_with_prob, normal_init
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, multi_apply
from hat.utils.saved_tensor import support_saved_tensor

__all__ = ["FCOSHead", "VehicleSideFCOSHead", "FCOSHeadWithConeInvasion"]

INF = 1e8


@OBJECT_REGISTRY.register
class FCOSHead(nn.Module):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`.

    Args:
        num_classes: Number of categories excluding the background category.
        in_strides: A list contains the strides of feature maps from backbone
            or neck.
        out_strides: A list contains the strides of this head will output.
        stride2channels: A stride to channel dict.
        upscale_bbox_pred: If true, upscale bbox pred by FPN strides.
        feat_channels: Number of hidden channels.
        stacked_convs: Number of stacking convs of the head.
        use_sigmoid: Whether the classification output is obtained using
            sigmoid.
        share_bn: Whether to share bn between multiple levels, default is
            share_bn.
        dequant_output: Whether to dequant output. Default: True
        int8_output: If True, output int8, otherwise output int32.
            Default: True.
        int16_output: If True, output int16, otherwise output int32.
            Default: False.
        nhwc_output: transpose output layout to nhwc.
        share_conv: Only the number of all stride channels is the same,
            share_conv can be True, branches share conv, otherwise not.
            Default: True.
        bbox_relu: Whether use relu for bbox. Default: True.
        use_plain_conv: If True, use plain conv rather than depth-wise  conv in
            some conv layers. This argument works when share_conv=True.
            Default: False.
        use_gn: If True, use group normalization instead of batch normalization
            in some conv layers. This argument works when share_conv=True.
            Default: False.
        use_scale: If True, add a scale layer to scale the predictions like
            what original FCOS does. This argument works when share_conv=True.
            Default: False.
        add_stride: If True, add extra out_strides. Sometimes the out_strides
            is not a subset of in_strides, for example, the in_strides is
            [4, 8, 16, 32, 64] but the out_strides is [8, 16, 32, 64, 128],
            then we need to add an extra stride 128 in this head. This argument
            works when share_conv=True. Default: False.
        skip_qtensor_check: if True, skip head qtensor check.
            The python grammar `assert` not support for TorchDynamo.
        output_dict: If True, forward(self) will output a dict.
        use_save_tensor: If true, turn off save tensor.
    """

    def __init__(
        self,
        num_classes: int,
        in_strides: Sequence[int],
        out_strides: Sequence[int],
        stride2channels: dict,
        upscale_bbox_pred: bool,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        use_sigmoid: bool = True,
        share_bn: bool = False,
        dequant_output: bool = True,
        int8_output: bool = True,
        int16_output=False,
        nhwc_output=False,
        share_conv: bool = True,
        bbox_relu: bool = True,
        use_plain_conv: bool = False,
        use_gn: bool = False,
        use_scale: bool = False,
        add_stride: bool = False,
        output_dict: bool = False,
        set_all_int16_qconfig=False,
        pred_reg_channel: int = 4,
        skip_qtensor_check: bool = False,
        use_save_tensor: bool = True,
    ):
        super(FCOSHead, self).__init__()
        if upscale_bbox_pred:
            assert dequant_output, (
                "dequant_output should be True to convert "
                "QTensor to Tensor when upscale_bbox_pred is True"
            )
        self.num_classes = num_classes
        self.in_strides = sorted(_as_list(in_strides))
        self.out_strides = sorted(_as_list(out_strides))
        self.add_stride = add_stride
        self.output_dict = output_dict
        self.set_all_int16_qconfig = set_all_int16_qconfig
        self.skip_qtensor_check = skip_qtensor_check
        self.use_save_tensor = use_save_tensor

        if not self.add_stride:
            assert set(self.out_strides).issubset(
                self.in_strides
            ), "out_strides must be a subset of in_strides"
        else:
            assert (
                len(set(self.out_strides).intersection(self.in_strides)) > 0
            ), "in_strides and out_strides must have overlap"
            # If self.out_strides (e.g. [8, 16, 32, 64, 128]) is not a subset
            # of self.in_strides (e.g. [4, 8, 16, 32, 64]), we need to add
            # extra strides (128 in this case).
            self.extra_strides = sorted(
                set(self.out_strides).difference(in_strides)
            )
            assert len(self.extra_strides) > 0 and min(
                self.extra_strides
            ) > max(self.in_strides), (
                "if add_stride, out_strides should have bigger strides than "
                "the maximum in_stride."
            )
            assert share_conv, "adding stride only works in share conv mode."

        self.feat_indices = [
            self.in_strides.index(stride)
            for stride in self.out_strides
            if stride in self.in_strides
        ]

        self.stride2channels = stride2channels
        self.in_channels = (
            [stride2channels[stride] for stride in self.in_strides]
            if not share_conv
            else stride2channels[self.in_strides[0]]
        )
        self.pred_reg_channel = pred_reg_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.use_sigmoid = use_sigmoid
        self.share_bn = share_bn
        self.upscale_bbox_pred = upscale_bbox_pred
        self.dequant_output = dequant_output
        self.int8_output = int8_output
        self.int16_output = int16_output
        assert not (
            int8_output and int16_output
        ), "int8_output and int16_output cannot be true at the same time."
        self.nhwc_output = nhwc_output
        self.share_conv = share_conv
        self.bbox_relu = bbox_relu
        assert self.share_bn is False if self.share_conv is False else True
        assert (
            len(set(stride2channels.values())) != 1 and not self.share_conv
        ) or len(set(stride2channels.values())) == 1
        self.background_label = num_classes
        if use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.use_plain_conv = use_plain_conv
        self.use_gn = use_gn
        self.use_scale = use_scale
        # self.use_plain_conv, self.use_gn, and self.use_scale works when
        # self.share_conv = True and self.share_bn = True
        assert (
            (self.share_conv and self.share_bn)
            if self.use_plain_conv or self.use_gn or self.use_scale
            else True
        )
        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.dequant = DeQuantStub()
        if self.share_conv:
            if self.share_bn:
                self._init_cls_convs()
                self._init_reg_convs()
                if self.use_scale:
                    self.scales = nn.ModuleList(
                        [Scale(1.0) for _ in self.out_strides]
                    )
            else:
                self._init_cls_reg_convs_with_independent_bn()
            if self.add_stride:
                self._init_extra_stride_convs()
            self._init_predictor()
        else:
            self._init_cls_no_shared_convs()
            self._init_reg_no_shared_convs()
            self._init_no_shared_predictor()

    def _init_cls_no_shared_convs(self):
        self.cls_convs_list = nn.ModuleList()
        for _ in range(self.stacked_convs):
            cls_convs = nn.ModuleList()
            for j in self.feat_indices:
                in_chn = self.in_channels[j]
                cls_convs.append(
                    SeparableConvModule2d(
                        in_chn,
                        in_chn,
                        kernel_size=3,
                        padding=1,
                        pw_norm_layer=nn.BatchNorm2d(in_chn),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                )
            self.cls_convs_list.append(cls_convs)

    def _init_reg_no_shared_convs(self):
        self.reg_convs_list = nn.ModuleList()
        for _ in range(self.stacked_convs):
            reg_convs = nn.ModuleList()
            for j in self.feat_indices:
                in_chn = self.in_channels[j]
                reg_convs.append(
                    SeparableConvModule2d(
                        in_chn,
                        in_chn,
                        kernel_size=3,
                        padding=1,
                        pw_norm_layer=nn.BatchNorm2d(in_chn),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                )
            self.reg_convs_list.append(reg_convs)

    def _init_no_shared_predictor(self):
        self.conv_cls = nn.ModuleList()
        self.conv_reg = nn.ModuleList()
        self.conv_centerness = nn.ModuleList()
        # relu6 maybe affect performance
        self.single_relu = nn.ReLU(inplace=True)
        for j in self.feat_indices:
            in_chn = self.in_channels[j]
            self.conv_cls.append(
                nn.Conv2d(in_chn, self.cls_out_channels, 3, padding=1)
            )
            self.conv_reg.append(
                nn.Conv2d(in_chn, self.pred_reg_channel, 3, padding=1)
            )
            self.conv_centerness.append(nn.Conv2d(in_chn, 1, 3, padding=1))

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                SeparableConvModule2d(
                    chn,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    pw_norm_layer=nn.GroupNorm(32, self.feat_channels)
                    if self.use_gn
                    else nn.BatchNorm2d(self.feat_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
                if not self.use_plain_conv
                else ConvModule2d(
                    chn,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    norm_layer=nn.GroupNorm(32, self.feat_channels)
                    if self.use_gn
                    else nn.BatchNorm2d(self.feat_channels),
                    act_layer=nn.ReLU(inplace=True),
                )
            )

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                SeparableConvModule2d(
                    chn,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    pw_norm_layer=nn.GroupNorm(32, self.feat_channels)
                    if self.use_gn
                    else nn.BatchNorm2d(self.feat_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
                if not self.use_plain_conv
                else ConvModule2d(
                    chn,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    norm_layer=nn.GroupNorm(32, self.feat_channels)
                    if self.use_gn
                    else nn.BatchNorm2d(self.feat_channels),
                    act_layer=nn.ReLU(inplace=True),
                )
            )

    def _init_extra_stride_convs(self):
        """Initialize extra fpn conv levels to add extra strides.

        This is generally used for adding an extra stride 128 to a fpn that
        outputs only a maximum stride 64.

        """
        self.extra_stride_convs = nn.ModuleList()
        for _ in self.extra_strides:
            chn = self.in_channels
            extra_conv = (
                SeparableConvModule2d(
                    chn,
                    chn,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    pw_norm_layer=nn.GroupNorm(32, chn)
                    if self.use_gn
                    else nn.BatchNorm2d(chn),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
                if not self.use_plain_conv
                else ConvModule2d(
                    chn,
                    chn,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                    norm_layer=nn.GroupNorm(32, chn)
                    if self.use_gn
                    else nn.BatchNorm2d(chn),
                    act_layer=None,
                )
            )
            self.extra_stride_convs.append(extra_conv)

    def _init_cls_reg_convs_with_independent_bn(self):
        """Initialize convs of cls head and reg head.

        depth-wise and point-wise convs are shared by all stride, but BN is
        independent, i.e. not shared, experiment shows that this will improve
        performance.
        """
        num_strides = len(self.out_strides)
        self.cls_convs = nn.ModuleList(
            [nn.ModuleList() for i in range(num_strides)]
        )
        self.reg_convs = nn.ModuleList(
            [nn.ModuleList() for i in range(num_strides)]
        )

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            for j in range(num_strides):
                if j == 0:
                    # to create new conv
                    cls_shared_dw_conv = None
                    cls_shared_pw_conv = None
                    reg_shared_dw_conv = None
                    reg_shared_pw_conv = None
                else:
                    # share convs of the first out stride, not create
                    cls_shared_dw_conv = self.cls_convs[0][i][0][0]
                    cls_shared_pw_conv = self.cls_convs[0][i][1][0]
                    reg_shared_dw_conv = self.reg_convs[0][i][0][0]
                    reg_shared_pw_conv = self.reg_convs[0][i][1][0]

                # construct cls_convs
                if cls_shared_dw_conv is None:
                    self.cls_convs[j].append(
                        SeparableConvModule2d(
                            chn,
                            self.feat_channels,
                            kernel_size=3,
                            padding=1,
                            pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                            pw_act_layer=nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.cls_convs[j].append(
                        nn.Sequential(
                            cls_shared_dw_conv,
                            cls_shared_pw_conv,
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )

                # construct reg_convs
                if reg_shared_dw_conv is None:
                    self.reg_convs[j].append(
                        SeparableConvModule2d(
                            chn,
                            self.feat_channels,
                            kernel_size=3,
                            padding=1,
                            pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                            pw_act_layer=nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.reg_convs[j].append(
                        nn.Sequential(
                            reg_shared_dw_conv,
                            reg_shared_pw_conv,
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1
        )
        self.conv_reg = nn.Conv2d(
            self.feat_channels, self.pred_reg_channel, 3, padding=1
        )
        # relu6 maybe affect performance
        self.single_relu = nn.ReLU(inplace=True)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

    def _init_weights(self):
        """Initialize weights of the head."""
        if self.share_conv:
            for m in self.cls_convs.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            for m in self.reg_convs.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            for m in self.conv_centerness.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            bias_cls = bias_init_with_prob(0.01)
            normal_init(self.conv_cls, std=0.01, bias=bias_cls)
            normal_init(self.conv_reg, std=0.01)
            normal_init(self.conv_centerness, std=0.01)
            if self.add_stride:
                for m in self.extra_stride_convs.modules():
                    if isinstance(m, nn.Conv2d):
                        normal_init(m, std=0.01)
        else:
            for m in self.cls_convs_list.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            for m in self.reg_convs_list.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            bias_cls = bias_init_with_prob(0.01)
            for m in self.conv_cls.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01, bias=bias_cls)
            for m in self.conv_reg.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            for m in self.conv_centerness.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)

    def forward_single(self, x, i, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            i (int): Index of feature level.
            stride (int): The corresponding stride for feature maps, only
                used to upscale bbox pred when self.upscale_bbox_pred
                is True.
        """
        cls_feat = x
        reg_feat = x
        if self.share_conv:
            if self.share_bn:
                for cls_layer in self.cls_convs:
                    cls_feat = cls_layer(cls_feat)
            else:
                for cls_layer in self.cls_convs[i]:
                    cls_feat = cls_layer(cls_feat)
            cls_score = self.conv_cls(cls_feat)

            if self.share_bn:
                for reg_layer in self.reg_convs:
                    reg_feat = reg_layer(reg_feat)
            else:
                for reg_layer in self.reg_convs[i]:
                    reg_feat = reg_layer(reg_feat)
            bbox_pred = self.conv_reg(reg_feat)
            if self.bbox_relu:
                bbox_pred = self.single_relu(bbox_pred)
            centerness = self.conv_centerness(reg_feat)
        else:
            for cls_layer in self.cls_convs_list:
                cls_feat = cls_layer[i](cls_feat)
            cls_score = self.conv_cls[i](cls_feat)

            for reg_layer in self.reg_convs_list:
                reg_feat = reg_layer[i](reg_feat)
            bbox_pred = self.conv_reg[i](reg_feat)
            if self.bbox_relu:
                bbox_pred = self.single_relu(bbox_pred)
            centerness = self.conv_centerness[i](reg_feat)

        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        if self.use_scale:
            bbox_pred = self.scales[i](bbox_pred).float()

        if self.dequant_output:
            cls_score = self.dequant(cls_score)
            bbox_pred = self.dequant(bbox_pred)
            centerness = self.dequant(centerness)

        if self.upscale_bbox_pred and self.training is not True:
            # Only used in eval mode when upscale_bbox_pred = True.
            # Because the ele-mul operation is not supported currently,
            # this part will be conduct in filter after dequant
            if not self.skip_qtensor_check:
                assert not isinstance(bbox_pred[0], horizon.qtensor.QTensor), (
                    "QTensor not support multiply op, you can set "
                    "dequant_output=True to convert QTensor to Tensor"
                )
            bbox_pred *= stride

        return cls_score, bbox_pred, centerness

    def forward(self, feats):
        if self.use_save_tensor and support_saved_tensor():
            # If using save tensor, every node must do forward and backward.
            # The following code avoid leaking gpu memory, when some
            # strides' feature do not backward.

            feats = _as_list(feats)
            ignore_feats = 0
            use_feats = []
            for i in range(len(feats)):
                if i in self.feat_indices:
                    use_feats.append(feats[i])
                else:
                    ignore_feats = ignore_feats + torch.sum(feats[i])
            feats = use_feats
            ignore_feats = ignore_feats * 0
            feats[0] = feats[0] + ignore_feats
        else:
            feats = [_as_list(feats)[index] for index in self.feat_indices]

        if self.add_stride:
            for i in range(len(self.extra_strides)):
                feats.append(self.extra_stride_convs[i](feats[-1]))
        cls_scores, bbox_preds, centernesses = [], [], []
        for i, feat in enumerate(feats):
            cls_score, bbox_pred, centerness = self.forward_single(
                feat, range(len(self.out_strides))[i], self.out_strides[i]
            )
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centernesses.append(centerness)
        if self.nhwc_output:
            for i in range(len(bbox_preds)):
                cls_scores[i] = cls_scores[i].permute(0, 2, 3, 1)
                bbox_preds[i] = bbox_preds[i].permute(0, 2, 3, 1)
                centernesses[i] = centernesses[i].permute(0, 2, 3, 1)
        output = (
            OrderedDict(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                centernesses=centernesses,
            )
            if self.output_dict
            else (cls_scores, bbox_preds, centernesses)
        )

        return output

    def fuse_model(self):
        def fuse_model_convs(convs):
            if self.share_conv:
                for modules in convs:
                    for m in modules:
                        if self.share_bn:
                            m.fuse_model()
                        else:
                            if isinstance(m, SeparableConvModule2d):
                                modules_to_fuse = [["1.0", "1.1", "1.2"]]
                            elif isinstance(m, nn.Sequential):
                                modules_to_fuse = [["1", "2", "3"]]
                            else:
                                raise NotImplementedError(
                                    f"Not support type{type(m)} to fuse"
                                )
                            horizon.quantization.fuse_conv_shared_modules(
                                m, modules_to_fuse, inplace=True
                            )
            else:
                for modules in convs:
                    for m in modules:
                        for n in m:
                            n.fuse_model()

        if self.use_plain_conv or self.use_gn or self.use_scale:
            raise NotImplementedError(
                "Not support type (ConvModule2d(its submodule), GroupNorm, "
                "Scale) to fuse."
            )

        if self.share_conv:
            fuse_model_convs(self.cls_convs)
            fuse_model_convs(self.reg_convs)
        else:
            fuse_model_convs(self.cls_convs_list)
            fuse_model_convs(self.reg_convs_list)

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        if self.int16_output:
            from hat.utils.qconfig_manager import QconfigMode

            qconfig_manager.set_qconfig_mode(QconfigMode.QAT)
            out_int16_qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
            self.conv_cls.qconfig = out_int16_qconfig
            self.conv_reg.qconfig = out_int16_qconfig
            self.conv_centerness.qconfig = out_int16_qconfig
        elif not self.int8_output:
            self.conv_cls.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.conv_reg.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.conv_centerness.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )

        if self.set_all_int16_qconfig:
            self.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                weight_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
                weight_calibration_qkwargs={"dtype": qint16},
            )

    def set_calibration_qconfig(self):
        if self.int16_output:
            self.conv_cls.qconfig = get_default_calib_qconfig(dtype="qint16")
            self.conv_reg.qconfig = get_default_calib_qconfig(dtype="qint16")
            self.conv_centerness.qconfig = get_default_calib_qconfig(
                dtype="qint16"
            )

        if self.set_all_int16_qconfig:
            self.qconfig = get_default_calib_qconfig(dtype="qint16")


@OBJECT_REGISTRY.register
class VehicleSideFCOSHead(nn.Module):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_strides (Sequence[int]): A list contains the strides of feature
            maps from backbone or neck.
        out_strides (Sequence[int]): A list contains the strides of this head
            will output.
        stride2channels (dict): A stride to channel dict.
        feat_channels (int): Number of hidden channels.
        stacked_convs (int): Number of stacking convs of the head.
        use_sigmoid (bool): Whether the classification output is obtained
            using sigmoid.
        share_bn (bool): Whether to share bn between multiple levels, default
            is share_bn.
        upscale_bbox_pred (bool): If true, upscale bbox pred by FPN strides.
        dequant_output (bool): Whether to dequant output. Default: True
        int8_output(bool): If True, output int8, otherwise output int32.
            Default: True
        share_conv(bool): Only the number of all stride channels is the same,
            share_conv can be True, branches share conv, otherwise not.
            Default: True
        use_plain_conv: If True, use plain conv rather than depth-wise  conv in
            some conv layers. This argument works when share_conv=True.
            Default: False.
        use_gn: If True, use group normalization instead of batch normalization
            in some conv layers. This argument works when share_conv=True.
            Default: False.
        use_scale: If True, add a scale layer to scale the predictions like
            what original FCOS does. This argument works when share_conv=True.
            Default: False.
        add_stride: If True, add extra out_strides. Sometimes the out_strides
            is not a subset of in_strides, for example, the in_strides is
            [4, 8, 16, 32, 64] but the out_strides is [8, 16, 32, 64, 128],
            then we need to add an extra stride 128 in this head. This argument
            works when share_conv=True. Default: False.
    """

    def __init__(
        self,
        num_classes,
        in_strides,
        out_strides,
        stride2channels,
        upscale_bbox_pred,
        feat_channels=256,
        stacked_convs=4,
        use_sigmoid=True,
        share_bn=False,
        dequant_output=True,
        int8_output=True,
        share_conv=True,
        enable_act=False,
        use_plain_conv=False,
        use_gn=False,
        use_scale=False,
        add_stride=False,
    ):
        super(VehicleSideFCOSHead, self).__init__()
        if upscale_bbox_pred:
            assert dequant_output, (
                "dequant_output should be True to convert "
                "QTensor to Tensor when upscale_bbox_pred is True"
            )
        self.num_classes = num_classes
        self.in_strides = sorted(_as_list(in_strides))
        self.out_strides = sorted(_as_list(out_strides))
        self.add_stride = add_stride

        if not self.add_stride:
            assert set(self.out_strides).issubset(
                self.in_strides
            ), "out_strides must be a subset of in_strides"
        else:
            assert (
                len(set(self.out_strides).intersection(self.in_strides)) > 0
            ), "in_strides and out_strides must have overlap"
            # If self.out_strides (e.g. [8, 16, 32, 64, 128]) is not a subset
            # of self.in_strides (e.g. [4, 8, 16, 32, 64]), we need to add
            # extra strides (128 in this case).
            self.extra_strides = sorted(
                set(self.out_strides).difference(in_strides)
            )
            assert len(self.extra_strides) > 0 and min(
                self.extra_strides
            ) > max(self.in_strides), (
                "if add_stride, out_strides should have bigger strides than "
                "the maximum in_stride."
            )
            assert share_conv, "adding stride only works in share conv mode."

        self.feat_indices = [
            self.in_strides.index(stride)
            for stride in self.out_strides
            if stride in self.in_strides
        ]

        self.stride2channels = stride2channels
        self.in_channels = (
            [stride2channels[stride] for stride in self.in_strides]
            if not share_conv
            else stride2channels[self.in_strides[0]]
        )
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.use_sigmoid = use_sigmoid
        self.share_bn = share_bn
        self.upscale_bbox_pred = upscale_bbox_pred
        self.dequant_output = dequant_output
        self.int8_output = int8_output
        self.share_conv = share_conv
        self.enable_act = enable_act
        self.use_plain_conv = use_plain_conv
        self.use_gn = use_gn
        self.use_scale = use_scale
        # self.use_plain_conv, self.use_gn, and self.use_scale works when
        # self.share_conv = True and self.share_bn = True
        assert (
            (self.share_conv and self.share_bn)
            if self.use_plain_conv or self.use_gn or self.use_scale
            else True
        )
        assert self.share_bn is False if self.share_conv is False else True
        assert (
            len(set(stride2channels.values())) != 1 and not self.share_conv
        ) or len(set(stride2channels.values())) == 1
        self.background_label = num_classes
        if use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.dequant = DeQuantStub()
        if self.share_conv:
            if self.share_bn:
                self._init_cls_convs()
                self._init_reg_convs()
                if self.use_scale:
                    self.scales = nn.ModuleList(
                        [Scale(1.0) for _ in self.out_strides]
                    )
            else:
                self._init_cls_reg_convs_with_independent_bn()
            if self.add_stride:
                self._init_extra_stride_convs()
            self._init_predictor()
        else:
            self._init_cls_no_shared_convs()
            self._init_reg_no_shared_convs()
            self._init_no_shared_predictor()

    def _init_cls_no_shared_convs(self):
        self.cls_convs_list = nn.ModuleList()
        for _ in range(self.stacked_convs):
            cls_convs = nn.ModuleList()
            for j in self.feat_indices:
                in_chn = self.in_channels[j]
                cls_convs.append(
                    SeparableConvModule2d(
                        in_chn,
                        in_chn,
                        kernel_size=3,
                        padding=1,
                        pw_norm_layer=nn.BatchNorm2d(in_chn),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                )
            self.cls_convs_list.append(cls_convs)

    def _init_reg_no_shared_convs(self):
        self.reg_convs_list = nn.ModuleList()
        for _ in range(self.stacked_convs):
            reg_convs = nn.ModuleList()
            for j in self.feat_indices:
                in_chn = self.in_channels[j]
                reg_convs.append(
                    SeparableConvModule2d(
                        in_chn,
                        in_chn,
                        kernel_size=3,
                        padding=1,
                        pw_norm_layer=nn.BatchNorm2d(in_chn),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                )
            self.reg_convs_list.append(reg_convs)

    def _init_no_shared_predictor(self):
        self.conv_cls = nn.ModuleList()
        self.conv_reg = nn.ModuleList()
        self.conv_alpha = nn.ModuleList()
        self.conv_centerness = nn.ModuleList()
        # relu6 maybe affect performance
        self.single_relu = nn.ReLU(inplace=False)
        self.single_tanh = nn.Tanh()
        for j in self.feat_indices:
            in_chn = self.in_channels[j]
            self.conv_cls.append(
                nn.Conv2d(in_chn, self.cls_out_channels, 3, padding=1)
            )
            self.conv_reg.append(nn.Conv2d(in_chn, 4, 3, padding=1))
            self.conv_alpha.append(nn.Conv2d(in_chn, 1, 3, padding=1))
            self.conv_centerness.append(nn.Conv2d(in_chn, 1, 3, padding=1))

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                SeparableConvModule2d(
                    chn,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    pw_norm_layer=nn.GroupNorm(32, self.feat_channels)
                    if self.use_gn
                    else nn.BatchNorm2d(self.feat_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
                if not self.use_plain_conv
                else ConvModule2d(
                    chn,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    norm_layer=nn.GroupNorm(32, self.feat_channels)
                    if self.use_gn
                    else nn.BatchNorm2d(self.feat_channels),
                    act_layer=nn.ReLU(inplace=True),
                )
            )

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                SeparableConvModule2d(
                    chn,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    pw_norm_layer=nn.GroupNorm(32, self.feat_channels)
                    if self.use_gn
                    else nn.BatchNorm2d(self.feat_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
                if not self.use_plain_conv
                else ConvModule2d(
                    chn,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    norm_layer=nn.GroupNorm(32, self.feat_channels)
                    if self.use_gn
                    else nn.BatchNorm2d(self.feat_channels),
                    act_layer=nn.ReLU(inplace=True),
                )
            )

    def _init_extra_stride_convs(self):
        """Initialize extra fpn conv levels to add extra strides.

        This is generally used for adding an extra stride 128 to a fpn that
        outputs only a maximum stride 64.

        """
        self.extra_stride_convs = nn.ModuleList()
        for _ in self.extra_strides:
            chn = self.in_channels
            extra_conv = (
                SeparableConvModule2d(
                    chn,
                    chn,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    pw_norm_layer=nn.GroupNorm(32, chn)
                    if self.use_gn
                    else nn.BatchNorm2d(chn),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
                if not self.use_plain_conv
                else ConvModule2d(
                    chn,
                    chn,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                    norm_layer=nn.GroupNorm(32, chn)
                    if self.use_gn
                    else nn.BatchNorm2d(chn),
                    act_layer=None,
                )
            )
            self.extra_stride_convs.append(extra_conv)

    def _init_cls_reg_convs_with_independent_bn(self):
        """Initialize convs of cls head and reg head.

        depth-wise and point-wise convs are shared by all stride, but BN is
        independent, i.e. not shared, experiment shows that this will improve
        performance.
        """
        num_strides = len(self.out_strides)
        self.cls_convs = nn.ModuleList(
            [nn.ModuleList() for i in range(num_strides)]
        )
        self.reg_convs = nn.ModuleList(
            [nn.ModuleList() for i in range(num_strides)]
        )

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            for j in range(num_strides):
                if j == 0:
                    # to create new conv
                    cls_shared_dw_conv = None
                    cls_shared_pw_conv = None
                    reg_shared_dw_conv = None
                    reg_shared_pw_conv = None
                else:
                    # share convs of the first out stride, not create
                    cls_shared_dw_conv = self.cls_convs[0][i][0][0]
                    cls_shared_pw_conv = self.cls_convs[0][i][1][0]
                    reg_shared_dw_conv = self.reg_convs[0][i][0][0]
                    reg_shared_pw_conv = self.reg_convs[0][i][1][0]

                # construct cls_convs
                if cls_shared_dw_conv is None:
                    self.cls_convs[j].append(
                        SeparableConvModule2d(
                            chn,
                            self.feat_channels,
                            kernel_size=3,
                            padding=1,
                            pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                            pw_act_layer=nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.cls_convs[j].append(
                        nn.Sequential(
                            cls_shared_dw_conv,
                            cls_shared_pw_conv,
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )

                # construct reg_convs
                if reg_shared_dw_conv is None:
                    self.reg_convs[j].append(
                        SeparableConvModule2d(
                            chn,
                            self.feat_channels,
                            kernel_size=3,
                            padding=1,
                            pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                            pw_act_layer=nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.reg_convs[j].append(
                        nn.Sequential(
                            reg_shared_dw_conv,
                            reg_shared_pw_conv,
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1
        )
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.conv_alpha = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        # relu6 maybe affect performance
        self.single_relu = nn.ReLU(inplace=False)
        self.single_tanh = nn.Tanh()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

    def _init_weights(self):
        """Initialize weights of the head."""
        if self.share_conv:
            for m in self.cls_convs.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            for m in self.reg_convs.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            for m in self.conv_centerness.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            bias_cls = bias_init_with_prob(0.01)
            normal_init(self.conv_cls, std=0.01, bias=bias_cls)
            normal_init(self.conv_reg, std=0.01)
            normal_init(self.conv_alpha, std=0.01)
            normal_init(self.conv_centerness, std=0.01)
            if self.add_stride:
                for m in self.extra_stride_convs.modules():
                    if isinstance(m, nn.Conv2d):
                        normal_init(m, std=0.01)
        else:
            for m in self.cls_convs_list.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            for m in self.reg_convs_list.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            bias_cls = bias_init_with_prob(0.01)
            for m in self.conv_cls.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01, bias=bias_cls)
            for m in self.conv_reg.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            for m in self.conv_alpha.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            for m in self.conv_centerness.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)

    def forward_single(self, x, i, stride):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            i (int): Index of feature level.
            stride (int): The corresponding stride for feature maps, only
                used to upscale bbox pred when self.upscale_bbox_pred
                is True.
        """
        cls_feat = x
        reg_feat = x
        if self.share_conv:
            if self.share_bn:
                for cls_layer in self.cls_convs:
                    cls_feat = cls_layer(cls_feat)
            else:
                for cls_layer in self.cls_convs[i]:
                    cls_feat = cls_layer(cls_feat)
            cls_score = self.conv_cls(cls_feat)

            if self.share_bn:
                for reg_layer in self.reg_convs:
                    reg_feat = reg_layer(reg_feat)
            else:
                for reg_layer in self.reg_convs[i]:
                    reg_feat = reg_layer(reg_feat)
            bbox_pred = self.conv_reg(reg_feat)
            bbox_pred = self.single_relu(bbox_pred)
            alpha_pred = self.conv_alpha(reg_feat)
            alpha_pred = self.single_tanh(alpha_pred)
            centerness = self.conv_centerness(reg_feat)
        else:
            for cls_layer in self.cls_convs_list:
                cls_feat = cls_layer[i](cls_feat)
            cls_score = self.conv_cls[i](cls_feat)

            for reg_layer in self.reg_convs_list:
                reg_feat = reg_layer[i](reg_feat)
            bbox_pred = self.conv_reg[i](reg_feat)
            if self.int8_output or self.enable_act:
                bbox_pred = self.single_relu(bbox_pred)
            alpha_pred = self.conv_alpha[i](reg_feat)
            if self.int8_output or self.enable_act:
                alpha_pred = self.single_tanh(alpha_pred)
            centerness = self.conv_centerness[i](reg_feat)

        if self.dequant_output:
            cls_score = self.dequant(cls_score)
            bbox_pred = self.dequant(bbox_pred)
            alpha_pred = self.dequant(alpha_pred)
            centerness = self.dequant(centerness)

        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        if self.use_scale:
            bbox_pred = self.scales[i](bbox_pred).float()

        if self.upscale_bbox_pred and self.training is not True:
            # Only used in eval mode when upscale_bbox_pred = True.
            # Because the ele-mul operation is not supported currently,
            # this part will be conduct in filter after dequant
            assert not isinstance(bbox_pred[0], horizon.qtensor.QTensor), (
                "QTensor not support multiply op, you can set "
                "dequant_output=True to convert QTensor to Tensor"
            )
            bbox_pred *= stride

        return cls_score, bbox_pred, alpha_pred, centerness

    def forward(self, feats):
        if bool(int(os.environ.get("HAT_USE_SAVEDTENSOR", "0"))):
            # If using save tensor, every node must do forward and backward.
            # The following code avoid leaking gpu memory, when some
            # strides' feature do not backward.

            feats = _as_list(feats)
            ignore_feats = 0
            use_feats = []
            for i in range(len(feats)):
                if i in self.feat_indices:
                    use_feats.append(feats[i])
                else:
                    ignore_feats = ignore_feats + torch.sum(feats[i])
            feats = use_feats
            ignore_feats = ignore_feats * 0
            feats[0] = feats[0] + ignore_feats
        else:
            feats = [_as_list(feats)[index] for index in self.feat_indices]

        if self.add_stride:
            for i in range(len(self.extra_strides)):
                feats.append(self.extra_stride_convs[i](feats[-1]))

        cls_scores, bbox_preds, alpha_preds, centernesses = multi_apply(
            self.forward_single,
            feats,
            range(len(self.out_strides)),
            self.out_strides,
        )

        return cls_scores, bbox_preds, alpha_preds, centernesses

    def fuse_model(self):
        def fuse_model_convs(convs):
            if self.share_conv:
                for modules in convs:
                    for m in modules:
                        if self.share_bn:
                            m.fuse_model()
                        else:
                            if isinstance(m, SeparableConvModule2d):
                                modules_to_fuse = [["1.0", "1.1", "1.2"]]
                            elif isinstance(m, nn.Sequential):
                                modules_to_fuse = [["1", "2", "3"]]
                            else:
                                raise NotImplementedError(
                                    f"Not support type{type(m)} to fuse"
                                )
                            horizon.quantization.fuse_conv_shared_modules(
                                m, modules_to_fuse, inplace=True
                            )
            else:
                for modules in convs:
                    for m in modules:
                        for n in m:
                            n.fuse_model()

        if self.use_plain_conv or self.use_gn or self.use_scale:
            raise NotImplementedError(
                "Not support type (ConvModule2d (its submodule), GroupNorm, "
                "Scale) to fuse."
            )

        if self.share_conv:
            fuse_model_convs(self.cls_convs)
            fuse_model_convs(self.reg_convs)
        else:
            fuse_model_convs(self.cls_convs_list)
            fuse_model_convs(self.reg_convs_list)

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        if not self.int8_output:
            self.conv_cls.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.conv_reg.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.conv_alpha.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.conv_centerness.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )


@OBJECT_REGISTRY.register
class FCOSHeadWithConeInvasion(FCOSHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`.

    Args:
        num_classes: Number of categories excluding the background category.
        in_strides: A list contains the strides of feature maps from backbone
            or neck.
        out_strides: A list contains the strides of this head will output.
        stride2channels: A stride to channel dict.
        upscale_bbox_pred: If true, upscale bbox pred by FPN strides.
        feat_channels: Number of hidden channels.
        stacked_convs: Number of stacking convs of the head.
        use_sigmoid: Whether the classification output is obtained using
            sigmoid.
        share_bn: Whether to share bn between multiple levels, default is
            share_bn.
        dequant_output: Whether to dequant output. Default: True
        int8_output: If True, output int8, otherwise output int32.
            Default: True.
        int16_output: If True, output int16, otherwise output int32.
            Default: False.
        nhwc_output: transpose output layout to nhwc.
        share_conv: Only the number of all stride channels is the same,
            share_conv can be True, branches share conv, otherwise not.
            Default: True.
        bbox_relu: Whether use relu for bbox. Default: True.
        invasion_scale_relu: Whether use relu for cone invasion scale.
            Default: True.
        use_plain_conv: If True, use plain conv rather than depth-wise  conv in
            some conv layers. This argument works when share_conv=True.
            Default: False.
        use_gn: If True, use group normalization instead of batch normalization
            in some conv layers. This argument works when share_conv=True.
            Default: False.
        use_scale: If True, add a scale layer to scale the predictions like
            what original FCOS does. This argument works when share_conv=True.
            Default: False.
        add_stride: If True, add extra out_strides. Sometimes the out_strides
            is not a subset of in_strides, for example, the in_strides is
            [4, 8, 16, 32, 64] but the out_strides is [8, 16, 32, 64, 128],
            then we need to add an extra stride 128 in this head. This argument
            works when share_conv=True. Default: False.
        skip_qtensor_check: if True, skip head qtensor check.
            The python grammar `assert` not support for TorchDynamo.
        output_dict: If True, forward(self) will output a dict.
        use_save_tensor: If true, turn off save tensor.
    """

    def __init__(
        self,
        num_classes: int,
        in_strides: Sequence[int],
        out_strides: Sequence[int],
        stride2channels: dict,
        upscale_bbox_pred: bool,
        upscale_invasion_scale: bool,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        use_sigmoid: bool = True,
        share_bn: bool = False,
        dequant_output: bool = True,
        int8_output: bool = True,
        int16_output=False,
        nhwc_output=False,
        share_conv: bool = True,
        bbox_relu: bool = True,
        invasion_scale_relu: bool = True,
        use_plain_conv: bool = False,
        use_gn: bool = False,
        use_scale: bool = False,
        add_stride: bool = False,
        output_dict: bool = False,
        set_all_int16_qconfig=False,
        pred_reg_channel: int = 4,
        skip_qtensor_check: bool = False,
        use_save_tensor: bool = True,
    ):
        super(FCOSHeadWithConeInvasion, self).__init__(
            num_classes,
            in_strides,
            out_strides,
            stride2channels,
            upscale_bbox_pred,
            feat_channels,
            stacked_convs,
            use_sigmoid,
            share_bn,
            dequant_output,
            int8_output,
            int16_output,
            nhwc_output,
            share_conv,
            bbox_relu,
            use_plain_conv,
            use_gn,
            use_scale,
            add_stride,
            output_dict,
            set_all_int16_qconfig,
            pred_reg_channel,
            skip_qtensor_check,
            use_save_tensor,
        )
        assert not share_conv, (
            "only support share_conv=False"
            + "in FCOSHeadWithConeInvasion task."
        )
        self.invasion_scale_relu = invasion_scale_relu
        self.upscale_invasion_scale = upscale_invasion_scale

    def _init_layers(self):
        """Initialize layers of the head."""
        self.dequant = DeQuantStub()
        self._init_cls_no_shared_convs()
        self._init_reg_no_shared_convs()
        self._init_invasion_status_no_shared_convs()
        self._init_invasion_scale_no_shared_convs()
        self._init_no_shared_predictor()

    def _init_invasion_status_no_shared_convs(self):
        self.invasion_status_convs_list = nn.ModuleList()
        for _ in range(self.stacked_convs):
            invasion_status_convs = nn.ModuleList()
            for j in self.feat_indices:
                in_chn = self.in_channels[j]
                invasion_status_convs.append(
                    SeparableConvModule2d(
                        in_chn,
                        in_chn,
                        kernel_size=3,
                        padding=1,
                        pw_norm_layer=nn.BatchNorm2d(in_chn),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                )
            self.invasion_status_convs_list.append(invasion_status_convs)

    def _init_invasion_scale_no_shared_convs(self):
        self.invasion_scale_convs_list = nn.ModuleList()
        for _ in range(self.stacked_convs):
            invasion_scale_convs = nn.ModuleList()
            for j in self.feat_indices:
                in_chn = self.in_channels[j]
                invasion_scale_convs.append(
                    SeparableConvModule2d(
                        in_chn,
                        in_chn,
                        kernel_size=3,
                        padding=1,
                        pw_norm_layer=nn.BatchNorm2d(in_chn),
                        pw_act_layer=nn.ReLU(inplace=True),
                    )
                )
            self.invasion_scale_convs_list.append(invasion_scale_convs)

    def _init_no_shared_predictor(self):
        self.conv_cls = nn.ModuleList()
        self.conv_reg = nn.ModuleList()
        self.conv_centerness = nn.ModuleList()
        self.conv_invasion_state = nn.ModuleList()
        self.conv_beside_valid = nn.ModuleList()
        self.conv_invasion_scale = nn.ModuleList()
        # relu6 maybe affect performance
        self.single_relu = nn.ReLU(inplace=True)
        for j in self.feat_indices:
            in_chn = self.in_channels[j]
            self.conv_cls.append(
                nn.Conv2d(in_chn, self.cls_out_channels, 3, padding=1)
            )
            self.conv_reg.append(
                nn.Conv2d(in_chn, self.pred_reg_channel, 3, padding=1)
            )
            self.conv_centerness.append(nn.Conv2d(in_chn, 1, 3, padding=1))
            self.conv_invasion_state.append(
                nn.Conv2d(in_chn, 1, 3, padding=1)
            )  # invasion state
            self.conv_beside_valid.append(
                nn.Conv2d(in_chn, 2, 3, padding=1)
            )  # beside valid conf
            self.conv_invasion_scale.append(
                nn.Conv2d(in_chn, 2, 3, padding=1)
            )  # invasion scale

    def _init_weights(self):
        """Initialize weights of the head."""
        init_module_list = [
            self.cls_convs_list,
            self.reg_convs_list,
            self.invasion_status_convs_list,
            self.invasion_scale_convs_list,
            self.conv_reg,
            self.conv_centerness,
            self.conv_invasion_scale,
            self.conv_invasion_state,
            self.conv_beside_valid,
        ]
        for modules in init_module_list:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        for m in self.conv_cls.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01, bias=bias_cls)

    def forward_single(self, x, i, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            i (int): Index of feature level.
            stride (int): The corresponding stride for feature maps, only
                used to upscale bbox pred when self.upscale_bbox_pred
                is True.
        """
        cls_feat = x
        reg_feat = x
        invasion_status_feat = x
        invasion_scale_feat = x

        for cls_layer in self.cls_convs_list:
            cls_feat = cls_layer[i](cls_feat)
        cls_score = self.conv_cls[i](cls_feat)

        for reg_layer in self.reg_convs_list:
            reg_feat = reg_layer[i](reg_feat)
        bbox_pred = self.conv_reg[i](reg_feat)
        for invasion_status_layer in self.invasion_status_convs_list:
            invasion_status_feat = invasion_status_layer[i](
                invasion_status_feat
            )
        for invasion_scale_layer in self.invasion_scale_convs_list:
            invasion_scale_feat = invasion_scale_layer[i](invasion_scale_feat)

        if self.bbox_relu:
            bbox_pred = self.single_relu(bbox_pred)
        centerness = self.conv_centerness[i](reg_feat)
        invasion_state = self.conv_invasion_state[i](invasion_status_feat)
        beside_valid_conf = self.conv_beside_valid[i](invasion_status_feat)
        invasion_scale = self.conv_invasion_scale[i](invasion_scale_feat)
        if self.invasion_scale_relu:
            invasion_scale = self.single_relu(invasion_scale)

        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        if self.use_scale:
            bbox_pred = self.scales[i](bbox_pred).float()

        if self.dequant_output:
            cls_score = self.dequant(cls_score)
            bbox_pred = self.dequant(bbox_pred)
            centerness = self.dequant(centerness)
            invasion_state = self.dequant(invasion_state)
            beside_valid_conf = self.dequant(beside_valid_conf)
            invasion_scale = self.dequant(invasion_scale)

        if self.upscale_bbox_pred and self.training is not True:
            # Only used in eval mode when upscale_bbox_pred = True.
            # Because the ele-mul operation is not supported currently,
            # this part will be conduct in filter after dequant
            if not self.skip_qtensor_check:
                assert not isinstance(bbox_pred[0], horizon.qtensor.QTensor), (
                    "QTensor not support multiply op, you can set "
                    "dequant_output=True to convert QTensor to Tensor"
                )
            bbox_pred *= stride

        if self.upscale_invasion_scale and self.training is not True:
            if not self.skip_qtensor_check:
                assert not isinstance(
                    invasion_scale[0], horizon.qtensor.QTensor
                ), (
                    "QTensor not support multiply op, you can set "
                    "dequant_output=True to convert QTensor to Tensor"
                )
            invasion_scale *= stride

        return (
            cls_score,
            bbox_pred,
            centerness,
            invasion_state,
            beside_valid_conf,
            invasion_scale,
        )

    def forward(self, feats):
        if self.use_save_tensor and support_saved_tensor():
            # If using save tensor, every node must do forward and backward.
            # The following code avoid leaking gpu memory, when some
            # strides' feature do not backward.

            feats = _as_list(feats)
            ignore_feats = 0
            use_feats = []
            for i in range(len(feats)):
                if i in self.feat_indices:
                    use_feats.append(feats[i])
                else:
                    ignore_feats = ignore_feats + torch.sum(feats[i])
            feats = use_feats
            ignore_feats = ignore_feats * 0
            feats[0] = feats[0] + ignore_feats
        else:
            feats = [_as_list(feats)[index] for index in self.feat_indices]

        if self.add_stride:
            for i in range(len(self.extra_strides)):
                feats.append(self.extra_stride_convs[i](feats[-1]))
        (
            cls_scores,
            bbox_preds,
            centernesses,
            invasion_states,
            beside_valid_confes,
            invasion_scales,
        ) = ([], [], [], [], [], [])
        for i, feat in enumerate(feats):
            (
                cls_score,
                bbox_pred,
                centerness,
                invasion_state,
                beside_valid_conf,
                invasion_scale,
            ) = self.forward_single(
                feat, range(len(self.out_strides))[i], self.out_strides[i]
            )
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centernesses.append(centerness)
            invasion_states.append(invasion_state)
            beside_valid_confes.append(beside_valid_conf)
            invasion_scales.append(invasion_scale)
        if self.nhwc_output:
            for i in range(len(bbox_preds)):
                cls_scores[i] = cls_scores[i].permute(0, 2, 3, 1)
                bbox_preds[i] = bbox_preds[i].permute(0, 2, 3, 1)
                centernesses[i] = centernesses[i].permute(0, 2, 3, 1)
                invasion_states[i] = invasion_states[i].permute(0, 2, 3, 1)
                beside_valid_confes[i] = beside_valid_confes[i].permute(
                    0, 2, 3, 1
                )
                invasion_scales[i] = invasion_scales[i].permute(0, 2, 3, 1)
        output = (
            OrderedDict(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                centernesses=centernesses,
                invasion_states=invasion_states,
                beside_valid_confes=beside_valid_confes,
                invasion_scales=invasion_scales,
            )
            if self.output_dict
            else (
                cls_scores,
                bbox_preds,
                centernesses,
                invasion_states,
                beside_valid_confes,
                invasion_scales,
            )
        )

        return output

    def fuse_model(self):
        def fuse_model_convs(convs):
            for modules in convs:
                for m in modules:
                    for n in m:
                        n.fuse_model()

        if self.use_plain_conv or self.use_gn or self.use_scale:
            raise NotImplementedError(
                "Not support type (ConvModule2d(its submodule), GroupNorm, "
                "Scale) to fuse."
            )

        fuse_model_convs(self.cls_convs_list)
        fuse_model_convs(self.reg_convs_list)
        fuse_model_convs(self.invasion_status_convs_list)
        fuse_model_convs(self.invasion_scale_convs_list)

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        # disable output quantization for last quanti layer.
        if self.int16_output:
            out_int16_qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
            self.conv_cls.qconfig = out_int16_qconfig
            self.conv_reg.qconfig = out_int16_qconfig
            self.conv_centerness.qconfig = out_int16_qconfig
            self.conv_invasion_state.qconfig = out_int16_qconfig
            self.conv_beside_valid.qconfig = out_int16_qconfig
            self.conv_invasion_scale.qconfig = out_int16_qconfig
        elif not self.int8_output:
            self.conv_cls.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.conv_reg.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.conv_centerness.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.conv_invasion_state.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.conv_beside_valid.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            self.conv_invasion_scale.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )

        if self.set_all_int16_qconfig:
            self.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                weight_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
                weight_calibration_qkwargs={"dtype": qint16},
            )

    def set_calibration_qconfig(self):
        if self.int16_output:
            self.conv_cls.qconfig = get_default_calib_qconfig(dtype="qint16")
            self.conv_reg.qconfig = get_default_calib_qconfig(dtype="qint16")
            self.conv_centerness.qconfig = get_default_calib_qconfig(
                dtype="qint16"
            )
            self.conv_invasion_state.qconfig = get_default_calib_qconfig(
                dtype="qint16"
            )
            self.conv_beside_valid.qconfig = get_default_calib_qconfig(
                dtype="qint16"
            )
            self.conv_invasion_scale.qconfig = get_default_calib_qconfig(
                dtype="qint16"
            )

        if self.set_all_int16_qconfig:
            self.qconfig = get_default_calib_qconfig(dtype="qint16")
