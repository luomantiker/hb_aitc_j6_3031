# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Tuple

import horizon_plugin_pytorch as horizon
import torch
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from torch import nn
from torch.quantization import DeQuantStub

from hat.models.weight_init import bias_init_with_prob, normal_init
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, multi_apply
from hat.utils.model_helpers import fx_wrap


@OBJECT_REGISTRY.register
class FCOS3DHead(nn.Module):
    """Anchor-free head used in FCOS3D.

    Args:
        num_classes: Number of categories excluding the background
            category.
        in_channels: Number of channels in the input feature map.
        feat_channels: Number of hidden channels.
            Used in child classes. Defaults to 256.
        stacked_convs: Number of stacking convs of the head.
        strides: Downsample factor of each feature map.
        group_reg_dims: The dimension of each regression
            target group. Default: (2, 1, 3, 1, 2).
        use_direction_classifier:
            Whether to add a direction classifier.
        pred_attrs: Whether to predict attributes.
            Defaults to False.
        num_attrs: The number of attributes to be predicted.
            Default: 9.
        cls_branch: Channels for classification branch.
            Default: (128, 64).
        reg_branch: Channels for regression branch.
            Default: (
                (128, 64),  # offset
                (128, 64),  # depth
                (64, ),  # size
                (64, ),  # rot
                ()  # velo
            ),
        dir_branch: Channels for direction
            classification branch. Default: (64, ).
        attr_branch: Channels for classification branch.
            Default: (64, ).
        centerness_branch: Channels for centerness branch.
            Default: (64, ).
        centerness_on_reg: If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: True.
        return_for_compiler: whether return the output using for compile.
        output_int32: whether the output is int32.
    """  # noqa: E501

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int,
        stacked_convs: int,
        strides: Tuple[int],
        group_reg_dims: Tuple[int],
        use_direction_classifier: bool,
        pred_attrs: bool,
        num_attrs: int,
        cls_branch: Tuple[int],
        reg_branch: Tuple[int],
        dir_branch: Tuple[int],
        attr_branch: Tuple[int],
        centerness_branch: Tuple[int],
        centerness_on_reg: Tuple[int],
        return_for_compiler=False,
        output_int32=True,
    ):
        super(FCOS3DHead, self).__init__()
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.group_reg_dims = group_reg_dims
        self.use_direction_classifier = use_direction_classifier
        self.pred_attrs = pred_attrs
        self.num_attrs = num_attrs
        self.cls_branch = cls_branch
        self.reg_branch = reg_branch
        self.dir_branch = dir_branch
        self.attr_branch = attr_branch
        self.centerness_branch = centerness_branch
        self.centerness_on_reg = centerness_on_reg
        self.return_for_compiler = return_for_compiler
        self.output_int32 = output_int32

        self.out_channels = []
        for reg_branch_channels in reg_branch:
            if len(reg_branch_channels) > 0:
                self.out_channels.append(reg_branch_channels[-1])
            else:
                self.out_channels.append(-1)
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()
        self.conv_centerness_prev = self._init_branch(
            conv_channels=self.centerness_branch,
            conv_strides=(1,) * len(self.centerness_branch),
        )
        self.conv_centerness = nn.Conv2d(self.centerness_branch[-1], 1, 1)
        self.dequant = DeQuantStub()
        self.cat = FloatFunctional()

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(len(self.strides)):
            cls_convs_per_stride = nn.ModuleList()
            for j in range(self.stacked_convs):
                if i == 0:
                    chn = self.in_channels if j == 0 else self.feat_channels
                    cls_dw_conv = nn.Conv2d(
                        in_channels=chn,
                        out_channels=chn,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=chn,
                    )
                    cls_pw_conv = nn.Conv2d(
                        in_channels=chn,
                        out_channels=self.feat_channels,
                        kernel_size=1,
                        stride=1,
                    )
                    cls_convs_per_stride.append(
                        nn.Sequential(
                            cls_dw_conv,
                            cls_pw_conv,
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    cls_convs_per_stride.append(
                        nn.Sequential(
                            self.cls_convs[0][j][0],
                            self.cls_convs[0][j][1],
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )
            self.cls_convs.append(cls_convs_per_stride)

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(len(self.strides)):
            reg_convs_per_stride = nn.ModuleList()
            for j in range(self.stacked_convs):
                if i == 0:
                    chn = self.in_channels if j == 0 else self.feat_channels
                    reg_dw_conv = nn.Conv2d(
                        in_channels=chn,
                        out_channels=chn,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=chn,
                    )
                    reg_pw_conv = nn.Conv2d(
                        in_channels=chn,
                        out_channels=self.feat_channels,
                        kernel_size=1,
                        stride=1,
                    )
                    reg_convs_per_stride.append(
                        nn.Sequential(
                            reg_dw_conv,
                            reg_pw_conv,
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    reg_convs_per_stride.append(
                        nn.Sequential(
                            self.reg_convs[0][j][0],
                            self.reg_convs[0][j][1],
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )

            self.reg_convs.append(reg_convs_per_stride)

    def _init_branch(self, conv_channels=(64), conv_strides=(1)):
        """Initialize conv layers as a prediction branch."""
        conv_before_pred = nn.ModuleList()
        if isinstance(conv_channels, int):
            conv_channels = [self.feat_channels] + [conv_channels]
            conv_strides = [conv_strides]
        else:
            conv_channels = [self.feat_channels] + list(conv_channels)
            conv_strides = list(conv_strides)
        for i in range(len(self.strides)):
            conv_before_pred_stride = nn.ModuleList()
            for j in range(len(conv_strides)):
                if i == 0:
                    dw_conv = nn.Conv2d(
                        in_channels=conv_channels[j],
                        out_channels=conv_channels[j],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=conv_channels[j],
                    )
                    pw_conv = nn.Conv2d(
                        in_channels=conv_channels[j],
                        out_channels=conv_channels[j + 1],
                        kernel_size=1,
                        stride=1,
                    )

                    conv_before_pred_stride.append(
                        nn.Sequential(
                            dw_conv,
                            pw_conv,
                            nn.BatchNorm2d(conv_channels[j + 1]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    conv_before_pred_stride.append(
                        nn.Sequential(
                            conv_before_pred[0][j][0],
                            conv_before_pred[0][j][1],
                            nn.BatchNorm2d(conv_channels[j + 1]),
                            nn.ReLU(inplace=True),
                        )
                    )

            conv_before_pred.append(conv_before_pred_stride)
        return conv_before_pred

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls_prev = self._init_branch(
            conv_channels=self.cls_branch,
            conv_strides=(1,) * len(self.cls_branch),
        )
        self.conv_cls = nn.Conv2d(
            self.cls_branch[-1], self.cls_out_channels, 1
        )

        self.conv_reg_prevs = nn.ModuleList()
        self.conv_regs = nn.ModuleList()
        for i in range(len(self.group_reg_dims)):
            reg_dim = self.group_reg_dims[i]
            reg_branch_channels = self.reg_branch[i]
            out_channel = self.out_channels[i]
            if len(reg_branch_channels) > 0:
                self.conv_reg_prevs.append(
                    self._init_branch(
                        conv_channels=reg_branch_channels,
                        conv_strides=(1,) * len(reg_branch_channels),
                    )
                )
                self.conv_regs.append(nn.Conv2d(out_channel, reg_dim, 1))
            else:
                self.conv_reg_prevs.append(None)
                self.conv_regs.append(
                    nn.Conv2d(self.feat_channels, reg_dim, 1)
                )
        if self.use_direction_classifier:
            self.conv_dir_cls_prev = self._init_branch(
                conv_channels=self.dir_branch,
                conv_strides=(1,) * len(self.dir_branch),
            )
            self.conv_dir_cls = nn.Conv2d(self.dir_branch[-1], 2, 1)

        if self.pred_attrs:
            self.conv_attr_prev = self._init_branch(
                conv_channels=self.attr_branch,
                conv_strides=(1,) * len(self.attr_branch),
            )
            self.conv_attr = nn.Conv2d(self.attr_branch[-1], self.num_attrs, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        for modules in [self.cls_convs, self.reg_convs, self.conv_cls_prev]:
            for m in modules[0]:  # m: nn.Sequential
                for n in m:
                    if isinstance(n, nn.Conv2d):
                        normal_init(n, std=0.01)
        for conv_reg_prev in self.conv_reg_prevs:
            if conv_reg_prev is None:
                continue
            for m in conv_reg_prev[0]:
                for n in m:
                    if isinstance(n, nn.Conv2d):
                        normal_init(n, std=0.01)

        if self.use_direction_classifier:
            for m in self.conv_dir_cls_prev[0]:
                for n in m:
                    if isinstance(n, nn.Conv2d):
                        normal_init(n, std=0.01)

        if self.pred_attrs:
            for m in self.conv_attr_prev[0]:
                for n in m:
                    if isinstance(n, nn.Conv2d):
                        normal_init(n, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        for conv_reg in self.conv_regs:
            normal_init(conv_reg, std=0.01)
        if self.use_direction_classifier:
            normal_init(self.conv_dir_cls, std=0.01, bias=bias_cls)
        if self.pred_attrs:
            normal_init(self.conv_attr, std=0.01, bias=bias_cls)

        for m in self.conv_centerness_prev[0]:
            for n in m:
                if isinstance(n, nn.Conv2d):
                    normal_init(n, std=0.01)
        normal_init(self.conv_centerness, std=0.01)

    def forward(self, feats):
        feats = _as_list(feats)
        return multi_apply(self.forward_single, feats, self.strides)

    def forward_single(self, x, stride):
        cls_feat = x
        reg_feat = x

        idx = self.strides.index(stride)
        for cls_layer in self.cls_convs[idx]:
            cls_feat = cls_layer(cls_feat)
        # clone the cls_feat for reusing the feature map afterwards
        clone_cls_feat = cls_feat
        for conv_cls_prev_layer in self.conv_cls_prev[idx]:
            clone_cls_feat = conv_cls_prev_layer(clone_cls_feat)
        cls_score = self.conv_cls(clone_cls_feat)

        for reg_layer in self.reg_convs[idx]:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = []
        for i in range(len(self.group_reg_dims)):
            # clone the reg_feat for reusing the feature map afterwards
            clone_reg_feat = reg_feat
            if len(self.reg_branch[i]) > 0:
                for conv_reg_prev_layer in self.conv_reg_prevs[i][idx]:
                    clone_reg_feat = conv_reg_prev_layer(clone_reg_feat)
            bbox_pred.append(self.conv_regs[i](clone_reg_feat))

        dir_cls_pred = None
        if self.use_direction_classifier:
            clone_reg_feat = reg_feat
            for conv_dir_cls_prev_layer in self.conv_dir_cls_prev[idx]:
                clone_reg_feat = conv_dir_cls_prev_layer(clone_reg_feat)
            dir_cls_pred = self.conv_dir_cls(clone_reg_feat)

        attr_pred = None
        if self.pred_attrs:
            # clone the cls_feat for reusing the feature map afterwards
            clone_cls_feat = cls_feat
            for conv_attr_prev_layer in self.conv_attr_prev[idx]:
                clone_cls_feat = conv_attr_prev_layer(clone_cls_feat)
            attr_pred = self.conv_attr(clone_cls_feat)

        if self.centerness_on_reg:
            clone_reg_feat = reg_feat
            for conv_centerness_prev_layer in self.conv_centerness_prev[idx]:
                clone_reg_feat = conv_centerness_prev_layer(clone_reg_feat)
            centerness = self.conv_centerness(clone_reg_feat)
        else:
            clone_cls_feat = cls_feat
            for conv_centerness_prev_layer in self.conv_centerness_prev[idx]:
                clone_cls_feat = conv_centerness_prev_layer(clone_cls_feat)
            centerness = self.conv_centerness(clone_cls_feat)

        cls_score = self.dequant(cls_score)
        bbox_pred = [self.dequant(x) for x in bbox_pred]
        dir_cls_pred = self.dequant(dir_cls_pred)
        attr_pred = self.dequant(attr_pred)
        centerness = self.dequant(centerness)
        if self.return_for_compiler:
            return (
                cls_score,
                *bbox_pred,
                dir_cls_pred,
                attr_pred,
                centerness,
            )

        bbox_pred = self._decode(bbox_pred, stride)

        return (
            cls_score,
            bbox_pred,
            dir_cls_pred,
            attr_pred,
            centerness,
        )

    @fx_wrap()
    def _decode(self, bbox_pred, stride):
        bbox_pred = torch.cat(bbox_pred, dim=1)
        clone_bbox = bbox_pred.clone()
        bbox_pred[:, :2] = clone_bbox[:, :2].float()
        bbox_pred[:, 2] = clone_bbox[:, 2].float()
        bbox_pred[:, 3:6] = clone_bbox[:, 3:6].float()

        bbox_pred[:, 2] = bbox_pred[:, 2].exp()
        bbox_pred[:, 3:6] = bbox_pred[:, 3:6].exp()
        if not self.training:
            # Note that this line is conducted only when testing
            bbox_pred[:, :2] *= stride
        return bbox_pred

    def fuse_model(self):
        def fuse_model_convs(convs: nn.ModuleList):
            for modules in convs:  # modules: nn.ModuleList
                for m in modules:  # m: nn.Sequential
                    if isinstance(m, nn.Sequential):
                        modules_to_fuse = [["1", "2", "3"]]
                    else:
                        raise NotImplementedError(
                            f"Not support type{type(m)} to fuse"
                        )
                    horizon.quantization.fuse_conv_shared_modules(
                        m, modules_to_fuse, inplace=True
                    )

        fuse_model_convs(self.cls_convs)
        fuse_model_convs(self.reg_convs)
        fuse_model_convs(self.conv_cls_prev)
        for module in self.conv_reg_prevs:
            if module is not None:
                fuse_model_convs(module)
        if self.use_direction_classifier:
            fuse_model_convs(self.conv_dir_cls_prev)
        if self.pred_attrs:
            fuse_model_convs(self.conv_attr_prev)
        fuse_model_convs(self.conv_centerness_prev)

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        if self.output_int32:
            self.conv_cls.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
            for module in self.conv_regs:
                module.qconfig = qconfig_manager.get_default_qat_out_qconfig()
            if self.use_direction_classifier:
                self.conv_dir_cls.qconfig = (
                    qconfig_manager.get_default_qat_out_qconfig()
                )
            if self.pred_attrs:
                self.conv_attr.qconfig = (
                    qconfig_manager.get_default_qat_out_qconfig()
                )
            self.conv_centerness.qconfig = (
                qconfig_manager.get_default_qat_out_qconfig()
            )
        else:
            self.conv_cls.qconfig = (
                qconfig_manager.default_qat_out_16bit_qconfig
            )
            for module in self.conv_regs:
                module.qconfig = qconfig_manager.default_qat_out_16bit_qconfig
            if self.use_direction_classifier:
                self.conv_dir_cls.qconfig = (
                    qconfig_manager.default_qat_out_16bit_qconfig
                )
            if self.pred_attrs:
                self.conv_attr.qconfig = (
                    qconfig_manager.default_qat_out_16bit_qconfig
                )
            self.conv_centerness.qconfig = (
                qconfig_manager.default_qat_out_16bit_qconfig()
            )
