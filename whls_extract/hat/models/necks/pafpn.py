# CopyRight: mmdetection
import copy
import logging
from typing import Dict, List

import horizon_plugin_pytorch.nn as nnF
import torch.nn as nn
import torch.nn.functional as F

from hat.models.base_modules.basic_mixvargenet_module import BasicMixVarGEBlock
from hat.models.base_modules.basic_vargnet_module import BasicVarGBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.weight_init import normal_init
from hat.registry import OBJECT_REGISTRY

__all__ = ["PAFPN", "VargPAFPN"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class PAFPN(nn.Module):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int | Dict): Output channels of each scale
        out_strides (List[int]): Stride of output feature map
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to\
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to\
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv\
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed:

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra\
            conv. Default: False.
        norm_cfg (dict): A dict of norm layer configuration. A typical norm_cfg
            can be {"norm_type": "gn", "num_groups": 32, "affine": True} or
            {"norm_type": "bn"}. Default: None.
            If norm_cfg is none, no norm layer is used.
            If norm_cfg["norm_type"] == "gn", the group norm layer is used.
            If norm_cfg["norm_type"] == "bn", the batch norm layer is used.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        out_strides,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        norm_cfg=None,
    ):
        super(PAFPN, self).__init__()

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        assert isinstance(out_channels, int) or isinstance(out_channels, dict)
        out_channels = (
            [out_channels[stride] for stride in out_strides]
            if isinstance(out_channels, dict)
            else [out_channels for _ in out_strides]
        )
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ("on_input", "on_lateral", "on_output")
        elif add_extra_convs:  # True
            self.add_extra_convs = "on_input"

        self.with_norm = norm_cfg is not None
        self.norm_type = norm_cfg["norm_type"] if self.with_norm else None
        self.norm_cfg = norm_cfg

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        def _get_norm_layer(num_channels):
            norm_cfg_copy = copy.deepcopy(self.norm_cfg)
            if self.norm_type == "gn":
                # remove the unused "norm_type" to fit norm layer parameters
                norm_cfg_copy.pop("norm_type")
                assert "num_groups" in norm_cfg_copy
                layer = nn.GroupNorm(
                    num_channels=num_channels,
                    **norm_cfg_copy,
                )
            elif self.norm_type == "bn":
                # remove the unused "norm_type" to fit norm layer parameters
                norm_cfg_copy.pop("norm_type")
                layer = nn.BatchNorm2d(num_channels, **norm_cfg_copy)
            else:
                layer = None
            return layer

        for i in range(self.start_level, self.backbone_end_level):
            out_channels = self.out_channels[i - self.start_level]
            norm_layer = _get_norm_layer(out_channels)
            l_conv = ConvModule2d(
                in_channels=in_channels[i],
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                padding=0,
                bias=True if not self.with_norm else False,
                norm_layer=norm_layer,
            )
            fpn_conv = ConvModule2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                groups=1,
                padding=1,
                bias=True if not self.with_norm else False,
                norm_layer=norm_layer,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = self.out_channels[
                        self.backbone_end_level - self.start_level + i - 1
                    ]
                out_channels = self.out_channels[
                    self.backbone_end_level - self.start_level + i
                ]
                norm_layer = _get_norm_layer(out_channels)
                extra_fpn_conv = ConvModule2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    groups=1,
                    padding=1,
                    bias=True if not self.with_norm else False,
                    norm_layer=norm_layer,
                )
                self.fpn_convs.append(extra_fpn_conv)

        self.upsampling = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            in_channels = self.out_channels[i - self.start_level]
            out_channels = self.out_channels[i - (self.start_level + 1)]
            norm_layer = _get_norm_layer(out_channels)
            up_conv = nn.Sequential(
                ConvModule2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    groups=1,
                    padding=0,
                    bias=True if not self.with_norm else False,
                    norm_layer=norm_layer,
                ),
                nnF.Interpolate(
                    scale_factor=2,
                    mode="bilinear",
                    recompute_scale_factor=True,
                ),
            )
            self.upsampling.append(up_conv)
        self.add = nn.quantized.FloatFunctional()

        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            in_channels = self.out_channels[i - (self.start_level + 1)]
            out_channels = self.out_channels[i - self.start_level]
            norm_layer = _get_norm_layer(out_channels)
            d_conv = ConvModule2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                groups=1,
                padding=1,
                bias=True if not self.with_norm else False,
                norm_layer=norm_layer,
            )
            pafpn_conv = ConvModule2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                groups=1,
                padding=1,
                bias=True if not self.with_norm else False,
                norm_layer=norm_layer,
            )
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

        self._init_weights()

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = self.add.add(
                self.upsampling[i - 1](laterals[i]), laterals[i - 1]
            )

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] = self.add.add(
                self.downsample_convs[i](inter_outs[i]), inter_outs[i + 1]
            )

        outs = []
        outs.append(inter_outs[0])
        outs.extend(
            [
                self.pafpn_convs[i - 1](inter_outs[i])
                for i in range(1, used_backbone_levels)
            ]
        )

        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == "on_input":
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == "on_lateral":
                    outs.append(
                        self.fpn_convs[used_backbone_levels](laterals[-1])
                    )
                elif self.add_extra_convs == "on_output":
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return outs

    def _init_weights(self):
        """Initialize the weights of PAFPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01, bias=0)

    def fuse_model(self):
        for m in self.lateral_convs:
            m.fuse_model()
        for m in self.fpn_convs:
            m.fuse_model()
        for m in self.downsample_convs:
            m.fuse_model()
        for m in self.pafpn_convs:
            m.fuse_model()
        for m in self.upsampling:
            m[0].fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        if self.norm_type == "gn":
            raise ValueError("QAT doesn't support GN!")
        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class VargPAFPN(nn.Module):
    """Path Aggregation Network with BasicVargNetBlock or BasicMixVargNetBlock.

    Args:
        in_channels: Number of input channels per scale.
        out_channels: Output channels of each scale
        out_strides: Stride of output feature map
        num_outs: Number of output scales.
        bn_kwargs: Dict for Bn layer.
        start_level: Index of the start input backbone level used to
            build the feature pyramid. Default is 0.
        end_level Index of the end input backbone level (exclusive) to
            build the feature pyramid.
            Default is -1, which means the last level.
        with_pafpn_conv: Choice whether to use a extra 3x3 conv_block to the
            out features. Default is False.
        varg_block_type: Choice varg block type from
            ["BasicVarGBlock", "BasicMixVarGEBlock"],
            Default is "BasicMixVarGEBlock".
        group_base: groupbase for varg block.
            Default is 16.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        out_strides: List[int],
        num_outs: int,
        bn_kwargs: Dict,
        start_level: int = 0,
        end_level: int = -1,
        with_pafpn_conv: bool = False,
        varg_block_type: str = "BasicMixVarGEBlock",
        group_base: int = 16,
    ):
        super(VargPAFPN, self).__init__()

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        assert isinstance(out_channels, int)
        out_channels = [out_channels for _ in out_strides]
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_pafpn_conv = with_pafpn_conv

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            out_channels = self.out_channels[i - self.start_level]
            l_conv = ConvModule2d(
                in_channels=in_channels[i],
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                padding=0,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            )
            fpn_conv = ConvModule2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                padding=0,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                act_layer=nn.ReLU(inplace=True),
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.upsampling = nn.ModuleList()
        self.conv_top_down_add = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            in_channels = self.out_channels[i - self.start_level]
            out_channels = self.out_channels[i - (self.start_level + 1)]
            up_op = nnF.Interpolate(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )
            self.upsampling.append(up_op)
            self.conv_top_down_add.append(nn.quantized.FloatFunctional())

        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.conv_bottom_up_add = nn.ModuleList()
        if self.with_pafpn_conv:
            self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            in_channels = self.out_channels[i - (self.start_level + 1)]
            out_channels = self.out_channels[i - self.start_level]
            if varg_block_type == "BasicVarGBlock":
                d_conv = BasicVarGBlock(
                    in_channels=in_channels,
                    mid_channels=out_channels,
                    out_channels=out_channels,
                    stride=2,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    factor=1,
                    group_base=group_base,
                    merge_branch=True,
                    bn_kwargs=bn_kwargs,
                )
            elif varg_block_type == "BasicMixVarGEBlock":
                d_conv = BasicMixVarGEBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    bias=True,
                    bn_kwargs=bn_kwargs,
                    factor=1,
                    conv1_group_base=group_base,
                )
            else:
                raise NotImplementedError(
                    f"varg_block_type {varg_block_type} not support now,"
                    "please choice from [BasicVarGBlock, BasicMixVargNetBlock]"
                )
            self.downsample_convs.append(d_conv)
            self.conv_bottom_up_add.append(nn.quantized.FloatFunctional())
            if self.with_pafpn_conv:
                pafpn_conv = ConvModule2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    groups=1,
                    padding=1,
                    norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                )
                self.pafpn_convs.append(pafpn_conv)

        self._init_weights()

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = self.conv_top_down_add[i - 1].add(
                self.upsampling[i - 1](laterals[i]), laterals[i - 1]
            )

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] = self.conv_bottom_up_add[i].add(
                self.downsample_convs[i](inter_outs[i]), inter_outs[i + 1]
            )

        if self.with_pafpn_conv:
            outs = []
            outs.append(inter_outs[0])
            outs.extend(
                [
                    self.pafpn_convs[i - 1](inter_outs[i])
                    for i in range(1, used_backbone_levels)
                ]
            )
        else:
            outs = inter_outs
        return outs

    def _init_weights(self):
        """Initialize the weights of PAFPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01, bias=0)

    def fuse_model(self):
        for m in self.lateral_convs:
            m.fuse_model()
        for m in self.fpn_convs:
            m.fuse_model()
        for m in self.downsample_convs:
            m.fuse_model()
        if self.with_pafpn_conv:
            for m in self.pafpn_convs:
                m.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
