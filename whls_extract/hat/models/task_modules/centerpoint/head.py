# Copyright (c) Horizon Robotics. All rights reserved.
import copy
from typing import Dict, List, Sequence

import torch.nn as nn
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_vargnet_module import BasicVarGBlock
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.base_modules.separable_conv_module import SeparableConvModule2d
from hat.models.weight_init import normal_init
from hat.registry import OBJECT_REGISTRY

__all__ = ["CenterPointHead"]


class TaskHead(nn.Module):
    """Task head for CenterHead.

    Args:
        in_channels: Input channels for conv_layer.
        heads: Conv information.
        head_conv_channels: Output channels.
            Default: 64.
        final_kernal: Kernal size for the last conv layer.
            Deafult: 1.
        init_bias: Initial bias. Default: -2.19.
        int8_output: Whether using int8 output.

    """

    def __init__(
        self,
        in_channels: int,
        heads: Dict,
        head_conv_channels: int,
        final_kernel: int = 1,
        init_bias: float = -2.19,
        int8_output: bool = False,
        bn_kwargs=None,
    ):
        super(TaskHead, self).__init__()
        self.heads = heads

        self.init_bias = init_bias
        self.int8_output = int8_output
        if bn_kwargs is None:
            bn_kwargs = {}
        self.dequant = DeQuantStub()
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for _ in range(num_conv - 1):
                conv_layers.append(
                    self._make_conv(
                        in_channels=c_in,
                        out_channels=head_conv_channels,
                        kernel_size=final_kernel,
                        stride=1,
                    )
                )
                c_in = head_conv_channels

            conv_layers.append(
                ConvModule2d(
                    in_channels=head_conv_channels,
                    out_channels=classes,
                    kernel_size=final_kernel,
                    padding=final_kernel // 2,
                    stride=1,
                    bias=True if head == "heatmap" else False,
                )
            )
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            head_convs = self.__getattr__(head)
            for m in head_convs.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            if head == "heatmap":
                head_convs[-1][0].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """
        Perform the forward pass for a given input.

        Args:
            x: Input feature to the model.

        Returns:
            ret_dict: A dictionary containing the outputs from different heads,
                    with each head's output being dequantized.
        """
        ret_dict = {}
        for head in self.heads:
            ret_dict[head] = self.dequant(self.__getattr__(head)(x))
        return ret_dict

    def fuse_model(self) -> None:
        """Perform model fusion on the modules."""

        for head in self.heads:
            head_convs = self.__getattr__(head)
            for m in head_convs:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if not self.int8_output:
            for head in self.heads:
                head_convs = self.__getattr__(head)
                head_convs[
                    -1
                ].qconfig = qconfig_manager.get_default_qat_out_qconfig()

    def _make_conv(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=True,
    ):
        """Make conv module."""

        return ConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=bias,
            norm_layer=nn.BatchNorm2d(out_channels),
            act_layer=nn.ReLU(inplace=True),
        )


class DepthwiseSeparableTaskHead(TaskHead):
    def _make_conv(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=True,
    ):
        """Make conv module."""
        return SeparableConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
            bias=bias,
            pw_norm_layer=nn.BatchNorm2d(out_channels),
            pw_act_layer=nn.ReLU(inplace=True),
        )


@OBJECT_REGISTRY.register
class CenterPointHead(nn.Module):
    """CenterPointHead module.

    Args:
        in_channels: In channels for each task.
        tasks: List of task info.
        share_conv_channels: Channels for share conv.
        share_conv_num: Number of convs for shared.
        common_heads: common head for each task.
        num_heatmap_convs: Number of heatmap convs.
        bn_kwargs : Kwargs of bn layer
        final_kernel: Kernerl size for final kernel.
    """

    def __init__(
        self,
        in_channels: int,
        tasks: List[dict],
        share_conv_channels: int,
        share_conv_num: int,
        common_heads: Dict,
        num_heatmap_convs: int = 2,
        bn_kwargs=None,
        **kwargs,
    ):
        super(CenterPointHead, self).__init__()

        self.bn_kwargs = bn_kwargs if bn_kwargs else {}
        self.shared_conv = nn.Sequential(
            *(
                self._make_conv(
                    in_channels=in_channels if i == 0 else share_conv_channels,
                    out_channels=share_conv_channels,
                    kernel_size=3,
                    stride=1,
                )
                for i in range(share_conv_num)
            )
        )

        self.task_heads = nn.ModuleList()

        num_classes = [len(t["class_names"]) for t in tasks]

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update({"heatmap": (num_cls, num_heatmap_convs)})
            task_head = self._make_task(
                in_channels=share_conv_channels,
                heads=heads,
                bn_kwargs=self.bn_kwargs,
                **kwargs,
            )
            self.task_heads.append(task_head)

    def _init_weights(self):
        """Initialize weights."""
        for m in self.shared_conv.modules:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def forward(self, feats):
        """
        Perform the forward pass for extracted features.

        Args:
            feats: Input feature(s) to the model. If a sequence of features is
                provided, only the first one will be used.

        Returns:
            rets: A list of outputs from the individual task heads.
        """
        rets = []
        if isinstance(feats, Sequence):
            feats = feats[0]
        feats = self.shared_conv(feats)
        for task in self.task_heads:
            rets.append(task(feats))

        return rets

    def _make_conv(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=True,
    ):
        """Make conv module."""

        return ConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=bias,
            norm_layer=nn.BatchNorm2d(out_channels),
            act_layer=nn.ReLU(inplace=True),
        )

    def _make_task(self, **kwargs):
        """Make task head module."""
        return TaskHead(**kwargs)

    def fuse_model(self) -> None:
        """Perform model fusion on the modules."""

        for m in self.shared_conv:
            if hasattr(m, "fuse_model"):
                m.fuse_model()
        for head in self.task_heads:
            if hasattr(head, "fuse_model"):
                head.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for head in self.task_heads:
            if hasattr(head, "set_qconfig"):
                head.set_qconfig()


@OBJECT_REGISTRY.register
class DepthwiseSeparableCenterPointHead(CenterPointHead):
    def _make_conv(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=True,
    ):
        """Make conv module."""
        pw_norm_layer = nn.BatchNorm2d(out_channels, **self.bn_kwargs)
        pw_act_layer = nn.ReLU(inplace=True)

        return SeparableConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
            bias=bias,
            pw_norm_layer=pw_norm_layer,
            pw_act_layer=pw_act_layer,
        )

    def _make_task(self, **kwargs):
        """Make task head module."""
        return DepthwiseSeparableTaskHead(**kwargs)


@OBJECT_REGISTRY.register
class VargCenterPointHead(CenterPointHead):
    def __init__(
        self,
        group_base=8,
        merge_branch=False,
        factor=2,
        dw_with_relu=True,
        pw_with_relu=False,
        **kwargs,
    ):
        self.group_base = group_base
        self.merge_branch = merge_branch
        self.factor = factor
        self.dw_with_relu = dw_with_relu
        self.pw_with_relu = pw_with_relu
        super(VargCenterPointHead, self).__init__(**kwargs)

    def _make_conv(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=True,
    ):
        return BasicVarGBlock(
            in_channels=in_channels,
            mid_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=bias,
            bn_kwargs=self.bn_kwargs,
            factor=self.factor,
            group_base=self.group_base,
            merge_branch=self.merge_branch,
            dw_with_relu=self.dw_with_relu,
            pw_with_relu=self.pw_with_relu,
        )

    def _make_task(self, **kwargs):
        return TaskHead(**kwargs)
