# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Sequence, Tuple

import horizon_plugin_pytorch.nn as hnn
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_henet_module import (
    BasicHENetStageBlock,
    S2DDown,
)
from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class HENet(nn.Module):
    """
    Module of HENet.

    Args:
        in_channels: The in_channels for the block.
        block_nums: Number of blocks in each stage.
        embed_dims: Output channels in each stage.
        attention_block_num: Number of attention blocks in each stage.
        mlp_ratios: Mlp expand ratios in each stage.
        mlp_ratio_attn: Mlp expand ratio in attention blocks.
        act_layer: activation layers type.
        use_layer_scale: Use a learnable scale factor in the residual branch.
        layer_scale_init_value: Init value of the learnable scale factor.
        num_classes: Number of classes for a Classifier.
        include_top: Whether to include output layer.
        flat_output: Whether to view the output tensor.
        extra_act: Use extra activation layers in each stage.
        final_expand_channel: Channel expansion before pooling.
        feature_mix_channel: Channel expansion is performed before head.
        block_cls: Basic block types in each stage.
        down_cls: Downsample block types in each stage.
        patch_embed: Stem conv style in the very beginning.
        quant_input: Add a quantization node at the beginning.
        dequant_output: Add a dequantization node at the end.
        stage_out_norm: Add a norm layer to stage outputs.
            Ignored if include_top is True.
    """

    def __init__(
        self,
        in_channels: int,
        block_nums: Tuple[int],
        embed_dims: Tuple[int],
        attention_block_num: Tuple[int],
        mlp_ratios: Tuple[int] = (2, 2, 2, 2),
        mlp_ratio_attn: int = 2,
        act_layer: Tuple[str] = ("nn.GELU", "nn.GELU", "nn.GELU", "nn.GELU"),
        use_layer_scale: Tuple[bool] = (True, True, True, True),
        layer_scale_init_value: float = 1e-5,
        num_classes: int = 1000,
        include_top: bool = True,
        flat_output: bool = True,
        extra_act: Tuple[bool] = (False, False, False, False),
        final_expand_channel: int = 0,
        feature_mix_channel: int = 0,
        block_cls: Tuple[str] = ("DWCB", "DWCB", "DWCB", "DWCB"),
        down_cls: Tuple[str] = ("S2DDown", "S2DDown", "S2DDown", "None"),
        patch_embed: str = "origin",
        quant_input: bool = True,
        dequant_output: bool = True,
        stage_out_norm: bool = True,
    ):
        super().__init__()

        self.final_expand_channel = final_expand_channel
        self.feature_mix_channel = feature_mix_channel
        self.stage_out_norm = stage_out_norm

        self.block_cls = block_cls

        self.include_top = include_top
        self.flat_output = flat_output

        if self.include_top:
            self.num_classes = num_classes

        if patch_embed in ["origin"]:
            self.patch_embed = nn.Sequential(
                ConvModule2d(
                    in_channels,
                    embed_dims[0] // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_layer=nn.BatchNorm2d(embed_dims[0] // 2),
                    act_layer=nn.ReLU(),
                ),
                ConvModule2d(
                    embed_dims[0] // 2,
                    embed_dims[0],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_layer=nn.BatchNorm2d(embed_dims[0]),
                    act_layer=nn.ReLU(),
                ),
            )
        elif patch_embed in ["conv_k5_s4"]:
            self.patch_embed = nn.Sequential(
                ConvModule2d(
                    in_channels,
                    embed_dims[0],
                    kernel_size=5,
                    stride=4,
                    padding=2,
                    norm_layer=nn.BatchNorm2d(embed_dims[0]),
                    act_layer=nn.GELU(),
                ),
            )
        elif patch_embed in ["conv_4_s4"]:
            self.patch_embed = nn.Sequential(
                ConvModule2d(
                    in_channels,
                    embed_dims[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                    norm_layer=nn.BatchNorm2d(embed_dims[0]),
                    act_layer=nn.GELU(),
                ),
            )

        stages = []
        downsample_block = []
        for block_idx, block_num in enumerate(block_nums):
            stages.append(
                BasicHENetStageBlock(
                    in_dim=embed_dims[block_idx],
                    block_num=block_num,
                    attention_block_num=attention_block_num[block_idx],
                    mlp_ratio=mlp_ratios[block_idx],
                    mlp_ratio_attn=mlp_ratio_attn,
                    act_layer=act_layer[block_idx],
                    use_layer_scale=use_layer_scale[block_idx],
                    layer_scale_init_value=layer_scale_init_value,
                    extra_act=extra_act[block_idx],
                    block_cls=block_cls[block_idx],
                )
            )
            if block_idx < len(block_nums) - 1:
                assert eval(down_cls[block_idx]) in [S2DDown], down_cls[
                    block_idx
                ]
                downsample_block.append(
                    eval(down_cls[block_idx])(
                        patch_size=2,
                        in_dim=embed_dims[block_idx],
                        out_dim=embed_dims[block_idx + 1],
                    )
                )
        self.stages = nn.ModuleList(stages)
        self.downsample_block = nn.ModuleList(downsample_block)

        if final_expand_channel in [0, None]:
            self.final_expand_layer = nn.Identity()
            self.norm = nn.BatchNorm2d(embed_dims[-1])
            last_channels = embed_dims[-1]
        else:
            self.final_expand_layer = ConvModule2d(
                embed_dims[-1],
                final_expand_channel,
                kernel_size=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(final_expand_channel),
                act_layer=eval(act_layer[-1])(),
            )
            last_channels = final_expand_channel

        if feature_mix_channel in [0, None]:
            self.feature_mix_layer = nn.Identity()
        else:
            self.feature_mix_layer = ConvModule2d(
                last_channels,
                feature_mix_channel,
                kernel_size=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(feature_mix_channel),
                act_layer=eval(act_layer[-1])(),
            )
            last_channels = feature_mix_channel

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = (
                nn.Linear(last_channels, num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        else:
            stage_norm = []
            for embed_dim in embed_dims:
                if self.stage_out_norm is True:
                    stage_norm.append(nn.BatchNorm2d(embed_dim))
                else:
                    stage_norm.append(nn.Identity())
            self.stage_norm = nn.ModuleList(stage_norm)

        self.up = hnn.Interpolate(
            scale_factor=2, mode="bilinear", recompute_scale_factor=True
        )
        self.quant = QuantStub() if quant_input else None
        self.dequant = DeQuantStub() if dequant_output else None

    def forward(self, x):
        if self.quant is not None:
            x = self.quant(x)
        if isinstance(x, Sequence) and len(x) == 1:
            x = x[0]

        x = self.patch_embed(x)
        outs = []
        for idx in range(len(self.stages)):
            x = self.stages[idx](x)
            if not self.include_top:
                x_normed = self.stage_norm[idx](x)
                if idx == 0:
                    outs.append(self.up(x_normed))
                outs.append(x_normed)
            if idx < len(self.stages) - 1:
                x = self.downsample_block[idx](x)

        if not self.include_top:
            return outs

        if self.final_expand_channel in [0, None]:
            x = self.norm(x)
        else:
            x = self.final_expand_layer(x)
        x = self.avgpool(x)
        x = self.feature_mix_layer(x)
        x = self.head(torch.flatten(x, 1))

        if self.dequant is not None:
            x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        if self.include_top:
            # disable output quantization for last quanti layer.
            self.head.qconfig = qconfig_manager.get_default_qat_out_qconfig()

    def fuse_model(self):
        for module in self.patch_embed:
            module.fuse_model()
        for block in self.downsample_block:
            block.fuse_model()
        for stage in self.stages:
            stage.fuse_model()
        if hasattr(self.final_expand_layer, "fuse_model"):
            self.final_expand_layer.fuse_model()
            block.fuse_model()
        if hasattr(self.feature_mix_layer, "fuse_model"):
            self.feature_mix_layer.fuse_model()
