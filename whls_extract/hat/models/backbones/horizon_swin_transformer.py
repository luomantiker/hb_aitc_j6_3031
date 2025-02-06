# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import LayerNorm as LayerNorm2d
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from hat.models.base_modules.basic_horizon_swin_module import (
    BasicLayer4d,
    PatchEmbedding4d,
    PatchMerging4d,
)
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.weight_init import trunc_normal
from hat.registry import OBJECT_REGISTRY

__all__ = ["HorizonSwinTransformer"]


@OBJECT_REGISTRY.register
class HorizonSwinTransformer(nn.Module):
    """A module of adjusted swin transformer, running faster on bpu.

    Args:
        depth_list: Depths of each Swin Transformer stage.
            for swin_T, the numbers could be [2, 2, 6, 2].
            for swin_S, swin_B, or swin_L, the numbers could be [2, 2, 18, 2].
        num_heads: Number of attention head of each stage.
            for swin_T or swin_S, the numbers could be [3, 6, 12, 24].
            for swin_B, the numbers could be [4, 8, 16, 32].
            for swin_L, the numbers could be [6, 12, 24, 48].
        num_classes: Num classes of output layer.
        patch_size: Patch size. Default: 4.
        in_channels: Number of input image channels. Default: 3.
        embedding_dims: Number of linear projection output channels.
            for swin_T or swin_S, the numbers could be 96.
            for swin_B, the number could be 128.
            for swin_L, the number could be 192.
        window_size: Window size. Default: 7.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias: Whether to add a learnable bias to query, key, value.
            Default: True.
        qk_scale: Override default qk scale of head_dim ** -0.5 if set.
        dropout_ratio: Dropout rate. Default: 0.
        attention_dropout_ratio: Attention dropout rate. Default: 0.
        drop_path_ratio: Stochastic depth rate. Default: 0.
        patch_norm: Whether to add normalization after patch embedding.
            Default: True.
        out_indices: Output from which stages.
        frozen_stages: Stages to be frozen (stop grad and set eval mode).
            Default: -1. -1 means not freezing any parameters.
        include_top: Whether to include output layer. Default: True.
        flat_output: Whether to view the output tensor. Default: True.
    """

    def __init__(
        self,
        depth_list: List[int],
        num_heads: List[int],
        num_classes: int = 1000,
        patch_size: Union[int, Tuple[int, int]] = 4,
        in_channels: int = 3,
        embedding_dims: int = 96,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        dropout_ratio: float = 0.0,
        attention_dropout_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        patch_norm: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        include_top: bool = True,
        flat_output: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.depth_list = depth_list
        self.num_layers = len(depth_list)
        self.embedding_dims = embedding_dims
        self.window_size = window_size

        self.num_features = int(embedding_dims * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.dropout_ratio = dropout_ratio
        self.attention_drop_ratio = attention_dropout_ratio
        self.include_top = include_top
        self.flat_output = flat_output
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.quant = QuantStub(scale=1.0 / 128.0)
        self.dequant = DeQuantStub()

        # split image into non-overlapping patches
        self.patch_embedding = PatchEmbedding4d(
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dims=embedding_dims,
            norm_layer=LayerNorm2d if patch_norm else None,
        )

        self.pos_drop = nn.Dropout2d(p=dropout_ratio)
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_ratio, sum(depth_list))
        ]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer4d(
                dim=int(embedding_dims * 2 ** i_layer),
                depth=depth_list[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dropout_ratio,
                attn_drop=attention_dropout_ratio,
                drop_path=dpr[
                    sum(depth_list[:i_layer]) : sum(depth_list[: i_layer + 1])
                ],
                norm_layer=LayerNorm2d,
                downsample=PatchMerging4d
                if (i_layer < self.num_layers - 1)
                else None,
            )
            self.layers.append(layer)

        num_features = [
            int(embedding_dims * 2 ** i) for i in range(self.num_layers)
        ]
        self.num_features = num_features
        # add a norm layer for each output
        for i_layer in out_indices:
            layer = LayerNorm2d([num_features[i_layer], 1, 1], dim=1)
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        if self.include_top:
            self.norm = LayerNorm2d([self.num_features[-1], 1, 1], dim=1)
            self.output = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                ConvModule2d(self.num_features[-1], num_classes, 1),
            )
        else:
            self.norm, self.output = None, None

        self._freeze_stages()
        self.apply(self.init_weights)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, m):
        """Initialize the weights in backbone."""

        if isinstance(m, nn.Linear):
            trunc_normal(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.quant(x)
        x = self.patch_embedding(x)
        x = self.pos_drop(x)  # B x C x H x W

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)
                outs.append(x_out)

        if not self.include_top:
            return outs
        x = self.norm(x)
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def fuse_model(self):
        pass

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.include_top:
            # disable output quantization for last quanti layer.
            getattr(
                self.output, "1"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()
