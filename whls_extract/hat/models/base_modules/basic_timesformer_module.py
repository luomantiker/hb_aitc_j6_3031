# Copyright (c) Horizon Robotics. All rights reserved.
import os
from collections import OrderedDict
from typing import Optional, Union

import horizon_plugin_pytorch.nn.quantized as quantized
import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.quantization import QuantStub

from .basic_swin_module import DropPath
from .mlp_module import MlpModule2d

__all__ = [
    "load_pretrained",
    "SpaceTimeBlock",
    "PatchEmbed",
]


def conv_filter(state_dict: Union[dict, OrderedDict], patch_size: int = 16):
    """Convert patch embedding weight from linear proj to conv."""

    out_dict = {}
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def load_state_dict(checkpoint_path: str, use_ema: bool = False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict_key = "state_dict"
        if isinstance(checkpoint, dict):
            if use_ema and "state_dict_ema" in checkpoint:
                state_dict_key = "state_dict_ema"
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith("module") else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif "model_state" in checkpoint:
            state_dict_key = "model_state"
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `model.` prefix
                name = k[6:] if k.startswith("model") else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        return state_dict
    else:
        raise FileNotFoundError()


def load_pretrained(
    model: nn.Module,
    num_classes: int = 1000,
    num_frames: int = 8,
    num_patches: int = 196,
    attention_type: str = "divided_space_time",
    pretrained_model: str = "",
    strict: bool = True,
):
    if len(pretrained_model) == 0:
        state_dict = {}
    else:
        try:
            state_dict = load_state_dict(pretrained_model)["model"]
        except Exception:
            state_dict = load_state_dict(pretrained_model)

    state_dict = conv_filter(state_dict)

    if num_classes != state_dict["head" + ".weight"].size(0):
        del state_dict["head" + ".weight"]
        del state_dict["head" + ".bias"]

    # Resizing the positional embeddings in case they don't match
    if num_patches + 1 != state_dict["pos_embed"].size(1):
        pos_embed = state_dict["pos_embed"]
        cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
        other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
        new_pos_embed = F.interpolate(
            other_pos_embed, size=(num_patches), mode="nearest"
        )
        new_pos_embed = new_pos_embed.transpose(1, 2)
        new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
        state_dict["pos_embed"] = new_pos_embed

    # Resizing time embeddings in case they don't match
    if "time_embed" in state_dict and num_frames != state_dict[
        "time_embed"
    ].size(1):
        time_embed = state_dict["time_embed"].transpose(1, 2)
        new_time_embed = F.interpolate(
            time_embed, size=(num_frames), mode="nearest"
        )
        state_dict["time_embed"] = new_time_embed.transpose(1, 2)

    # Initializing temporal attention
    if attention_type == "divided_space_time":
        new_state_dict = state_dict.copy()
        for key in state_dict:
            if "blocks" in key and "attn" in key:
                new_key = key.replace("attn", "temporal_attn")
                if new_key not in state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[new_key] = state_dict[new_key]
            if "blocks" in key and "norm1" in key:
                new_key = key.replace("norm1", "temporal_norm1")
                if new_key not in state_dict:
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[new_key] = state_dict[new_key]
        state_dict = new_state_dict

    new_state_dict = state_dict.copy()
    for key in state_dict:
        if "blocks" in key and "mlp.fc1" in key:
            new_key = key.replace("mlp.fc1", "mlp.0")
            new_state_dict[new_key] = state_dict[key]
            del new_state_dict[key]
        if "blocks" in key and "mlp.fc2" in key:
            new_key = key.replace("mlp.fc2", "mlp.2")
            new_state_dict[new_key] = state_dict[key]
            del new_state_dict[key]
    state_dict = new_state_dict

    # Loading the weights
    model.load_state_dict(state_dict, strict=False)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[int] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        with_qkv: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        scale = qk_scale or head_dim ** -0.5
        self.register_buffer("scale", torch.Tensor([scale]))
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.mul = quantized.FloatFunctional()
        self.matmul = quantized.FloatFunctional()
        self.attn_matmul = quantized.FloatFunctional()
        self.scale_quant = QuantStub()

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(
                0, 2, 1, 3
            )
            q, k, v = qkv, qkv, qkv
        k = k.permute(0, 1, 3, 2)
        scale = self.scale_quant(self.scale)
        attn = self.mul.mul(self.matmul.matmul(q, k), scale)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = self.attn_matmul.matmul(attn, v)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class SpaceTimeBlock(nn.Module):
    """Basic block of TimeSformer."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[int] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        attention_type: str = "divided_space_time",
    ):
        super().__init__()
        self.attention_type = attention_type
        assert attention_type in [
            "divided_space_time",
            "space_only",
            "joint_space_time",
        ]

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Temporal Attention Parameters
        if self.attention_type == "divided_space_time":
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.temporal_fc = nn.Linear(dim, dim)

        # drop path
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpModule2d(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer(),
            drop_ratio=drop,
        )

        self.temporal_add = quantized.FloatFunctional()
        self.spatial_cat = quantized.FloatFunctional()
        self.cls_token_mean = quantized.FloatFunctional()
        self.mlp_spatial_cat = quantized.FloatFunctional()
        self.mlp_temporal_cat = quantized.FloatFunctional()
        self.mlp_add1 = quantized.FloatFunctional()
        self.mlp_add2 = quantized.FloatFunctional()

    def forward(self, x, B, T, W):
        if self.attention_type in ["space_only", "joint_space_time"]:
            x = self.mlp_add1.add(x, self.drop_path(self.attn(self.norm1(x))))
            x = self.mlp_add2.add(x, self.drop_path(self.mlp(self.norm2(x))))
            return x
        elif self.attention_type == "divided_space_time":
            # Temporal
            xt = x[:, 1:, :]
            C = xt.shape[-1]
            xt = xt.reshape(-1, T, C)
            res_temporal = self.drop_path(
                self.temporal_attn(self.temporal_norm1(xt))
            )
            res_temporal = res_temporal.reshape(B, -1, C)
            res_temporal = self.temporal_fc(res_temporal)
            xt = self.temporal_add.add(x[:, 1:, :], res_temporal)

            # Spatial
            init_cls_token = x[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = cls_token.reshape(-1, C).unsqueeze(1)
            xs = xt
            xs = xs.reshape(B, -1, T, C).permute(0, 2, 1, 3)
            HW = xs.shape[2]
            xs = xs.reshape(-1, HW, C)
            xs = self.spatial_cat.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            # Taking care of CLS token
            cls_token = res_spatial[:, 0, :]
            cls_token = cls_token.reshape(B, T, C)
            cls_token = self.cls_token_mean.mean(
                cls_token, 1, keepdim=True
            )  # averaging for every frame
            res_spatial = res_spatial[:, 1:, :]
            res_spatial = res_spatial.reshape(B, T, -1, C).permute(0, 2, 1, 3)
            res_spatial = res_spatial.reshape(B, -1, C)
            res = res_spatial
            x = xt

            # Mlp
            x1 = self.mlp_temporal_cat.cat((init_cls_token, x), 1)
            x2 = self.mlp_spatial_cat.cat((cls_token, res), 1)
            x = self.mlp_add1.add(x1, x2)
            x = self.mlp_add2.add(x, self.drop_path(self.mlp(self.norm2(x))))
            return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        img_size: Input image size.
        path_size: Patch size.
        in_chans: Input channels.
        embed_dim: Embedded dimension.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        x = self.proj(x)
        N, C, H, W = x.shape
        x = x.reshape(N, C, -1).permute(0, 2, 1)
        return x, T, W
