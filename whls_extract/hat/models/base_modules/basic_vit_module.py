import collections
from typing import Callable, Optional

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.base_modules.basic_swin_module import DropPath
from hat.models.base_modules.mlp_module import MlpModule2d as Mlp

__all__ = ["PatchEmbed", "Attention", "LayerScale", "ViTBlock", "ResPostBlock"]


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return (x, x)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple(
                [s // p for s, p in zip(self.img_size, self.patch_size)]
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        # flatten spatial dim and transpose to channels last,
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention module.

    Args:
        dim: Dimensionality of input and output tensor.
        num_heads: Number of attention heads.
        qkv_bias: If True, include bias to query, key,
                and value projections.
        qk_norm: If True, apply normalization to query and key.
        attn_drop: Dropout probability applied to attention weights.
        proj_drop: Dropout probability applied to output projections.
        norm_layer: Normalization layer applied to the attention scores.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.mul_scale = FloatFunctional()

    def get_xshape(self, x: torch.Tensor):
        B, N, C = x.shape
        return B, N, C, x

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.split(1, 0)
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)  # qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = self.mul_scale.mul_scalar(q, self.scale)
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.quant = QuantStub()
        self.mul = FloatFunctional()

    def forward(self, x):
        gamma = self.quant(self.gamma)
        return self.mul.mul(x, gamma)


class ViTBlock(nn.Module):
    """Vision Transformer Block, consisting of attention and MLP layers.

    Args:
        dim: Dimensionality of input and output tensor.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden layer dimension to input dimension.
        qkv_bias: If True, include bias to query, key,
                    and value projections.
        qk_norm: If True, apply normalization to query and key.
        proj_drop: Dropout probability applied to projections.
        attn_drop: Dropout probability applied to attention weights.
        init_values: Initial value for LayerScale,
                if None, uses identity function.
        drop_path: Dropout probability applied to the attention
                and MLP pathways.
        act_layer: Activation function applied to the MLP hidden layer.
        norm_layer: Normalization layer applied after the attention
                and MLP layers.
        mlp_layer: MLP layer architecture.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float = None,
        drop_path: float = 0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        mlp_layer: Callable = Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.drop_path1 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_channels=dim,
            hidden_channels=int(dim * mlp_ratio),
            act_layer=act_layer(),
            drop_ratio=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.add1 = FloatFunctional()
        self.add2 = FloatFunctional()

    def forward(self, x: torch.Tensor):
        x = self.add1.add(
            x, self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        )
        x = self.add2.add(
            x, self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        )
        return x


class ResPostBlock(nn.Module):
    """Residual Post-attention Block in a Transformer architecture.

    Args:
        dim: Dimensionality of input and output tensor.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden layer dimension to input dimension.
        qkv_bias: If True, include bias to query, key,
                and value projections.
        qk_norm: If True, apply normalization to query and key.
        proj_drop: Dropout probability applied to projections.
        attn_drop: Dropout probability applied to attention weights.
        init_values: Initial value for LayerNorm weights,
                if None, uses default initialization.
        drop_path: Dropout probability applied to the attention
                and MLP pathways.
        act_layer: Activation function applied to the MLP hidden layer.
        norm_layer: Normalization layer applied after the attention
                and MLP layers.
        mlp_layer: MLP layer architecture.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float = None,
        drop_path: float = 0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        mlp_layer: Callable = Mlp,
    ):
        super().__init__()
        self.init_values = init_values

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        self.mlp = mlp_layer(
            in_channels=dim,
            hidden_channels=int(dim * mlp_ratio),
            act_layer=act_layer(),
            drop_ratio=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.add1 = FloatFunctional()
        self.add2 = FloatFunctional()
        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes
        # for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x: torch.Tensor):
        x = self.add1.add(x, self.drop_path1(self.norm1(self.attn(x))))
        x = self.add2.add(x, self.drop_path2(self.norm2(self.mlp(x))))
        return x
