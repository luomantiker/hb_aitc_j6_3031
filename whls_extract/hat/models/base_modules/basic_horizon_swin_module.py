# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Optional, Tuple, Union

import horizon_plugin_pytorch.nn.quantized as quantized
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.nn import LayerNorm as LayerNorm2d
from horizon_plugin_pytorch.nn.functional import window_partition as horizon_wp
from horizon_plugin_pytorch.nn.functional import window_reverse as horizon_wr
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.base_modules.basic_swin_module import DropPath
from hat.models.weight_init import trunc_normal
from hat.utils.model_helpers import fx_wrap

__all__ = [
    "MlpModule2d",
    "PatchEmbedding4d",
    "PatchMerging4d",
    "WindowAttention4d",
    "SwinBasicLayer4d",
    "BasicLayer4d",
]


class MlpModule2d(nn.Sequential):
    """A mlp block that bundles two fc layers.

    Fc layer is made of conv2d instead of linear module.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Number of hidden channels.
        out_channels: Number of output channels.
        act_layer: Activation layer. Default: None.
        drop_ratio: Dropout ratio of output. Default: 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: Optional[nn.Module] = None,
        drop_ratio: float = 0.0,
    ):
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        fc1 = nn.Conv2d(in_channels, hidden_channels, 1)
        fc2 = nn.Conv2d(hidden_channels, out_channels, 1)
        if drop_ratio > 0.0:
            drop1 = nn.Dropout2d(drop_ratio)
            drop2 = nn.Dropout2d(drop_ratio)
        else:
            drop1, drop2 = None, None
        layer_list = [fc1, act_layer, drop1, fc2, drop2]
        self.layer_list = [layer for layer in layer_list if layer is not None]
        super(MlpModule2d, self).__init__(*self.layer_list)


class PatchEmbedding4d(nn.Module):
    """Image to Patch Embedding of HorizonSwinTransformer.

    Args:
        patch_size: Patch token size. Default: 4.
        in_channels: Number of input image channels. Default: 3.
        embedding_dims: Number of linear projection output channels.
            Default: 96.
        norm_layer: Normalization layer. Default: None.
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 4,
        in_channels: int = 3,
        embedding_dims: int = 96,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embedding_dims, patch_size, stride=patch_size
        )
        self.norm_layer = norm_layer
        if norm_layer is not None:
            self.norm = norm_layer([embedding_dims, 1, 1], dim=1)
        else:
            self.norm = None

        self.in_chans = in_channels
        self.embed_dim = embedding_dims

        self.patch_size = (
            (patch_size, patch_size)
            if isinstance(patch_size, int)
            else patch_size
        )

    @fx_wrap()
    def forward_pad(self, x):
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(
                x, (0, self.patch_size[1] - W % self.patch_size[1], 0, 0)
            )
        if H % self.patch_size[0] != 0:
            x = F.pad(
                x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0])
            )
        return x

    def forward(self, x):
        # padding
        x = self.forward_pad(x)

        x = self.proj(x)  # B x C x H x W
        if self.norm is not None:
            x = self.norm(x)  # B x C x H x W
        return x


class PatchMerging4d(nn.Module):
    """Patch Merging Layer of HorizonSwinTransformer.

    Args:
        dim: Number of input channels.
        norm_layer: Normalization layer. Default: None.
    """

    def __init__(
        self,
        dim: int,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, 1, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer([4 * dim, 1, 1], dim=1)
        else:
            self.norm = None
        self.cat = quantized.FloatFunctional()

    @fx_wrap()
    def _forward_pad(self, x):
        # x: B C H W
        # padding
        H, W = x.size(2), x.size(3)
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, W % 2, 0, H % 2))
        return x

    def forward(self, x):
        x = self._forward_pad(x)

        x0 = x[:, :, 0::2, :][:, :, :, 0::2]  # B C H/2 W/2
        x1 = x[:, :, 1::2, :][:, :, :, 0::2]  # B C H/2 W/2
        x2 = x[:, :, 0::2, :][:, :, :, 1::2]  # B C H/2 W/2
        x3 = x[:, :, 1::2, :][:, :, :, 1::2]  # B C H/2 W/2
        x = self.cat.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2

        if self.norm is not None:
            x = self.norm(x)
        x = self.reduction(x)
        return x


class WindowAttention4d(nn.Module):
    """
    Windows based multi-head self attention module with relative position bias.

    It supports both of shifted and non-shifted window.

    Args:
        dim: Number of input channels.
        window_size: The height and width of the window.
        num_heads: Number of attention heads.
        qkv_bias:  If True, add a learnable bias to query, key, value.
            Default: True.
        qk_scale: Override default qk scale of head_dim ** -0.5 if set.
        attention_drop_ratio: Dropout ratio of attention weight. Default: 0.0.
        proj_drop_ratio: Dropout ratio of output. Default: 0.0.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attention_drop_ratio: float = 0.0,
        proj_drop_ratio: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        scale = qk_scale or head_dim ** -0.5
        self.register_buffer("scale", torch.Tensor([scale]))

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads
            )
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window # noqa E501
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += (
            self.window_size[0] - 1
        )  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer(
            "relative_position_index", relative_position_index
        )

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attention_drop = nn.Dropout2d(attention_drop_ratio)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout2d(proj_drop_ratio)

        trunc_normal(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

        self.mul = quantized.FloatFunctional()
        self.matmul = quantized.FloatFunctional()
        self.add = quantized.FloatFunctional()
        self.mask_add = quantized.FloatFunctional()
        self.attn_matmul = quantized.FloatFunctional()
        self.quant = QuantStub(scale=None)
        self.scale_quant = QuantStub()

    @fx_wrap()
    def _get_shape(self, x):
        B, C, H, W = x.shape
        N = H * W
        return B, C, H, W, N

    @fx_wrap()
    def _gen_qkv(self, qkv, B, C, H, W):
        # new 4dims
        qkv = qkv.permute(0, 2, 3, 1)  # BxHxWx3C
        q, k, v = qkv[..., :C], qkv[..., C : C * 2], qkv[..., C * 2 :]  # BHWC
        q = q.reshape(B, H * W, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)
        k = k.reshape(B, H * W, self.num_heads, C // self.num_heads)
        k = k.permute(0, 2, 1, 3)
        v = v.reshape(B, H * W, self.num_heads, C // self.num_heads)
        v = v.permute(0, 2, 1, 3)
        return q, k, v

    @fx_wrap()
    def _merge_mask(self, attention, mask, B):
        if mask is not None:
            nW = mask.shape[0]
            mask = mask.repeat(B // nW, 1, 1, 1)
            attention = self.mask_add.add(attention, mask)
        return attention

    def forward(self, x, mask=None):
        B, C, H, W, N = self._get_shape(x)

        qkv = self.qkv(x)

        # new 4dims
        q, k, v = self._gen_qkv(qkv, B, C, H, W)

        scale = self.scale_quant(self.scale)
        q = self.mul.mul(q, scale)
        attention = self.matmul.matmul(q, k, x_trans=False, y_trans=True)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = (
            relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        )  # 1, nH, Wh*Ww, Wh*Ww
        relative_position_bias = self.quant(relative_position_bias)
        attention = self.add.add(attention, relative_position_bias)

        attention = self._merge_mask(attention, mask, B)
        attention = self.softmax(attention)

        attention = self.attention_drop(attention)
        x = self.attn_matmul.matmul(attention, v)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C, 1)
        x = x.permute(0, 2, 1, 3)

        x = x.reshape(B, C, H, W).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBasicLayer4d(nn.Module):
    """Basic block of HorizonSwinTransformer.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size.
        shift_size: Shift size for SW-MSA. Default: 0.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: 4.0
        qkv_bias: If True, add a learnable bias to query, key, value.
            Default: True.
        qk_scale: Override default qk scale of head_dim ** -0.5 if set.
        drop: Dropout rate. Default: 0.0
        attn_drop: Attention dropout rate. Default: 0.0
        drop_path: Stochastic depth rate. Default: 0.0
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = LayerNorm2d([dim, 1, 1], dim=1)
        self.attn = WindowAttention4d(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attention_drop_ratio=attn_drop,
            proj_drop_ratio=drop,
        )

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = LayerNorm2d([dim, 1, 1], dim=1)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpModule2d(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=nn.GELU(),
            drop_ratio=drop,
        )

        self.H = None
        self.W = None

        self.add = quantized.FloatFunctional()
        self.add2 = quantized.FloatFunctional()
        self.attn_quant = QuantStub()

    @fx_wrap()
    def forward_pre(self, x, mask_matrix):
        _, C, H, W = x.shape

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        _, _, Hp, Wp = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        shifted_x = shifted_x.permute(0, 2, 3, 1)
        x_windows = horizon_wp(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.permute(0, 3, 1, 2)

        if attn_mask is not None:
            attn_mask = self.attn_quant(attn_mask)
        return x_windows, attn_mask, H, W, Hp, Wp, C, pad_r, pad_b

    @fx_wrap()
    def forward_post(self, attn_windows, H, W, Hp, Wp, C, pad_r, pad_b):

        # merge windows
        attn_windows = attn_windows.view(
            -1, C, self.window_size, self.window_size
        )
        attn_windows = attn_windows.permute(0, 2, 3, 1)
        shifted_x = horizon_wr(
            attn_windows, self.window_size, Hp, Wp
        )  # B H' W' C
        shifted_x = shifted_x.permute(0, 3, 1, 2)  # B C H' W'

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(2, 3),
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()
        return x

    def forward(self, x, mask_matrix):
        shortcut = x
        x = self.norm1(x)  # B, C, H, W

        x_windows, attn_mask, H, W, Hp, Wp, C, pad_r, pad_b = self.forward_pre(
            x, mask_matrix
        )

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=attn_mask
        )  # nW*B, C, window_size, window_size,

        x = self.forward_post(attn_windows, H, W, Hp, Wp, C, pad_r, pad_b)

        # FFN
        x = self.add.add(shortcut, self.drop_path(x))
        x = self.add2.add(x, self.drop_path(self.mlp(self.norm2(x))))
        return x


class BasicLayer4d(nn.Module):
    """A basic layer for one stage of HorizonSwinTransformer.

    Args:
        dim: Number of feature channels
        depth: Depths of this stage.
        num_heads: Number of attention head.
        window_size: Local window size. Default: 7.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias: If True, add a learnable bias to query, key, value.
        qk_scale: Override default qk scale of head_dim ** -0.5 if set.
        drop: Dropout rate. Default: 0.0.
        attn_drop: Attention dropout rate. Default: 0.0.
        drop_path: Stochastic depth rate. Default: 0.0.
        norm_layer: Normalization layer.
        downsample: Downsample layer at the end of the layer.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Optional[nn.Module] = None,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinBasicLayer4d(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    @fx_wrap()
    def _gen_mask(self, x):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, C, H, W).
            H, W: Spatial resolution of the input feature.
        """
        H, W = x.size(2), x.size(3)

        # calculate attention mask for SW-MSA
        if self.shift_size > 0:
            Hp = int(np.ceil(H / self.window_size)) * self.window_size
            Wp = int(np.ceil(W / self.window_size)) * self.window_size
            img_mask = torch.zeros(
                (1, 1, Hp, Wp), device=x.device
            )  # 1 1 Hp Wp
            h_slices = (
                slice(0, Hp - self.window_size),
                slice(Hp - self.window_size, Hp - self.shift_size),
                slice(Hp - self.shift_size, None),
            )
            w_slices = (
                slice(0, Wp - self.window_size),
                slice(Wp - self.window_size, Wp - self.shift_size),
                slice(Wp - self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, :, h, w] = cnt
                    cnt += 1

            img_mask = img_mask.permute(0, 2, 3, 1)
            mask_windows = horizon_wp(img_mask, self.window_size)
            mask_windows = mask_windows.permute(0, 3, 1, 2)

            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
            attn_mask = attn_mask.unsqueeze(1)
        else:
            attn_mask = None

        return attn_mask

    def forward(self, x):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, C, H, W).
            H, W: Spatial resolution of the input feature.
        """

        attn_mask = self._gen_mask(x)

        for blk in self.blocks:
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x
