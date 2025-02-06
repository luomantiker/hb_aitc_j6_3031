# Copyright (c) Horizon Robotics. All rights reserved.

import math
from typing import Optional, Tuple, Union

import horizon_plugin_pytorch.nn.quantized as quantized
import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.weight_init import trunc_normal
from hat.utils.model_helpers import fx_wrap
from .mlp_module import MlpModule2d

__all__ = [
    "DropPath",
    "PatchEmbedding",
    "PatchMerging",
    "WindowAttention",
    "SwinBasicLayer",
    "BasicLayer",
]


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device
    )
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Args:
        drop_prob: Stochastic depth rate.
        check_skip: Skip QTensor check when compiling
            models with trt and torchdynamo. Default: False.
    """

    def __init__(
        self, drop_prob: Optional[float] = None, check_skip: bool = False
    ):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.check_skip = check_skip

    @fx_wrap()
    def forward(self, x):
        if not self.check_skip:
            if isinstance(x, QTensor):
                return x
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding.

    Args:
        patch_size: Patch token size. Default: 4.
        in_channels: Number of input image channels. Default: 3.
        embedding_dims: Number of linear projection output channels.
            Default: 96.
        norm_layer: Normalization layer. Default: None
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
            self.norm = norm_layer(embedding_dims)
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
    def forward(self, x):
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(
                x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0])
            )

        x = self.proj(x)  # B x C x H x W
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2)
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer.

    Args:
        dim: Number of input channels.
        norm_layer: Normalization layer. Default: None.
        use_native_op: Use torch's native operators when compiling
            models with trt and torchdynamo. Default: False.
    """

    def __init__(
        self,
        dim: int,
        norm_layer: Optional[nn.Module] = None,
        use_native_op=False,
    ):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.cat = quantized.FloatFunctional()
        self.use_native_op = use_native_op

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(-1, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            H += H % 2
            W += W % 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        if self.use_native_op:
            x = torch.cat([x0, x1, x2, x3], -1)
        else:
            x = self.cat.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(-1, H * W // 4, 4 * C)  # B H/2*W/2 4*C

        if self.norm is not None:
            x = self.norm(x)
        x = self.reduction(x)
        return x


def window_partition(x, window_size):
    """Window patition.

    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        -1, H // window_size, window_size, W // window_size, window_size, C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """Windows reverse to 4dims.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    C = windows.shape[3]

    x = windows.view(
        -1, H // window_size, W // window_size, window_size, window_size, C
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


class WindowAttention(nn.Module):
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
        use_native_op: Use torch's native operators when compiling
            models with trt and torchdynamo. Default: False.
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
        use_native_op: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attention_drop = nn.Dropout(attention_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        trunc_normal(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

        self.mul = quantized.FloatFunctional()
        self.matmul = quantized.FloatFunctional()
        self.add = quantized.FloatFunctional()
        self.mask_add = quantized.FloatFunctional()
        self.attn_matmul = quantized.FloatFunctional()
        self.quant = QuantStub(scale=None)
        self.use_native_op = use_native_op

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(-1, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q.mul(self.scale)
        k = k.transpose(-2, -1).contiguous()
        if self.use_native_op:
            attention = torch.matmul(q, k)
        else:
            attention = self.matmul.matmul(q, k)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = relative_position_bias.unsqueeze(0)

        relative_position_bias = self.quant(relative_position_bias)
        if self.use_native_op:
            attention = torch.add(attention, relative_position_bias)
        else:
            attention = self.add.add(attention, relative_position_bias)
        # attention = attention + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # nW = mask.shape[0]
            attention = attention.view(-1, mask.shape[0], self.num_heads, N, N)
            # attention = attention + mask.unsqueeze(1).unsqueeze(0)
            if self.use_native_op:
                attention = torch.add(
                    attention, mask.unsqueeze(1).unsqueeze(0)
                )
            else:
                attention = self.mask_add.add(
                    attention, mask.unsqueeze(1).unsqueeze(0)
                )
            attention = attention.view(-1, self.num_heads, N, N)
            attention = self.softmax(attention)
        else:
            attention = self.softmax(attention)

        attention = self.attention_drop(attention)
        # x = (attention @ v).transpose(1, 2).reshape(B, N, C)
        if self.use_native_op:
            x = torch.matmul(attention, v)
        else:
            x = self.attn_matmul.matmul(attention, v)
        x = x.transpose(1, 2).reshape(-1, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self):
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"  # noqa E501


class SwinBasicLayer(nn.Module):
    """Basic swin transformer block.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size.
        shift_size: Shift size for SW-MSA.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: 4.0
        qkv_bias: If True, add a learnable bias to query, key, value.
            Default: True.
        qk_scale: Override default qk scale of head_dim ** -0.5 if set.
        drop: Dropout rate. Default: 0.0
        attn_drop: Attention dropout rate. Default: 0.0
        drop_path: Stochastic depth rate. Default: 0.0
        use_native_op: Use torch's native operators when compiling
            models with trt and torchdynamo. Default: False
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
        use_native_op: bool = False,
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

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attention_drop_ratio=attn_drop,
            proj_drop_ratio=drop,
            use_native_op=use_native_op,
        )

        self.drop_path = (
            DropPath(drop_path, check_skip=use_native_op)
            if drop_path > 0.0
            else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpModule2d(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=nn.GELU(),
            drop_ratio=drop,
        )

        self.add = quantized.FloatFunctional()
        self.add2 = quantized.FloatFunctional()
        self.attn_quant = QuantStub()
        self.use_native_op = use_native_op
        self.replace_roll = use_native_op

    def forward(self, x, mask_matrix, h, w):
        H, W = h, w

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(-1, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            if self.use_native_op:
                # dynamic_shape_pad_entension does not support padding
                # lengths greater than 4.
                x = x.permute(0, 3, 1, 2)
                x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
                x = x.permute(0, 2, 3, 1)
            else:
                x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            # replace roll with two cat equivalents for torchdynamo
            if self.replace_roll:
                shifted_x = torch.cat(
                    (
                        x[:, self.shift_size :, :, :],
                        x[:, : self.shift_size, :, :],
                    ),
                    dim=1,
                )
                shifted_x = torch.cat(
                    (
                        shifted_x[:, :, self.shift_size :, :],
                        shifted_x[:, :, : self.shift_size, :],
                    ),
                    dim=2,
                )
            else:
                shifted_x = torch.roll(
                    x,
                    shifts=(-self.shift_size, -self.shift_size),
                    dims=(1, 2),
                )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if attn_mask is not None:
            attn_mask = self.attn_quant(attn_mask)
        attn_windows = self.attn(
            x_windows, mask=attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp
        )  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            # To use torchdynamo, replace roll with two cat equivalents
            if self.replace_roll:
                shifted_x = torch.cat(
                    (
                        shifted_x[:, -self.shift_size :, :, :],
                        shifted_x[:, : -self.shift_size, :, :],
                    ),
                    dim=1,
                )
                x = torch.cat(
                    (
                        shifted_x[:, :, -self.shift_size :, :],
                        shifted_x[:, :, : -self.shift_size, :],
                    ),
                    dim=2,
                )
            else:
                x = torch.roll(
                    shifted_x,
                    shifts=(self.shift_size, self.shift_size),
                    dims=(1, 2),
                )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(-1, H * W, C)

        # FFN
        # x = shortcut + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.use_native_op:
            x = torch.add(shortcut, self.drop_path(x))
            x = torch.add(x, self.drop_path(self.mlp(self.norm2(x))))
        else:
            x = self.add.add(shortcut, self.drop_path(x))
            x = self.add2.add(x, self.drop_path(self.mlp(self.norm2(x))))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, \
                num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, \
                mlp_ratio={self.mlp_ratio}"
        )


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

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
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        downsample: Downsample layer at the end of the layer.
        use_native_op: Use torch's native operators when compiling
            models with trt and torchdynamo. Default: False
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
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        use_native_op: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinBasicLayer(
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
                    use_native_op=use_native_op,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                dim=dim,
                norm_layer=norm_layer,
                use_native_op=use_native_op,
            )
        else:
            self.downsample = None

    @fx_wrap()
    def forward(self, x, H, W):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(math.ceil(H / self.window_size)) * self.window_size
        Wp = int(math.ceil(W / self.window_size)) * self.window_size

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] += cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(
            -1, self.window_size * self.window_size
        )
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)
        ).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_mask, H, W)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W
