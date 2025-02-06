# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional

import torch

try:
    from horizon_plugin_pytorch.nn import ChannelScale2d
except ImportError:
    ChannelScale2d = None
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn

from .conv_module import ConvModule2d

__all__ = [
    "S2DDown",
    "DWCB",
    "GroupDWCB",
    "AltDWCB",
    "AttentionBlock",
    "LinearAttention",
    "BasicHENetStageBlock",
]


class S2DDown(nn.Module):
    """
    A module that performs a spatial-to-deep downsampling operation.

    Args:
        in_dim: The number of input channels.
        out_dim: The number of output channels.
        patch_size: The size of the patch for the convolution operation.
        padding: Optional padding for the convolution operation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        patch_size: int,
        padding: Optional[int] = 0,
    ):
        super().__init__()

        # Equivalent operation that merges downsampling and pointwise conv.
        self.proj = ConvModule2d(
            in_dim,
            out_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=padding,
            norm_layer=nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.proj(x)
        return x

    def fuse_model(self):
        self.proj.fuse_model()


class DWCB(nn.Module):
    """
    A Depthwise Convolutional Block with pointwise convolutions.

    Args:
        dim: The number of input and output channels.
        hidden_dim: The number of hidden channels for the pointwise
            convolution.
        kernel_size: The size of the kernel for the depthwise convolution.
        act_layer: A string representing the activation layer to be used.
        use_layer_scale: A boolean indicating if layer scale is to be used.
        extra_act: An optional activation layer after the block's main path.
        block_idx: An optional index for the block, used for alternate
            convolution patterns.
    """

    def __init__(  # noqa: D205
        self,
        dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        act_layer: str = "nn.GELU",
        use_layer_scale: bool = True,
        extra_act: Optional[bool] = False,
        block_idx: Optional[int] = 0,
    ):
        super().__init__()

        self.extra_act = (
            eval(act_layer)() if extra_act is True else nn.Identity()
        )

        self.dwconv = ConvModule2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            norm_layer=nn.BatchNorm2d(dim),
        )
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = eval(act_layer)()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            assert (
                ChannelScale2d is not None
            ), "require horizon_plugin_pytorch>=2.4.4"
            self.layer_scale = ChannelScale2d(dim)

        self.add = FF()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = self.add.add(input, self.layer_scale(x))
        else:
            x = self.add.add(input, x)
        x = self.extra_act(x)
        return x

    def fuse_model(self):
        self.dwconv.fuse_model()


class GroupDWCB(nn.Module):
    """
    A Depthwise Convolutional Block with grouped pointwise convolutions.

    Args:
        dim: The number of input and output channels.
        hidden_dim: The number of hidden channels for the pointwise
            convolution.
        kernel_size: The size of the kernel for the depthwise convolution.
        act_layer: A string representing the activation layer to be used.
        use_layer_scale: A boolean indicating if layer scale is to be used.
        extra_act: An optional activation layer after the block's main path.
        block_idx: An optional index for the block, used for alternate
            convolution patterns.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        act_layer: str = "nn.GELU",
        use_layer_scale: bool = True,
        extra_act: Optional[bool] = False,
        block_idx: Optional[int] = 0,
    ):
        super().__init__()

        self.extra_act = (
            eval(act_layer)() if extra_act is True else nn.Identity()
        )

        # Other settings are recommended to be validated first.
        group_width = {
            64: 64,
            128: 64,
            192: 64,
            384: 64,
            256: 128,
            48: 48,
            96: 48,
        }[dim]

        self.dwconv = ConvModule2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            norm_layer=nn.BatchNorm2d(dim),
        )
        self.pwconv1 = nn.Conv2d(
            dim, hidden_dim, kernel_size=1, groups=dim // group_width
        )
        self.act = eval(act_layer)()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            assert (
                ChannelScale2d is not None
            ), "require horizon_plugin_pytorch>=2.4.4"
            self.layer_scale = ChannelScale2d(dim)

        self.add = FF()
        self.add_dw = FF()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = self.add.add(input, self.layer_scale(x))
        else:
            x = self.add.add(input, x)
        x = self.extra_act(x)
        return x

    def fuse_model(self):
        self.dwconv.fuse_model()


class AltDWCB(nn.Module):
    """
    An Alternating Depthwise Convolutional Block with varying kernel sizes.

    Args:
        dim: The number of input and output channels.
        hidden_dim: The number of hidden channels for the pointwise
            convolution.
        kernel_size: The size of the kernel for the depthwise convolution.
        act_layer: A string representing the activation layer to be used.
        use_layer_scale: A boolean indicating if layer scale is to be used.
        extra_act: An optional activation layer after the block's main path.
        block_idx: An optional index for the block, used for determining the
            kernel size pattern.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        act_layer: str = "nn.GELU",
        use_layer_scale: bool = True,
        extra_act: Optional[bool] = False,
        block_idx: Optional[int] = 0,
    ):
        super().__init__()

        self.extra_act = (
            eval(act_layer)() if extra_act is True else nn.Identity()
        )

        # Alternate between (1, n) and (n, 1) convolutions.
        if block_idx % 2 == 0:
            dw_k, dw_p = (1, 5), (0, 2)
        else:
            dw_k, dw_p = (5, 1), (2, 0)

        self.dwconv = nn.Sequential(
            ConvModule2d(
                dim,
                dim,
                kernel_size=dw_k,
                padding=dw_p,
                groups=dim,
                norm_layer=nn.BatchNorm2d(dim),
            ),
        )

        self.pwconv1 = ConvModule2d(
            dim, hidden_dim, kernel_size=1, act_layer=eval(act_layer)()
        )
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            assert (
                ChannelScale2d is not None
            ), "require horizon_plugin_pytorch>=2.4.4"
            self.layer_scale = ChannelScale2d(dim)

        self.add = FF()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = self.add.add(input, self.layer_scale(x))
        else:
            x = self.add.add(input, x)
        x = self.extra_act(x)
        return x

    def fuse_model(self):
        if isinstance(self.dwconv, ConvModule2d):
            self.dwconv.fuse_model()
        else:
            for module in self.dwconv:
                module.fuse_model()
        self.pwconv1.fuse_model()


class ConvMlp(nn.Module):
    """
    A Convolutional MLP module for feature transformation.

    Args:
        dim: The number of input and output channels.
        hidden_dim: The number of hidden channels.
        act_layer: A string representing the activation layer to be used.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        act_layer: str = "nn.GELU",
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = eval(act_layer)()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class LinearAttention(nn.Module):
    """
    A Linear Attention module.

    Args:
        dim: The feature dimension.
        dim_head: The dimension of each attention head.
        num_heads: The number of attention heads.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        num_heads: int,
    ):
        super().__init__()
        self.query = nn.Linear(dim, dim_head * num_heads)
        self.key = nn.Linear(dim, dim_head * num_heads)
        self.query_weight = nn.Parameter(torch.randn(dim_head * num_heads, 1))
        self.scale = dim_head ** -0.5
        self.projection = nn.Linear(dim_head * num_heads, dim_head * num_heads)
        self.output = nn.Linear(dim_head * num_heads, dim_head)
        self.matmul = FF()
        self.mul_scalar = FF()
        self.sum = FF()
        self.add = FF()
        self.mul1 = FF()
        self.mul2 = FF()
        self.quant = QuantStub()

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        query_weight = self.matmul.matmul(query, self.quant(self.query_weight))
        query_weight = self.mul_scalar.mul_scalar(query_weight, self.scale)
        attn_weight = self.sum.sum(
            self.mul1.mul(query_weight, query), dim=1, keepdim=True
        )
        attn_weight = attn_weight.repeat(1, key.shape[1], 1)

        out = self.add.add(
            self.projection(self.mul2.mul(attn_weight, key)), query
        )

        out = self.output(out)

        return out


class AttentionBlock(nn.Module):
    """
    An Attention Block that applies a convolution, attention, and MLP.

    Args:
        dim: The number of input and output channels.
        mlp_ratio: The ratio of the hidden dimension to the input dimension
            for the MLP.
        act_layer: A string representing the activation layer to be used.
        use_layer_scale: A boolean indicating if layer scale is to be used.
        layer_scale_init_value: The initial value for the layer scale
            parameter.
        extra_act: An optional activation layer after the block's main path.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 2.0,
        act_layer: str = "nn.GELU",
        use_layer_scale: bool = True,
        layer_scale_init_value: Optional[float] = 1e-5,
        extra_act: Optional[bool] = False,
    ):
        super().__init__()

        self.conv = DWCB(
            dim=dim,
            hidden_dim=dim,
            kernel_size=3,
            act_layer=act_layer,
            use_layer_scale=True,
            extra_act=extra_act,
        )
        self.attn = LinearAttention(dim=dim, dim_head=dim, num_heads=1)
        self.mlp = ConvMlp(
            dim=dim, hidden_dim=int(dim * mlp_ratio), act_layer=act_layer
        )
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_attn = nn.Parameter(
                layer_scale_init_value
                * torch.ones(dim).unsqueeze(-1).unsqueeze(-1),
                requires_grad=True,
            )
            self.layer_scale_mlp = nn.Parameter(
                layer_scale_init_value
                * torch.ones(dim).unsqueeze(-1).unsqueeze(-1),
                requires_grad=True,
            )
            self.quant_attn = QuantStub()
            self.quant_mlp = QuantStub()
            self.mul_attn = FF()
            self.mul_mlp = FF()
        self.add_attn = FF()
        self.add_mlp = FF()

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        if self.use_layer_scale and isinstance(self.attn, LinearAttention):
            x = self.add_attn.add(
                x,
                self.mul_attn.mul(
                    self.quant_attn(self.layer_scale_attn),
                    self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C))
                    .reshape(B, H, W, C)
                    .permute(0, 3, 1, 2),
                ),
            )
            x = self.add_mlp.add(
                x,
                self.mul_mlp.mul(
                    self.quant_mlp(self.layer_scale_mlp), self.mlp(x)
                ),
            )
        elif self.use_layer_scale and isinstance(self.attn, DWCB):
            x = self.add_attn.add(
                x,
                self.mul_attn.mul(
                    self.quant_attn(self.layer_scale_attn), self.attn(x)
                ),
            )
            x = self.add_mlp.add(
                x,
                self.mul_mlp.mul(
                    self.quant_mlp(self.layer_scale_mlp), self.mlp(x)
                ),
            )
            x = self.add_mlp.add(
                x,
                self.mul_mlp.mul(
                    self.quant_mlp(self.layer_scale_mlp), self.mlp(x)
                ),
            )
        else:
            x = self.add_attn.add(
                x,
                self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C))
                .reshape(B, H, W, C)
                .permute(0, 3, 1, 2),
            )
            x = self.add_mlp.add(x, self.mlp(x))
        return x

    def fuse_model(self):
        self.conv.fuse_model()


class BasicHENetStageBlock(nn.Module):
    """
    A basic building block for the HENet stage.

    Args:
        in_dim: The number of input channels.
        block_num: The total number of blocks in the stage.
        attention_block_num: The number of attention blocks in the stage.
        mlp_ratio: The ratio of the hidden dimension to the input dimension
            for the MLP in the DWCB.
        mlp_ratio_attn: The ratio for the MLP in the Attention Block.
        act_layer: A string representing the activation layer to be used.
        use_layer_scale: A boolean indicating if layer scale is to be used.
        layer_scale_init_value: The initial value for the layer scale
            parameter.
        extra_act: An optional activation layer after the block's main path.
        block_cls: A string representing the class of the block to be used.
    """

    def __init__(
        self,
        in_dim: int,
        block_num: int,
        attention_block_num: int,
        mlp_ratio: float = 2.0,
        mlp_ratio_attn: float = 2.0,
        act_layer: str = "nn.GELU",
        use_layer_scale: bool = True,
        layer_scale_init_value: Optional[float] = 1e-5,
        extra_act: Optional[bool] = False,
        block_cls: str = "DWCB",
    ):
        super().__init__()
        block = []

        for block_idx in range(block_num - attention_block_num):
            block.append(
                eval(block_cls)(
                    dim=in_dim,
                    hidden_dim=int(mlp_ratio * in_dim),
                    kernel_size=3,
                    act_layer=act_layer,
                    use_layer_scale=use_layer_scale,
                    extra_act=extra_act,
                    block_idx=block_idx,
                )
            )
        for _ in range(attention_block_num):
            block.append(
                AttentionBlock(
                    dim=in_dim,
                    mlp_ratio=mlp_ratio_attn,
                    act_layer=act_layer,
                    use_layer_scale=True,
                    layer_scale_init_value=layer_scale_init_value,
                    extra_act=extra_act,
                )
            )

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

    def fuse_model(self):
        for m in self.block:
            m.fuse_model()
