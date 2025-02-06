# Copyright (c) Horizon Robotics, All rights reserved.

import math
from typing import Optional, Tuple

import horizon_plugin_pytorch as hopp
import torch
from torch import nn

from hat.models.base_modules.conv_module import ConvModule2d


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention.

    基于 ConvModule2d 构造的 MultiHeadAttention.

    Args:
        n_head: 多头注意力模块的头数.
        n_feat: 总共有多少维输出, 要求 n_feat % n_head 可整除.
        dropout_rate: 对 attention map进行dropout的比例. Defaults to 0.
        neg_inf: 指定作为-inf 的实际数值，默认为 -float("inf")
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float = 0,
        scores_fill_value=-float("inf"),
    ):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.head = n_head
        self.denom = 1.0 / math.sqrt(self.d_k)
        self.scores_fill_value = scores_fill_value
        # query key value 变换线性层
        self.q_conv = ConvModule2d(
            n_feat,
            n_feat,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )
        self.k_conv = ConvModule2d(
            n_feat,
            n_feat,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )
        self.v_conv = ConvModule2d(
            n_feat,
            n_feat,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )
        # 特征输出层
        self.out_conv = ConvModule2d(
            n_feat,
            n_feat,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )
        self.softmax = nn.Softmax(dim=2)
        # Q * K^T 矩阵运算辅助算子
        self.score_matmul = hopp.nn.quantized.FloatFunctional()
        # sqrt(d_k) 辅助算子
        self.div = hopp.nn.quantized.FloatFunctional()
        # attention dropout
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        # score * V 矩阵运算辅助算子
        self.att_matmul = hopp.nn.quantized.FloatFunctional()

        self.cat1 = hopp.nn.quantized.FloatFunctional()
        self.cat2 = hopp.nn.quantized.FloatFunctional()
        self.cat3 = hopp.nn.quantized.FloatFunctional()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        cache: torch.Tensor = None,
    ):
        """Multihead Attention 训练过程中前向的过程.

        Args:
            query: (B, C, 1, T1).
            key: (B, C, 1, T2)
            value: (B, C, 1, T2)
            mask: (B, C, 1, T2) or (B, C, T1, T2)

        Returns:
            经过计算得到的特征矩阵. (B, C, 1, T1)
        """

        # 计算 Q, K, V [B, H, Dk, T]
        query, key, value, new_cache = self._forward_qkv(
            query, key, value, cache
        )  # [B, H, C, T]

        # 计算 scores
        scores = self._calculate_scores(query, key)
        if mask is not None and mask.size(0) > 0:  # training mode
            # 根据 mask 整理 attention 的数值
            # 因为 scores 是转置, 所以 mask 也要转置
            mask = mask.permute((0, 1, 3, 2)).eq(0).detach()
            scores = scores.masked_fill(mask, self.scores_fill_value)
            attn = self.softmax(scores).masked_fill(mask, 0.0)
        else:  # eval mode
            attn = self.softmax(scores)

        # 进行 attention 计算
        x = self._forward_attn(attn, value)

        # 输出层
        out = self.out_conv(x)
        return out, new_cache

    def _forward_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """得到 query, key 和 value 的矩阵.

        Args:
            query: 用于计算query的特征矩阵
            key: 用于计算key的特征矩阵
            value: 用于计算value的特征矩阵

        Returns:
            经过计算和变形后的 query, key, value 矩阵(实际上是他们的转置).
            (B, H, Dk, T)
        """  # Forward Q K V
        n_batch, _, _, _ = query.size()
        # b, h, dk, t1
        query = self.q_conv(query).view(n_batch, self.head, self.d_k, -1)
        # b, h, dk, t2
        key = self.k_conv(key)  # b, c, 1, t1
        value = self.v_conv(value)  # b, c, 1, t2
        if cache is not None and cache.size(1) > 1:
            # caceh: b, c, 1, t      c = 64*2*att_head
            k_cache, v_cache = torch.split(cache, cache.size(1) // 2, dim=1)
            # b c 1 t1
            key = self.cat1.cat((k_cache, key), dim=3)
            value = self.cat2.cat((v_cache, value), dim=3)
        else:
            _ = self.cat1.cat((key, key), dim=3)
            _ = self.cat2.cat((value, value), dim=3)

        new_cache = self.cat3.cat((key, value), dim=1)  # b, c, 1, t

        key = key.view(n_batch, self.head, self.d_k, -1)  # b, h, c, t
        value = value.view(n_batch, self.head, self.d_k, -1)  # b, h, c, t
        return query, key, value, new_cache

    def _calculate_scores(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        r"""计算query和key矩阵乘的得分.

        .. math::

            Q \cdot K^T == K \cdot Q^T

        Args:
            query: query的转置, (B, H, Dk, T1)
            key: key的转置, (B, H, Dk, T2)

        Returns:
            计算后的得分, (B, H, T2, T1)
        """
        key = key.transpose(2, 3)
        scores = self.score_matmul.matmul(key, query)
        scores = self.div.mul_scalar(scores, self.denom)
        return scores

    def _forward_attn(self, attn, value):
        r"""将注意力map应用到value上.

        .. math::

        attention \cdot V = V^T \cdot attention^T

        Args:
            attn: 注意力map的转置
            value: value矩阵的转置

        Returns:
            通过注意力得到的特征. (B, h*d_k, 1, T2)
        """
        n_batch = value.size(0)
        p_attn = self.dropout(attn)
        # b, h, d_k, t1
        x = self.att_matmul.matmul(value, p_attn)
        x = x.view(n_batch, self.d_k * self.head, 1, -1)
        return x

    def trace(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        """trace.

        在进行保存 torch.jit.trace 时, 不使用 mask 操作.

        Args:
            query: (B, C, 1, T1), 用来当做 query 的特征.
            key: (B, C, 1, T2), 用来当做 key 的特征.
            value: (B, C, 1, T2), 用来当做 value 的特征.

        Returns:
            经过计算得到的特征矩阵. (B, C, 1, T1)
        """
        query, key, value = self._forward_qkv(query, key, value)
        scores = self._calculate_scores(query, key)
        attn = self.softmax(scores)
        x = self._forward_attn(attn, value)
        out = self.out_conv(x)
        return out

    def fuse_model(self):
        self.q_conv.fuse_model()
        self.k_conv.fuse_model()
        self.v_conv.fuse_model()
        self.out_conv.fuse_model()


class PositionwiseFeedForward(nn.Module):
    """Positionwise Feed Forward层.

    前向线性运算, 两层网络结构, 输入维度和输出维度相同. 基于ConvModule2d实现.

    Args:
        idim: 输入维度.
        hidden_dim: 中间输出的维度.
        dropout_rate: Dropout比例. 默认是 0.
        activation (torch.nn.Module): Activation function

    """

    def __init__(
        self,
        idim: int,
        hidden_dim: int,
        activation: nn.Module,
        dropout_rate: float = 0,
    ):
        """Construct a PositionwiseFeedForward object."""
        super().__init__()

        self.conv_1 = ConvModule2d(
            idim,
            hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=None,
            act_layer=activation,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.conv_2 = ConvModule2d(
            hidden_dim,
            idim,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=None,
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """训练过程中, feed foward网络前向的过程.

        Args:
            xs: 输入特征举证. (B, C, 1, T)

        Returns:
            输出矩阵, (B, C, 1, T)

        """
        xs = self.conv_1(xs)
        xs = self.dropout(xs)
        xs = self.conv_2(xs)
        return xs

    def fuse_model(self):
        self.conv_1.fuse_model()
        self.conv_2.fuse_model()


class CausalConvolutionModule(nn.Module):
    """Conformer单元中的ConvolutionModule.

    采用推理卷积(Causal Convolution)方式实现.

    Args:
        in_channels: 输入通道数.
        kernel_size: 中间层的kernel size大小
        activation: 激活层.
        bias: 卷积层是否使用偏置项. 默认为 True.
        bn_kwargs: BatchNorm 的超参dict, key 包含 eps, momentum, affine 等.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        activation: Optional[nn.Module],
        bias: bool = True,
    ):
        super().__init__()

        self.pointwise_conv1 = ConvModule2d(
            in_channels,
            in_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            norm_layer=None,
            act_layer=None,
        )

        self.lorder = kernel_size - 1

        self.depthwise_conv = ConvModule2d(
            in_channels,
            in_channels,
            (1, kernel_size),
            stride=1,
            padding=0,
            groups=in_channels,
            bias=bias,
            norm_layer=None,
            act_layer=None,
        )

        self.norm = hopp.nn.layer_norm.LayerNorm((in_channels, -1, -1), dim=1)
        self.activation = activation
        self.glu = torch.nn.GLU(dim=1)

        self.pointwise_conv2 = ConvModule2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            norm_layer=None,
        )

        self.cat = hopp.nn.quantized.FloatFunctional()

    def forward(
        self, x: torch.Tensor, mask_pad: torch.Tensor, cache
    ) -> torch.Tensor:
        # ! pytorch_horizon_plugin 中的 cat 算子需要被训练, 不能单独在预测使用
        # ! 在 trace 过程中, 需要 cat(cache, x)
        # ! 所以这里手动cat一个zero向量, 保证 cat 得到训练
        b, c, _, t = x.size()
        if mask_pad is not None:
            if len(mask_pad.size()) == 3:
                mask_pad = mask_pad.unsqueeze(1)
            x = x.masked_fill(mask_pad.eq(0), 0.0)  # B 384 1 T
        if cache is None or cache.size(-1) == 0:
            cache = torch.zeros(
                (b, c, 1, self.lorder), dtype=torch.float32, device=x.device
            )
        x = self.cat.cat((cache, x), dim=3)
        new_cache = x
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        if mask_pad is not None:
            x = x.masked_fill(mask_pad.eq(0), 0.0)
        x = x.masked_fill(mask_pad.eq(0), 0.0)
        return x, new_cache

    def trace(
        self, x: torch.Tensor, cache: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache is None:
            x = torch.nn.functional.pad(x, [6, 0], mode="constant", value=0)
        else:
            x = self.cat.cat((cache, x), dim=3)
        new_cache = x
        x = self.pointwise_conv1(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv2(x)
        return x, new_cache

    def fuse_model(self):
        self.pointwise_conv1.fuse_model()
        self.depthwise_conv.fuse_model()
        self.pointwise_conv2.fuse_model()
