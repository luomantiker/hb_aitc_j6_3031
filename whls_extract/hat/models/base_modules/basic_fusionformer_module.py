# Copyright (c) Horizon Robotics, All rights reserved.

import logging
import math
from copy import deepcopy
from typing import Optional, Tuple

import horizon_plugin_pytorch as hopp
import torch
from horizon_plugin_pytorch.qtensor import QTensor
from torch import nn

try:
    from horizon_plugin_pytorch import quantization

    fuser_func = quantization.fuse_known_modules
except Warning:
    logging.warning(
        "Please install horizon_plugin_pytorch first, otherwise use "
        "pytorch official quantification"
    )
    from torch.quantization.fuse_modules import fuse_known_modules

    fuser_func = fuse_known_modules


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer.

    eq: score(x_i, x_j) = e^(THETA(x_i)^T dot PHI(x_j))
        where THETA is linear_q and PHI is linear_k

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        scores_fill_value=-float("inf"),
    ):
        """Construct an MultiHeadAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1),
            nn.BatchNorm2d(n_feat),
        )
        self.linear_k = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1),
            nn.BatchNorm2d(n_feat),
        )
        self.linear_v = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1),
            nn.BatchNorm2d(n_feat),
        )
        self.linear_out = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1),
            nn.BatchNorm2d(n_feat),
        )
        self.activation = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        # 适配 horizon_pytorch_plugin 初始化为具体的数字
        self.denom = 1.0 / math.sqrt(self.d_k)
        self.scores_fill_value = scores_fill_value
        # attention dropout
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)

        # Q * K^T 矩阵运算辅助算子
        self.score_matmul = hopp.nn.quantized.FloatFunctional()
        # sqrt(d_k) 辅助算子
        self.div = hopp.nn.quantized.FloatFunctional()
        # score * V 矩阵运算辅助算子
        self.att_matmul = hopp.nn.quantized.FloatFunctional()
        # concat cache 辅助算子
        self.cat1 = hopp.nn.quantized.FloatFunctional()
        self.cat2 = hopp.nn.quantized.FloatFunctional()
        self.cat3 = hopp.nn.quantized.FloatFunctional()

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self,
            [["linear_q.0", "linear_q.1"]],
            inplace=True,
            fuser_func=fuser_func,
        )
        torch.quantization.fuse_modules(
            self,
            [["linear_k.0", "linear_k.1"]],
            inplace=True,
            fuser_func=fuser_func,
        )
        torch.quantization.fuse_modules(
            self,
            [["linear_v.0", "linear_v.1"]],
            inplace=True,
            fuser_func=fuser_func,
        )
        torch.quantization.fuse_modules(
            self,
            [["linear_out.0", "linear_out.1"]],
            inplace=True,
            fuser_func=fuser_func,
        )

    def forward_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        NOTE(xcsong): Input shape [N, C, H, W] is required by torch.Conv2d,
            the reason for choosing `W` as time-axis is to avoid unnecessary
            padding, see `width alignment` for more details:
            https://horizonrobotics.feishu.cn/docs/doccncwfEKITYRcQVA4Bx3nfPQf#9jhvRq  # noqa: E501

        Args:
            query (torch.Tensor): Query tensor (#batch, n_feat, 1, time1).
            key (torch.Tensor):   Key tensor   (#batch, n_feat, 1, time2).
            value (torch.Tensor): Value tensor (#batch, n_feat, 1, time2).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, head, time2, d_k).

        """
        n_batch, n_feat, _, time1 = query.size()
        _, _, _, time2 = key.size()
        q = self.linear_q(query)
        k = self.linear_k(key)  # .view(n_batch, self.h, self.d_k, time2)
        v = self.linear_v(value)  # .view(n_batch, self.h, self.d_k, time2)
        if cache is not None:
            k_cache, v_cache = torch.split(cache, cache.size(1) // 2, dim=1)
            k = self.cat1.cat((k_cache, k), dim=3)
            v = self.cat2.cat((v_cache, v), dim=3)
            time2 = time2 + k.size(3)
        else:
            _ = self.cat1.cat((key, key), dim=3)
            _ = self.cat2.cat((value, value), dim=3)

        new_cache = self.cat3.cat((key, value), dim=1)  # b, c, 1, t

        q = q.view(n_batch, self.h, self.d_k, time1)
        v = v.view(n_batch, self.h, self.d_k, time2)
        k = k.view(n_batch, self.h, self.d_k, time2)
        # q = q.transpose(2, 3)  # (batch, head, time1, d_k)
        # k = k.transpose(2, 3)  # (batch, head, time2, d_k)
        # v = v.transpose(2, 3)  # (batch, head, time2, d_k)
        return q, k, v, new_cache

    def caculate_scores(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention scores.

        Args:
            query (torch.Tensor): Query tensor (#batch, n_head, time1, d_k).
            key (torch.Tensor):   Key tensor   (#batch, n_head, time2, d_k).

        Returns:
            torch.Tensor: Score tensor (#batch, n_head, time1, time2).

        """
        key = key.transpose(2, 3)  # (batch, head, d_k, time2)
        scores = self.score_matmul.matmul(key, query)
        scores = self.div.mul_scalar(scores, self.denom)
        return scores

    def forward_attention(
        self, value: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, n_feat, 1, time1).
                weighted by the attention score (#batch, n_head, time1, time2).

        """
        n_batch, _, _, time1 = scores.size()
        if mask is not None and mask.size(0) > 0:  # training mode
            mask = mask.permute((0, 1, 3, 2)).eq(0).detach()
            scores = scores.masked_fill(mask, self.scores_fill_value)
            attn = self.activation(scores).masked_fill(mask, 0.0)
        else:  # eval mode
            attn = self.activation(scores)  # (batch, n_head, time1, time2)
        p_attn = self.dropout(attn)
        x = self.att_matmul.matmul(value, p_attn)
        # B, head, T1, d_k -> B head*d_k, 1, T1
        x = x.view(n_batch, self.d_k * self.h, 1, time1)
        # (batch, n_feat, 1, time1)
        return self.linear_out(x)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, n_feat, 1, time1).
            key (torch.Tensor):   Key tensor   (#batch, n_feat, 1, time2).
            value (torch.Tensor): Value tensor (#batch, n_feat, 1, time2).
            mask (torch.Tensor):  Mask tensor  (#batch, 1, time1, time2)
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2).

        Returns:
            torch.Tensor: Output tensor (#batch, n_feat, 1, time1).
            torch.Tensor: Cache tensor (1, head, cache_t + time2, d_k * 2).

        """
        # Step-1: forward qkv
        q, k, v, new_cache = self.forward_qkv(query, key, value, cache)

        # Step-2: calculate scores, (#batch, n_head, time1, time2)
        scores = self.caculate_scores(q, k)

        # Step-3: forward attention, (#batch, n_feat, 1, time1)
        x = self.forward_attention(v, scores, mask)
        return x, new_cache


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer. fully-convolutional version.

    FeedForward are applied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function.

    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),  # noqa: B008
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()

        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(idim, hidden_units, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(hidden_units),
            activation,
            torch.nn.Dropout(dropout_rate),
        )

        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_units, idim, kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(idim),
        )

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self,
            [["conv_1.0", "conv_1.1", "conv_1.2"]],
            inplace=True,
            fuser_func=fuser_func,
        )
        torch.quantization.fuse_modules(
            self,
            [["conv_2.0", "conv_2.1"]],
            inplace=True,
            fuser_func=fuser_func,
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, D, 1, T)

        Returns:
            output tensor, (B, D, 1, T)

        """
        return self.conv_2(self.conv_1(xs))


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model. Conv2d version."""

    def __init__(
        self,
        in_channels: int,
        inner_channels: int,
        kernel_size: int = 7,
        activation: nn.Module = nn.ReLU(),  # noqa: B008
        causal: bool = False,
        bias: bool = True,
    ):
        """Construct an ConvolutionModule object.

        Args:
            in_channels (int): The number of input channels.
            inner_channels (int): Inner channels of depthwise_conv.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not

        """
        super().__init__()

        self.pointwise_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inner_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.BatchNorm2d(inner_channels),
            deepcopy(activation),
        )

        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
            self.pad = nn.ConstantPad2d(
                padding=(self.lorder, 0, 0, 0),
                value=0.0,
            )
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
            self.pad = nn.Identity()

        # TODO(xcsong): Remove activation in dw_conv may be beneficial,
        #               ref: Xception, section4.7.
        # NOTE(xcsong): Different from wenet.transformer.convolution, we use
        #               ReLu instead of GLU to activate the output of pw_conv1,
        #               thus the input-channels of dw_conv should be
        #               2*in_channels (aka inner_channels).
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                inner_channels,
                inner_channels,
                (1, kernel_size),
                stride=1,
                padding=(0, padding),
                groups=inner_channels,
                bias=bias,
            ),
            nn.BatchNorm2d(inner_channels),
            deepcopy(activation),
        )

        self.pointwise_conv2 = nn.Sequential(
            nn.Conv2d(
                inner_channels,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.BatchNorm2d(in_channels),
        )

        # 辅助算子
        self.cat = hopp.nn.quantized.FloatFunctional()

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self,
            [["pointwise_conv1.0", "pointwise_conv1.1", "pointwise_conv1.2"]],
            inplace=True,
            fuser_func=fuser_func,
        )
        torch.quantization.fuse_modules(
            self,
            [["depthwise_conv.0", "depthwise_conv.1", "depthwise_conv.2"]],
            inplace=True,
            fuser_func=fuser_func,
        )
        torch.quantization.fuse_modules(
            self,
            [["pointwise_conv2.0", "pointwise_conv2.1"]],
            inplace=True,
            fuser_func=fuser_func,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,  # noqa: B008
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, in_channels, 1, time).
            mask_pad (torch.Tensor): used for batch padding
                (#batch, 1, 1, time)
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, in_channels, 1, cachetime)

        Returns:
            torch.Tensor: Output tensor (#batch, in_channels, 1, time).
            torch.Tensor: Cache tensor (#batch, in_channels, 1, cachetime).

        """
        if mask_pad is not None:
            x = x.masked_fill(mask_pad.eq(0), 0.0)

        if self.lorder > 0:
            if cache is None:
                b, c, _, t = x.size()
                cache = torch.zeros((b, c, 1, self.lorder), device=x.device)
                if isinstance(x, QTensor):
                    cache = QTensor(
                        cache,
                        scale=x.q_scale(),
                        dtype=x.dtype,
                        per_channel_axis=x.q_per_channel_axis(),
                    )
            # else:
            # assert cache.size(0) == x.size(0)  # batch
            # assert cache.size(1) == x.size(1)  # channel
            x = self.cat.cat((cache, x), dim=3)  # (b, c, 1, cache_t + t)
        new_cache = x

        x = self.pointwise_conv1(x)  # (batch, inner_channel, 1, cache_t + t)
        x = self.depthwise_conv(x)  # (batch, inner_channel, 1, t)
        x = self.pointwise_conv2(x)  # (batch, in_channel,    1, t)
        if mask_pad is not None:
            x = x.masked_fill(mask_pad.eq(0), 0.0)

        return x, new_cache
