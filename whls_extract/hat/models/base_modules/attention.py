# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import math
import warnings
from typing import Optional

import horizon_plugin_pytorch as horizon
import horizon_plugin_pytorch.nn.quantized as quantized
import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.quantization import FixedScaleObserver, QuantStub
from torch import Tensor
from torch.nn.init import constant_, xavier_uniform_

from hat.utils.model_helpers import fx_wrap

__all__ = [
    "HorizonMultiheadAttention",
    "MultiheadAttention",
    "MultiScaleDeformableAttention4Dim",
]

logger = logging.getLogger(__name__)


class HorizonMultiheadAttention(nn.Module):
    """modify torch.nn.MultiheadAttention to support quantization.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads.
            Note that ``embed_dim`` will be split across ``num_heads``,
            i.e. each head will have dimension ``embed_dim // num_heads``.
        dropout: Dropout probability. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers.
            Default: ``True``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        scale = self.head_dim ** -0.5
        self.register_buffer("scale", torch.Tensor([scale]))

        # define q k v projection layers
        self.q_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=bias)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=bias)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=bias)

        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=bias)

        self.attention_drop = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.mul = quantized.FloatFunctional()
        self.matmul = quantized.FloatFunctional()
        self.add = quantized.FloatFunctional()
        self.mask_add = quantized.FloatFunctional()
        self.attn_matmul = quantized.FloatFunctional()
        self.scale_quant = QuantStub()
        self.attn_mask_quant = QuantStub(scale=1)

        self._reset_parameters()

    def _reset_parameters(self):
        # follow nn.MultiheadAttention to init parameters
        in_proj_weight = torch.concat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight)
        )
        nn.init.xavier_uniform_(in_proj_weight)
        w_q, w_k, w_v = in_proj_weight.chunk(3)
        self.q_proj.weight, self.k_proj.weight, self.v_proj.weight = (
            nn.Parameter(w_q),
            nn.Parameter(w_k),
            nn.Parameter(w_v),
        )

        if self.bias:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    @fx_wrap()
    def checkdim(self, embed_dim, key, value):
        assert (
            embed_dim == self.embed_dim
        ), f"was expecting embedding dimension of {self.embed_dim}, \
            but got {embed_dim}"
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

    @fx_wrap()
    def prep_attention_mask(
        self,
        attn_mask,
        tgt_h,
        tgt_w,
        src_h,
        src_w,
        bsz,
        key_padding_mask,
    ):

        if attn_mask is not None:
            if not isinstance(attn_mask, QTensor):
                if attn_mask.dtype == torch.uint8:
                    warnings.warn(
                        "Byte tensor for attn_mask is deprecated."
                        "Use bool tensor instead."
                    )
                    attn_mask = attn_mask.to(torch.bool)
                else:
                    assert (
                        attn_mask.is_floating_point()
                        or attn_mask.dtype == torch.bool
                    ), (
                        "Only float, byte, and bool types are supported"
                        f"for attn_mask, not {attn_mask.dtype}"
                    )
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_h * tgt_w, src_h * src_w)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape},"
                        "but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (
                    bsz * self.num_heads,
                    tgt_h * tgt_w,
                    src_h * src_w,
                )
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape},"
                        f"but should be {correct_3d_size}."
                    )
            elif attn_mask.dim() == 4:
                correct_4d_size = (
                    bsz * self.num_heads,
                    tgt_h,
                    tgt_w,
                    src_h * src_w,
                )
                if attn_mask.shape != correct_4d_size:
                    raise RuntimeError(
                        f"The shape of the 4D attn_mask is {attn_mask.shape},"
                        f"but should be {correct_4d_size}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )

        # prep key padding mask
        if (
            key_padding_mask is not None
            and key_padding_mask.dtype == torch.uint8
        ):
            warnings.warn(
                "Byte tensor for key_padding_mask is deprecated."
                "Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        return attn_mask, key_padding_mask

    @fx_wrap()
    def merge_key_padding_attention_masks(
        self,
        key_padding_mask,
        bsz,
        src_len,
        attn_mask,
    ):
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bsz,
                src_len,
            ), f"expecting key_padding_mask shape of {(bsz, src_len)}, \
                but got {key_padding_mask.shape}"
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(bsz * self.num_heads, 1, src_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(
                    key_padding_mask, float("-100")
                )

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-100"))
            attn_mask = new_attn_mask
        if attn_mask is not None:
            if not isinstance(attn_mask, QTensor):
                attn_mask = self.attn_mask_quant(attn_mask)
        return attn_mask

    @fx_wrap()
    def mask_attention(
        self,
        attention,
        attn_mask,
        tgt_h,
        tgt_w,
        src_h,
        src_w,
    ):
        if attn_mask is None:
            return attention
        if attn_mask.dim() != 4:
            attn_mask = attn_mask.contiguous().unsqueeze(1)
            if attn_mask.shape[2] != 1:
                attn_mask = attn_mask.view(-1, tgt_h, tgt_w, src_h * src_w)
        attention = self.mask_add.add(attention, attn_mask)
        return attention

    @fx_wrap()
    def _view_qkv(self, bsz, src_h, src_w, tgt_h, tgt_w, q, k, v):
        q = (
            q.contiguous()
            .view(bsz * self.num_heads, self.head_dim, tgt_h, tgt_w)
            .permute(0, 2, 3, 1)
        )
        k = k.contiguous().view(
            bsz * self.num_heads, 1, self.head_dim, src_h * src_w
        )
        v = (
            v.contiguous()
            .view(bsz * self.num_heads, 1, self.head_dim, src_h * src_w)
            .permute(0, 1, 3, 2)
        )
        return q, k, v

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        # set up shape vars
        bsz, embed_dim, tgt_h, tgt_w = query.shape
        _, _, src_h, src_w = key.shape

        self.checkdim(embed_dim, key, value)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # prep attention mask

        attn_mask, key_padding_mask = self.prep_attention_mask(
            attn_mask,
            tgt_h,
            tgt_w,
            src_h,
            src_w,
            bsz,
            key_padding_mask,
        )

        q, k, v = self._view_qkv(bsz, src_h, src_w, tgt_h, tgt_w, q, k, v)

        # update source sequence length after adjustments
        src_len = k.size(3)

        # merge key padding and attention masks
        attn_mask = self.merge_key_padding_attention_masks(
            key_padding_mask,
            bsz,
            src_len,
            attn_mask,
        )

        # q = q * self.scale
        # attention = (q @ k.transpose(-2, -1))
        scale = self.scale_quant(self.scale)
        q = self.mul.mul(
            q, scale
        )  # [bsz*self.num_heads, tgt_h, tgt_w, self.head_dim]

        attention = self.matmul.matmul(
            q, k
        )  # [bsz*self.num_heads, tgt_h, tgt_w, src_h*src_w]

        attention = self.mask_attention(
            attention,
            attn_mask,
            tgt_h,
            tgt_w,
            src_h,
            src_w,
        )
        attention = self.softmax(attention)

        attention = self.attention_drop(attention)
        # output = (attention @ v)
        attn_output = self.attn_matmul.matmul(
            attention, v
        )  # [bsz*self.num_heads, tgt_h, tgt_w, self.head_dim]
        attn_output = (
            attn_output.permute(0, 3, 1, 2)
            .contiguous()
            .view(bsz, embed_dim, tgt_h, tgt_w)
        )
        attn_output = self.out_proj(attn_output)

        return attn_output, None

    def fuse_model(self):
        pass


class MultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    Implemente MultiheadAttention with identity connection,
    and position embedding is also passed as input.

    Args:
        embed_dim: The embedding dimension for attention.
        num_heads: The number of attention heads.
        attn_drop: A Dropout layer on attn_output_weights.
        proj_drop: A Dropout layer after `MultiheadAttention`.
        batch_first: if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
            **kwargs,
        )
        self.query_pos_add = FloatFunctional()
        self.key_pos_add = FloatFunctional()
        self.identity_add = FloatFunctional()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        identity: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_pos: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query: Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key: Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value: Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity: The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos: The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos: The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`,
                then `query_pos` will be used for `key_pos`.
            attn_mask: ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask: ByteTensor with shape `(bs, num_key)` indicates
                which elements within `key` to be ignored in attention.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    logger.warning(
                        f"position encoding of key is"
                        f"missing in {self.__class__.__name__}."
                    )
        if query_pos is not None:
            query = self.query_pos_add.add(query, query_pos)
        if key_pos is not None:
            key = self.key_pos_add.add(key, key_pos)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]

        return self.identity_add.add(identity, self.proj_drop(out))


class MultiScaleDeformableAttention4Dim(nn.Module):
    """4Dim version for MultiScaleDeformableAttention4Dim.

    Args:
        d_model: The feature dimension.
        n_levels: The num of featuremap.
        n_heads: Parallel attention heads.
        n_points: The num points for each head sample.
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):

        super().__init__()
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_model = d_model
        self.sampling_offsets = nn.ModuleList()
        self.sampling_mul1 = nn.ModuleList()
        self.sampling_mul2 = nn.ModuleList()
        self.sampling_add1 = nn.ModuleList()
        self.sampling_add2 = nn.ModuleList()
        self.sampling_mul3 = nn.ModuleList()
        self.value_projs = nn.ModuleList()

        for _ in range(self.n_levels):
            self.sampling_offsets.append(
                nn.Conv2d(self.d_model, n_heads * n_points * 2, 1)
            )
            self.value_projs.append(nn.Conv2d(self.d_model, self.d_model, 1))
            self.sampling_mul1.append(FF())
            self.sampling_mul2.append(FF())
            self.sampling_add1.append(FF())
            self.sampling_add2.append(FF())
            self.sampling_mul3.append(FF())
        self.attention_weights = nn.Conv2d(
            self.d_model, n_heads * n_levels * n_points, 1
        )
        self.attention_weights_mul = FF()
        self.attention_weights_sum = FF()
        self.output_proj = nn.Conv2d(self.d_model, self.d_model, 1)
        self.cat = FF()
        qin16_max = 32767 - (-32768)

        self.quant_shape = QuantStub(scale=1.0 / qin16_max)
        self.softmax = torch.nn.Sigmoid()
        self._reset_parameters()

    def _reset_parameters(self):
        grid_head = self.n_heads
        thetas = torch.arange(grid_head, dtype=torch.float32) * (
            2.0 * math.pi / grid_head
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(grid_head, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        for j in range(self.n_levels):
            with torch.no_grad():
                self.sampling_offsets[j].bias = nn.Parameter(
                    grid_init[:, j, :, :].reshape(-1)
                )
            constant_(self.sampling_offsets[j].weight.data, 0.0)

            xavier_uniform_(self.value_projs[j].weight.data)
            constant_(self.value_projs[j].bias.data, 0.0)

        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    @fx_wrap
    def _get_div_shape(self, values):
        div_shape = []
        for i in range(self.n_levels):
            v_h, v_w = values[i].shape[-2:]
            div_shape.append(
                torch.tensor([1 / v_w, 1 / v_h], dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(2)
            )
        return div_shape

    @fx_wrap
    def _get_sample_grid(
        self,
        sampling_offsets_tmp,
        bs,
        len_h,
        len_w,
        reference_points,
        i,
        div_shape,
    ):
        sampling_offsets_tmp = sampling_offsets_tmp.view(
            bs * self.n_heads * self.n_points, 2, len_h, len_w
        )
        if reference_points.shape[1] == 2:
            sampling_offsets_tmp = self.sampling_mul1[i].mul(
                sampling_offsets_tmp,
                self.quant_shape(div_shape[i].to(sampling_offsets_tmp.device)),
            )
            sampling_offsets_tmp = sampling_offsets_tmp.view(
                bs, self.n_heads, self.n_points, 2, len_h, len_w
            )
            reference_points = reference_points.view(bs, 1, 1, 2, len_h, len_w)

            sampling_locations = self.sampling_add1[i].add(
                reference_points, sampling_offsets_tmp
            )

        elif reference_points.shape[1] == 4:
            (
                reference_tmp_xy,
                reference_tmp_wh,
            ) = reference_points.split((2), dim=1)
            sampling_offsets_tmp = self.sampling_mul1[i].mul_scalar(
                sampling_offsets_tmp, 0.5 / self.n_points
            )
            sampling_offsets_tmp = sampling_offsets_tmp.view(
                bs, self.n_heads, self.n_points, 2, len_h, len_w
            )
            reference_tmp_wh = reference_tmp_wh.view(bs, 1, 1, 2, len_h, len_w)
            reference_tmp_xy = reference_tmp_xy.view(bs, 1, 1, 2, len_h, len_w)
            sampling_offsets_tmp = self.sampling_mul3[i].mul(
                sampling_offsets_tmp, reference_tmp_wh
            )
            sampling_locations = self.sampling_add1[i].add(
                reference_tmp_xy, sampling_offsets_tmp
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be "
                "2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        sampling_grids_tmp = self.sampling_mul2[i].mul_scalar(
            sampling_locations, 2
        )
        sampling_grids = self.sampling_add2[i].add_scalar(
            sampling_grids_tmp, -1
        )
        sampling_grids = sampling_grids.view(
            bs * self.n_heads, self.n_points, 2, len_h * len_w
        )

        sampling_grids = sampling_grids.permute(0, 1, 3, 2)

        return sampling_grids

    def forward(self, query, reference_points, values):

        div_shape = self._get_div_shape(values)
        bs, C, len_h, len_w = query.size()

        sampling_value_list = []

        for i in range(self.n_levels):
            sampling_offsets_tmp = self.sampling_offsets[i](query)

            sampling_grids = self._get_sample_grid(
                sampling_offsets_tmp,
                bs,
                len_h,
                len_w,
                reference_points,
                i,
                div_shape,
            )

            value_h, value_w = values[i].shape[-2:]
            value_l_ = self._get_value_l(i, values, bs, value_h, value_w)
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grids,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_value_list.append(sampling_value_l_)

        sampling_value_all = self.cat.cat(sampling_value_list, dim=2)

        attention_weights = self.attention_weights(query)
        attention_weights = self._view_attention_weights(
            attention_weights,
            bs,
            len_h,
            len_w,
        )
        attention_weights = self.softmax(attention_weights)

        attention_result = self.attention_weights_mul.mul(
            sampling_value_all, attention_weights
        )
        attentiom_result1 = self.attention_weights_sum.sum(
            attention_result, dim=2, keepdim=True
        )
        attentiom_result1 = attentiom_result1.view(
            bs, self.d_model, len_h, len_w
        )
        output = self.output_proj(attentiom_result1)
        return output

    @fx_wrap()
    def _get_value_channel(self):
        return self.d_model // self.n_heads

    def _get_value_l(self, i, values, bs, value_h, value_w):
        return self.value_projs[i](values[i]).view(
            bs * self.n_heads,
            self._get_value_channel(),
            value_h,
            value_w,
        )  # [bs, d_m, h, w]  -> #[bs*head, d_m//head, h,w]

    @fx_wrap()
    def _view_attention_weights(self, attention_weights, bs, len_h, len_w):
        return attention_weights.view(
            bs * self.n_heads, 1, self.n_levels * self.n_points, len_h * len_w
        )

    def fuse_model(self):
        pass

    def set_qconfig(self):
        qint16_qconfig = horizon.quantization.get_default_qat_qconfig(
            dtype="qint16",
        )
        qint8_qconfig = horizon.quantization.get_default_qat_qconfig(
            dtype="qint8",
        )
        int16_modules_list = [
            self.sampling_offsets,
            self.sampling_add2,
            self.sampling_mul1,
            self.sampling_mul2,
            self.sampling_add1,
            self.sampling_mul3,
        ]
        for module in int16_modules_list:
            for layer in module:
                layer.qconfig = qint16_qconfig

        for layer in self.value_projs:
            layer.qconfig = qint8_qconfig

        self.attention_weights.qconfig = qint16_qconfig
        self.attention_weights_mul.qconfig = qint16_qconfig
        self.attention_weights_sum.qconfig = qint16_qconfig
        self.output_proj.qconfig = qint8_qconfig
        self.softmax.qconfig = qint16_qconfig
        self.cat.qconfig = qint16_qconfig

        qin16_max = 32767 - (-32768)
        self.quant_shape.qconfig = (
            horizon.quantization.get_default_qat_qconfig(
                dtype="qint16",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": 1.0 / qin16_max,
                },
            )
        )
