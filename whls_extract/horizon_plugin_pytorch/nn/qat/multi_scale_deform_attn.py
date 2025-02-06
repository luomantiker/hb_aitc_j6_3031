import math
from typing import Optional, Union

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.init import constant_, xavier_uniform_

from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.nn.multi_scale_deform_attn import (
    MultiScaleDeformableAttention as FloatMultiScaleDeformableAttention,
)
from horizon_plugin_pytorch.nn.qat import FloatFunctional, Linear, QuantStub
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .softmax import SegmentLUTSoftmax

__all__ = ["MultiScaleDeformableAttention"]


class MultiScaleDeformableAttention(nn.Module):
    _FLOAT_MODULE = FloatMultiScaleDeformableAttention
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims: The embedding dimension of Attention.
            Default: 256.
        num_heads: Parallel attention heads. Default: 8.
        num_levels: The number of feature map used in
            Attention. Default: 4.
        num_points: The number of sampling points for
            each query in each head. Default: 4.
        dropout: A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first: Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        value_proj_ratio: The expansion ratio of value_proj.
            Default: 1.0.
        split_weight_mul: Whether split attention weight mul onto each level
            outputs. Enable this can reduce memory usage in qat training.
        split_batch: Whether Compute each batch at a time. Enable this can
            reduce memory usage in qat training.
    """

    @typechecked
    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
        batch_first: bool = False,
        value_proj_ratio: float = 1.0,
        split_weight_mul: bool = False,
        split_batch: bool = False,
        qconfig=None,
    ):
        super().__init__()
        assert qconfig is not None, "qconfig must be provided"
        assert (
            qconfig.activation is not None
        ), "qconfig.activation must be provided"
        assert qconfig.weight is not None, "qconfig.weight must be provided"

        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )

        from horizon_plugin_pytorch.quantization.qconfig import (
            replace_qconfig_dtype,
        )

        int8_qconfig = replace_qconfig_dtype(qconfig, qint8)
        int16_qconfig = replace_qconfig_dtype(qconfig, qint16)
        self.batch_first = batch_first

        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.split_weight_mul = split_weight_mul
        self.split_batch = split_batch
        self.dropout = nn.Dropout(dropout)
        self.pos_add = FloatFunctional(qconfig=int8_qconfig)

        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = Linear(
            embed_dims, value_proj_size, bias=True, qconfig=int8_qconfig
        )
        self.mask_quant = QuantStub(scale=1, qconfig=int8_qconfig)
        self.mask_mul = FloatFunctional(qconfig=int8_qconfig)
        self.sampling_offsets = Linear(
            embed_dims,
            num_heads * num_levels * num_points * 2,
            bias=True,
            qconfig=int16_qconfig,
        )

        self.attention_weights = Linear(
            embed_dims,
            num_heads * num_levels * num_points,
            bias=True,
            qconfig=int16_qconfig,
        )
        self.softmax = SegmentLUTSoftmax(
            dim=-1, min_sub_out=-12, qconfig=int16_qconfig
        )

        self.quant_normalizer = QuantStub(qconfig=int16_qconfig)
        self.sampling_mul1 = FloatFunctional(qconfig=int16_qconfig)
        self.sampling_add1 = FloatFunctional(qconfig=int16_qconfig)
        self.sampling_mul2 = FloatFunctional(qconfig=int16_qconfig)
        self.sampling_mul3 = FloatFunctional(qconfig=int16_qconfig)
        self.sampling_add2 = FloatFunctional(qconfig=int16_qconfig)
        self.sampling_cat = FloatFunctional(qconfig=int16_qconfig)
        if self.split_weight_mul:
            self.attention_weight_mul = nn.ModuleList(
                (
                    FloatFunctional(qconfig=int16_qconfig)
                    for _ in range(self.num_levels)
                )
            )
            self.attention_weight_pre_sum = nn.ModuleList(
                (
                    FloatFunctional(qconfig=int16_qconfig)
                    for _ in range(self.num_levels)
                )
            )
        else:
            self.attention_weight_mul = FloatFunctional(qconfig=int16_qconfig)
        self.attention_weight_sum = FloatFunctional(qconfig=int16_qconfig)

        self.output_proj = Linear(
            value_proj_size, embed_dims, bias=True, qconfig=int8_qconfig
        )
        self.residual_add = FloatFunctional(qconfig=qconfig)

        if self.split_batch:
            self.batch_cat = FloatFunctional(qconfig=int16_qconfig)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize Module Parameters."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32, device=device
        ) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def _multi_scale_deformable_attn(
        self,
        value: Union[Tensor, QTensor],
        value_spatial_shapes: Tensor,
        sampling_locations: Union[Tensor, QTensor],
        attention_weights: Union[Tensor, QTensor],
    ) -> Tensor:
        """Fundamental implementation of multi-scale deformable attention.

        Args:
            value: The value has shape
                (bs, num_keys, num_heads, embed_dims//num_heads)
            value_spatial_shapes: Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations: The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights: The weight of sampling points
                used when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),

        Returns:
            Tensor: shape of (bs, num_queries, embed_dims)
        """
        # Convert Tensor to list to avoid multi cuda sync on enumerate.
        value_spatial_shapes = value_spatial_shapes.cpu().numpy().tolist()

        bs, _, num_heads, embed_dims = value.shape
        (
            _,
            num_queries,
            num_heads,
            num_levels,
            num_points,
            _,
        ) = sampling_locations.shape
        value_list = value.split(
            [int(H_ * W_) for H_, W_ in value_spatial_shapes],  # noqa: N806
            dim=1,
        )
        del value
        sampling_grids = self.sampling_mul3.mul_scalar(sampling_locations, 2)
        del sampling_locations
        sampling_grids = self.sampling_add2.add_scalar(sampling_grids, -1)
        sampling_value_list = []

        for level, (H_, W_) in enumerate(value_spatial_shapes):  # noqa: N806
            # bs, H_*W_, num_heads, embed_dims ->
            # bs, H_*W_, num_heads*embed_dims ->
            # bs, num_heads*embed_dims, H_*W_ ->
            # bs*num_heads, embed_dims, H_, W_
            value_l_ = (
                value_list[level]
                .reshape(bs, H_ * W_, -1)
                .transpose(1, 2)
                .reshape(bs * num_heads, embed_dims, H_, W_)
            )
            # bs, num_queries, num_heads, num_points, 2 ->
            # bs, num_heads, num_queries, num_points, 2 ->
            # bs*num_heads, num_queries, num_points, 2
            sampling_grid_l_ = (
                sampling_grids[:, :, :, level : (level + 1)]
                .transpose(1, 2)
                .reshape(bs * num_heads, num_queries, num_points, 2)
            )
            # bs*num_heads, embed_dims, num_queries, num_points
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            del value_l_
            del sampling_grid_l_

            if self.split_weight_mul:
                # (bs*num_heads, embed_dims, num_queries, num_points) *
                # (bs*num_heads, 1, num_queries, num_points) ->
                # (bs*num_heads, embed_dims, num_queries, num_points)
                sampling_value_l_ = self.attention_weight_mul[level].mul(
                    sampling_value_l_,
                    attention_weights[:, :, :, level, :]
                    .transpose(1, 2)
                    .reshape(bs * num_heads, 1, num_queries, num_points),
                )

                # (bs*num_heads, embed_dims, num_queries, num_points) ->
                # (bs*num_heads, embed_dims, num_queries, 1)
                sampling_value_l_ = self.attention_weight_pre_sum[level].sum(
                    sampling_value_l_, dim=-1, keepdim=True
                )

            sampling_value_list.append(sampling_value_l_)

        # if self.split_weight_mul:
        # (bs*num_heads, embed_dims, num_queries, 1) ->
        # (bs*num_heads, embed_dims, num_queries, num_levels)
        # else:
        # (bs*num_heads, embed_dims, num_queries, num_points) ->
        # (bs*num_heads, embed_dims, num_queries, num_levels*num_points)
        sampling_value_all = self.sampling_cat.cat(sampling_value_list, dim=-1)
        del sampling_value_list

        if not self.split_weight_mul:
            sampling_value_all = self.attention_weight_mul.mul(
                sampling_value_all,
                attention_weights.transpose(1, 2).reshape(
                    bs * num_heads, 1, num_queries, num_levels * num_points
                ),
            )

        # bs*num_heads, embed_dims, num_queries, 1
        output = self.attention_weight_sum.sum(
            sampling_value_all, dim=-1, keepdim=True
        ).view(bs, num_heads * embed_dims, num_queries)

        return output.transpose(1, 2)

    @typechecked
    def forward(
        self,
        query: Union[Tensor, QTensor],
        key: Optional[Union[Tensor, QTensor]] = None,
        value: Optional[Union[Tensor, QTensor]] = None,
        identity: Optional[Union[Tensor, QTensor]] = None,
        query_pos: Optional[Union[Tensor, QTensor]] = None,
        key_padding_mask: Optional[Tensor] = None,
        reference_points: Optional[Union[Tensor, QTensor]] = None,
        spatial_shapes: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query: Query of Transformer with shape
                (num_query, bs, embed_dims).
            key: The key tensor with shape
                (num_key, bs, embed_dims).
            value: The value tensor with shape
                (num_key, bs, embed_dims).
            identity: The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos: The positional encoding for `query`.
                Default: None.
            key_padding_mask: ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points:  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (bs, num_query, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes: Spatial shape of features in
                different levels. int tensor with shape (num_levels, 2),
                last dimension represents (h, w).
        Returns:
            Tensor: the same shape with query.
        """

        assert reference_points is not None, "reference_points must be given!"
        assert spatial_shapes is not None, "spatial_shapes must be given!"
        assert not isinstance(
            spatial_shapes, QTensor
        ), "spatial_shapes cannot be quantized!"
        assert spatial_shapes.dtype in [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ], (
            "spatial_shapes must be int tensor, "
            f"but get {spatial_shapes.dtype}!"
        )

        if value is None:
            value = query

        if identity is None:
            identity = query

        if query_pos is not None:
            query = self.pos_add.add(query, query_pos)

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (
            spatial_shapes[:, :1] * spatial_shapes[:, 1:]
        ).sum() == num_value, (
            "sum of H * W in `spatial_shapes` must be equal with num_value!"
        )
        value = self.value_proj(value)
        if key_padding_mask is not None:
            assert not isinstance(
                key_padding_mask, QTensor
            ), "key_padding_mask cannot be quantized!"
            key_padding_mask = 1 - key_padding_mask[..., None].float()
            key_padding_mask = self.mask_quant(key_padding_mask)
            value = self.mask_mul.mul(value, key_padding_mask)
            del key_padding_mask
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        del query
        attention_weights = self.softmax(attention_weights)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            offset_normalizer = self.quant_normalizer(1.0 / offset_normalizer)
            sampling_offsets = self.sampling_mul1.mul(
                sampling_offsets,
                offset_normalizer[None, None, None, :, None, :],
            )
            # transpose to meet hbdk broadcast rules
            sampling_locations = self.sampling_add1.add(
                reference_points[:, :, None, None, :, :],
                sampling_offsets.transpose(3, 4),
            ).transpose(3, 4)
        elif reference_points.shape[-1] == 4:
            # bs, num_query, num_heads, num_levels, num_points, 2 ->
            # bs, num_query, num_heads, num_points, num_levels, 2
            sampling_offsets = self.sampling_mul2.mul_scalar(
                sampling_offsets, 0.5 / self.num_points
            ).transpose(3, 4)
            # input1: bs, num_query, num_heads, num_points, num_levels, 2
            # input2: bs, num_query,         1,          1, num_levels, 2
            sampling_offsets = self.sampling_mul1.mul(
                sampling_offsets, reference_points[:, :, None, None, :, 2:]
            )
            # input1: bs, num_query, num_heads, num_points, num_levels, 2
            # input2: bs, num_query,         1,          1, num_levels, 2
            sampling_locations = self.sampling_add1.add(
                sampling_offsets, reference_points[:, :, None, None, :, :2]
            ).transpose(3, 4)
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        del sampling_offsets

        if self.split_batch:
            bsz = value.size(0)
            output_list = []
            for i in range(bsz):
                output_list.append(
                    self._multi_scale_deformable_attn(
                        value[i : i + 1],
                        spatial_shapes,
                        sampling_locations[i : i + 1],
                        attention_weights[i : i + 1],
                    )
                )
            output = self.batch_cat.cat(output_list)
        else:
            output = self._multi_scale_deformable_attn(
                value, spatial_shapes, sampling_locations, attention_weights
            )

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.residual_add.add(self.dropout(output), identity)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict.

        Args: `mod` a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_deform_attn = cls(
            mod.embed_dims,
            mod.num_heads,
            mod.num_levels,
            mod.num_points,
            mod.dropout.p,
            mod.batch_first,
            mod.value_proj_ratio,
            mod.split_weight_mul,
            mod.split_batch,
            qconfig,
        )
        qat_deform_attn.value_proj.weight = mod.value_proj.weight
        qat_deform_attn.value_proj.bias = mod.value_proj.bias
        qat_deform_attn.sampling_offsets.weight = mod.sampling_offsets.weight
        qat_deform_attn.sampling_offsets.bias = mod.sampling_offsets.bias
        qat_deform_attn.attention_weights.weight = mod.attention_weights.weight
        qat_deform_attn.attention_weights.bias = mod.attention_weights.bias
        qat_deform_attn.output_proj.weight = mod.output_proj.weight
        qat_deform_attn.output_proj.bias = mod.output_proj.bias
        return qat_deform_attn
