import math
import warnings
from typing import Any, List, Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    from mmcv.cnn.bricks.transformer import (
        MultiheadAttention,
        MultiScaleDeformableAttention,
    )
    from mmcv.ops.multi_scale_deform_attn import (
        MultiScaleDeformableAttnFunction,
        multi_scale_deformable_attn_pytorch,
    )

    OBJECT_REGISTRY.register(MultiScaleDeformableAttention)
    OBJECT_REGISTRY.register(MultiheadAttention)

except ImportError:
    MultiScaleDeformableAttnFunction = None
    multi_scale_deformable_attn_pytorch = None


@OBJECT_REGISTRY.register
class BevDeformableTemporalAttention(nn.Module):
    """An attention module used in BEVFormer.

    Args:
        embed_dims: The embedding dimension of Attention.
            Default: 256.
        num_heads: Parallel attention heads. Default: 64.
        num_levels: The number of feature map used in
            Attention. Default: 4.
        num_points: The number of sampling points for
            each query in each head. Default: 4.
        im2col_step: The step used in image_to_column.
            Default: 64.
        dropout: A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first: Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        qv_cat: if True to concat query and value.
    """

    @require_packages("mmcv")
    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        num_bev_queue: int = 2,
        im2col_step: int = 64,
        bev_h: int = 200,
        bev_w: int = 200,
        dropout: float = 0.1,
        batch_first: bool = False,
        qv_cat: bool = True,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.bev_h = bev_h
        self.bev_w = bev_w

        def _is_power_of_2(n: int) -> bool:
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(
                        n, type(n)
                    )
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.num_bev_queue = num_bev_queue
        self.qv_cat = qv_cat

        self.sampling_offsets = nn.Conv2d(
            embed_dims * num_bev_queue if qv_cat else embed_dims,
            num_bev_queue * num_heads * num_levels * num_points * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.attention_weights = nn.Conv2d(
            embed_dims * num_bev_queue if qv_cat else embed_dims,
            num_bev_queue * num_heads * num_levels * num_points,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize for Parameters of Module."""

        nn.init.kaiming_normal_(self.sampling_offsets.weight)
        nn.init.kaiming_normal_(self.attention_weights.weight)

        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        identity: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        reference_points: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        pre_bev_feat: torch.Tensor = None,
        pre_ref_points: torch.Tensor = None,
        start_of_sequence: torch.Tensor = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query : Query of Transformer with shape
                (num_query, bs, embed_dims).
            key : The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value : The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity : The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos : The positional encoding for `query`.
                Default: None.
            key_pos : The positional encoding for `key`. Default
                None.
            reference_points :  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes : Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index : The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            pre_bev_feat: Previous frame's BEV feat.
            pre_ref_points: refernce_points in current frame to previous frame.

        Returns:
            forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if pre_bev_feat is not None:
            pre_bev_feat[:, start_of_sequence] = query[:, start_of_sequence]
            value = torch.cat([pre_bev_feat, query], 1)
        else:
            value = torch.cat([query, query], 1)

        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos

        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        if self.qv_cat:
            query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)

        value = value.view(
            self.num_bev_queue * bs, num_value, self.num_heads, -1
        )

        query_bev = query.permute(0, 2, 1).view(
            bs,
            self.num_bev_queue * self.embed_dims
            if self.qv_cat
            else self.embed_dims,
            self.bev_h,
            self.bev_w,
        )  # torch.Size([2, 512, 200, 200])
        sampling_offsets = self.sampling_offsets(query_bev).flatten(2, 3)
        sampling_offsets = sampling_offsets.permute(0, 2, 1).view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
            2,
        )

        attention_weights = self.attention_weights(query_bev).flatten(2, 3)
        attention_weights = attention_weights.permute(0, 2, 1).view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels * self.num_points,
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
        )

        attention_weights = (
            attention_weights.permute(3, 0, 1, 2, 4, 5)
            .reshape(
                bs * self.num_bev_queue,
                num_query,
                self.num_heads,
                self.num_levels,
                self.num_points,
            )
            .contiguous()
        )
        sampling_offsets = (
            sampling_offsets.permute(3, 0, 1, 2, 4, 5, 6)
            .reshape(
                bs * self.num_bev_queue,
                num_query,
                self.num_heads,
                self.num_levels,
                self.num_points,
                2,
            )
            .contiguous()
        )

        if len(reference_points.shape) == 3:
            reference_points = reference_points[:, :, None, :]
        if pre_ref_points is not None:
            ref_2d = torch.cat(
                [pre_ref_points, reference_points[:, :, 0:1, :2]], 0
            )
        else:
            ref_2d = torch.cat(
                [
                    reference_points[:, :, 0:1, :2],
                    reference_points[:, :, 0:1, :2],
                ],
                0,
            )

        if ref_2d.shape[-1] == 2:  # 1, 100, 3

            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            # four refs 1, 10000, 4, 3. must match 4 levels or 1 levels

            sampling_locations = (
                ref_2d[:, :, None, :, None, :2]
                + sampling_offsets
                / offset_normalizer[None, None, None, :, None, :]
            )

        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        output = MultiScaleDeformableAttnFunction.apply(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )

        output = output.permute(1, 2, 0)
        output = (output[..., :bs] + output[..., bs:]) / self.num_bev_queue
        output = output.permute(2, 0, 1)
        output = self.output_proj(output)

        output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@OBJECT_REGISTRY.register
class BevSpatialCrossAtten(nn.Module):
    """An attention module used in Detr3d.

    Args:
        pc_range: point cloud range.
        deformable_attention: Module for deformable cross attn.
        embed_dims: The embedding dimension of Attention.
            Default: 256.
        num_refs: Number of reference points in head dimension.
            Default: 4.
        num_cams: The number of cameras. Default: 6.
        num_points: The number of sampling points for
            each query in each head. Default: 4.
        dropout: A Dropout layer on `inp_identity`.
            Default: 0..

    """

    @require_packages("mmcv")
    def __init__(
        self,
        pc_range: List[float],
        deformable_attention: nn.Module,
        embed_dims: int = 256,
        num_refs: int = 4,
        num_cams: int = 6,
        dropout: float = 0.1,
    ):
        super(BevSpatialCrossAtten, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        self.deformable_attention = deformable_attention
        self.embed_dims = embed_dims

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.num_refs = num_refs
        self.num_cams = num_cams

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize for Parameters of Module."""

        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        bev_reference_points: Optional[torch.Tensor] = None,
        mlvl_feats_spatial_shapes: Optional[torch.Tensor] = None,
        mlvl_feats_level_start_index: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward Function of Detr3DCrossAtten.

        Args:
            query: Query of Transformer with shape
                (num_query, bs, embed_dims).
            key: The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value: The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            query_pos: The positional encoding for `query`.
                Default: None.
            bev_reference_points:  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            mlvl_feats_spatial_shapes: Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            mlvl_feats_level_start_index: The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            residual: The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        query = query.permute(1, 0, 2)  # torch.Size([1, 40000, 256])
        slots = torch.zeros_like(query)

        value = value.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        reference_points = bev_reference_points.clone()

        img_metas = kwargs["img_metas"]

        # from hat.utils.forkedpdb import set_trace; set_trace()
        lidar2img_shape = (
            list(img_metas["lidar2img"].shape)
            if "lidar2img" in img_metas
            else list(img_metas["T_vcs2img"].shape)
        )
        lidar2img_shape[2] = 4
        lidar2img = reference_points.new_zeros(lidar2img_shape)
        lidar2img[..., 3, 3] = 1.0
        lidar2img[..., :3, :4] = (
            img_metas["lidar2img"]
            if "lidar2img" in img_metas
            else img_metas["T_vcs2img"]
        )

        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
            + self.pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
            + self.pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2])
            + self.pc_range[2]
        )
        # reference_points (B, num_query, Nref, 4)
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )
        B, num_query, num_ref = reference_points.size()[:3]
        # reference_points = reference_points.view()

        num_cam = lidar2img.size(1)
        reference_points = (
            reference_points.view(B, 1, num_query * num_ref, 4)
            .repeat(1, num_cam, 1, 1)
            .unsqueeze(-1)
        )
        lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(
            1, 1, num_query * num_ref, 1, 1
        )
        reference_points_cam = torch.matmul(
            lidar2img, reference_points
        ).squeeze(-1)
        eps = 1e-5
        mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )
        reference_points_cam[..., 0] /= img_metas["img_shape"][..., 1:2]
        reference_points_cam[..., 1] /= img_metas["img_shape"][..., 0:1]

        mask = (
            mask
            & (reference_points_cam[..., 0:1] > 0.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
        )
        # mask = mask.view(B, num_cam, 1, 1, num_query*num_ref, 1)
        mask = mask.view(B, num_cam, num_query, num_ref).permute(1, 0, 2, 3)
        # mask = mask.view(B*num_cam, num_query*num_ref, 1, 1, 1)
        mask = torch.nan_to_num(mask)

        reference_points_cam = reference_points_cam.view(
            B, num_cam, num_query, num_ref, 2
        ).permute(1, 0, 2, 3, 4)

        indexes = []
        for _, mask_per_img in enumerate(mask):
            for j in range(B):
                index_query_per_img = (
                    mask_per_img[j].sum(-1).nonzero().squeeze(-1)
                )
                indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries.
        # This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [B * num_cam, max_len, self.embed_dims]
        )
        reference_points_rebatch = reference_points_cam.new_zeros(
            [B * num_cam, max_len, num_ref, 2]
        )

        for i, reference_points_per_img in enumerate(reference_points_cam):
            for j in range(B):
                index_query_per_img = indexes[i * B + j]
                queries_rebatch[
                    j * num_cam + i, : len(index_query_per_img)
                ] = query[j, index_query_per_img]
                reference_points_rebatch[
                    j * num_cam + i, : len(index_query_per_img)
                ] = reference_points_per_img[j, index_query_per_img]

        queries = self.deformable_attention(
            query=queries_rebatch,
            value=value,
            spatial_shapes=mlvl_feats_spatial_shapes,
            reference_points=reference_points_rebatch,
            level_start_index=mlvl_feats_level_start_index,
        )

        for idx, index_query_per_img in enumerate(indexes):
            i = idx // B
            j = idx - i * B
            slots[j, index_query_per_img] += queries[
                j * self.num_cams + i, : len(index_query_per_img)
            ]

        count = mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)
        slots = slots.permute(1, 0, 2)

        return self.dropout(slots) + identity


@OBJECT_REGISTRY.register
class MSDeformableAttention3D(nn.Module):
    """An attention module used in BEVFormer based on Deformable-Detr. \
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims: The embedding dimension of Attention.
            Default: 256.
        num_heads: Parallel attention heads. Default: 64.
        num_levels: The number of feature map used in
            Attention. Default: 4.
        num_points: The number of sampling points for
            each query in each head. Default: 4.
        im2col_step: The step used in image_to_column.
            Default: 64.
        batch_first: Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    @require_packages("mmcv")
    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
        im2col_step: int = 64,
        batch_first: bool = True,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n: int) -> bool:
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(
                        n, type(n)
                    )
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize for Parameters of Module."""
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query: Query of Transformer with shape
                ( bs, num_query, embed_dims).
            value: The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            query_pos: The positional encoding for `query`.
                Default: None.
            reference_points:  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes: Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index: The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = (
                sampling_offsets
                / offset_normalizer[None, None, None, :, None, :]
            )
            (
                bs,
                num_query,
                num_heads,
                num_levels,
                num_all_points,
                xy,
            ) = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs,
                num_query,
                num_heads,
                num_levels,
                num_all_points // num_Z_anchors,
                num_Z_anchors,
                xy,
            )
            sampling_locations = reference_points + sampling_offsets
            (
                bs,
                num_query,
                num_heads,
                num_levels,
                num_points,
                num_Z_anchors,
                xy,
            ) = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy
            )

        elif reference_points.shape[-1] == 4:
            raise ValueError(
                "Last dim of reference_points == 4 is not supported now."
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


@OBJECT_REGISTRY.register
class ObjectDetr3DCrossAtten(nn.Module):
    """An attention module used in Detr3d.

    Args:
        embed_dims : The embedding dimension of Attention.
            Default: 256.
        num_heads : Parallel attention heads. Default: 64.
        num_levels : The number of feature map used in
            Attention. Default: 4.
        num_points : The number of sampling points for
            each query in each head. Default: 4.
        im2col_step : The step used in image_to_column.
            Default: 64.
        dropout : A Dropout layer on `inp_identity`.
            Default: 0..

    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        im2col_step: int = 64,
        pc_range: List[float] = None,
        dropout: float = 0.1,
        batch_first: bool = False,
    ):
        super(ObjectDetr3DCrossAtten, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(
                        n, type(n)
                    )
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * self.num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * self.num_levels * num_points
        )

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.batch_first = batch_first

        self.init_weights()

    def init_weights(self):
        """Initialize for Parameters of Module."""

        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        bev_feat_shapes: Optional[torch.Tensor] = None,
        bev_feat_level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward Function of Detr3DCrossAtten.

        Args:
            query: Query of Transformer with shape
                (num_query, bs, embed_dims).
            key: The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value: The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual: The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos: The positional encoding for `query`.
                Default: None.
            key_pos: The positional encoding for `key`. Default
                None.
            reference_points:  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            level_start_index: The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if value is None:
            value = key
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        bs, num_query, _ = query.size()
        bs, num_value, _ = value.size()

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        offset_normalizer = torch.stack(
            [bev_feat_shapes[..., 1], bev_feat_shapes[..., 0]], -1
        )
        sampling_locations = (
            reference_points[:, :, None, None, None, :2]
            + sampling_offsets
            / offset_normalizer[None, None, None, :, None, :]
        )

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value,
                bev_feat_shapes,
                bev_feat_level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            output = multi_scale_deformable_attn_pytorch(
                value,
                bev_feat_shapes,
                sampling_locations,
                attention_weights,
            )

        output = self.output_proj(output)
        output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
