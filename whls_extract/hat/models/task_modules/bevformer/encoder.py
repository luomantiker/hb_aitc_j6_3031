# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import List, Tuple

import torch
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor, nn
from torch.nn.init import normal_

from hat.models.task_modules.bevformer.attention import (
    HorizonMultiScaleDeformableAttention,
    HorizonMultiScaleDeformableAttention3D,
    HorizonSpatialCrossAttention,
    HorizonTemporalSelfAttention,
)
from hat.models.task_modules.bevformer.utils import FFN, get_clone_module
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = [
    "BEVFormerEncoder",
    "BEVFormerEncoderLayer",
    "SingleBEVFormerEncoder",
    "SingleBEVFormerEncoderLayer",
]


@OBJECT_REGISTRY.register
class BEVFormerEncoder(nn.Module):
    """The basic structure of BEVFormerEncoder.

    Args:
        bev_h: The height of bevfeat.
        bev_w: The width of bevfeat.
        embed_dims: The embedding dimension of Attention.
        encoder_layer: The encoder layer.
        use_cams_embeds: Whether to use camera embeds.
        num_feature_levels: Num of feats.
        num_cams: The num of camera.
        num_layers: The num of encoder layers.
        return_intermediate: Whether to return intermediate outputs.
    """

    def __init__(
        self,
        bev_h: int,
        bev_w: int,
        embed_dims: int,
        encoder_layer: nn.Module,
        use_cams_embeds: bool = True,
        num_feature_levels: int = 1,
        num_cams: int = 6,
        num_layers: int = 3,
        return_intermediate: bool = False,
    ):

        super(BEVFormerEncoder, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.embed_dims = embed_dims
        self.use_cams_embeds = use_cams_embeds

        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims)
        )

        self.add_canbus = FF()
        self.cat_pre_cur_query = FF()

        self.addcams_embeds = nn.ModuleList()
        self.add_level_embeds = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            self.addcams_embeds.append(FF())
            self.add_level_embeds.append(FF())

        self.quant_cams_embeds = QuantStub()
        self.quant_level_embeds = QuantStub()
        self.cat_fl = FF()

        self.layers = get_clone_module(encoder_layer, num_layers)
        self.return_intermediate = return_intermediate
        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if (
                isinstance(m, HorizonSpatialCrossAttention)
                or isinstance(m, HorizonMultiScaleDeformableAttention3D)
                or isinstance(m, HorizonTemporalSelfAttention)
                or isinstance(m, HorizonMultiScaleDeformableAttention)
            ):
                m.init_weights()

        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    @fx_wrap()
    def get_encoder_inputs(
        self,
        hybird_ref_2d: Tensor,
        bev_query: Tensor,
        bev_pos: Tensor,
        prev_bev: Tensor,
        mlvl_feats: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Get encoder layer inputs."""
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, _, _ = hybird_ref_2d.shape
        prev_bev = self.cat_pre_cur_query.cat(
            [prev_bev, bev_query], dim=1
        ).reshape(bs, len_bev, -1)

        feat_flatten = []
        spatial_shapes = []

        level_embed_spilted = self.quant_level_embeds(self.level_embeds).split(
            1, dim=0
        )
        cams_embeds = (
            self.quant_cams_embeds(self.cams_embeds).unsqueeze(1).unsqueeze(1)
        )

        for lvl, feat in enumerate(mlvl_feats):
            bs, _, _, h, w = feat.shape
            spatial_shape = (w, h)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = self.addcams_embeds[lvl].add(feat, cams_embeds)
            feat = self.add_level_embeds[lvl].add(
                feat, level_embed_spilted[lvl].unsqueeze(0).unsqueeze(0)
            )
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = self.cat_fl.cat(feat_flatten, 2)
        cross_spatial_shapes = torch.tensor(
            spatial_shapes, dtype=torch.float32, device=feat_flatten.device
        )
        sa_spatial_shapes = torch.tensor(
            [[self.bev_w, self.bev_h]], device=feat_flatten.device
        ).to(torch.float32)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

        return (
            bev_query,
            feat_flatten,
            bev_pos,
            hybird_ref_2d,
            cross_spatial_shapes,
            sa_spatial_shapes,
            prev_bev,
        )

    def forward(
        self,
        bev_query: Tensor,
        mlvl_feats: List[Tensor],
        bev_pos: Tensor,
        prev_bev: Tensor,
        hybird_ref_2d: Tensor,
        queries_rebatch_grid: Tensor,
        restore_bev_grid: Tensor,
        reference_points_rebatch: Tensor,
        bev_pillar_counts: Tensor,
    ) -> Tensor:
        """Forward BEVFormerEncoder.

        Args:
            bev_query: Input bev query.
            mlvl_feats: Input multi-cameta features.
            bev_pos: bev query pos embed.
            prev_bev: Previous frame bev feat.
            hybird_ref_2d: Hybird refpoints for temporal attention.
            queries_rebatch_grid: The grid for gridsample to sample bev \
                query for rebatch.
            restore_bev_grid: The grid for gridsample to restore bev query.
            reference_points_rebatch: The refpoints for spatial attention.
            bev_pillar_counts: Each bev pillar corresponds to the \
                number of cameras.
        """

        (
            bev_query,
            feat_flatten,
            bev_pos,
            hybird_ref_2d,
            cross_spatial_shapes,
            sa_spatial_shapes,
            prev_bev,
        ) = self.get_encoder_inputs(
            hybird_ref_2d, bev_query, bev_pos, prev_bev, mlvl_feats
        )

        output = bev_query
        intermediate = []
        for _, layer in enumerate(self.layers):
            output = layer(
                query=bev_query,
                key=feat_flatten,
                value=feat_flatten,
                query_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                sa_spatial_shapes=sa_spatial_shapes,
                cross_spatial_shapes=cross_spatial_shapes,
                prev_bev=prev_bev,
                queries_rebatch_grid=queries_rebatch_grid,
                restore_bev_grid=restore_bev_grid,
                reference_points_rebatch=reference_points_rebatch,
                bev_pillar_counts=bev_pillar_counts,
            )
            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return intermediate

        return output

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        for layer in self.layers:
            if hasattr(layer, "set_qconfig"):
                layer.set_qconfig()

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""
        for layer in self.layers:
            if hasattr(layer, "fuse_model"):
                layer.fuse_model()


@OBJECT_REGISTRY.register
class BEVFormerEncoderLayer(nn.Module):
    """The basic structure of BEVFormerEncoderLayer.

    Args:
        selfattention: The self attention module.
        crossattention: The cross attention module.
        embed_dims: The embedding dimension of Attention.
        dropout: Probability of an element to be zeroed.
    """

    def __init__(
        self,
        selfattention: nn.Module,
        crossattention: nn.Module,
        embed_dims: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sa = selfattention
        self.sa_norm = nn.LayerNorm(embed_dims)
        self.ca = crossattention
        self.ca_norm = nn.LayerNorm(embed_dims)
        self.ffn = FFN(embed_dims, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        query_pos: Tensor = None,
        ref_2d: Tensor = None,
        sa_spatial_shapes: Tensor = None,
        cross_spatial_shapes: Tensor = None,
        prev_bev: Tensor = None,
        queries_rebatch_grid=None,
        restore_bev_grid=None,
        reference_points_rebatch=None,
        bev_pillar_counts=None,
    ):
        """Foward BEVFormerEncoderLayer."""
        query = self.sa(
            query=query,
            value=prev_bev,
            query_pos=query_pos,
            reference_points=ref_2d,
            spatial_shapes=sa_spatial_shapes,
        )
        query = self.sa_norm(query)

        query = self.ca(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            spatial_shapes=cross_spatial_shapes,
            queries_rebatch_grid=queries_rebatch_grid,
            restore_bev_grid=restore_bev_grid,
            reference_points_rebatch=reference_points_rebatch,
            bev_pillar_counts=bev_pillar_counts,
        )

        query = self.ca_norm(query)

        query = self.ffn(query, query)
        query = self.ffn_norm(query)
        return query

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        if hasattr(self.sa, "set_qconfig"):
            self.sa.set_qconfig()
        if hasattr(self.ca, "set_qconfig"):
            self.ca.set_qconfig()
        if hasattr(self.ffn, "set_qconfig"):
            self.ffn.set_qconfig()

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""
        if hasattr(self.sa, "fuse_model"):
            self.sa.fuse_model()
        if hasattr(self.ca, "fuse_model"):
            self.ca.fuse_model()
        if hasattr(self.ffn, "fuse_model"):
            self.ffn.fuse_model()


@OBJECT_REGISTRY.register
class SingleBEVFormerEncoder(BEVFormerEncoder):
    """The basic structure of BEVFormerEncoder for single frame."""

    @fx_wrap()
    def get_encoder_inputs(
        self,
        mlvl_feats: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Get encoder layer inputs."""

        feat_flatten = []
        spatial_shapes = []

        level_embed_spilted = self.quant_level_embeds(self.level_embeds).split(
            1, dim=0
        )
        cams_embeds = (
            self.quant_cams_embeds(self.cams_embeds).unsqueeze(1).unsqueeze(1)
        )

        for lvl, feat in enumerate(mlvl_feats):
            bs, _, _, h, w = feat.shape
            spatial_shape = (w, h)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = self.addcams_embeds[lvl].add(feat, cams_embeds)
            feat = self.add_level_embeds[lvl].add(
                feat, level_embed_spilted[lvl].unsqueeze(0).unsqueeze(0)
            )
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = self.cat_fl.cat(feat_flatten, 2)
        cross_spatial_shapes = torch.tensor(
            spatial_shapes, dtype=torch.float32, device=feat_flatten.device
        )
        sa_spatial_shapes = torch.tensor(
            [[self.bev_w, self.bev_h]], device=feat_flatten.device
        ).to(torch.float32)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

        return (
            feat_flatten,
            cross_spatial_shapes,
            sa_spatial_shapes,
        )

    def forward(
        self,
        bev_query: Tensor,
        mlvl_feats: List[Tensor],
        bev_pos: Tensor,
        ref_2d: Tensor,
        queries_rebatch_grid: Tensor,
        restore_bev_grid: Tensor,
        reference_points_rebatch: Tensor,
        bev_pillar_counts: Tensor,
    ) -> Tensor:
        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        (
            feat_flatten,
            cross_spatial_shapes,
            sa_spatial_shapes,
        ) = self.get_encoder_inputs(mlvl_feats)

        output = bev_query
        intermediate = []
        for _, layer in enumerate(self.layers):
            output = layer(
                query=bev_query,
                key=feat_flatten,
                value=feat_flatten,
                query_pos=bev_pos,
                ref_2d=ref_2d,
                sa_spatial_shapes=sa_spatial_shapes,
                cross_spatial_shapes=cross_spatial_shapes,
                queries_rebatch_grid=queries_rebatch_grid,
                restore_bev_grid=restore_bev_grid,
                reference_points_rebatch=reference_points_rebatch,
                bev_pillar_counts=bev_pillar_counts,
            )
            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return intermediate

        return output


@OBJECT_REGISTRY.register
class SingleBEVFormerEncoderLayer(BEVFormerEncoderLayer):
    """The basic structure of BEVFormerEncoderLayer for single frame."""

    def forward(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        query_pos: Tensor = None,
        ref_2d: Tensor = None,
        sa_spatial_shapes: Tensor = None,
        cross_spatial_shapes: Tensor = None,
        queries_rebatch_grid: Tensor = None,
        restore_bev_grid: Tensor = None,
        reference_points_rebatch: Tensor = None,
        bev_pillar_counts: Tensor = None,
    ):
        query = self.sa(
            query,
            query,
            query_pos=query_pos,
            reference_points=ref_2d,
            spatial_shapes=sa_spatial_shapes,
        )
        query = self.sa_norm(query)

        query = self.ca(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            spatial_shapes=cross_spatial_shapes,
            queries_rebatch_grid=queries_rebatch_grid,
            restore_bev_grid=restore_bev_grid,
            reference_points_rebatch=reference_points_rebatch,
            bev_pillar_counts=bev_pillar_counts,
        )

        query = self.ca_norm(query)

        query = self.ffn(query, query)
        query = self.ffn_norm(query)
        return query
