import math
from typing import List, Optional

import torch
import torch.nn as nn
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.base_modules.mlp_module import FFN
from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager
from hat.utils.model_helpers import fx_wrap
from .layers import (
    BaseTransformerLayer,
    MultiheadAttention,
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
)

__all__ = [
    "DeformableDetrTransformerEncoder",
    "DeformableDetrTransformerDecoder",
    "DeformableDetrTransformer",
]


@OBJECT_REGISTRY.register
class DeformableDetrTransformerEncoder(TransformerLayerSequence):
    """Transformer encoder designed for the Deformable DETR architecture.

    This encoder uses multi-scale deformable attention mechanisms to enhance
    feature extraction for object detection tasks.

    Args:
        embed_dim: Dimensionality of the input embeddings.
        num_heads: Number of attention heads.
        feedforward_dim: Dimensionality of the feedforward network model.
        attn_dropout: Dropout rate for the attention layers.
        ffn_dropout: Dropout rate for the feedforward network.
        num_layers: Number of transformer layers.
        post_norm: Apply layer normalization post the transformer layers.
        num_feature_levels: Number of feature levels for multi-scale attention.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        post_norm: bool = False,
        num_feature_levels: int = 4,
    ):
        super(DeformableDetrTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                embed_dim=embed_dim,
                attn=MultiScaleDeformableAttention(
                    embed_dims=embed_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                    num_levels=num_feature_levels,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    num_fcs=2,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(
        self,
        query: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
    ):
        """Forward function.

        Args:
            query: Input tensor for the transformer encoder.
            query_pos Positional encodings for the query.
            query_key_padding_mask: Padding mask for keys in the query.
            spatial_shapes: Spatial shapes for multi-scale feature maps.
            reference_points: Reference points for deformable attention.

        """

        for layer in self.layers:
            query = layer(
                query,
                None,
                None,
                query_pos=query_pos,
                query_key_padding_mask=query_key_padding_mask,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


@OBJECT_REGISTRY.register
class DeformableDetrTransformerDecoder(TransformerLayerSequence):
    """Transformer decoder the Deformable DETR model.

    It incorporates multi-head and multi-scale deformable attention mechanisms.

    Args:
        embed_dim: Dimension of input embeddings.
        num_heads: Number of attention heads.
        feedforward_dim: Dimension of the feedforward network.
        attn_dropout: Dropout rate for attention layers.
        ffn_dropout: Dropout rate for feedforward network.
        num_layers: Number of decoder layers.
        return_intermediate: Whether to return intermediate outputs.
        num_feature_levels: Number of feature levels for multi-scale attention.
        as_two_stage: Enables two-stage decoder mode.
        with_box_refine: Enables bounding box refinement.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        return_intermediate: bool = True,
        num_feature_levels: int = 4,
        as_two_stage: bool = False,
        with_box_refine: bool = False,
    ):
        super(DeformableDetrTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                embed_dim=embed_dim,
                attn=[
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=True,
                    ),
                    MultiScaleDeformableAttention(
                        embed_dims=embed_dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True,
                        num_levels=num_feature_levels,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=(
                    "self_attn",
                    "norm",
                    "cross_attn",
                    "norm",
                    "ffn",
                    "norm",
                ),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate
        self.as_two_stage = as_two_stage
        self.with_box_refine = with_box_refine
        self.bbox_embed = None
        self.class_embed = None

        if self.with_box_refine:
            self.ref_mul = nn.ModuleList()
            self.ref_add = nn.ModuleList()
            self.tmp_cat = nn.ModuleList()
            self.sigmoid1 = nn.ModuleList()
            for _ in range(len(self.layers)):
                self.ref_mul.append(FloatFunctional())
                self.ref_add.append(FloatFunctional())
                self.tmp_cat.append(FloatFunctional())
                self.sigmoid1.append(nn.Sigmoid())
        else:
            self.ref_mul = FloatFunctional()
            self.sigmoid1 = nn.Sigmoid()
        self.bbox_embed = None
        self.class_embed = None
        self.valid_quant = QuantStub()

    @fx_wrap()
    def add_ref2(self, i: int, tmp: torch.Tensor, reference: torch.Tensor):
        a, b = tmp[..., :2], tmp[..., 2:]
        a = self.ref_add[i].add(a, reference)
        tmp = self.tmp_cat[i].cat([a, b], dim=-1)
        return tmp

    @fx_wrap()
    def gen_valid_ratio(self, valid_ratios: torch.Tensor):
        valid_ratio2 = self.valid_quant(valid_ratios)[:, None]
        valid_ratio4 = None
        if self.bbox_embed is not None or self.as_two_stage:
            valid_ratio4 = torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            valid_ratio4 = self.valid_quant(valid_ratio4)
        return valid_ratio2, valid_ratio4

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_masks: Optional[torch.Tensor] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points_unact: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,  # nlvl, 2
        valid_ratios: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward function.

        Args:
            query: The main query tensor to the transformer decoder.
            key: Tensor used as the 'key' in the attention mechanism.
            value: Tensor used as the 'value' in the attention mechanism.
            query_pos: Positional encodings for the query tensor.
            key_pos: Positional encodings for the key tensor.
            attn_masks:  attention masks that can be used to prevent the model
                from attending to certain positions within the sequence,
                often used for masking out padding tokens.
            query_key_padding_mask: Mask to prevent attention to certain
                    positions in the query.
            key_padding_mask: Mask to prevent attention to certain positions
                    in the key tensor.
            reference_points_unact: Unactivated reference points for
                    deformable attention.
            spatial_shapes: Spatial shapes of feature maps.
            valid_ratios: Ratios of valid points in spatial shapes.
        """
        output = query
        valid_ratio2, valid_ratio4 = self.gen_valid_ratio(valid_ratios)

        if self.bbox_embed is None:
            valid_ratios = valid_ratio4 if self.as_two_stage else valid_ratio2
            reference_points = self.sigmoid1(reference_points_unact)
            reference_points_input = self.ref_mul.mul(
                reference_points[:, :, None], valid_ratios
            )

        intermediate = []
        intermediate_tmp = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if self.bbox_embed is not None:
                reference_points = self.sigmoid1[layer_idx](
                    reference_points_unact
                )
                if layer_idx == 0 and not self.as_two_stage:
                    reference_points_input = self.ref_mul[layer_idx].mul(
                        reference_points[:, :, None], valid_ratio2
                    )
                else:
                    reference_points_input = self.ref_mul[layer_idx].mul(
                        reference_points[:, :, None], valid_ratio4
                    )

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,  # nlvl, 2
                **kwargs,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                intermediate_tmp.append(tmp)
                if self.as_two_stage or layer_idx > 0:
                    new_reference_points = self.ref_add[layer_idx].add(
                        tmp, reference_points_unact
                    )
                else:
                    new_reference_points = self.add_ref2(
                        layer_idx, tmp, reference_points_unact
                    )
                reference_points_unact = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points_unact)

        if self.return_intermediate:
            return (
                intermediate,
                intermediate_tmp,
                intermediate_reference_points,
            )

        return output, reference_points_unact

    def set_qconfig(self):
        modules_list = [
            self.sigmoid1,
            self.ref_mul,
            self.valid_quant,
        ]
        if self.with_box_refine:
            modules_list.append(self.ref_add)
        for module in modules_list:
            module.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
        for layer in self.layers:
            layer.attentions[
                0
            ].identity_add.qconfig = qconfig_manager.get_qconfig(  # noqa E501
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
            layer.attentions[
                1
            ].qconfig = qconfig_manager.get_qconfig(  # noqa E501
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
            layer.ffns[0].layers[
                -2
            ].qconfig = qconfig_manager.get_qconfig(  # noqa E501
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
            layer.ffns[
                0
            ].add_identity_op.qconfig = qconfig_manager.get_qconfig(  # noqa E501
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )


@OBJECT_REGISTRY.register
class DeformableDetrTransformer(nn.Module):
    """Transformer module for Deformable DETR.

    Args:
        encoder: encoder module.
        decoder: decoder module.
        as_two_stage: whether to use two-stage transformer.
        num_feature_levels: number of feature levels.
        two_stage_num_proposals: number of proposals in two-stage transformer.
            Only used when as_two_stage is True.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        num_feature_levels: int = 4,
        as_two_stage: bool = False,
        two_stage_num_proposals: int = 300,
    ):
        super(DeformableDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.as_two_stage = as_two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        self.embed_dim = self.encoder.embed_dim

        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dim)
        )

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
            self.enc_output_norm = nn.LayerNorm(self.embed_dim)
            self.pos_trans = nn.Linear(self.embed_dim * 2, self.embed_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dim * 2)
            self.ref4_quant = QuantStub()
            self.add_proposal = FloatFunctional()

        else:
            self.reference_points = nn.Linear(self.embed_dim, 2)
            self.query_quant = QuantStub()
            self.query_pos_quant = QuantStub()

        self.feat_cat = FloatFunctional()

        self.lvl_pos_embed_quant = QuantStub()
        self.ref_enc_quant = QuantStub()
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                if hasattr(m, "init_weights"):
                    m.init_weights()
                elif hasattr(m, "_reset_parameters"):
                    m._reset_parameters()
        if not self.as_two_stage:
            nn.init.xavier_normal_(self.reference_points.weight.data, gain=1.0)
            nn.init.constant_(self.reference_points.bias.data, 0.0)
        nn.init.normal_(self.level_embeds)

    @fx_wrap()
    def gen_encoder_output_proposals(
        self,
        memory_padding_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        reference_points: torch.Tensor,
    ):
        """Generate 4d reference points for decoder input.

        Args:
            memory_padding_mask: Feature mask with shape (bs, feat_len).
            spatial shapes: Multi scale features scatial shapes tensor with
                shape like (num_levels, 2)
            reference_points: 2d reference points with shape (bs, feat_len, 2)
        """
        N = memory_padding_mask.shape[0]
        proposals = []
        wh_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            wh = (
                torch.ones([N, H * W, 2]).to(reference_points.device)
                * 0.05
                * (2.0 ** lvl)
            )
            wh_list.append(wh)
        wh = torch.cat(wh_list, dim=1)
        proposals = torch.cat([reference_points, wh], dim=-1)

        output_proposals = proposals
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).all(-1, keepdim=True)

        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if self.training:
            output_proposals = output_proposals.masked_fill(
                memory_padding_mask.unsqueeze(-1), 12
            )
            output_proposals = output_proposals.masked_fill(
                ~output_proposals_valid, 12
            )
        return output_proposals, output_proposals_valid

    @fx_wrap()
    def gen_encoder_output_memory(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        output_proposals_valid: torch.Tensor,
    ):
        """Filter memory with input mask and valid proposal mask.

        Args:
            memory: Feature mask with shape (bs, feat_len, embed_dim).
            memory_padding_mask: Input feature mask with shape (bs, feat_len)
            output_proposals_valid: proposal mask with shape (bs, feat_len, 1)
        """
        output_memory = memory
        if self.training:
            output_memory = output_memory.masked_fill(
                memory_padding_mask.unsqueeze(-1), float(0)
            )
            output_memory = output_memory.masked_fill(
                ~output_proposals_valid, float(0)
            )
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory

    @fx_wrap()
    def get_reference_points(
        self, spatial_shapes: torch.Tensor, valid_ratios: torch.Tensor
    ):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes: The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios: The ratios of valid points on the feature map,
                has shape (bs, num_levels, 2)

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        device = spatial_shapes.device
        for lvl, (H, W) in enumerate(spatial_shapes):
            # generate reference points
            ref_y, ref_x = torch.meshgrid(
                torch.arange(0.5, H, 1, dtype=torch.float32, device=device),
                torch.arange(0.5, W, 1, dtype=torch.float32, device=device),
            )

            # normalize reference points
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H
            )
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W
            )
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)

        # concat all the reference points together: [bs, all hw, 2]
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, all h*w, 4, 2]
        # 每个reference points都在每个level feature上有个初始的采样点
        multi_level_reference_points = (
            reference_points[:, :, None] * valid_ratios[:, None]
        )
        return reference_points, multi_level_reference_points

    def get_valid_ratio(self, mask: torch.Tensor):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(
        self,
        proposals: torch.Tensor,
        num_pos_feats: int = 128,
        temperature: int = 10000,
    ):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device
        )
        dim_t = temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats
        )
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)
        return pos

    @fx_wrap()
    def before_encoder(
        self,
        multi_level_feats: List[torch.Tensor],
        multi_level_masks: List[torch.Tensor],
        multi_level_pos_embeds: List[torch.Tensor],
    ):
        """Prepare encoder inputs.

        This method flattens and transposes the input features and embeddings,
        calculates valid ratios, and generates reference points for each level.
        It's used to prepare the data for the deformable attention mechanism in
        the encoder.

        Args:
            multi_level_feats: List of feature maps from different levels, each
                has shape (feat_h, feat_w)
            multi_level_masks: List of masks corresponding to the feature maps.
            multi_level_pos_embeds: List of positional embeddings for each
                    feature map level.
        """
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            valid_ratio = self.get_valid_ratio(mask)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
            valid_ratios.append(valid_ratio)
        feat_flatten = self.feat_cat.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        valid_ratios = torch.stack(valid_ratios, 1)
        (
            reference_points,
            multi_level_reference_points,
        ) = self.get_reference_points(spatial_shapes, valid_ratios)
        multi_level_reference_points = self.ref_enc_quant(
            multi_level_reference_points
        )
        lvl_pos_embed_flatten = self.lvl_pos_embed_quant(lvl_pos_embed_flatten)

        return (
            feat_flatten,
            lvl_pos_embed_flatten,
            mask_flatten,
            spatial_shapes,
            valid_ratios,
            reference_points,
            multi_level_reference_points,
        )

    @fx_wrap()
    def before_decoder(
        self,
        memory: torch.Tensor,
        mask_flatten: torch.Tensor,
        spatial_shapes: torch.Tensor,
        reference_points: torch.Tensor,
        query_embed: torch.Tensor,
    ):
        """Process inputs before feeding them to the decoder.

        Args:
            memory: The encoded memory tensor from the encoder with
                shape (bs, feat_len, embed_dim).
            mask_flatten: Flattened mask tensor with shape (bs, feat_len).
            spatial_shapes: The spatial shapes of feature maps.
            reference_points: 2d Reference points for deformable attention
                with shape (bs, feat_len, 2).
            query_embed: Tensor containing query embeddings, required unless in
                two-stage mode.
        """
        bs, _, c = memory.shape
        if self.as_two_stage:
            (
                output_proposals,
                output_proposals_valid,
            ) = self.gen_encoder_output_proposals(
                mask_flatten, spatial_shapes, reference_points
            )
            output_memory = self.gen_encoder_output_memory(
                memory, mask_flatten, output_proposals_valid
            )

            enc_outputs_class = self.decoder.class_embed[
                self.decoder.num_layers
            ](output_memory)
            enc_outputs_coord_unact = self.add_proposal.add(
                self.decoder.bbox_embed[self.decoder.num_layers](
                    output_memory
                ),
                self.ref4_quant(output_proposals),
            )

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], topk, dim=1
            )[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )
            topk_coords_unact = topk_coords_unact.detach()
            reference_points_unact = topk_coords_unact

            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))
            )
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
            return (
                query,
                query_pos,
                reference_points_unact,
                enc_outputs_class,
                enc_outputs_coord_unact,
            )
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            query_pos = self.query_pos_quant(query_pos)
            reference_points_unact = self.reference_points(query_pos)
            query = self.query_quant(query)

            return query, query_pos, reference_points_unact, None, None

    def forward(
        self,
        multi_level_feats: List[torch.Tensor],
        multi_level_masks: List[torch.Tensor],
        multi_level_pos_embeds: List[torch.Tensor],
        query_embed: Optional[torch.Tensor] = None,
    ):
        """Forward pass for the Deformable DETR model.

        Args:
            multi_level_feats: Feature maps from various levels.
            multi_level_masks: Corresponding masks for the feature maps.
            multi_level_pos_embeds: Positional embeddings for
                    each feature map level.
            query_embed: Query embeddings for the decoder. required unless in
                    two-stage mode.
        """
        assert self.as_two_stage or query_embed is not None

        (
            feat_flatten,
            lvl_pos_embed_flatten,
            mask_flatten,
            spatial_shapes,
            valid_ratios,
            reference_points,
            multi_level_reference_points,
        ) = self.before_encoder(
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
        )

        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=multi_level_reference_points,
        )

        (
            query,
            query_pos,
            init_reference_unact,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.before_decoder(
            memory,
            mask_flatten,
            spatial_shapes,
            reference_points,
            query_embed,
        )

        # decoder
        inter_states, inter_bbox_out, inter_references = self.decoder(
            query=query,  # bs, num_queries, embed_dims
            key=None,
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=query_pos,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points_unact=init_reference_unact,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            valid_ratios=valid_ratios,  # bs, nlvl, 2
        )

        inter_references_out_unact = inter_references
        if self.as_two_stage:
            return (
                inter_states,
                inter_bbox_out,
                init_reference_unact,
                inter_references_out_unact,
                enc_outputs_class,
                enc_outputs_coord_unact,
            )
        return (
            inter_states,
            inter_bbox_out,
            init_reference_unact,
            inter_references_out_unact,
            None,
            None,
        )

    def set_qconfig(self):

        modules_list = [
            self.ref_enc_quant,
        ]
        length = self.decoder.num_layers
        if self.as_two_stage:
            modules_list.append(self.ref4_quant)
            modules_list.append(self.add_proposal)
            modules_list.append(self.enc_output)
            modules_list.append(self.decoder.bbox_embed[length].layers[-1])
        else:
            modules_list.append(self.reference_points)
        for module in modules_list:
            module.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
        for module in [self.encoder, self.decoder]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
