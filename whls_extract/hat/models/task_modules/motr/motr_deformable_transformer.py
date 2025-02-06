import math

import horizon_plugin_pytorch as horizon
import horizon_plugin_pytorch.nn.quantized as quantized
import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.nn import LayerNorm as LayerNorm2d
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import FixedScaleObserver, QuantStub
from torch.nn.init import constant_, normal_, xavier_uniform_
from torch.quantization import DeQuantStub

from hat.models.base_modules.attention import (
    HorizonMultiheadAttention as MultiheadAttention,
)
from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.task_modules.motr.motr_utils import _get_clones
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["MotrDeformableTransformer"]


@OBJECT_REGISTRY.register
class MotrDeformableTransformer(nn.Module):
    """Implements the motr deformable transformer.

    Args:
        pos_embed: The feature pos embed module.
        d_model: The feature dimension.
        nhead: Parallel attention heads.
        num_queries: The number of query.
        num_encoder_layers: Number of `TransformerEncoderLayer`.
        num_decoder_layers: Number of `TransformerDecoderLayer`.
        dim_feedforward: The hidden dimension for FFNs used in both
            encoder and decoder.
        dropout: Probability of an element to be zeroed. Default 0.1.
        return_intermediate_dec: Whether to return the intermediate
            output from each TransformerDecoderLayer or only the last
            TransformerDecoderLayer. Default False.
        num_feature_levels: The num of featuremap.
        enc_n_points: The num of encoder deformable attention points.
        dec_n_points: The num of decoder deformable attention points.
        extra_track_attn: Whether enable track attention.
    """

    def __init__(
        self,
        pos_embed: nn.Module,
        d_model: int = 256,
        nhead: int = 8,
        num_queries: int = 300,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        return_intermediate_dec: bool = False,
        num_feature_levels: int = 1,
        enc_n_points: int = 4,
        dec_n_points: int = 4,
        extra_track_attn: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.return_intermediate_dec = return_intermediate_dec

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        self.pos_embed = pos_embed

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=dec_n_points,
            extra_track_attn=extra_track_attn,
            num_queries=self.num_queries,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            self.d_model,
            return_intermediate_dec,
        )

        self.cat_tracked_empty_refs = quantized.FloatFunctional()
        self.sigmoid = torch.nn.Sigmoid()
        self.reference_points_encoder_quant = QuantStub()

        self.pos_quant = nn.ModuleList()
        self.pos_embed_add = nn.ModuleList()
        self.pos_embed_quant = nn.ModuleList()

        for _ in range(num_feature_levels):
            self.pos_quant.append(QuantStub(scale=None))
            self.pos_embed_add.append(quantized.FloatFunctional())
            self.pos_embed_quant.append(QuantStub(scale=None))

        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, d_model)
        )

        self.cat_src_flatten = quantized.FloatFunctional()
        self.cat_lvl_pos_embed = quantized.FloatFunctional()
        self.dequant = DeQuantStub()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() == 4:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MotrDeformAttn):
                m._reset_parameters()

        for idx, bbox_layer in enumerate(self.decoder.bbox_embed):
            if idx == 0:
                nn.init.constant_(
                    bbox_layer[-1].conv_list[0].bias.data[2:], -2.0
                )
            else:
                nn.init.constant_(bbox_layer[-1].conv_list[0].bias.data, 0)
            nn.init.constant_(bbox_layer[-1].conv_list[0].weight.data, 0.0)

        normal_(self.level_embed)

    @fx_wrap
    def _get_feature_mask(self, feat):
        mask = (
            torch.zeros(feat.shape[-2:], device=feat.device)
            .to(torch.bool)
            .unsqueeze(0)
        )
        return mask

    @fx_wrap
    def _get_pos_emb(self, mask):
        return self.pos_embed(mask)

    @fx_wrap
    def _get_lvl_pos_embed(self, pos_embeds_list, feat_idx, pos_embed):

        pos_embed_tmp = pos_embeds_list[feat_idx].view(1, -1, 1, 1)
        lvl_pos_embed = self.pos_embed_add[feat_idx].add(
            pos_embed, self.pos_embed_quant[feat_idx](pos_embed_tmp)
        )
        return lvl_pos_embed

    def forward(self, srcs, query_embed, ref_pts, tgt_mask, track_mask):
        assert query_embed is not None
        masks = []
        spatial_shapes = []
        lvl_pos_embed_flatten = []
        pos_embeds_list = torch.split(self.level_embed, 1, dim=0)
        for feat_idx, feat in enumerate(srcs):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            mask = self._get_feature_mask(feat)

            masks.append(mask)
            pos_l = self._get_pos_emb(mask)
            pos_embed = self.pos_quant[feat_idx](pos_l)
            lvl_pos_embed = self._get_lvl_pos_embed(
                pos_embeds_list, feat_idx, pos_embed
            )
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        reference_points_encoder = self.get_reference_points(
            spatial_shapes, masks, device=feat.device
        )

        reference_points_tmp = reference_points_encoder.permute(0, 3, 2, 1)
        reference_points_tmp = torch.split(reference_points_tmp, 1, dim=2)
        reference_points_tmp = reference_points_tmp[0]

        reference_points_tmp = reference_points_tmp.reshape(
            reference_points_tmp.shape[0],
            reference_points_tmp.shape[1],
            feat.shape[2],
            feat.shape[3],
        )
        reference_points_encoder = self.reference_points_encoder_quant(
            reference_points_tmp
        )

        memory = self.encoder(
            srcs[0],
            reference_points_encoder,
            lvl_pos_embed_flatten[0],
        )
        memory = [memory]

        tgt = query_embed

        init_reference_points = ref_pts

        reference_points = self.sigmoid(init_reference_points)

        hs, new_reference_points_before_sigmoid = self.decoder(
            tgt,
            reference_points,
            init_reference_points,
            memory,
            query_embed,
            tgt_mask,
            track_mask,
        )

        return hs, new_reference_points_before_sigmoid

    @fx_wrap
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0]).unsqueeze(0)
        valid_W = torch.sum(~mask[:, 0, :]).unsqueeze(0)

        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @fx_wrap
    def get_reference_points(self, spatial_shapes, masks, device):

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.int64, device=device
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H_ - 0.5, H_, dtype=torch.float32, device=device
                ),
                torch.linspace(
                    0.5, W_ - 0.5, W_, dtype=torch.float32, device=device
                ),
            )
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H_
            )
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W_
            )
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def fuse_model(self):
        if hasattr(self.encoder, "fuse_model"):
            self.encoder.fuse_model()
        if hasattr(self.decoder, "fuse_model"):
            self.decoder.fuse_model()

    def set_qconfig(self):

        modules_list = [
            self.sigmoid,
            self.reference_points_encoder_quant,
        ]
        for module in modules_list:
            module.qconfig = horizon.quantization.get_default_qat_qconfig(
                dtype="qint16"
            )

        if hasattr(self.encoder, "set_qconfig"):
            self.encoder.set_qconfig()
        if hasattr(self.decoder, "set_qconfig"):
            self.decoder.set_qconfig()


class DeformableTransformerDecoderLayer(nn.Module):
    """Implements the motr deformable transformer decoder layer.

    Args:
        d_model: The feature dimension.
        d_ffn: The hidden dimension for FFNs used in both
            encoder and decoder.
        dropout: Probability of an element to be zeroed. Default 0.1.
        n_levels: The num of featuremap.
        n_heads: Parallel attention heads.
        n_points: The num of decoder deformable attention points.
        extra_track_attn: Whether enable track attention.
        num_queries: The num of query.
    """

    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        n_levels=4,
        n_heads=8,
        n_points=4,
        extra_track_attn=False,
        num_queries=300,
    ):
        super().__init__()

        self.num_head = n_heads
        self.num_queries = num_queries
        # cross attention
        self.cross_attn = MotrDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm2d(normalized_shape=[d_model, 1, 1], dim=1)

        self.self_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm2d(normalized_shape=[d_model, 1, 1], dim=1)

        self.linear1 = ConvModule2d(
            in_channels=d_model,
            out_channels=d_ffn,
            kernel_size=1,
            bias=True,
            act_layer=nn.ReLU(inplace=True),
        )

        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Conv2d(d_ffn, d_model, 1)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = LayerNorm2d(normalized_shape=[d_model, 1, 1], dim=1)

        # update track query_embed
        self.extra_track_attn = extra_track_attn
        if self.extra_track_attn:
            self.update_attn = MultiheadAttention(
                d_model, n_heads, bias=False, dropout=dropout
            )
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = LayerNorm2d(normalized_shape=[d_model, 1, 1], dim=1)

        self.add_pos1 = quantized.FloatFunctional()
        self.add_pos2 = quantized.FloatFunctional()
        self.add_pos3 = quantized.FloatFunctional()
        self.add_ffn = quantized.FloatFunctional()

        self.add_track_attn = quantized.FloatFunctional()
        self.cat_track_tgt = quantized.FloatFunctional()
        self.add_self_attn = quantized.FloatFunctional()
        self.add_cross_attn = quantized.FloatFunctional()

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.linear1(tgt)))
        tgt = self.add_ffn.add(tgt, self.dropout4(tgt2))
        tgt = self.norm3(tgt)
        return tgt

    def _forward_self_attn(self, tgt, query_pos, tgt_mask, track_mask):
        if self.extra_track_attn:
            tgt = self._forward_track_attn(tgt, query_pos, track_mask)
        q = k = self.add_pos2.add(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask)[0]
        tgt = self.add_self_attn.add(tgt, self.dropout2(tgt2))
        tgt = self.norm2(tgt)
        return tgt

    @fx_wrap()
    def _get_split_c(self, q):
        return q.shape[2] // 2

    def _forward_track_attn(self, tgt, query_pos, track_mask):

        q = self.add_pos1.add(tgt, query_pos)

        split_c = self._get_split_c(q)
        tgt_tmp1, tgt_tmp2 = torch.split(tgt, split_c, dim=2)
        q_tmp1, q_tmp2 = torch.split(q, split_c, dim=2)
        k_tmp2 = q_tmp2
        tgt2 = self.update_attn(
            q_tmp2,
            k_tmp2,
            tgt_tmp2,
            attn_mask=track_mask,
        )[0]

        tgt2 = self.add_track_attn.add(tgt_tmp2, self.dropout5(tgt2))
        tgt_norm = self.norm4(tgt2)
        tgt = self.cat_track_tgt.cat([tgt_tmp1, tgt_norm], dim=2)
        return tgt

    def forward(
        self, tgt, query_pos, reference_points, src, tgt_mask, track_mask
    ):
        # self attention
        tgt = self._forward_self_attn(tgt, query_pos, tgt_mask, track_mask)
        # cross attention
        tgt2 = self.cross_attn(
            self.add_pos3.add(tgt, query_pos), reference_points, src
        )
        tgt = self.add_cross_attn.add(tgt, self.dropout1(tgt2))
        tgt = self.norm1(tgt)
        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt

    def fuse_model(self):
        self.linear1.fuse_model()

    def set_qconfig(self):
        modules_list = [
            self.norm1,
            self.norm2,
            self.linear1,
            self.linear2,
            self.norm3,
            self.self_attn.matmul,
            self.self_attn.attn_matmul,
            self.self_attn.out_proj,
            self.update_attn.matmul,
            self.update_attn.attn_matmul,
            self.update_attn.out_proj,
        ]
        for module in modules_list:
            module.qconfig = horizon.quantization.get_default_qat_qconfig(
                dtype="qint16"
            )

        fix_scale_modules_list = [
            self.self_attn.attn_mask_quant,
            self.update_attn.attn_mask_quant,
        ]
        for module in fix_scale_modules_list:
            module.qconfig = horizon.quantization.get_default_qat_qconfig(
                dtype="qint8",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": 1.0,
                },
            )
        if hasattr(self.cross_attn, "set_qconfig"):
            self.cross_attn.set_qconfig()


class DeformableTransformerDecoder(nn.Module):
    """Implements the motr deformable transformer decoder module.

    Args:
        decoder_layer: The decoder layer.
        num_layers: The num decoder layer of decoder module.
        hidden_dim: The hidden dimension for querypos and bbox pred.
        return_intermediate_dec: Whether to return the intermediate
            output from each TransformerDecoderLayer or only the last
            TransformerDecoderLayer. Default False.
    """

    def __init__(
        self, decoder_layer, num_layers, hidden_dim, return_intermediate=False
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        bbox_embed = nn.Sequential(
            ConvModule2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                bias=True,
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                bias=True,
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                in_channels=hidden_dim,
                out_channels=4,
                kernel_size=1,
                bias=True,
            ),
        )
        self.bbox_embed = _get_clones(bbox_embed, num_layers)

        self.new_reference_points_sigmoids = _get_clones(
            nn.Sigmoid(), num_layers - 1
        )
        self.reference_points_add = nn.ModuleList()
        self.mul_pos_scale = nn.ModuleList()
        self.query_pos_scale = nn.ModuleList()
        self.query_sineembed_pos = nn.ModuleList()

        for _ in range(num_layers):
            self.reference_points_add.append(FF())
            self.query_sineembed_pos.append(SinPos())

        for _ in range(num_layers - 1):
            self.mul_pos_scale.append(FF())
            self.query_pos_scale.append(
                nn.Sequential(
                    ConvModule2d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=1,
                        bias=True,
                        act_layer=nn.ReLU(inplace=True),
                    ),
                    ConvModule2d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=1,
                        bias=True,
                    ),
                )
            )

    def forward(
        self,
        tgt,
        reference_points,
        init_reference_points,
        src,
        query_pos,
        tgt_mask,
        track_mask,
    ):
        output = tgt
        intermediate = []
        intermediate_reference_points_before_sigmoids = []
        new_reference_points_before_sigmoid = init_reference_points
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            raw_query_pos = self.query_sineembed_pos[lid](
                reference_points_input
            )
            if lid != 0:
                pos_scale = self.query_pos_scale[lid - 1](output)
                query_pos = self.mul_pos_scale[lid - 1].mul(
                    pos_scale, raw_query_pos
                )
            else:
                query_pos = raw_query_pos
            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                tgt_mask,
                track_mask,
            )

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)

                new_reference_points = self.reference_points_add[lid].add(
                    tmp, new_reference_points_before_sigmoid
                )
                new_reference_points_before_sigmoid = new_reference_points
                if lid < len(self.layers) - 1:
                    new_reference_points = self.new_reference_points_sigmoids[
                        lid
                    ](new_reference_points)

                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points_before_sigmoids.append(
                    new_reference_points_before_sigmoid
                )

        if self.return_intermediate:
            return intermediate, intermediate_reference_points_before_sigmoids

        return [output], [new_reference_points_before_sigmoid]

    def fuse_model(self):

        modules_list = [
            self.bbox_embed,
            self.query_pos_scale,
        ]
        for module in modules_list:
            for m in module:
                for _m in m:
                    if hasattr(_m, "fuse_model"):
                        _m.fuse_model()

        modules_list = [
            self.layers,
            self.query_sineembed_pos,
        ]
        for module in modules_list:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):

        modules_list = [
            self.new_reference_points_sigmoids,
            self.query_pos_scale,
            self.mul_pos_scale,
            self.reference_points_add,
        ]
        for module in modules_list:
            module.qconfig = horizon.quantization.get_default_qat_qconfig(
                dtype="qint16"
            )

        modules_list = [
            self.layers,
            self.query_sineembed_pos,
        ]
        for module in modules_list:
            for layers in module:
                if hasattr(layers, "set_qconfig"):
                    layers.set_qconfig()
        for m in self.bbox_embed:
            m[-1].qconfig = horizon.quantization.get_default_qat_qconfig(
                dtype="qint16"
            )


class DeformableTransformerEncoderLayer(nn.Module):
    """Implements the motr deformable transformer encoder layer.

    Args:
        d_model: The feature dimension.
        d_ffn: The hidden dimension for FFNs used.
        dropout: Probability of an element to be zeroed. Default 0.1.
        n_levels: The num of featuremap.
        n_heads: Parallel attention heads.
        n_points: The num of decoder deformable attention points.
    """

    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MotrDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm2d(normalized_shape=[d_model, 1, 1], dim=1)

        self.linear1 = ConvModule2d(
            in_channels=d_model,
            out_channels=d_ffn,
            kernel_size=1,
            bias=True,
            act_layer=nn.ReLU(inplace=True),
        )

        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = ConvModule2d(
            in_channels=d_ffn,
            out_channels=d_model,
            kernel_size=1,
            bias=True,
        )

        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = LayerNorm2d(normalized_shape=[d_model, 1, 1], dim=1)
        self.add_pos = quantized.FloatFunctional()
        self.add_att = quantized.FloatFunctional()
        self.add_ffn = quantized.FloatFunctional()

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.linear1(src)))
        src = self.add_ffn.add(src, self.dropout3(src2))
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points):
        # self attention
        src1 = self.add_pos.add(src, pos)
        src2 = self.self_attn(src1, reference_points, [src])
        src = self.add_att.add(src, self.dropout1(src2))
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)
        return src

    def fuse_model(self):
        self.linear1.fuse_model()
        self.linear2.fuse_model()

    def set_qconfig(self):
        modules_list = [
            self.norm1,
            self.norm2,
            self.linear1,
            self.linear2,
            self.add_pos,
            self.add_att,
            self.add_ffn,
        ]
        for module in modules_list:
            module.qconfig = horizon.quantization.get_default_qat_qconfig(
                dtype="qint16"
            )
        if hasattr(self.self_attn, "set_qconfig"):
            self.self_attn.set_qconfig()


class DeformableTransformerEncoder(nn.Module):
    """Implements the motr deformable transformer encoder module.

    Args:
        encoder_layer: The encoder layer.
        num_layers: The num encoder layer of encoder module.
    """

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, reference_points, pos):
        output = src
        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                pos,
                reference_points,
            )

        return output

    def fuse_model(self):
        for module in self.layers:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        for module in self.layers:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


class SinPos(nn.Module):
    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 512,
        out_dim: int = 256,
        num_pos_feats: int = 128,
        temperature: int = 20,
    ):
        super().__init__()
        self.add_pos = FF()
        self.mul_dim_t = FF()
        self.mul_scale = FF()
        self.cat_1 = FF()
        self.cat_2 = FF()
        self.sin_model = horizon.nn.Sin()
        self.cos_model = horizon.nn.Cos()
        self.conv2 = nn.Sequential(
            ConvModule2d(
                in_channels=in_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                bias=True,
                act_layer=nn.ReLU(inplace=True),
            ),
            ConvModule2d(
                in_channels=hidden_dim,
                out_channels=out_dim,
                kernel_size=1,
                bias=True,
            ),
        )
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.quant_shape = QuantStub()

    @fx_wrap
    def _get_dim_t(self, device):
        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=device,
        )
        dim_t = 1 / (
            self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        )
        dim_t = dim_t.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return dim_t

    def forward(self, reference_points):
        bs, n_points, refs_h, refs_w = reference_points.size()
        reference_points = reference_points.unsqueeze(2)
        scale = 2 * math.pi
        dim_t = self._get_dim_t(device=reference_points.device)
        dim_t = self.quant_shape(dim_t)

        xywh_embed = self.mul_scale.mul_scalar(reference_points, scale)
        xywh_pos = self.mul_dim_t.mul(xywh_embed, dim_t)

        pos_x = self.cat_1.cat(
            (
                self.sin_model(xywh_pos[:, :, 0::2]),
                self.cos_model(xywh_pos[:, :, 1::2]),
            ),
            dim=3,
        )

        output = pos_x.view(reference_points.shape[0], -1, refs_h, refs_w)
        output = self.conv2(output)
        return output

    def fuse_model(self):
        for m in self.conv2:
            if hasattr(m, "fuse_model"):
                m.fuse_model()

    def set_qconfig(self):
        self.qconfig = horizon.quantization.get_default_qat_qconfig(
            dtype="qint16"
        )
        self.sin_model.qconfig = horizon.quantization.get_default_qat_qconfig(
            dtype="qint8"
        )
        self.cos_model.qconfig = horizon.quantization.get_default_qat_qconfig(
            dtype="qint8"
        )


class MotrDeformAttn(nn.Module):
    """Modify the deformable attntion for motr.

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
        grid_head = 8
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
                align_corners=True,
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
