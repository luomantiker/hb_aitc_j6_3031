# Copyright (c) Horizon Robotics. All rights reserved.

import math

import horizon_plugin_pytorch as horizon
import horizon_plugin_pytorch.nn.quantized as quantized
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import FixedScaleObserver, QuantStub
from torch.quantization import DeQuantStub

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.task_modules.motr.motr_utils import _get_clones
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["MotrHead"]


@OBJECT_REGISTRY.register
class MotrHead(nn.Module):
    """Implements the MOTR head.

    Args:
        transformer: transformer module.
        num_classes: Number of categories excluding the background.
        in_channels: Number of channels in the input featuremaps.
        max_per_img: max number of object in single image.
    """

    def __init__(
        self,
        transformer: nn.Module,
        num_classes: int = 1,
        in_channels: int = 2048,
        max_per_img: int = 100,
    ):
        super(MotrHead, self).__init__()
        self.num_queries = max_per_img
        self.transformer = transformer
        hidden_dim = self.transformer.d_model
        self.return_intermediate_dec = self.transformer.return_intermediate_dec
        self.num_classes = num_classes

        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.refpoint_embed = nn.Embedding(self.num_queries, 4)

        input_proj_list = []
        self.in_channels = in_channels
        for in_channel in self.in_channels:
            input_proj_list.append(
                nn.Sequential(
                    ConvModule2d(
                        in_channels=in_channel,
                        out_channels=hidden_dim,
                        kernel_size=1,
                        bias=True,
                    )
                )
            )

        self.input_proj = nn.ModuleList(input_proj_list)

        self.class_embed = ConvModule2d(
            in_channels=hidden_dim,
            out_channels=num_classes,
            kernel_size=1,
            bias=True,
        )

        num_pred = transformer.decoder.num_layers

        self.class_embed = _get_clones(self.class_embed, num_pred)

        for class_emd in self.class_embed:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            class_emd.conv_list[0].bias.data = (
                torch.ones(num_classes) * bias_value
            )

        self.query_embed_weight_quant = QuantStub(scale=None)
        self.ref_quant = QuantStub(scale=None)
        self.query_embed_mask_quant = QuantStub(scale=1.0)

        self.query_tracked_quant = QuantStub(scale=None)
        self.query_tracked_ref_pts_quant = QuantStub(scale=None)
        self.query_tracked_mask_quant = QuantStub(scale=1.0)

        self.cat_tracked_empty_query = quantized.FloatFunctional()
        self.cat_tracked_empty_mask = quantized.FloatFunctional()

        self.mask_matmul = quantized.FloatFunctional()
        self.mask_add_scalar = quantized.FloatFunctional()
        self.mask_mul_scalar = quantized.FloatFunctional()

        self.bbox_add1 = quantized.FloatFunctional()
        self.bbox_add2 = quantized.FloatFunctional()
        self.bbox_cat1 = quantized.FloatFunctional()
        self.cat_tracked_empty_refs = quantized.FloatFunctional()
        self.dequant = DeQuantStub()

        self._init_weights()

    def _init_weights(self):
        for proj in self.input_proj:
            for m in proj.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=1)
                    nn.init.constant_(m.bias, 0)

    @fx_wrap
    def _get_embedding_mask_query(self, batch_size, mask_query):

        mask_query_embedding_weight = torch.ones(
            (batch_size, 1, 1, mask_query.shape[3]),
            dtype=torch.float,
            device=mask_query.device,
        )
        return mask_query_embedding_weight

    def forward(self, feats, query_pos, ref_pts, mask_query):
        srcs = []
        for idx_feat, feat in enumerate(feats):
            srcs.append(self.input_proj[idx_feat](feat))

        batch_size = feats[0].shape[0]
        query_pos = self.query_tracked_quant(query_pos)
        ref_pts = self.query_tracked_ref_pts_quant(ref_pts)
        mask_query = self.query_tracked_mask_quant(mask_query)

        mask_query_embedding_weight = self._get_embedding_mask_query(
            batch_size, mask_query
        )

        mask_query_embedding_weight = self.query_embed_mask_quant(
            mask_query_embedding_weight
        )
        mask_query_cat = self.cat_tracked_empty_mask.cat(
            [mask_query_embedding_weight, mask_query], dim=3
        )
        tgt_mask1 = mask_query_cat.reshape(
            batch_size, 1, 1, mask_query_cat.shape[3]
        )
        tgt_mask2 = mask_query_cat.reshape(
            batch_size, 1, mask_query_cat.shape[3], 1
        )
        tgt_mask = self.mask_matmul.matmul(tgt_mask2, tgt_mask1)
        tgt_mask = self.mask_mul_scalar.mul_scalar(tgt_mask, 100)
        tgt_mask = self.mask_add_scalar.add_scalar(tgt_mask, -100)
        tgt_mask = tgt_mask.squeeze(1).squeeze(0)

        (
            query_pos_cat,
            init_reference_points,
            tgt_mask,
            track_mask_tmp4,
        ) = self._build_query(tgt_mask, query_pos, ref_pts, batch_size)

        hs, new_reference_points_before_sigmoid = self.transformer(
            srcs,
            query_pos_cat,
            ref_pts=init_reference_points,
            tgt_mask=tgt_mask,
            track_mask=track_mask_tmp4,
        )

        outputs_coords = []
        outputs_classes = []
        if self.return_intermediate_dec:
            for lvl in range(len(hs)):
                outputs_class = self.class_embed[lvl](hs[lvl])
                outputs_classes.append(self.dequant(outputs_class))
                outputs_coords.append(
                    self.dequant(new_reference_points_before_sigmoid[lvl])
                )
        else:
            outputs_class = self.class_embed[-1](hs[0])
            outputs_classes.append(self.dequant(outputs_class))
            outputs_coords.append(
                self.dequant(new_reference_points_before_sigmoid[0])
            )
        out_hs = self.dequant(hs[-1])
        return outputs_classes, outputs_coords, out_hs

    @fx_wrap()
    def _build_query(self, tgt_mask, query_pos, ref_pts, batch_size):

        track_mask_tmp1, track_mask_tmp2 = torch.split(
            tgt_mask, tgt_mask.shape[0] // 2, dim=0
        )
        track_mask_tmp3, track_mask_tmp4 = torch.split(
            track_mask_tmp2, track_mask_tmp2.shape[1] // 2, dim=1
        )

        query_embed_reshape = (
            self.query_embed.weight.transpose(0, 1)
            .contiguous()
            .view(
                batch_size,
                self.query_embed.weight.shape[1],
                2,
                self.num_queries // 2,
            )
        )

        query_pos_embedding_weight = self.query_embed_weight_quant(
            query_embed_reshape
        )

        query_pos_cat = self.cat_tracked_empty_query.cat(
            [query_pos_embedding_weight, query_pos], dim=2
        )

        reference_points_init = self.ref_quant(
            self.refpoint_embed.weight.transpose(0, 1)
            .contiguous()
            .view(
                batch_size,
                self.refpoint_embed.weight.shape[1],
                2,
                self.num_queries // 2,
            )
        )
        init_reference_points = self.cat_tracked_empty_refs.cat(
            [reference_points_init, ref_pts], dim=2
        )
        return (
            query_pos_cat,
            init_reference_points,
            tgt_mask,
            track_mask_tmp4,
        )

    def fuse_model(self):
        for m in self.input_proj:
            for _m in m:
                if hasattr(_m, "fuse_model"):
                    _m.fuse_model()
        if hasattr(self.transformer, "fuse_model"):
            self.transformer.fuse_model()

    def set_qconfig(self):

        self.query_embed.qconfig = None
        self.refpoint_embed.qconfig = None
        int16_modules_list = [
            self.query_tracked_ref_pts_quant,
            self.ref_quant,
            self.cat_tracked_empty_refs,
            self.query_tracked_quant,
            self.query_embed_weight_quant,
            self.cat_tracked_empty_query,
        ]
        for module in int16_modules_list:
            module.qconfig = horizon.quantization.get_default_qat_qconfig(
                dtype="qint16"
            )

        fixscale_int8_modules_list = [
            self.query_embed_mask_quant,
            self.query_tracked_mask_quant,
        ]
        for module in fixscale_int8_modules_list:
            module.qconfig = horizon.quantization.get_default_qat_qconfig(
                dtype="qint8",
                activation_qkwargs={
                    "observer": FixedScaleObserver,
                    "scale": 1.0,
                },
            )

        for m in self.class_embed:
            m.qconfig = horizon.quantization.get_default_qat_out_qconfig()
        if hasattr(self.transformer, "set_qconfig"):
            self.transformer.set_qconfig()
