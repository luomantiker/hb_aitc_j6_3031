# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict, List, Tuple

import torch
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import Linear
from torch.quantization import DeQuantStub

from hat.models.task_modules.bevformer.utils import (
    FFN,
    bias_init_with_prob,
    get_clone_module,
    xavier_init,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = ["BEVFormerDetDecoder"]


@OBJECT_REGISTRY.register
class BEVFormerDetDecoder(nn.Module):
    """The basic structure of BEVFormerDetDecoder.

    Args:
        decoder: Decoder module.
        embed_dims: The embedding dimension of Attention.
        num_reg_fcs: The num of reg fc.
        num_pred: The num of pred.
        bev_h: The height of bevfeat.
        bev_w: The width of bevfeat.
        code_size: Code size of bboxes.
        num_query: The num of object query.
        num_classes: The num of classes.
        pc_range: VCS range or point cloud range.
        is_compile: Whether for compile.
        post_process: Post process module.
        criterion: The num of camera.
        group_detr: The num group for query.
    """

    def __init__(
        self,
        decoder: nn.Module,
        embed_dims: int = 256,
        num_reg_fcs: int = 2,
        num_pred: int = 6,
        bev_h: int = 30,
        bev_w: int = 30,
        code_size: int = 10,
        num_query: int = 900,
        num_classes: int = 10,
        pc_range: List[float] = None,
        is_compile: bool = False,
        post_process: nn.Module = None,
        criterion: nn.Module = None,
        group_detr: int = 1,
    ):
        super(BEVFormerDetDecoder, self).__init__()
        self.num_query = num_query * group_detr
        self.group_detr = group_detr
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.decoder = decoder
        self.num_reg_fcs = num_reg_fcs
        self.cls_out_channels = num_classes
        self.num_classes = num_classes
        self.code_size = code_size
        self.num_pred = num_pred
        self.pc_range = pc_range
        self.is_compile = is_compile
        self.post_process = post_process
        self.criterion = criterion

        self._init_layers()
        self.init_weights()

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.

        num_pred = self.num_pred

        self.cls_branches = get_clone_module(fc_cls, num_pred)
        self.reg_branches = get_clone_module(reg_branch, num_pred)

        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.query_embedding = nn.Embedding(
            self.num_query, self.embed_dims * 2
        )

        self.quant_object_query_embed = QuantStub()
        self.sigmoid = torch.nn.Sigmoid()
        self.dequant = DeQuantStub()

    def init_weights(self) -> None:
        """Initialize weights."""
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)
        xavier_init(self.reference_points, distribution="uniform", bias=0.0)

    @fx_wrap()
    def _post_process(self, data: Dict, outputs: Dict) -> Dict:
        """Post process."""
        if self.training:
            loss_dict = self.criterion(outputs, data)
            return loss_dict
        else:
            if self.post_process is None:
                return outputs
            results = self.post_process(outputs)
            return results

    @autocast(enabled=False)
    @fx_wrap()
    def get_outputs(
        self,
        outputs_classes: List[Tensor],
        reference_out: List[Tensor],
        bbox_outputs: List[Tensor],
        bev_embed: Tensor,
    ) -> Dict:
        """Get the outputs."""
        new_outputs_coords = []
        for lvl in range(len(outputs_classes)):
            reference = reference_out[lvl].float()
            tmp = bbox_outputs[lvl].float()
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
                + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
                + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2])
                + self.pc_range[2]
            )
            new_outputs_coords.append(tmp)

        outputs_classes = torch.stack(outputs_classes).float()
        outputs_coords = torch.stack(new_outputs_coords).float()
        preds_dicts = {
            "bev_embed": bev_embed.float(),
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }
        return preds_dicts

    @fx_wrap()
    def get_spatial_shapes(self) -> Tensor:
        """Get the spatial shapes."""
        return torch.tensor([[self.bev_w, self.bev_h]], dtype=torch.float32)

    @fx_wrap()
    def get_object_query_embed(self) -> Tensor:
        """Get object query."""
        object_query_embed = self.quant_object_query_embed(
            self.query_embedding.weight
        )

        if not self.training:
            object_query_embed = object_query_embed[
                : self.num_query // self.group_detr
            ]
        return object_query_embed

    def bev_decoder(self, bev_embed: Tensor) -> List[Tensor]:
        """Decode the bev feat for object detection."""
        bs = bev_embed.size(0)
        object_query_embed = self.get_object_query_embed()
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1
        )
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        attn_mask = None
        init_reference_points = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed_compile_out = bev_embed
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references, bbox_outputs = self.decoder(
            query=query,
            key=bev_embed,
            value=bev_embed,
            query_pos=query_pos,
            reg_branches=self.reg_branches,
            spatial_shapes=self.get_spatial_shapes().to(query.device),
            init_reference_points=init_reference_points,
            attn_mask=attn_mask,
        )

        inter_references_out = inter_references

        outputs_classes = []
        outputs_coords = []
        reference_out = []

        for lvl in range(len(inter_states)):
            if lvl == 0:
                reference = init_reference_points
            else:
                reference = inter_references_out[lvl - 1]

            outputs_class = self.cls_branches[lvl](
                inter_states[lvl].permute(1, 0, 2)
            )
            tmp = bbox_outputs[lvl]
            outputs_classes.append(self.dequant(outputs_class))
            outputs_coords.append(self.dequant(tmp))
            reference_out.append(self.dequant(reference))

        if self.is_compile:
            bev_embed = self.dequant(bev_embed_compile_out)
            return (
                bev_embed,
                outputs_classes[-1],
                reference_out[-1],
                outputs_coords[-1],
            )
        else:
            bev_embed = self.dequant(bev_embed)
            return (bev_embed, outputs_classes, reference_out, outputs_coords)

    @fx_wrap()
    def get_bev_embed(self, bev_embed: Dict) -> Tensor:
        if bev_embed.dim() == 4:
            bev_embed = bev_embed.flatten(2).permute(0, 2, 1).contiguous()
        return bev_embed

    def forward(self, bev_embed: Tensor, data: Dict = None) -> Dict:
        """Forward BEVFormerDetDecoder."""
        bev_embed = self.get_bev_embed(bev_embed)
        outputs = self.bev_decoder(bev_embed)
        if self.is_compile:
            return outputs
        bev_embed, outputs_classes, reference_out, outputs_coords = outputs
        outputs = self.get_outputs(
            outputs_classes, reference_out, outputs_coords, bev_embed
        )
        return self._post_process(data, outputs)

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        from hat.utils import qconfig_manager

        for _, m in enumerate(self.cls_branches):
            m[0].qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )
            m[3].qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )
            m[-1].qconfig = qconfig_manager.get_default_qat_out_qconfig()

        for m in self.reg_branches:
            m[-1].qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )

        self.query_embedding.qconfig = None
        self.reg_branches[-1][
            -1
        ].qconfig = qconfig_manager.get_default_qat_out_qconfig()

        int16_module = [
            self.reference_points,
            self.sigmoid,
        ]
        for m in int16_module:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )
        if hasattr(self.decoder, "set_qconfig"):
            self.decoder.set_qconfig()

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""
        from horizon_plugin_pytorch import quantization

        for i in range(len(self.reg_branches)):
            torch.quantization.fuse_modules(
                self,
                [f"reg_branches.{i}.0", f"reg_branches.{i}.1"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )
            torch.quantization.fuse_modules(
                self,
                [f"reg_branches.{i}.2", f"reg_branches.{i}.3"],
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )

        if hasattr(self.decoder, "fuse_model"):
            self.decoder.fuse_model()


@OBJECT_REGISTRY.register
class DetectionTransformerDecoder(nn.Module):
    """The basic structure of DetectionTransformerDecoder.

    Args:
        num_layers: The num of encoder layers.
        decoder_layer: The decoder layer.
        return_intermediate: Whether to return intermediate outputs.
    """

    def __init__(
        self,
        num_layers: int = 3,
        decoder_layer: nn.Module = None,
        return_intermediate: bool = False,
    ):
        super().__init__()
        self.layers = get_clone_module(decoder_layer, num_layers)
        self.return_intermediate = return_intermediate
        self.num_layers = num_layers

        self.reference_points_add1 = nn.ModuleList()
        self.reference_points_add2 = nn.ModuleList()
        self.reference_points_cat = nn.ModuleList()
        self.new_reference_points_sigmoids = nn.ModuleList()
        for _ in range(num_layers):
            self.reference_points_add1.append(FF())
            self.reference_points_add2.append(FF())
            self.reference_points_cat.append(FF())
            self.new_reference_points_sigmoids.append(nn.Sigmoid())

    def forward(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        query_pos: Tensor = None,
        reg_branches: nn.ModuleList = None,
        spatial_shapes: Tensor = None,
        init_reference_points: Tensor = None,
        attn_mask: Tensor = None,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Foward DetectionTransformerDecoder."""
        output = query
        intermediate = []
        intermediate_reference_points_before_sigmoids = []
        bbox_outputs = []

        new_reference_points_before_sigmoid = init_reference_points
        reference_points = init_reference_points

        for lid, layer in enumerate(self.layers):

            reference_points = self.new_reference_points_sigmoids[lid](
                reference_points
            )
            reference_points_input = reference_points[..., :2].unsqueeze(2)

            output = layer(
                output,
                reference_points=reference_points_input,
                key=key,
                value=value,
                query_pos=query_pos,
                spatial_shapes=spatial_shapes,
                attn_masks=attn_mask,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                if lid != (len(self.layers) - 1):
                    new_reference_points1 = self.reference_points_add1[
                        lid
                    ].add(
                        tmp[..., :2],
                        new_reference_points_before_sigmoid[..., :2],
                    )
                    new_reference_points2 = self.reference_points_add2[
                        lid
                    ].add(
                        tmp[..., 4:5],
                        new_reference_points_before_sigmoid[..., 2:3],
                    )
                    new_reference_points = self.reference_points_cat[lid].cat(
                        [new_reference_points1, new_reference_points2], dim=-1
                    )
                    new_reference_points_before_sigmoid = new_reference_points
                reference_points = new_reference_points_before_sigmoid.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points_before_sigmoids.append(
                    new_reference_points_before_sigmoid
                )
                bbox_outputs.append(tmp)

        if self.return_intermediate:
            return (
                intermediate,
                intermediate_reference_points_before_sigmoids,
                bbox_outputs,
            )

        return [output], [new_reference_points_before_sigmoid], [tmp]

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        from hat.utils import qconfig_manager

        for layer in self.layers:
            if hasattr(layer, "set_qconfig"):
                layer.set_qconfig()

        int16_modules = [
            self.reference_points_add1,
            self.reference_points_add2,
            self.reference_points_cat,
            self.new_reference_points_sigmoids,
        ]
        for m in int16_modules:
            for _m in m:
                _m.qconfig = qconfig_manager.get_qconfig(
                    activation_qat_qkwargs={"dtype": qint16},
                    activation_calibration_qkwargs={
                        "dtype": qint16,
                    },
                    activation_calibration_observer="mix",
                )

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""
        for layer in self.layers:
            if hasattr(layer, "fuse_model"):
                layer.fuse_model()


@OBJECT_REGISTRY.register
class DetrTransformerDecoderLayer(nn.Module):
    """The basic structure of DetrTransformerDecoderLayer.

    Args:
        crossattention: The cross attention module.
        selfattention: The self attention module.
        embed_dims: The embedding dimension of Attention.
        dropout: Probability of an element to be zeroed.
        num_heads: Parallel attention heads.
    """

    def __init__(
        self,
        crossattention: nn.Module,
        selfattention: nn.Module = None,
        embed_dims: int = 256,
        dropout: float = 0.1,
        num_heads: int = 8,
    ):
        super().__init__()
        self.sa = selfattention
        if self.sa is None:
            self.sa = torch.nn.MultiheadAttention(
                embed_dim=embed_dims, num_heads=num_heads, dropout=dropout
            )
        self.sa_norm = nn.LayerNorm(embed_dims)
        self.ca = crossattention
        self.ca_norm = nn.LayerNorm(embed_dims)
        self.ffn = FFN(embed_dims, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(embed_dims)
        self.add_pos1 = FF()
        self.add_pos2 = FF()
        self.add_self_attn = FF()
        self.dropout1 = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        query_pos: Tensor = None,
        attn_masks: Tensor = None,
        reference_points: Tensor = None,
        spatial_shapes: Tensor = None,
    ) -> Tensor:
        """Foward DetrTransformerDecoderLayer."""
        if query_pos is not None:
            sa_query = self.add_pos1.add(query, query_pos)
        else:
            sa_query = query
        sa_key = sa_query
        sa_query = self.sa(
            sa_query,
            key=sa_key,
            value=query,
            attn_mask=attn_masks,
        )[0]

        query = self.add_self_attn.add(query, self.dropout1(sa_query))
        query = self.sa_norm(query)
        query = self.ca(
            query,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
        )
        query = self.ca_norm(query)
        query = self.ffn(query, query)
        query = self.ffn_norm(query)
        return query

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        from hat.utils import qconfig_manager

        if hasattr(self.sa, "set_qconfig"):
            self.sa.set_qconfig()

        if hasattr(self.ca, "set_qconfig"):
            self.ca.set_qconfig()
        if hasattr(self.ffn, "set_qconfig"):
            self.ffn.set_qconfig()
        int16_module = [
            self.add_self_attn,
        ]
        for m in int16_module:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""
        if hasattr(self.sa, "fuse_model"):
            self.sa.fuse_model()
        if hasattr(self.ca, "fuse_model"):
            self.ca.fuse_model()
        if hasattr(self.ffn, "fuse_model"):
            self.ffn.fuse_model()
