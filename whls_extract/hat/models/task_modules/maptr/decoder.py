# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import List, Optional

import torch
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import Linear
from torch.quantization import DeQuantStub

from hat.core.box_utils import box_corner_to_center
from hat.models.task_modules.bevformer.attention import (
    HorizonMultiScaleDeformableAttention,
    HorizonMultiScaleDeformableAttention3D,
    HorizonSpatialCrossAttention,
    HorizonTemporalSelfAttention,
)
from hat.models.task_modules.bevformer.utils import (
    bias_init_with_prob,
    get_clone_module,
    xavier_init,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = ["MapTRPerceptionDecoder", "MapTRDecoder"]


@OBJECT_REGISTRY.register
class MapTRPerceptionDecoder(nn.Module):
    """The basic structure of MapTRPerceptionDecoder.

    Args:
        decoder: Decoder module.
        embed_dims: Dimension of embedding vectors.
        num_reg_fcs: Number of fully connected layers for regression.
        bev_h: Bird's Eye View (BEV) height.
        bev_w: Bird's Eye View (BEV) width.
        code_size: Size of the code representing a detection.
        num_classes: Number of classes for detection.
        pc_range: Point cloud range.
        is_deploy: Flag for deployment mode.
        post_process: Post-processing module.
        criterion: Loss module.
        num_vec: Number of vectors.
        num_pts_per_vec: Number of points per vector.
        num_pts_per_gt_vec: Number of points per ground truth vector.
        query_embed_type: Type of query embedding.
        transform_method: Method for transformation.
        gt_shift_pts_pattern: Pattern for ground truth shift points.
        dir_interval: Interval for direction calculation.
    """

    def __init__(
        self,
        decoder: nn.Module,
        embed_dims: int,
        num_reg_fcs: int = 2,
        bev_h: int = 30,
        bev_w: int = 30,
        code_size: int = 10,
        num_classes: int = 10,
        pc_range: Optional[List[float]] = None,
        is_deploy: bool = False,
        post_process: Optional[nn.Module] = None,
        criterion: Optional[nn.Module] = None,
        num_vec: int = 20,
        num_pts_per_vec: int = 2,
        num_pts_per_gt_vec: int = 2,
        query_embed_type: str = "all_pts",
        transform_method: str = "minmax",
        gt_shift_pts_pattern: str = "v0",
        dir_interval: int = 1,
    ):
        super(MapTRPerceptionDecoder, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.decoder = decoder
        self.num_reg_fcs = num_reg_fcs
        self.cls_out_channels = num_classes
        self.code_size = code_size
        self.pc_range = pc_range
        self.is_deploy = is_deploy
        self.post_process = post_process
        self.criterion = criterion

        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
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

        num_pred = self.decoder.num_layers

        self.cls_branches = get_clone_module(fc_cls, num_pred)
        self.reg_branches = get_clone_module(reg_branch, num_pred)

        self.reference_points = nn.Linear(self.embed_dims, 2)

        if self.query_embed_type == "all_pts":
            self.query_embedding = nn.Embedding(
                self.num_query, self.embed_dims * 2
            )
        elif self.query_embed_type == "instance_pts":
            self.query_embedding = None
            self.instance_embedding = nn.Embedding(
                self.num_vec, self.embed_dims * 2
            )
            self.pts_embedding = nn.Embedding(
                self.num_pts_per_vec, self.embed_dims * 2
            )

        self.quant_object_query_embed = QuantStub()
        self.sigmoid = torch.nn.Sigmoid()
        self.dequant = DeQuantStub()

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)
        xavier_init(self.reference_points, distribution="uniform", bias=0.0)

    @autocast(enabled=False)
    @fx_wrap()
    def _post_process(self, data, outputs):
        if self.training:
            loss_dict = self.criterion(outputs, data)
            return loss_dict
        else:
            if self.post_process is None:
                return outputs
            results = self.post_process(outputs, data)
            return results

    @autocast(enabled=False)
    @fx_wrap()
    def get_outputs(self, outputs_classes, reference_out, bbox_outputs):

        # bev_embed, outputs_classes, reference_out, bbox_outputs = outputs
        new_outputs_coords = []
        outputs_pts_coords = []

        for lvl in range(len(outputs_classes)):
            reference = reference_out[lvl].float()
            # reference = inverse_sigmoid(reference)
            tmp = bbox_outputs[lvl].float()
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid()
            outputs_coord, outputs_pts_coord = self.transform_box(tmp)

            new_outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)

        outputs_classes = torch.stack(outputs_classes).float()
        outputs_coords = torch.stack(new_outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)

        preds_dicts = {
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "all_pts_preds": outputs_pts_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
        }
        # print("h"*50)
        # print("bev_embed shape: ", bev_embed.dtype)
        # print("outputs_classes shape: ", outputs_classes.dtype)
        # print("outputs_coords shape: ", outputs_coords.dtype)
        # print("outputs_pts_coords shape: ", outputs_pts_coords.dtype)
        return preds_dicts

    @fx_wrap()
    def transform_box(self, pts, y_first=False):
        pts_reshape = pts.view(
            pts.shape[0], self.num_vec, self.num_pts_per_vec, 2
        )
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == "minmax":
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = box_corner_to_center(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def bev_decoder(self, bev_embed):
        bs = bev_embed.size(0)

        if self.query_embed_type == "all_pts":
            object_query_embeds = self.query_embedding.weight
        elif self.query_embed_type == "instance_pts":
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1)

        object_query_embed = self.quant_object_query_embed(object_query_embeds)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1
        )
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        init_reference_points = reference_points
        reference_points = self.sigmoid(reference_points)

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references, outputs = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=self.reg_branches,
            spatial_shapes=torch.tensor(
                [[self.bev_w, self.bev_h]],
                device=query.device,
                dtype=torch.long,
            ),
            # spatial_shapes=self.getspatial_shapes(query),
            init_reference_points=init_reference_points,
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
                inter_states[lvl]
                .permute(1, 0, 2)
                .view(bs, self.num_vec, self.num_pts_per_vec, -1)
                .mean(2)
            )
            tmp = outputs[lvl]
            outputs_classes.append(self.dequant(outputs_class))
            outputs_coords.append(self.dequant(tmp))
            reference_out.append(self.dequant(reference))

        if self.is_deploy:
            return (
                outputs_classes[-1],
                reference_out[-1],
                outputs_coords[-1],
            )
        else:
            return outputs_classes, reference_out, outputs_coords

    def forward(self, bev_embed, mlvl_feats=None, data=None):

        outputs = self.bev_decoder(bev_embed)
        if self.is_deploy:
            return outputs
        outputs_classes, reference_out, outputs_coords = outputs
        outputs = self.get_outputs(
            outputs_classes, reference_out, outputs_coords
        )
        # outputs["bev_embed"] = bev_embed
        return self._post_process(data, outputs)


@OBJECT_REGISTRY.register
class MapTRDecoder(nn.Module):
    """Implement the decoder in MapTR transformer.

    Args:
        num_layers: Number of decoder layers.
        decoder_layer: Decoder layer module.
        return_intermediate: Whether to return intermediate outputs.
    """

    def __init__(
        self,
        num_layers: int = 3,
        decoder_layer: Optional[nn.Module] = None,
        return_intermediate: bool = False,
    ):
        super().__init__()
        self.layers = get_clone_module(decoder_layer, num_layers)
        self.return_intermediate = return_intermediate
        self.num_layers = num_layers

        self.reference_points_add1 = nn.ModuleList()
        self.new_reference_points_sigmoids = nn.ModuleList()

        for _ in range(num_layers):
            self.reference_points_add1.append(FF())
            self.new_reference_points_sigmoids.append(nn.Sigmoid())

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

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        reference_points=None,
        reg_branches=None,
        spatial_shapes=None,
        init_reference_points=None,
        **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points_before_sigmoids = []
        new_reference_points_before_sigmoid = init_reference_points
        outputs = []

        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2
            )  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                reference_points=reference_points_input,
                key=key,
                value=value,
                query_pos=query_pos,
                spatial_shapes=spatial_shapes,
                **kwargs,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                new_reference_points = self.reference_points_add1[lid].add(
                    tmp[..., :2], new_reference_points_before_sigmoid[..., :2]
                )

                new_reference_points_before_sigmoid = new_reference_points

                new_reference_points = self.new_reference_points_sigmoids[lid](
                    new_reference_points
                )
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points_before_sigmoids.append(
                    new_reference_points_before_sigmoid
                )
                outputs.append(tmp)

        if self.return_intermediate:
            return (
                intermediate,
                intermediate_reference_points_before_sigmoids,
                outputs,
            )

        return [output], [new_reference_points_before_sigmoid], [tmp]
