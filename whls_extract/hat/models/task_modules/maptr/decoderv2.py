# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
from typing import Dict, List, Optional

import torch
from horizon_plugin_pytorch.fx.jit_scheme import Tracer
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import Linear
from torch.quantization import DeQuantStub

from hat.core.box_utils import box_corner_to_center
from hat.models.task_modules.bevformer.utils import (
    FFN,
    bias_init_with_prob,
    get_clone_module,
    xavier_init,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = [
    "MapTRPerceptionDecoderv2",
]


@OBJECT_REGISTRY.register
class MapTRPerceptionDecoderv2(nn.Module):
    """The basic structure of the MapTR perception decoderv2.

    Args:
        decoder: Decoder module.
        embed_dims: Dimension of the embeddings.
        queue_length: Length of the queue for input data.
        num_cam: Number of cameras. Default is 6.
        num_reg_fcs: Number of fc layers for regression. Default is 2.
        bev_h: Height of the bird's-eye view. Default is 30.
        bev_w: Width of the bird's-eye view. Default is 30.
        num_vec_one2one: Number of one-to-one vectors. Default is 50.
        num_vec_one2many: Number of one-to-many vectors. Default is 0.
        k_one2many: K value for one-to-many vectors. Default is 0.
        lambda_one2many: Lambda value for one-to-many vectors. Default is 1.
        num_pts_per_vec: Number of points per vector. Default is 2.
        num_pts_per_gt_vec: Number of points per gt vector. Default is 2.
        query_embed_type: Type of query embedding. Default is "all_pts".
        transform_method: Method for transformation. Default is "minmax".
        gt_shift_pts_pattern: Gt shift points pattern. Default is "v0".
        dir_interval: Direction interval. Default is 1.
        code_size: Size of the code. Default is 10.
        num_classes: Number of classes. Default is 10.
        pc_range: Point cloud range.
        is_deploy: Whether the model is in deployment mode. Default is False.
        post_process: Post-process module.
        criterion: Loss module.
        aux_seg: Auxiliary segmentation config.
    """

    def __init__(
        self,
        decoder: nn.Module,
        embed_dims: int,
        queue_length: int,
        num_cam: int = 6,
        num_reg_fcs: int = 2,
        bev_h: int = 30,
        bev_w: int = 30,
        num_vec_one2one: int = 50,
        num_vec_one2many: int = 0,
        k_one2many: int = 0,
        lambda_one2many: int = 1,
        num_pts_per_vec: int = 2,
        num_pts_per_gt_vec: int = 2,
        query_embed_type: str = "all_pts",
        transform_method: str = "minmax",
        gt_shift_pts_pattern: str = "v0",
        dir_interval: int = 1,
        code_size: int = 10,
        num_classes: int = 10,
        pc_range: Optional[list] = None,
        is_deploy: bool = False,
        post_process: Optional[nn.Module] = None,
        criterion: Optional[nn.Module] = None,
        aux_seg: Optional[Dict] = None,
    ):
        super(MapTRPerceptionDecoderv2, self).__init__()
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

        num_vec = num_vec_one2one + num_vec_one2many
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval
        self.num_vec_one2one = num_vec_one2one
        self.num_vec_one2many = num_vec_one2many
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many

        if aux_seg is None:
            aux_seg = {
                "use_aux_seg": False,
                "bev_seg": False,
                "pv_seg": False,
                "seg_classes": 1,
                "feat_down_sample": 32,
            }
        self.aux_seg = aux_seg
        self.queue_length = queue_length
        self.num_cam = num_cam

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

        self.seg_head = None
        self.pv_seg_head = None
        if self.aux_seg["use_aux_seg"]:
            if not (self.aux_seg["bev_seg"] or self.aux_seg["pv_seg"]):
                raise ValueError("aux_seg must have bev_seg or pv_seg")
            if self.aux_seg["bev_seg"]:
                self.seg_head = nn.Sequential(
                    nn.Conv2d(
                        self.embed_dims,
                        self.embed_dims,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    # nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        self.embed_dims,
                        self.aux_seg["seg_classes"],
                        kernel_size=1,
                        padding=0,
                    ),
                )
            if self.aux_seg["pv_seg"]:
                self.pv_seg_head = nn.ModuleList()
                for _ in range(len(self.aux_seg["feat_down_sample"])):
                    self.pv_seg_head.append(
                        nn.Sequential(
                            nn.Conv2d(
                                self.embed_dims,
                                self.embed_dims,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                            # nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(
                                self.embed_dims,
                                self.aux_seg["seg_classes"],
                                kernel_size=1,
                                padding=0,
                            ),
                        )
                    )

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
        if self.training and Tracer.is_tracing() is False:
            losses = {}
            loss_one2one = self.criterion(outputs, data)
            losses.update(loss_one2one)

            k_one2many = self.k_one2many
            gt_bboxes_list = data["seq_meta"][0]["gt_instances"]
            gt_labels_list = data["seq_meta"][0]["gt_labels_map"]
            multi_gt_bboxes_3d = copy.deepcopy(gt_bboxes_list)
            multi_gt_labels_3d = copy.deepcopy(gt_labels_list)
            for i, (each_gt_bboxes_3d, each_gt_labels_3d) in enumerate(
                zip(multi_gt_bboxes_3d, multi_gt_labels_3d)
            ):
                each_gt_bboxes_3d.instance_list = (
                    each_gt_bboxes_3d.instance_list * k_one2many
                )
                multi_gt_labels_3d[i] = each_gt_labels_3d.repeat(k_one2many)
                multi_gt_bboxes_3d[i] = each_gt_bboxes_3d
            # import ipdb;ipdb.set_trace()
            one2many_outs = outputs["one2many_outs"]
            one2many_data = {
                "seq_meta": [
                    {
                        "gt_instances": multi_gt_bboxes_3d,
                        "gt_labels_map": multi_gt_labels_3d,
                    }
                ]
            }
            loss_dict_one2many = self.criterion(one2many_outs, one2many_data)

            lambda_one2many = self.lambda_one2many
            for key, value in loss_dict_one2many.items():
                if key + "_one2many" in losses.keys():
                    losses[key + "_one2many"] += value * lambda_one2many
                else:
                    losses[key + "_one2many"] = value * lambda_one2many
            # import ipdb;ipdb.set_trace()
            return losses
        else:
            if self.post_process is None:
                return outputs
            results = self.post_process(outputs, data)
            return results

    @autocast(enabled=False)
    @fx_wrap()
    def get_outputs(
        self,
        outputs_classes: List[Tensor],
        reference_out: List[Tensor],
        bbox_outputs: List[Tensor],
        outputs_seg=None,
        outputs_pv_seg=None,
    ) -> Dict:

        # bev_embed, outputs_classes, reference_out, bbox_outputs = outputs
        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_pts_coords_one2one = []

        outputs_classes_one2many = []
        outputs_coords_one2many = []
        outputs_pts_coords_one2many = []

        for lvl in range(len(outputs_classes)):
            reference = reference_out[lvl].float()
            # reference = inverse_sigmoid(reference)
            tmp = bbox_outputs[lvl].float()
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid()
            outputs_coord, outputs_pts_coord = self.transform_box(tmp)
            outputs_class = outputs_classes[lvl].float()

            outputs_classes_one2one.append(
                outputs_class[:, 0 : self.num_vec_one2one]
            )
            outputs_coords_one2one.append(
                outputs_coord[:, 0 : self.num_vec_one2one]
            )
            outputs_pts_coords_one2one.append(
                outputs_pts_coord[:, 0 : self.num_vec_one2one]
            )

            outputs_classes_one2many.append(
                outputs_class[:, self.num_vec_one2one :]
            )
            outputs_coords_one2many.append(
                outputs_coord[:, self.num_vec_one2one :]
            )
            outputs_pts_coords_one2many.append(
                outputs_pts_coord[:, self.num_vec_one2one :]
            )

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_pts_coords_one2one = torch.stack(outputs_pts_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)
        outputs_pts_coords_one2many = torch.stack(outputs_pts_coords_one2many)

        preds_dicts = {
            "all_cls_scores": outputs_classes_one2one,
            "all_bbox_preds": outputs_coords_one2one,
            "all_pts_preds": outputs_pts_coords_one2one,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
            "seg": outputs_seg.float() if outputs_seg is not None else None,
            "pv_seg": [seg.float() for seg in outputs_pv_seg]
            if outputs_pv_seg is not None
            else None,
            "one2many_outs": {
                "all_cls_scores": outputs_classes_one2many,
                "all_bbox_preds": outputs_coords_one2many,
                "all_pts_preds": outputs_pts_coords_one2many,
                "enc_cls_scores": None,
                "enc_bbox_preds": None,
                "enc_pts_preds": None,
                "seg": None,
                "pv_seg": None,
            },
        }

        return preds_dicts

    @fx_wrap()
    def transform_box(self, pts, y_first=False):
        """
        Convert the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        num_vec = self.get_num_vec()
        pts_reshape = pts.view(pts.shape[0], num_vec, self.num_pts_per_vec, 2)
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

    @fx_wrap()
    def get_num_vec(self):
        if self.training:
            num_vec = self.num_vec
        else:
            num_vec = self.num_vec_one2one
        return num_vec

    @autocast(enabled=False)
    def bev_decoder(self, bev_embed, mlvl_feats):
        bs = bev_embed.size(0)
        num_vec = self.get_num_vec()

        if self.query_embed_type == "all_pts":
            object_query_embeds = self.query_embedding.weight
        elif self.query_embed_type == "instance_pts":
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight[
                0:num_vec
            ].unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1)

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = (
            torch.zeros(
                [
                    num_vec,
                    num_vec,
                ]
            )
            .bool()
            .to(bev_embed.device)
        )
        self_attn_mask[
            self.num_vec_one2one :,
            0 : self.num_vec_one2one,
        ] = True
        self_attn_mask[
            0 : self.num_vec_one2one,
            self.num_vec_one2one :,
        ] = True

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

        inter_states, inter_references, offsets = self.decoder(
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
            init_reference_points=init_reference_points,
            self_attn_mask=self_attn_mask,
            num_vec=num_vec,
            num_pts_per_vec=self.num_pts_per_vec,
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
                .view(bs, num_vec, self.num_pts_per_vec, -1)
                .mean(2)
            )
            tmp = offsets[lvl]
            outputs_classes.append(self.dequant(outputs_class))
            outputs_coords.append(self.dequant(tmp))
            reference_out.append(self.dequant(reference))

        outputs_seg = None
        outputs_pv_segs = None
        if self.aux_seg["use_aux_seg"]:
            seg_bev_embed = (
                bev_embed.permute(1, 0, 2)
                .view(bs, self.bev_h, self.bev_w, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            if self.aux_seg["bev_seg"]:
                seg_bev_embed = self.dequant(seg_bev_embed).float()
                outputs_seg = self.seg_head(seg_bev_embed)

            if self.aux_seg["pv_seg"]:
                outputs_pv_segs = []
                for i, (feat_idx, _) in enumerate(
                    self.aux_seg["feat_down_sample"]
                ):
                    feats = mlvl_feats[feat_idx]
                    feats = feats.view(
                        (
                            self.queue_length,
                            bs,
                            self.num_cam,
                        )
                        + feats.shape[1:]
                    )[0:1]
                    feats = self.dequant(feats.flatten(0, 2)).float()
                    outputs_pv_seg = self.pv_seg_head[i](feats)
                    outputs_pv_seg = outputs_pv_seg.view(
                        (bs, self.num_cam) + outputs_pv_seg.shape[1:]
                    )
                    outputs_pv_segs.append(outputs_pv_seg)

        if self.is_deploy:
            return (
                outputs_classes[-1],
                reference_out[-1],
                outputs_coords[-1],
            )
        else:
            return (
                outputs_classes,
                reference_out,
                outputs_coords,
                outputs_seg,
                outputs_pv_segs,
            )

    @autocast(enabled=False)
    def forward(self, bev_embed, mlvl_feats, data=None):

        outputs = self.bev_decoder(bev_embed, mlvl_feats)
        if self.is_deploy:
            return outputs
        (
            outputs_classes,
            reference_out,
            outputs_coords,
            outputs_seg,
            outputs_pv_seg,
        ) = outputs
        outputs = self.get_outputs(
            outputs_classes,
            reference_out,
            outputs_coords,
            outputs_seg,
            outputs_pv_seg,
        )
        # outputs["bev_embed"] = bev_embed
        return self._post_process(data, outputs)

    def set_qconfig(self):
        aux_mods = [
            self.seg_head,
            self.pv_seg_head,
            self.criterion,
            self.post_process,
        ]

        for m in aux_mods:
            if m is not None:
                m.qconfig = None

    def fuse_model(self):
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
class DecoupledDetrTransformerDecoderLayer(nn.Module):
    """Implements decoder layer in MapTRv2 transformer.

    Args:
        crossattention: Cross-attention module.
        embed_dims: Dimension of the embeddings. Default is 256.
        dropout: Dropout rate. Default is 0.1.
        num_heads: Number of attention heads. Default is 8.
    """

    def __init__(
        self,
        crossattention: nn.Module,
        embed_dims: int = 256,
        dropout: float = 0.1,
        num_heads: int = 8,
    ):
        super().__init__()
        self.sa_vecs = torch.nn.MultiheadAttention(
            embed_dim=embed_dims, num_heads=num_heads, dropout=dropout
        )
        self.sa_vecs_norm = nn.LayerNorm(embed_dims)
        self.sa_pts = torch.nn.MultiheadAttention(
            embed_dim=embed_dims, num_heads=num_heads, dropout=dropout
        )
        self.sa_pts_norm = nn.LayerNorm(embed_dims)
        self.ca = crossattention
        self.ca_norm = nn.LayerNorm(embed_dims)
        self.ffn = FFN(embed_dims, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(embed_dims)
        self.add_pos1 = FF()
        self.add_pos2 = FF()
        self.add_pos3 = FF()
        self.add_self_attn1 = FF()
        self.add_self_attn2 = FF()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        # key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        **kwargs,
    ):

        num_vec = kwargs["num_vec"]
        num_pts_per_vec = kwargs["num_pts_per_vec"]

        # inter-ins self-attention
        identity = query
        n_pts, n_batch, n_dim = query.shape
        query = query.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(
            1, 2
        )
        query_pos = query_pos.view(
            num_vec, num_pts_per_vec, n_batch, n_dim
        ).flatten(1, 2)
        sa_query_vecs = self.add_pos1.add(query, query_pos)
        query = self.sa_vecs(
            sa_query_vecs,
            key=sa_query_vecs,
            value=query,
            # query_pos=query_pos,
            # key_pos=key_pos,
            attn_mask=kwargs["self_attn_mask"],
            # key_padding_mask=key_padding_mask,
        )[0]
        query = query.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(
            0, 1
        )
        query_pos = query_pos.view(
            num_vec, num_pts_per_vec, n_batch, n_dim
        ).flatten(0, 1)
        query = self.add_self_attn1.add(identity, self.dropout1(query))
        query = self.sa_vecs_norm(query)

        # intra-ins self-attention
        identity = query
        query = (
            query.view(num_vec, num_pts_per_vec, n_batch, n_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
            .flatten(1, 2)
        )
        query_pos = (
            query_pos.view(num_vec, num_pts_per_vec, n_batch, n_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
            .flatten(1, 2)
        )
        sa_query_pts = self.add_pos2.add(query, query_pos)
        query = self.sa_pts(
            sa_query_pts,
            key=sa_query_pts,
            value=query,
            # query_pos=query_pos,
            # key_pos=key_pos,
            attn_mask=attn_masks,
            # key_padding_mask=key_padding_mask,
        )[0]
        query = (
            query.view(num_pts_per_vec, num_vec, n_batch, n_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
            .flatten(0, 1)
        )
        query_pos = (
            query_pos.view(num_pts_per_vec, num_vec, n_batch, n_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
            .flatten(0, 1)
        )
        query = self.add_self_attn2.add(identity, self.dropout2(query))
        query = self.sa_pts_norm(query)

        # cross-attention
        query = self.ca(
            query,
            value,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
        )

        query = self.ca_norm(query)
        query = self.ffn(query, query)
        query = self.ffn_norm(query)
        return query
