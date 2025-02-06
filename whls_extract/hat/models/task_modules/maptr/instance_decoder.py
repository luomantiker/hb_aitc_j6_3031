# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
from typing import Dict, List, Optional

import torch
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import Linear
from torch.quantization import DeQuantStub

from hat.core.box_utils import box_corner_to_center
from hat.models.task_modules.bevformer.attention import (
    HorizonMultiPointDeformableAttention,
)
from hat.models.task_modules.bevformer.utils import (
    bias_init_with_prob,
    get_clone_module,
    xavier_init,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = [
    "MapInstanceDetectorHead",
    "MapInstanceDecoder",
]


@OBJECT_REGISTRY.register
class MapInstanceDetectorHead(nn.Module):
    """The basic structure of the MapTROE perception decoderv2.

    This head is designed to handle instance queries exclusively, removing
        the dependency on point queries.

    Args:
        in_channels: The channels of input bev features. Default is 128.
        queue_length: Length of the queue for input data.
        num_cam: Number of cameras. Default is 6.
        bev_h: Height of the bird's-eye view. Default is 30.
        bev_w: Width of the bird's-eye view. Default is 30.
        decoder: Decoder module.
        embed_dims: Dimension of the embeddings.
        num_vec_one2one: Number of one-to-one vectors. Default is 50.
        num_vec_one2many: Number of one-to-many vectors. Default is 0.
        num_vec: Number of queries for calibration. Default is None.
        k_one2many: K value for one-to-many vectors. Default is 0.
        lambda_one2many: Lambda value for one-to-many vectors. Default is 1.
        num_pts_per_vec: Number of points per vector. Default is 2.
        num_pts_per_gt_vec: Number of points per gt vector. Default is 2.
        transform_method: Method for transformation. Default is "minmax".
        gt_shift_pts_pattern: Gt shift points pattern. Default is "v0".
        code_size: Size of the coordinate. Default is 2.
        num_classes: Number of classes. Default is 10.
        post_process: Post-process module.
        criterion: Loss module.
        aux_seg: Auxiliary segmentation config.
        is_deploy: Whether the model is in deployment mode. Default is False.
    """

    def __init__(
        self,
        in_channels: int = 128,
        queue_length=1,
        num_cam: int = 6,
        bev_h: int = 30,
        bev_w: int = 30,
        decoder: nn.Module = None,
        embed_dims: int = 512,
        num_vec_one2one: int = 50,
        num_vec_one2many: int = 0,
        num_vec: int = None,
        k_one2many: int = 0,
        lambda_one2many: int = 1,
        num_pts_per_vec: int = 2,
        num_pts_per_gt_vec: int = 2,
        transform_method: str = "minmax",
        gt_shift_pts_pattern: str = "v0",
        code_size: int = 2,
        num_classes: int = 10,
        post_process: Optional[nn.Module] = None,
        criterion: Optional[nn.Module] = None,
        aux_seg: Optional[Dict] = None,
        is_deploy: bool = False,
    ):
        super(MapInstanceDetectorHead, self).__init__()
        self.in_channels = in_channels
        self.queue_length = queue_length
        self.num_cam = num_cam
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.decoder = decoder
        self.embed_dims = embed_dims
        self.num_vec_one2one = num_vec_one2one
        self.num_vec_one2many = num_vec_one2many
        self.num_vec = num_vec
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many
        self.num_points = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        self.code_size = code_size
        self.cls_out_channels = num_classes
        self.post_process = post_process
        self.criterion = criterion
        self.is_deploy = is_deploy
        self.num_query = num_vec_one2one + num_vec_one2many
        if aux_seg is None:
            aux_seg = {
                "use_aux_seg": False,
                "bev_seg": False,
                "pv_seg": False,
                "seg_classes": 1,
                "feat_down_sample": 32,
            }
        self.aux_seg = aux_seg

        self._init_embedding()
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        """Initialize cls branch, reg branch and aux modules of head."""
        self.input_proj = nn.Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1
        )

        cls_branch = nn.Sequential(
            Linear(self.embed_dims, self.cls_out_channels)
        )

        reg_branch = [
            Linear(self.embed_dims, 2 * self.embed_dims),
            nn.LayerNorm(2 * self.embed_dims),
            nn.ReLU(),
            Linear(2 * self.embed_dims, 2 * self.embed_dims),
            nn.LayerNorm(2 * self.embed_dims),
            nn.ReLU(),
            Linear(2 * self.embed_dims, self.num_points * self.code_size),
        ]
        reg_branch = nn.Sequential(*reg_branch)

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.

        num_layers = self.decoder.num_layers
        cls_branches = nn.ModuleList([cls_branch for _ in range(num_layers)])
        reg_branches = nn.ModuleList([reg_branch for _ in range(num_layers)])

        self.reg_branches = reg_branches
        self.cls_branches = cls_branches
        self.sigmoid = torch.nn.Sigmoid()

        self.seg_head = None
        self.pv_seg_head = None
        if self.aux_seg["use_aux_seg"]:
            if not (self.aux_seg["bev_seg"] or self.aux_seg["pv_seg"]):
                raise ValueError("aux_seg must have bev_seg or pv_seg")
            if self.aux_seg["bev_seg"]:
                self.seg_head = nn.Sequential(
                    nn.Conv2d(
                        self.in_channels,
                        self.in_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    # nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        self.in_channels,
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
                                self.in_channels,
                                self.in_channels,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                            # nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(
                                self.in_channels,
                                self.aux_seg["seg_classes"],
                                kernel_size=1,
                                padding=0,
                            ),
                        )
                    )

        self.quant_object_query_embed = QuantStub()
        self.dequant = DeQuantStub()

    def init_weights(self):
        """Initialize weights of the head."""
        for p in self.input_proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        xavier_init(self.reference_points, distribution="uniform", bias=0.0)

        for m in self.reg_branches:
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

        bias_init = bias_init_with_prob(0.01)
        if isinstance(self.cls_branches, nn.ModuleList):
            for m in self.cls_branches:
                if hasattr(m, "bias"):
                    nn.init.constant_(m.bias, bias_init)
        else:
            m = self.cls_branches
            nn.init.constant_(m.bias, bias_init)

    def _init_embedding(self):
        """Initialize embeddings of the head."""
        # query_embed
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

        self.reference_points = nn.Linear(self.embed_dims, self.num_points * 2)

    def _prepare_context(self, bev_features):
        """Prepare bev features for further processing.

        Args:
            bev_features: The input bev_features.

        Return:
            The processed bev features.
        """
        B, C, H, W = bev_features.shape
        bev_features = self.input_proj(bev_features)
        assert list(bev_features.shape) == [B, self.embed_dims, H, W]
        return bev_features

    @autocast(enabled=False)
    @fx_wrap()
    def _post_process(self, data, outputs):
        if self.training:
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
        outputs_seg=None,
        outputs_pv_seg=None,
    ) -> Dict:

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_pts_coords_one2one = []

        outputs_classes_one2many = []
        outputs_coords_one2many = []
        outputs_pts_coords_one2many = []

        for lvl in range(len(outputs_classes)):
            tmp = reference_out[lvl].float()

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
        if self.num_vec is None:
            num_vec = self.get_num_vec()
        else:
            num_vec = self.num_vec
        pts_reshape = pts.view(pts.shape[0], num_vec, self.num_points, 2)
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
            num_vec = self.num_vec_one2one + self.num_vec_one2many
        else:
            num_vec = self.num_vec_one2one
        return num_vec

    @autocast(enabled=False)
    def bev_decoder(self, bev_features, mlvl_feats):
        bs, _, c = bev_features.shape
        bev_features = (
            bev_features.reshape(bs, self.bev_h, self.bev_w, c)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        bev_seg_features = bev_features.clone()
        bev_features = self._prepare_context(bev_features)

        if self.num_vec is None:
            num_vec = self.get_num_vec()
        else:
            num_vec = self.num_vec

        # make attn mask
        self_attn_mask = (
            torch.zeros(
                [
                    num_vec,
                    num_vec,
                ]
            )
            .bool()
            .to(bev_features.device)
        )
        self_attn_mask[
            self.num_vec_one2one :,
            0 : self.num_vec_one2one,
        ] = True
        self_attn_mask[
            0 : self.num_vec_one2one,
            self.num_vec_one2one :,
        ] = True

        pos_embed = None
        query_embedding = (
            (self.query_embedding.weight[0:num_vec]).unsqueeze(0)
        ).repeat(
            bs, 1, 1
        )  # [B, num_q, embed_dims]

        query_embedding = self.quant_object_query_embed(query_embedding)

        init_reference_points = self.reference_points(
            query_embedding
        )  # (bs, num_q, 2*num_pts)
        init_reference_points = self.sigmoid(init_reference_points)
        init_reference_points = init_reference_points.view(
            -1, num_vec, self.num_points, 2
        )  # (bs, num_q, num_pts, 2)

        bev_features = bev_features.flatten(2).permute(2, 0, 1)
        query_embedding = query_embedding.permute(1, 0, 2)

        inter_states, init_reference_points, inter_references = self.decoder(
            query=query_embedding,
            key=None,
            value=bev_features,
            query_pos=pos_embed,
            reference_points=init_reference_points,
            reg_branches=self.reg_branches,
            spatial_shapes=torch.tensor(
                [[self.bev_w, self.bev_h]],
                device=query_embedding.device,
                dtype=torch.long,
            ),
            self_attn_mask=self_attn_mask,
        )

        outputs_classes = []
        reference_out = []
        for lvl in range(len(inter_states)):
            reg_points = inter_references[lvl]
            outputs_class = self.cls_branches[lvl](inter_states[lvl])
            outputs_classes.append(self.dequant(outputs_class))
            reference_out.append(self.dequant(reg_points))

        outputs_seg = None
        outputs_pv_segs = None
        if self.aux_seg["use_aux_seg"]:
            seg_bev_embed = bev_seg_features.contiguous()
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
            )
        else:
            return (
                outputs_classes,
                reference_out,
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
            outputs_seg,
            outputs_pv_seg,
        ) = outputs
        outputs = self.get_outputs(
            outputs_classes,
            reference_out,
            outputs_seg,
            outputs_pv_seg,
        )
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


@OBJECT_REGISTRY.register
class MapInstanceDecoder(nn.Module):
    """Implement the decoder in MapInstance head.

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
        self.new_reference_points_sigmoids = nn.ModuleList()
        for _ in range(num_layers):
            self.new_reference_points_sigmoids.append(nn.Sigmoid())

        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, HorizonMultiPointDeformableAttention):
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
        predict_refine=False,
        **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []
        init_reference_points = reference_points
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(
                2
            )  # bs, num_query, num_level, num_pts, 2
            output = layer(
                output,
                reference_points=reference_points_input,
                key=key,
                value=value,
                query_pos=query_pos,
                spatial_shapes=spatial_shapes,
                attn_masks=kwargs["self_attn_mask"],
            )
            if reg_branches is not None:
                reg_points = reg_branches[lid](output.permute(1, 0, 2))

                bs, num_queries, num_points2 = reg_points.shape
                reg_points = self.new_reference_points_sigmoids[lid](
                    reg_points
                )  # (bs, num_q, num_points, 2)
                new_reference_points = reg_points.view(
                    bs, num_queries, num_points2 // 2, 2
                )
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output.permute(1, 0, 2))
                intermediate_reference_points.append(reg_points)

        if self.return_intermediate:
            return (
                intermediate,
                init_reference_points,
                intermediate_reference_points,
            )

        return [output], [init_reference_points], [reference_points]
