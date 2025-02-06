# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict, List, Optional

import torch
from horizon_plugin_pytorch.fx.jit_scheme import Tracer
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = [
    "SparseMapPerceptionDecoder",
]


@OBJECT_REGISTRY.register
class SparseMapPerceptionDecoder(nn.Module):
    """The sparse structure of the MapTR perception decoder.

    Args:
        decoder: Decoder module.
        embed_dims: Dimension of the embeddings.
        num_cam: Number of cameras. Default is 6.
        num_vec_one2one: Number of one-to-one vectors. Default is 50.
        num_vec_one2many: Number of one-to-many vectors. Default is 0.
        k_one2many: K value for one-to-many vectors. Default is 0.
        lambda_one2many: Lambda value for one-to-many vectors. Default is 1.
        num_pts_per_vec: Number of points per vector. Default is 2.
        transform_method: Method for transformation. Default is "minmax".
        is_deploy: Whether the model is in deployment mode. Default is False.
        post_process: Post-process module.
        criterion: Loss module.
        aux_seg: Auxiliary segmentation config.
        depth_branch: Depth branch module.
    """

    def __init__(
        self,
        decoder: nn.Module,
        embed_dims: int,
        num_cam: int = 6,
        num_vec_one2one: int = 50,
        num_vec_one2many: int = 0,
        k_one2many: int = 0,
        lambda_one2many: int = 1,
        num_pts_per_vec: int = 2,
        transform_method: str = "minmax",
        is_deploy: bool = False,
        post_process: Optional[nn.Module] = None,
        criterion: Optional[nn.Module] = None,
        aux_seg: Optional[Dict] = None,
        depth_branch: Optional[nn.Module] = None,
    ):
        super(SparseMapPerceptionDecoder, self).__init__()
        self.embed_dims = embed_dims
        self.decoder = decoder
        self.is_deploy = is_deploy
        self.post_process = post_process
        self.criterion = criterion

        self.transform_method = transform_method
        num_vec = num_vec_one2one + num_vec_one2many
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
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
        self.num_cam = num_cam
        self.depth_branch = None
        if self.aux_seg["use_aux_seg"] and self.aux_seg["dense_depth"]:
            self.depth_branch = depth_branch

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        self.pv_seg_head_layers = nn.ModuleList()
        if self.aux_seg["use_aux_seg"]:
            if not (self.aux_seg["pv_seg"] or self.aux_seg["dense_depth"]):
                raise ValueError("aux_seg must have pv_seg or dense_depth")
            if self.aux_seg["pv_seg"]:
                for _ in range(len(self.aux_seg["feat_down_sample"])):
                    self.pv_seg_head_layers.append(
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

        self.dequant = DeQuantStub()

    @autocast(enabled=False)
    def _post_process(self, data, outputs):
        if self.training and Tracer.is_tracing() is False:
            losses = {}
            loss_one2one = self.criterion(outputs, data)
            losses.update(loss_one2one)

            if "dense_depth" in outputs:
                losses["loss_dense_depth"] = self.depth_branch.loss(
                    outputs["dense_depth"], data
                )
            return losses
        else:
            if self.post_process is None:
                return outputs
            results = self.post_process(outputs, data)
            return results

    @autocast(enabled=False)
    def get_outputs(
        self,
        outputs_classes: List[Tensor],
        reference_out: List[Tensor],
        outputs_pv_seg=None,
        dense_depth=None,
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
            outputs_coord, outputs_pts_coord = self.transform_box(reference)
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
            "pv_seg": [seg.float() for seg in outputs_pv_seg]
            if outputs_pv_seg is not None
            else None,
            "dense_depth": [depth.float() for depth in dense_depth]
            if dense_depth is not None
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
            # bbox = box_corner_to_center(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def get_num_vec(self):
        if self.training:
            num_vec = self.num_vec
        else:
            num_vec = self.num_vec_one2one
        return num_vec

    # @autocast(enabled=False)
    def sparse_decoder(self, mlvl_feats, data, is_deploy=False):

        model_outs = self.decoder(mlvl_feats, data, compiler_model=is_deploy)
        if self.aux_seg["use_aux_seg"]:
            if self.aux_seg["pv_seg"]:
                bs = mlvl_feats[0].shape[0] // self.num_cam
                outputs_pv_segs = []
                for i, (feat_idx, _) in enumerate(
                    self.aux_seg["feat_down_sample"]
                ):
                    _, _, feat_h, feat_w = mlvl_feats[feat_idx].shape
                    feats = self.dequant(mlvl_feats[feat_idx])
                    outputs_pv_seg = self.pv_seg_head_layers[i](feats)
                    outputs_pv_seg = outputs_pv_seg.view(
                        bs, self.num_cam, -1, feat_h, feat_w
                    )
                    outputs_pv_segs.append(outputs_pv_seg)
                model_outs.update({"outputs_pv_seg": outputs_pv_segs})
            if self.aux_seg["dense_depth"]:
                depths = self.depth_branch(mlvl_feats, data)
                model_outs.update({"dense_depth": depths})

        return model_outs

    # @autocast(enabled=False)
    def forward(self, bev_embed, mlvl_feats, data=None):

        outputs = self.sparse_decoder(
            mlvl_feats, data, is_deploy=self.is_deploy
        )
        if self.is_deploy:
            return outputs
        outputs_classes = outputs["classification"]
        reference_out = outputs["prediction"]
        outputs_pv_seg = outputs.get("outputs_pv_seg", None)
        dense_depth = outputs.get("dense_depth", None)
        outputs = self.get_outputs(
            outputs_classes,
            reference_out,
            outputs_pv_seg,
            dense_depth,
        )
        # outputs["bev_embed"] = bev_embed
        return self._post_process(data, outputs)

    def set_qconfig(self):
        aux_mods = [
            self.pv_seg_head_layers,
            self.depth_branch,
            self.criterion,
            self.post_process,
        ]

        for m in aux_mods:
            if m is not None:
                m.qconfig = None
