# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Callable, Dict, Optional

import torch
from easydict import EasyDict
from torch import nn
from torch.cuda.amp import autocast

from hat.core.box_utils import box_corner_to_center
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

try:
    from hbdk4.compiler import load
    from horizon_plugin_pytorch.quantization.hbdk4 import (
        get_hbir_input_flattener,
        get_hbir_output_unflattener,
    )
except ImportError:
    load = None
    get_hbir_input_flattener = None
    get_hbir_output_unflattener = None

logger = logging.getLogger(__name__)

__all__ = ["Motr", "MotrIrInfer"]


@OBJECT_REGISTRY.register
class Motr(nn.Module):
    """The basic structure of Motr.

    Args:
        backbone: backbone module.
        neck: neck module.
        head: head module with transformer architecture.
        criterion: loss module.
        post_process: post process module.
        track_embed: track embed module.
        compile_motr: Whether to compile motr model.
        compile_qim: Whether to compile qim model
        num_query_h: The num of h dim for query reshape.
        batch_size: batch size
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        criterion: nn.Module = None,
        post_process: nn.Module = None,
        track_embed: nn.Module = None,
        compile_motr: bool = False,
        compile_qim: bool = False,
        num_query_h: int = 2,
        batch_size: int = 1,
    ):
        super(Motr, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.criterion = criterion
        self.post_process = post_process
        self.track_embed = track_embed
        self.seq_names = []
        self.tracked_instance = None
        self.frame_id = 1
        self.compile_motr = compile_motr
        self.compile_qim = compile_qim
        num_queries, dim = self.head.query_embed.weight.shape
        self.num_queries = num_queries
        self.queries_dim = dim
        self.num_query_h = num_query_h
        assert self.num_queries % self.num_query_h == 0, (
            "num_queries must be divisible by num_query_h, but get "
            "{} vs {}".format(self.num_queries, self.num_query_h)
        )
        self.num_query_w = self.num_queries // self.num_query_h
        self.batch_size = batch_size
        self.max_frame_num = 5

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck."""
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x

    @autocast(enabled=False)
    def _generate_empty_tracks(self):
        """Genrate empty track instance for detection new object."""
        track_instances = EasyDict()
        device = self.head.query_embed.weight.device
        track_instances.output_embedding = torch.zeros(
            (self.num_queries, self.queries_dim), device=device
        )
        track_instances.obj_idxes = torch.full(
            (self.num_queries,), -1, dtype=torch.long, device=device
        )
        track_instances.matched_gt_idxes = torch.full(
            (self.num_queries,), -1, dtype=torch.long, device=device
        )
        track_instances.disappear_time = torch.zeros(
            (self.num_queries,), dtype=torch.long, device=device
        )
        track_instances.iou = torch.zeros(
            (self.num_queries,), dtype=torch.float, device=device
        )
        track_instances.scores = torch.zeros(
            (self.num_queries,), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (self.num_queries,), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (self.num_queries, 4), dtype=torch.float, device=device
        )
        track_instances.pred_boxes_unsigmoid = torch.zeros(
            (self.num_queries, 4), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (self.num_queries, self.head.num_classes),
            dtype=torch.float,
            device=device,
        )
        track_instances.mask_query = torch.ones(
            (self.num_queries,), dtype=torch.float, device=device
        )
        track_instances.ref_pts = self.head.refpoint_embed.weight
        track_instances.query_pos = self.head.query_embed.weight
        return track_instances

    @autocast(enabled=False)
    def _generate_fake_tracks(
        self,
    ):
        """Genrate fake track instance for padding object."""
        track_instances = self._generate_empty_tracks()
        query_pos = torch.zeros_like(track_instances["query_pos"])
        mask_query = torch.zeros_like(track_instances["mask_query"])
        ref_pts = torch.zeros_like(track_instances["ref_pts"])
        track_instances["query_pos"] = query_pos
        track_instances["mask_query"] = mask_query
        track_instances["ref_pts"] = ref_pts

        return track_instances

    @autocast(enabled=False)
    @fx_wrap()
    def _prepare_single_image_forward(self, seq_data, frame_id):
        """Prepare track_pos, ref_pts and mask_query."""
        num_frame = len(seq_data["img"])
        if frame_id >= num_frame:
            return None, None, None, None

        seq_name = seq_data["seq_name"][frame_id]
        if not self.training:
            if seq_name not in self.seq_names:
                self.seq_names.append(seq_name)
                self.tracked_instance = self._generate_fake_tracks()
                self.frame_id = 1
        track_pos = self.tracked_instance.query_pos.transpose(0, 1).reshape(
            1, -1, self.num_query_h, self.num_query_w
        )
        ref_pts = self.tracked_instance.ref_pts.transpose(0, 1).reshape(
            1, -1, self.num_query_h, self.num_query_w
        )
        mask_query = self.tracked_instance.mask_query.reshape(1, 1, 1, -1)
        return track_pos, ref_pts, mask_query, seq_name

    @autocast(enabled=False)
    @fx_wrap()
    def _prepare_seq_forward(self, seq_data):
        """Prepare single seq data when train."""
        if self.training:
            targets = self.prepare_targets(seq_data)
            self.criterion.initialize_for_single_clip()
            self.tracked_instance = self._generate_fake_tracks()
            return targets
        return None

    def forward(self, data: Dict):
        if self.compile_motr:
            feats = self.extract_feat(data["img"])
            outputs_classes, outputs_coords, out_hs = self.head(
                [feats[-1]],
                data["query_pos"],
                data["ref_pts"],
                data["mask_query"],
            )
            return outputs_classes[-1], outputs_coords[-1], out_hs
        elif self.compile_qim:
            embed_query_pos = self.track_embed(
                data["query_pos"],
                data["output_embedding"],
                data["mask_query"],
            )
            return embed_query_pos
        else:
            model_outs = {}
            for i in range(self.batch_size):
                seq_data = data["frame_data_list"][i]
                targets = self._prepare_seq_forward(seq_data)
                for frame_id in range(self.max_frame_num):
                    (
                        track_pos,
                        ref_pts,
                        mask_query,
                        seq_name,
                    ) = self._prepare_single_image_forward(seq_data, frame_id)

                    if seq_name is None:
                        break
                    feats = self.extract_feat(seq_data["img"][frame_id])

                    (
                        outputs_classes_head,
                        outputs_coords_head,
                        out_hs,
                    ) = self.head([feats[-1]], track_pos, ref_pts, mask_query)

                    model_outs = self._frame_post_process(
                        seq_data,
                        targets,
                        frame_id,
                        seq_name,
                        outputs_classes_head,
                        outputs_coords_head,
                        out_hs,
                        model_outs,
                    )

                model_outs = self._sample_post_process(model_outs)

            return model_outs

    @autocast(enabled=False)
    @fx_wrap()
    def _sample_post_process(
        self,
        model_outs,
    ):
        if self.training:
            seq_losses = self.criterion()
            loss_weight = self.criterion.weight_dict
            for key, value in seq_losses.items():
                if key in model_outs:
                    model_outs[key].append(value * loss_weight[key])
                else:
                    model_outs[key] = []
                    model_outs[key].append(value * loss_weight[key])
        return model_outs

    @autocast(enabled=False)
    @fx_wrap()
    def _frame_post_process(
        self,
        seq_data,
        targets,
        frame_id,
        seq_name,
        outputs_classes_head,
        outputs_coords_head,
        out_hs,
        model_outs,
    ):
        empty_track_instance = self._generate_empty_tracks()
        fake_track_instance = self._generate_fake_tracks()
        (padding_track_instance, pad_len, frame_outs,) = self.post_process(
            track_instances=self.tracked_instance,
            empty_track_instance=empty_track_instance,
            fake_track_instance=fake_track_instance,
            out_hs=out_hs,
            outputs_classes_head=outputs_classes_head,
            outputs_coords_head=outputs_coords_head,
            criterion=self.criterion,
            targets=targets,
            seq_data=seq_data,
            frame_id=frame_id,
            seq_frame_id=self.frame_id,
            seq_name=seq_name,
        )
        if not self.training:
            model_outs[seq_data["img_name"][frame_id]] = frame_outs
        if pad_len:
            (
                pd_query_pos,
                pd_output_embedding,
                pd_mask_query,
            ) = self.prepare_track_embed_input(padding_track_instance)
            embed_query_pos = self.track_embed(
                pd_query_pos,
                pd_output_embedding,
                pd_mask_query,
            )
            padding_track_instance = self._pocess_track_embed_output(
                padding_track_instance,
                pad_len,
                embed_query_pos,
            )

        self.tracked_instance = padding_track_instance
        self.frame_id += 1
        return model_outs

    @autocast(enabled=False)
    def _pocess_track_embed_output(
        self, padding_track_instance, pad_len, embed_query_pos
    ):
        embed_query_pos = (
            embed_query_pos.reshape(1, 1, embed_query_pos.shape[1], -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .squeeze(0)
            .squeeze(0)
        )
        ref_pts_tmp = (
            padding_track_instance.pred_boxes_unsigmoid.detach().clone()
        )
        ref_pts_tmp[pad_len:] = 0.0
        embed_query_pos[pad_len:] = 0.0
        padding_track_instance.query_pos = embed_query_pos
        padding_track_instance.ref_pts = ref_pts_tmp

        return padding_track_instance

    @autocast(enabled=False)
    def prepare_track_embed_input(self, tracked_instance):
        query_pos = tracked_instance.query_pos.unsqueeze(0).unsqueeze(0)
        output_embedding = tracked_instance.output_embedding.unsqueeze(
            0
        ).unsqueeze(0)
        mask_query = tracked_instance.mask_query.reshape(1, 1, 1, -1)
        return query_pos, output_embedding, mask_query

    @autocast(enabled=False)
    def prepare_targets(self, targets):
        boxes = targets["gt_bboxes"]
        shapes = targets["img_shape"]
        gt_classes = targets["gt_classes"]
        gt_ids = targets["gt_ids"]
        frame_gts = []
        for shape, boxes_per_image, gt_class, gt_id in zip(
            shapes, boxes, gt_classes, gt_ids
        ):
            h, w = shape[-2:]
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=boxes_per_image.device
            )
            gt_boxes = boxes_per_image / image_size_xyxy
            gt_boxes = box_corner_to_center(gt_boxes)

            each_frame_gt = EasyDict()
            each_frame_gt.gt_bboxes = gt_boxes
            each_frame_gt.gt_classes = gt_class
            each_frame_gt.gt_ids = gt_id
            frame_gts.append(each_frame_gt)
        return frame_gts

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head, self.track_embed]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.backbone, self.neck, self.head, self.track_embed]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


class FakeHead(nn.Module):
    def __init__(self, num_queries, hidden_dim, num_classes):
        super(FakeHead, self).__init__()
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.refpoint_embed = nn.Embedding(num_queries, 4)
        self.num_classes = num_classes


@OBJECT_REGISTRY.register
class MotrIrInfer(Motr):
    """The basic structure of MotrIrInfer.

    Args:
        ir_model: The ir model.
        qim_model_path: The path of qim hbir model.
        post_process: post process module.
        num_query_h: The num of h dim for query reshape.
        batch_size: batch size.
        num_queries: The num of query.
        queries_dim: The dim of query.
        LoadCheckpoint: LoadCheckpoint func.
        num_classes: Num class.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        qim_ir_model: nn.Module,
        post_process: nn.Module = None,
        num_query_h: int = 2,
        batch_size: int = 1,
        num_queries: int = 256,
        queries_dim: int = 256,
        LoadCheckpoint: Optional[Callable] = None,
        num_classes: int = 1,
    ):
        super(Motr, self).__init__()
        self.post_process = post_process
        self.seq_names = []
        self.tracked_instance = None
        self.frame_id = 1
        self.num_queries = num_queries
        self.queries_dim = queries_dim
        self.num_query_h = num_query_h
        assert self.num_queries % self.num_query_h == 0, (
            "num_queries must be divisible by num_query_h, but get "
            "{} vs {}".format(self.num_queries, self.num_query_h)
        )
        self.num_query_w = self.num_queries // self.num_query_h
        self.max_frame_num = 1
        self.batch_size = batch_size

        self.model = ir_model
        self.qim_model = qim_ir_model
        self.head = FakeHead(self.num_queries, self.queries_dim, num_classes)

        self.loadcheckpoint = LoadCheckpoint

        self.data_device = "cpu"

        if self.loadcheckpoint is not None:
            self = self.loadcheckpoint(self)

    def forward(self, data: Dict):

        model_outs = {}
        for i in range(self.batch_size):
            seq_data = data["frame_data_list"][i]
            targets = self._prepare_seq_forward(seq_data)
            for frame_id in range(self.max_frame_num):
                (
                    track_pos,
                    ref_pts,
                    mask_query,
                    seq_name,
                ) = self._prepare_single_image_forward(seq_data, frame_id)

                if seq_name is None:
                    break
                model_inputs = {
                    "img": seq_data["img"][frame_id],
                    "query_pos": track_pos,
                    "mask_query": mask_query,
                    "ref_pts": ref_pts,
                }
                hbir_output = self.model(model_inputs)
                (
                    outputs_classes_head,
                    outputs_coords_head,
                    out_hs,
                ) = hbir_output

                model_outs = self._frame_post_process(
                    seq_data,
                    targets,
                    frame_id,
                    seq_name,
                    outputs_classes_head,
                    outputs_coords_head,
                    out_hs,
                    model_outs,
                )

            model_outs = self._sample_post_process(model_outs)

            return model_outs

    def _generate_empty_tracks(self):
        """Genrate empty track instance for detection new object."""
        track_instances = EasyDict()
        device = self.head.query_embed.weight.device
        track_instances.output_embedding = torch.zeros(
            (self.num_queries, self.queries_dim), device=device
        )
        track_instances.obj_idxes = torch.full(
            (self.num_queries,), -1, dtype=torch.long, device=device
        )
        track_instances.matched_gt_idxes = torch.full(
            (self.num_queries,), -1, dtype=torch.long, device=device
        )
        track_instances.disappear_time = torch.zeros(
            (self.num_queries,), dtype=torch.long, device=device
        )
        track_instances.iou = torch.zeros(
            (self.num_queries,), dtype=torch.float, device=device
        )
        track_instances.scores = torch.zeros(
            (self.num_queries,), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (self.num_queries,), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (self.num_queries, 4), dtype=torch.float, device=device
        )
        track_instances.pred_boxes_unsigmoid = torch.zeros(
            (self.num_queries, 4), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (self.num_queries, self.head.num_classes),
            dtype=torch.float,
            device=device,
        )
        track_instances.mask_query = torch.ones(
            (self.num_queries,), dtype=torch.float, device=device
        )
        track_instances.ref_pts = self.head.refpoint_embed.weight
        track_instances.query_pos = self.head.query_embed.weight
        return track_instances

    def _frame_post_process(
        self,
        seq_data,
        targets,
        frame_id,
        seq_name,
        outputs_classes_head,
        outputs_coords_head,
        out_hs,
        model_outs,
    ):
        empty_track_instance = self._generate_empty_tracks()
        fake_track_instance = self._generate_fake_tracks()
        (padding_track_instance, pad_len, frame_outs,) = self.post_process(
            track_instances=self.tracked_instance,
            empty_track_instance=empty_track_instance,
            fake_track_instance=fake_track_instance,
            out_hs=out_hs,
            outputs_classes_head=[outputs_classes_head],
            outputs_coords_head=[outputs_coords_head],
            criterion=None,
            targets=targets,
            seq_data=seq_data,
            frame_id=frame_id,
            seq_frame_id=self.frame_id,
            seq_name=seq_name,
        )
        if not self.training:
            model_outs[seq_data["img_name"][frame_id]] = frame_outs
        if pad_len:
            (
                pd_query_pos,
                pd_output_embedding,
                pd_mask_query,
            ) = self.prepare_track_embed_input(padding_track_instance)

            qim_inputs = {
                "output_embedding": pd_output_embedding,
                "query_pos": pd_query_pos,
                "mask_query": pd_mask_query,
            }

            embed_query_pos = self.qim_model(qim_inputs)
            padding_track_instance = self._pocess_track_embed_output(
                padding_track_instance,
                pad_len,
                embed_query_pos,
            )

        self.tracked_instance = padding_track_instance
        self.frame_id += 1
        return model_outs
