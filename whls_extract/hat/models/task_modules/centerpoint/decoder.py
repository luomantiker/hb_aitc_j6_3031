# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from hat.core.box3d_utils import bev3d_nms
from hat.core.circle_nms_jit import circle_nms
from hat.core.nus_box3d_utils import bbox_bev2ego
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


__all__ = ["CenterPointDecoder"]


@OBJECT_REGISTRY.register
class CenterPointDecoder(nn.Module):
    """The CenterPoint Decoder.

    Args:
        class_names: List of calss name for detection task
        tasks: List of tasks
        bev_size: Bev view size.
        norm_bbox: Whether using normalize for dim of bbox.
        max_num: Maximun number for bboxes of single task.
        use_max_pool: Whether using max pool as nms.
        max_pool_kernel: Kernel size if using max pool for nms.
        out_size_factor: Factor for output bbox.
        score_threshold: Treshold for filtering bbox of low score.
        nms_type: Which NMS type used for single task.
                  Choose ["rotate", ""circle"]
        min_radius: Min radius for circle nms.
        nms_threshold: NMS threshold.
        pre_max_size: Max size before nms.
        post_max_size: Max size after nms.
        decode_to_ego: Whether decoding to ego coordinate.
    """

    def __init__(
        self,
        class_names: List[str],
        tasks: List[Dict],
        bev_size: Tuple[float],
        norm_bbox: bool = True,
        max_num: int = 50,
        use_max_pool: bool = True,
        max_pool_kernel: Optional[int] = 3,
        out_size_factor: int = 4,
        score_threshold: float = 0.1,
        nms_type: Optional[List[str]] = None,
        min_radius: Optional[List[int]] = None,
        nms_threshold: float = None,
        pre_max_size: int = 1000,
        post_max_size: int = 100,
        decode_to_ego: bool = True,
    ):
        super(CenterPointDecoder, self).__init__()
        self.class_names = class_names
        self.max_num = max_num
        self.out_size_factor = out_size_factor
        self.tasks = tasks
        self.score_threshold = score_threshold
        self.bev_size = bev_size
        self.grid_size = [
            bev_size[0] * 2 / bev_size[2],
            bev_size[1] * 2 / bev_size[2],
        ]
        self.norm_bbox = norm_bbox
        self.decode_to_ego = decode_to_ego
        self.use_max_pool = use_max_pool
        self.max_pool_kernel = max_pool_kernel

        if nms_type is None:
            nms_type = [
                "rotate",
                "rotate",
                "rotate",
                "rotate",
                "rotate",
                "rotate",
            ]
        self.nms_type = nms_type

        if min_radius is None:
            min_radius = [4, 12, 10, 1, 0.85, 0.175]
        self.min_radius = min_radius

        if nms_threshold is None:
            nms_threshold = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.nms_threshold = nms_threshold

        self.pre_max_size = pre_max_size
        self.post_max_size = post_max_size

    def _topk(self, scores, K=80):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
        topk_inds = topk_inds % (height * width)
        topk_ys = (
            (topk_inds.float() / torch.tensor(width, dtype=torch.float))
            .int()
            .float()
        )
        topk_xs = (topk_inds % width).int().float()
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()

        topk_inds = self._gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind
        ).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(
            batch, K
        )
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(
            batch, K
        )
        topk_coords = torch.stack([topk_xs, topk_ys], dim=2)
        return topk_score, topk_inds, topk_clses, topk_coords

    def _gather_feat(self, feat, ind):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        return feat

    def _decode_task(self, preds, task):
        heatmap = preds["heatmap"].sigmoid()
        bbox_preds = []
        for key in ["reg", "height", "dim", "rot"]:
            bbox_preds.append(preds[key])
        if "vel" in preds:
            bbox_preds.append(preds["vel"])
        bbox = torch.cat(bbox_preds, dim=1)

        if self.use_max_pool:
            kernel = self.max_pool_kernel
            pad = (kernel - 1) // 2
            max_heatmap = F.max_pool2d(
                heatmap, (kernel, kernel), stride=1, padding=(pad, pad)
            )
            heatmap *= (heatmap == max_heatmap).float()

        scores, inds, clses, coords = self._topk(heatmap, K=self.max_num)
        bbox = bbox.permute((0, 2, 3, 1)).view(
            bbox.shape[0], -1, bbox.shape[1]
        )
        bbox = self._gather_feat(bbox, inds)
        reg = bbox[..., 0:2]
        xy = coords + reg
        hei = bbox[..., 2].unsqueeze(-1)
        dim = bbox[..., 3:6]
        if self.norm_bbox:
            dim = torch.exp(dim)
        # rotation value and direction label
        rot_sine = bbox[..., 6].unsqueeze(-1)
        rot_cosine = bbox[..., 7].unsqueeze(-1)
        rot = torch.atan2(rot_sine, rot_cosine)

        xy *= self.out_size_factor
        if bbox.shape[2] == 10:
            vel = bbox[..., 8:]
            box_preds = torch.cat([xy, hei, dim, rot, vel], dim=2)
        else:  # exist velocity, nuscene format
            box_preds = torch.cat([xy, hei, dim, rot], dim=2)

        return box_preds, scores, clses

    def _adjust_cat(self, cats, task):
        for idx, cat in enumerate(cats):
            cats[idx] = self.class_names.index(task["class_names"][cat])
        return cats

    def _decode_to_ego(self, bboxes_preds):
        ego_bboxes = []
        for bboxes in bboxes_preds:
            if len(bboxes) != 0:
                ego_bboxes_batch = bbox_bev2ego(bboxes, self.bev_size)
            else:
                ego_bboxes_batch = [torch.ones(11)]
            ego_bboxes.append(ego_bboxes_batch)

        return ego_bboxes

    def forward(
        self, preds: Sequence[torch.Tensor], meta_data: Dict[str, Any]
    ):
        rets = []
        for task_id, task in enumerate(self.tasks):
            ret_task = []
            box_preds, scores, clses = self._decode_task(preds[task_id], task)

            for each_preds, each_scores, each_clses in zip(
                box_preds, scores, clses
            ):
                thresh_mask = each_scores > self.score_threshold
                each_preds = each_preds[thresh_mask]
                each_scores = each_scores[thresh_mask]
                each_clses = each_clses[thresh_mask]
                valid_mask = (
                    (each_preds[..., 0] > 0)
                    & (each_preds[..., 0] < self.grid_size[1])
                    & (each_preds[..., 1] > 0)
                    & (each_preds[..., 1] < self.grid_size[0])
                )
                each_preds = each_preds[valid_mask]
                each_scores = each_scores[valid_mask]
                each_clses = each_clses[valid_mask]
                each_preds = each_preds[: self.pre_max_size]
                each_scores = each_scores[: self.pre_max_size]
                each_clses = each_clses[: self.pre_max_size]
                if len(each_preds) == 0:
                    ret = []
                else:
                    if self.use_max_pool is False:
                        if self.nms_type[task_id] == "circle":
                            centers = each_preds[:, :2]
                            boxes = torch.cat(
                                [centers, each_scores.view(-1, 1)], dim=1
                            )
                            keep = torch.tensor(
                                circle_nms(
                                    boxes.detach().cpu().numpy(),
                                    self.min_radius[task_id],
                                ),
                                dtype=torch.long,
                                device=boxes.device,
                            )
                        else:
                            keep = bev3d_nms(
                                each_preds[..., :7],
                                each_scores,
                                self.nms_threshold[task_id],
                            )
                        each_preds = each_preds[keep]
                        each_scores = each_scores[keep]
                        each_clses = each_clses[keep]
                    each_preds = each_preds[: self.post_max_size]
                    each_scores = each_scores.unsqueeze(-1)
                    each_clses = self._adjust_cat(each_clses, task).unsqueeze(
                        -1
                    )
                    ret = torch.cat(
                        [each_preds, each_scores, each_clses], dim=1
                    )
                ret_task.append(ret)
            rets.append(ret_task)
        final_rets = []
        batch = preds[0]["heatmap"].shape[0]
        for b in range(batch):
            ret_batch = []
            for ret in rets:
                if len(ret) == 0 or len(ret[b]) == 0:
                    continue
                ret_batch.append(ret[b])
            if len(ret_batch) != 0:
                ret_batch = torch.cat(ret_batch)
            final_rets.append(ret_batch)
        if self.decode_to_ego:
            final_rets = self._decode_to_ego(final_rets)
        return final_rets
