# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict, List

import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast

from hat.models.task_modules.bevformer.utils import denormalize_bbox
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class BevFormerProcess(nn.Module):
    """The basic structure of BevFormerProcess.

    Args:
        pc_range: VCS range or point cloud range.
        post_center_range: Limit of the center.
        max_num: Max number to be kept.
        score_threshold: Threshold to filter boxes based on score.
        num_classes: The num of classes.
    """

    def __init__(
        self,
        pc_range: List[float],
        post_center_range: List[float] = None,
        max_num: int = 100,
        score_threshold: float = None,
        num_classes: int = 10,
    ):
        super(BevFormerProcess, self).__init__()
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def decode_single(self, cls_scores: Tensor, bbox_preds: Tensor) -> Tensor:
        """Decode bboxes."""

        max_num = self.max_num
        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.reshape(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device
            )
            mask = (
                final_box_preds[..., :3] >= self.post_center_range[:3]
            ).all(1)
            mask &= (
                final_box_preds[..., :3] <= self.post_center_range[3:]
            ).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions = torch.cat(
                [boxes3d, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1
            ).detach()

        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )
        return predictions

    @autocast(enabled=False)
    def forward(self, preds_dicts: Dict) -> List:
        """Forward BevFormerProcess."""
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(all_cls_scores[i], all_bbox_preds[i])
            )
        return predictions_list
