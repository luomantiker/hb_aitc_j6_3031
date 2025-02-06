# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast

from hat.core.box_utils import box_center_to_corner
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = box_center_to_corner(bboxes)
    bboxes[..., 0::2] = (
        bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0]
    )
    bboxes[..., 1::2] = (
        bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    )

    return bboxes


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = (
        pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    )
    new_pts[..., 1:2] = (
        pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    )
    return new_pts


class MapTRNMSFreeCoder(object):
    """Bbox coder for NMS-free detector.

    Args:
        pc_range: Range of point cloud.
        post_center_range: Limit of the center.
            Default: None.
        max_num: Max number to be kept. Default: 100.
        score_threshold: Threshold to filter boxes based on score.
            Default: None.
        num_classes: Number of classes. Default: 10.
    """

    def __init__(
        self,
        pc_range: List[float],
        post_center_range: List[float] = None,
        max_num: int = 100,
        score_threshold: float = None,
        num_classes: int = 10,
        pred_absolute_points: bool = False,
    ):
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.pred_absolute_points = pred_absolute_points

    def encode(self):

        pass

    def decode_single(
        self, cls_scores: Tensor, bbox_preds: Tensor, pts_preds: Tensor
    ) -> Dict:
        """Decode bboxes.

        Args:
            cls_scores: Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            pts_preds: Outputs from the regression head. \
                Shape [num_query, 9].
        Returns:
            Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
        pts_preds = pts_preds[bbox_index]

        if self.pred_absolute_points:
            final_box_preds = bbox_preds
            final_pts_preds = pts_preds
        else:
            final_box_preds = denormalize_2d_bbox(bbox_preds, self.pc_range)
            final_pts_preds = denormalize_2d_pts(
                pts_preds, self.pc_range
            )  # num_q,num_p,2

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
                final_box_preds[..., :4] >= self.post_center_range[:4]
            ).all(1)
            mask &= (
                final_box_preds[..., :4] <= self.post_center_range[4:]
            ).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            pts = final_pts_preds[mask]
            labels = final_preds[mask]
            predictions_dict = {
                "bboxes": boxes3d,
                "scores": scores,
                "labels": labels,
                "pts": pts,
            }

        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )
        return predictions_dict

    def decode(self, preds_dicts):
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        all_pts_preds = preds_dicts["all_pts_preds"][-1]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(
                    all_cls_scores[i], all_bbox_preds[i], all_pts_preds[i]
                )
            )
        return predictions_list


@OBJECT_REGISTRY.register
class MapTRPostProcess(nn.Module):
    """Post-processing module for MapTR.

    Args:
        pc_range: Range of the point cloud. Default: None.
        post_center_range: Limit of the center. Default: None.
        max_num: Maximum number of boxes to keep. Default: 100.
        score_threshold: Threshold to filter boxes based on score.
            Default: None.
        num_classes: Number of classes. Default: 10.
    """

    def __init__(
        self,
        pc_range: Optional[List[float]] = None,
        post_center_range: Optional[List[float]] = None,
        max_num: int = 100,
        score_threshold: Optional[float] = None,
        num_classes: int = 10,
        pred_absolute_points: bool = False,
    ):
        super(MapTRPostProcess, self).__init__()

        self.bbox_coder = MapTRNMSFreeCoder(
            pc_range,
            post_center_range=post_center_range,
            max_num=max_num,
            score_threshold=score_threshold,
            num_classes=num_classes,
            pred_absolute_points=pred_absolute_points,
        )
        self.pc_range = pc_range

    @autocast(enabled=False)
    def forward(self, preds_dicts, img_metas):
        if "one2many_outs" in preds_dicts:
            preds_dicts.pop("one2many_outs")
        if "pv_seg" in preds_dicts:
            preds_dicts.pop("pv_seg")
        if "dense_depth" in preds_dicts:
            preds_dicts.pop("dense_depth")
        for k, v in preds_dicts.items():
            if v is not None:
                preds_dicts[k] = v.float()

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        bbox_results = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            # code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds["scores"]
            labels = preds["labels"]
            pts = preds["pts"]

            result_dict = {
                "boxes_3d": bboxes.to("cpu"),
                "scores_3d": scores.cpu(),
                "labels_3d": labels.cpu(),
                "pts_3d": pts.to("cpu"),
            }

            bbox_results.append(result_dict)

        # import pdb;pdb.set_trace()
        return bbox_results
