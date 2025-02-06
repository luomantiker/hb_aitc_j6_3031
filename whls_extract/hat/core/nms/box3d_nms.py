# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Optional, Tuple

import torch
from torch import Tensor

from hat.core.box3d_utils import bev3d_nms
from hat.utils.package_helper import require_packages

try:
    from horizon_plugin_pytorch.functional import nms_rotated
except ImportError:
    nms_rotated = None

logger = logging.getLogger(__name__)


def box2d3d_multiclass_nms(
    mlvl_bboxes3d: Tensor,
    mlvl_bboxes3d_for_nms: Tensor,
    mlvl_scores3d: Tensor,
    score_thr: float,
    max_num: Optional[int] = None,
    nms_thr: Optional[float] = None,
    mlvl_dir_scores: Optional[Tensor] = None,
    mlvl_attr_scores: Optional[Tensor] = None,
    mlvl_centers2d: Optional[Tensor] = None,
    mlvl_scores2d: Optional[Tensor] = None,
    mlvl_bboxes2d: Optional[Tensor] = None,
    do_bev3d_nms: bool = False,
    do_nms_bev: bool = False,
    filter_by_scores2d: bool = False,
) -> Tuple[Tensor, ...]:
    """Multi-class nms for 3D boxes.

    Args:
        mlvl_bboxes3d: Multi-level boxes with shape (N, M). M is the dimensions
            of boxes.
        mlvl_bboxes3d_for_nms: Multi-level boxes with shape
            (N, 5) ([x1, y1, x2, y2, ry]). N is the number of boxes.
        mlvl_scores3d: Multi-level boxes with shape (N, C + 1). N is the number
            of boxes. C is the number of classes.
        score_thr: Score thredhold to filter boxes with low confidence.
        max_num: Maximum number of boxes that will be kept.
        nms_thr: NMS score threshold.
        mlvl_dir_scores: Multi-level scores of direction classifier.
            Defaults to None.
        mlvl_attr_scores (torch.Tensor, optional): Multi-level scores
            of attribute classifier. Defaults to None.
        mlvl_centers2d: Multi-level projected 2D center. Defaults is None.
        mlvl_bboxes2d: Multi-level 2D bounding boxes. Defaults is None.
        do_bev3d_nms: Whether to use bev3d nms to get kept bbox mask.
        do_nms_bev: Whether to use bev nms to get kept bbox mask.
        filter_by_scores2d: Whether filter bbox by scores2d.

    Returns:
        Return results after nms, including 3D
            bounding boxes, scores, labels, direction scores, attribute
            scores (optional) and 2D bounding boxes (optional).
    """
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    # because mlvl_scores expand default ones_like matrix
    if filter_by_scores2d:
        assert (
            mlvl_scores2d is not None
        ), "mlvl_scores2d cannot be None when filtered by score2d"
        scores = mlvl_scores2d
    else:
        scores = mlvl_scores3d
    num_classes = scores.shape[1] - 1
    bboxes3d = []
    bboxes2d = []
    scores2d = []
    scores3d = []
    labels = []
    dir_scores = []
    center2d_prj = []
    attr_scores = []
    for i in range(0, num_classes):
        # get bboxes and scores of this class
        cls_inds = scores[:, i] > score_thr
        if not cls_inds.any():
            continue

        _scores3d = mlvl_scores3d[cls_inds, i]
        _bboxes_for_nms = mlvl_bboxes3d_for_nms[cls_inds, :]  # noqa

        # test no nms
        selected = _scores3d > score_thr

        if do_bev3d_nms:
            selected = bev3d_nms(_bboxes_for_nms, _scores3d, nms_thr)
        if do_nms_bev:
            selected = nms_bev(_bboxes_for_nms, _scores3d, nms_thr)
        _mlvl_bboxes3d = mlvl_bboxes3d[cls_inds, :]
        bboxes3d.append(_mlvl_bboxes3d[selected])

        scores3d.append(_scores3d[selected])
        cls_label = mlvl_bboxes3d.new_full(
            (len(selected),), i, dtype=torch.long
        )
        labels.append(cls_label)

        if mlvl_scores2d is not None:
            _scores2d = mlvl_scores2d[cls_inds, i]
            scores2d.append(_scores2d[selected])
        if mlvl_dir_scores is not None:
            _mlvl_dir_scores = mlvl_dir_scores[cls_inds]
            dir_scores.append(_mlvl_dir_scores[selected])
        if mlvl_bboxes2d is not None:
            _mlvl_bboxes2d = mlvl_bboxes2d[cls_inds]
            bboxes2d.append(_mlvl_bboxes2d[selected])
        if mlvl_centers2d is not None:
            _mlvl_center2d = mlvl_centers2d[cls_inds]
            center2d_prj.append(_mlvl_center2d[selected])
        if mlvl_attr_scores is not None:
            _mlvl_attr_scores = mlvl_attr_scores[cls_inds]
            attr_scores.append(_mlvl_attr_scores[selected])

    if bboxes3d:
        bboxes3d = torch.cat(bboxes3d, dim=0)
        scores3d = torch.cat(scores3d, dim=0)
        labels = torch.cat(labels, dim=0)
        if mlvl_scores2d is not None:
            scores2d = torch.cat(scores2d, dim=0)
        if mlvl_dir_scores is not None:
            dir_scores = torch.cat(dir_scores, dim=0)
        if mlvl_bboxes2d is not None:
            bboxes2d = torch.cat(bboxes2d, dim=0)
        if mlvl_centers2d is not None:
            center2d_prj = torch.cat(center2d_prj, dim=0)
        if mlvl_attr_scores is not None:
            attr_scores = torch.cat(attr_scores, dim=0)
        if max_num is not None and bboxes3d.shape[0] > max_num:
            _, inds = scores3d.sort(descending=True)
            inds = inds[:max_num]
            bboxes3d = bboxes3d[inds, :]
            labels = labels[inds]
            scores3d = scores3d[inds]
            if mlvl_scores2d is not None:
                scores2d = scores2d[inds]
            if mlvl_dir_scores is not None:
                dir_scores = dir_scores[inds]
            if mlvl_bboxes2d is not None:
                bboxes2d = bboxes2d[inds]
            if mlvl_centers2d is not None:
                center2d_prj = center2d_prj[inds]
            if mlvl_attr_scores is not None:
                attr_scores = attr_scores[inds]

    else:
        bboxes3d = mlvl_scores3d.new_zeros((0, mlvl_bboxes3d.size(-1)))
        scores3d = mlvl_scores3d.new_zeros((0,))
        labels = mlvl_scores3d.new_zeros((0,), dtype=torch.long)
        if mlvl_scores2d is not None:
            scores2d = mlvl_scores3d.new_zeros((0,))
        if mlvl_dir_scores is not None:
            dir_scores = mlvl_scores3d.new_zeros((0,))
        if mlvl_bboxes2d is not None:
            bboxes2d = mlvl_scores3d.new_zeros((0, 4))
        if mlvl_centers2d is not None:
            center2d_prj = mlvl_scores3d.new_zeros((0, 2))
        if mlvl_attr_scores is not None:
            attr_scores = mlvl_scores3d.new_zeros((0,))

    results = (labels, scores2d, bboxes2d, scores3d, bboxes3d, center2d_prj)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores,)
    else:
        results = results + (None,)
    if mlvl_attr_scores is not None:
        results = results + (attr_scores,)
    else:
        results = results + (None,)
    return results


# This function duplicates functionality of mmcv.ops.iou_3d.nms_bev
# from mmcv<=1.5, but using cuda ops from mmcv.ops.nms.nms_rotated.
# Nms api will be unified in mmdetection3d one day.
@require_packages("horizon_plugin_pytorch>=1.2.3")
def nms_bev(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    thresh: float,
    pre_max_size: Optional[int] = None,
    post_max_size: Optional[int] = None,
) -> torch.Tensor:
    """NMS function GPU implementation (for BEV boxes). The overlap of two \
    boxes for IoU calculation is defined as the exact overlapping area of the \
    two boxes. In this function, one can also set ``pre_max_size`` and \
    ``post_max_size``.

    Args:
        boxes: Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores: Scores of boxes with the shape of [N].
        thresh: Overlap threshold of NMS.
        pre_max_size: Max size of boxes before NMS.
            Default: None.
        post_max_size: Max size of boxes after NMS.
            Default: None.

    Returns:
        Indexes after NMS.
    """  # noqa
    assert boxes.size(1) == 5, "Input boxes shape should be [N, 5]"
    order = scores.sort(0, descending=True)[1]
    if pre_max_size is not None:
        order = order[:pre_max_size]
    boxes = boxes[order].contiguous()
    scores = scores[order]

    # xyxyr -> back to xywhr
    # note: better skip this step before nms_bev call in the future
    boxes = torch.stack(
        (
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2,
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
            boxes[:, 4],
        ),
        dim=-1,
    )

    keep = nms_rotated(boxes, scores, thresh)[1]
    keep = order[keep]
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def nms_bev_multiclass(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    thresh: float,
    num_max: int = 500,
):
    """NMS function GPU implementation (for BEV boxes). The overlap of two \
    boxes for IoU calculation is defined as the exact overlapping area of the \
    two boxes. In this function, one can also set ``pre_max_size`` and \
    ``post_max_size``.

    Args:
        boxes: Input boxes with the shape of [N, 5]
        scores: Scores of boxes with the shape of [N].
        labels: Labels of boxes with the shape of [N].
        num_classes: number of classes.
        thresh: Overlap threshold of NMS.

    Returns:
        torch.Tensor: Indexes after NMS.
    """  # noqa
    assert boxes.size(1) == 5, "Input boxes shape should be [N, 5]"

    all_scores = []
    all_keep = []
    order = scores.sort(0, descending=True)[1]
    boxes = boxes[order].contiguous()
    scores = scores[order].contiguous()
    labels = labels[order].contiguous()

    for idx in range(num_classes):
        cls_idx = labels == idx
        scores_cls = scores[cls_idx]
        if scores_cls.shape[0] == 0:
            continue
        boxes_cls = boxes[cls_idx]
        order_cls = order[cls_idx]

        keep = nms_rotated(boxes_cls, scores_cls, thresh)[1]
        scores_cls = scores_cls[keep]
        order_cls = order_cls[keep]

        all_scores.append(scores_cls)
        all_keep.append(order_cls)
    all_scores = torch.cat(all_scores)
    all_keep = torch.cat(all_keep)
    all_order = all_scores.sort(0, descending=True)[1]
    all_keep = all_keep[all_order]
    all_keep = all_keep[:num_max]
    return all_keep
