import logging
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hat.models.task_modules.maptr.criterion import (
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
)
from hat.registry import OBJECT_REGISTRY

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

logger = logging.getLogger(__name__)

__all__ = [
    "MapTRAssigner",
]


def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
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


@OBJECT_REGISTRY.register
class MapTRAssigner(nn.Module):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_cost: The classification cost module. Default: None.
        pts_cost: The points cost module. Default: None.
        pc_range: The point cloud range. Default: None.
        pred_absolute_points: Whether to predict absolute points.
            Default: False.
    """

    def __init__(
        self,
        cls_cost: nn.Module = None,
        pts_cost: nn.Module = None,
        pc_range: List[float] = None,
        pred_absolute_points: bool = False,
    ):
        super(MapTRAssigner, self).__init__()
        self.cls_cost = cls_cost
        self.pts_cost = pts_cost
        self.pc_range = pc_range
        self.pred_absolute_points = pred_absolute_points

    def assign(
        self,
        bbox_pred: Tensor,
        cls_pred: Tensor,
        pts_pred: Tensor,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
        gt_pts: Tensor,
        gt_bboxes_ignore: Tensor = None,
        eps: Union[int, float] = 1e-7,
    ) -> Tuple[int, Tensor, None, Tensor, Tensor]:
        """Compute one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
        between predictions and gts, treat this prediction as foreground
        and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred: Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in the range [0, 1].
                Shape [num_query, 4].
            cls_pred: Predicted classification logits.
                Shape [num_query, num_class].
            pts_pred: Predicted points. Shape [num_query, num_pts, 2].
            gt_bboxes: Ground truth boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels: Labels of `gt_bboxes`. Shape (num_gt,).
            gt_pts: Ground truth points.
                Shape [num_gt, num_orders, num_pts_per_gtline, 2].
            gt_bboxes_ignore: Ignored ground truth bboxes. Default: None.
            eps: A value added to the denominator for numerical stability.
                Default: 1e-7.

        Returns:
            num_gts: The number of ground truth boxes.
            assigned_gt_inds: The indices of assigned ground truth boxes.
            None: Reserved for future use.
            assigned_labels: The labels of assigned ground truth boxes.
            order_index: The order index tensor.
        """
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        assert (
            bbox_pred.shape[-1] == 4
        ), "Only support bbox pred shape is 4 dims"
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full(
            (num_bboxes,), -1, dtype=torch.long
        )
        assigned_labels = bbox_pred.new_full(
            (num_bboxes,), -1, dtype=torch.long
        )
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return num_gts, assigned_gt_inds, None, assigned_labels, None

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)

        _, num_orders, num_pts_per_gtline, num_coords = gt_pts.shape
        normalized_gt_pts = normalize_2d_pts(gt_pts, self.pc_range)
        num_pts_per_predline = pts_pred.size(1)
        if num_pts_per_predline != num_pts_per_gtline:
            pts_pred_interpolated = F.interpolate(
                pts_pred.permute(0, 2, 1),
                size=(num_pts_per_gtline),
                mode="linear",
                align_corners=True,
            )
            pts_pred_interpolated = pts_pred_interpolated.permute(
                0, 2, 1
            ).contiguous()
        else:
            pts_pred_interpolated = pts_pred
        if self.pred_absolute_points:
            pts_pred_interpolated = normalize_2d_pts(
                pts_pred_interpolated, self.pc_range
            )
        # num_q, num_pts, 2 <-> num_gt, num_pts, 2
        pts_cost_ordered = self.pts_cost(
            pts_pred_interpolated, normalized_gt_pts
        )
        pts_cost_ordered = pts_cost_ordered.view(
            num_bboxes, num_gts, num_orders
        )
        pts_cost, order_index = torch.min(pts_cost_ordered, 2)

        # weighted sum of above costs
        cost = cls_cost + pts_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError(
                'Please run "pip install scipy" ' "to install scipy first."
            )
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device
        )
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device
        )

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return num_gts, assigned_gt_inds, None, assigned_labels, order_index
