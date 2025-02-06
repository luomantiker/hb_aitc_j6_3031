from abc import ABCMeta
from typing import List

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from hat.core.nus_box3d_utils import bbox3d_nus_transform
from hat.registry import OBJECT_REGISTRY
from .assign_result import AssignResult


@OBJECT_REGISTRY.register
class HungarianBBoxAssigner3D(metaclass=ABCMeta):
    """Compute one-to-one matching between predictions and ground truth. \
        This class computes an assignment between the targets and \
        the predictions based on the costs. The costs are weighted \
        sum of three components: classification cost, regression L1 \
        cost and regression iou cost. The targets don't include the \
        no_object, so generally there are more predictions than targets. \
        After the one-to-one matching, the un-matched are treated as \
        backgrounds. Thus each query prediction will be assigned \
        with `0` or a positive integer indicating the ground truth index: \
        - 0: negative sample, no assigned gt. \
        - positive integer: positive sample, index (1-based) of assigned gt.

    Args:
        cls_cost: Classification cost.
        reg_cost: Regression cost.
        pc_range: vcs range or point cloud range.
    """

    def __init__(
        self,
        cls_cost: nn.Module,
        reg_cost: nn.Module,
        pc_range: List[float] = None,
    ):
        self.cls_cost = cls_cost
        self.reg_cost = reg_cost
        self.pc_range = pc_range

    def assign(
        self,
        bbox_pred: torch.Tensor,
        cls_pred: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> AssignResult:
        """Compute one-to-one matching based on the weighted costs. \
            This method assign each query prediction to a ground truth or \
            background. The `assigned_gt_inds` with -1 means don't care, \
            0 means negative sample, and positive number is the \
            index (1-based) of assigned gt. \
            The assignment is done in the following steps, the order matters. \
            1. assign every prediction to -1 \
            2. compute the weighted costs \
            3. do Hungarian matching on CPU based on the costs \
            4. assign all to 0 (background) first, then for each matched pair \
            between predictions and gts, treat this prediction as foreground \
            and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred : Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred : Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes : Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels : Label of `gt_bboxes`, shape (num_gt,).
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
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
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels
            )

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost

        normalized_gt_bboxes = bbox3d_nus_transform(gt_bboxes)

        reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])

        # weighted sum of above two costs
        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu().numpy()
        cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)

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
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels
        )
