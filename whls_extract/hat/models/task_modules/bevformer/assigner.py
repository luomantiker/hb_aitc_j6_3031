import logging
from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn

from hat.models.task_modules.bevformer.utils import normalize_bbox
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class BevFormerHungarianAssigner3D(nn.Module):
    """The basic structure of BevFormerHungarianAssigner3D.

    Args:
        cls_cost: classification cost module.
        reg_cost: regression cost module.
    """

    @require_packages("scipy", raise_msg="Please `pip3 install scipy`")
    def __init__(
        self,
        cls_cost: nn.Module,
        reg_cost: nn.Module,
    ):
        super(BevFormerHungarianAssigner3D, self).__init__()
        self.cls_cost = cls_cost
        self.reg_cost = reg_cost

    def assign(
        self,
        bbox_pred: Tensor,
        cls_pred: Tensor,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
    ) -> Tuple[Tensor]:
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full(
            (num_bboxes,), -1, dtype=torch.long
        )
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return assigned_gt_inds

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost

        normalized_gt_bboxes = normalize_bbox(gt_bboxes)

        reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])

        # weighted sum of above two costs
        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()

        cost = np.where(np.isneginf(cost), -1e8, cost)
        cost = np.where(np.isinf(cost), 1e8, cost)
        cost = np.where(np.isnan(cost), 0.0, cost)

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

        return assigned_gt_inds
