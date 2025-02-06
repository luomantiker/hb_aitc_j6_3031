# Copyright (c) Horizon Robotics. All rights reserved.

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from hat.core.box_utils import box_center_to_corner
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    from torchvision.ops.boxes import box_area
except ImportError:
    box_area = None

__all__ = ["HungarianMatcher", "generalized_box_iou"]


@require_packages("torchvision")
def box_iou(boxes1, boxes2, eps):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter + eps

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2, eps=1e-6):
    """Generalized IoU from https://giou.stanford.edu/.

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2, eps)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1] + eps

    return iou - (area - union) / area


@OBJECT_REGISTRY.register
class HungarianMatcher(nn.Module):
    """Compute an assignment between targets and predictions.

    For efficiency reasons, the targets don't include the no_object.
    Because of this, in general, there are more predictions than targets.
    In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    Args:
        cost_class: weight of the classification error.
        cost_bbox: weight of the L1 error of the bbox coordinates.
        cost_giou: weight of the giou loss of the bounding box.
        use_focal: whether to use focal loss.
        alpha: A weighting factor for pos-sample, (1-alpha) is for
            neg-sample.
        gamma: Gamma used in focal loss to compress the contribution
            of easy examples.
    Returns:
        A list, containing tuples of (index_i, index_j) where:
            index_i is the indices of the selected predictions (in order)
            index_j is the indices of the selected targets (in order)
        For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        use_focal: bool = False,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.alpha = alpha
        self.gamma = gamma
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, data):
        """Perform the matching.

        Args:
            outputs: a dict containing at least these entries:
                "pred_logits": Tensor of dim [bs, num_queries, num_classes]
                "pred_boxes": Tensor of dim [bs, num_queries, 4]

            data: a dict containing at least these entries:
                "gt_classes": Tensor of dim [num_target_boxes]
                "boxes": Tensor of dim [num_target_boxes, 4]
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        else:
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(
            0, 1
        )  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat(data["gt_classes"]).long()
        tgt_bbox = torch.cat(data["gt_bboxes"]).float()

        # Compute the classification cost.
        # Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching,
        # so it can be ommitted.
        if self.use_focal:
            neg_cost_class = (
                (1 - self.alpha)
                * (out_prob ** self.gamma)
                * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha
                * ((1 - out_prob) ** self.gamma)
                * (-(out_prob + 1e-8).log())
            )
            cost_class = (
                pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            )
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_center_to_corner(out_bbox).float(),
            box_center_to_corner(tgt_bbox),
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(boxes) for boxes in data["gt_bboxes"]]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
