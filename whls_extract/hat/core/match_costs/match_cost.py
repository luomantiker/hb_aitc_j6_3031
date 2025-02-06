import torch

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class BBox3DL1Cost(object):
    """BBox3DL1Cost.

    Args:
        weight: loss_weight
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def __call__(
        self, bbox_pred: torch.Tensor, gt_bboxes: torch.Tensor
    ) -> torch.Tensor:
        """Calculate 3d box l1 cost.

        Args:
            bbox_pred: Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes: Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@OBJECT_REGISTRY.register
class FocalLossCost:
    """FocalLossCost.

    Args:
        weight: loss_weight
        alpha: focal_loss alpha
        gamma: focal_loss gamma
        eps: default 1e-12
        binary_input: Whether the input is binary,
           default False.
    """

    def __init__(
        self,
        weight: float = 1.0,
        alpha: float = 0.25,
        gamma: int = 2,
        eps: float = 1e-12,
        binary_input: bool = False,
    ):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def _focal_loss_cost(
        self, cls_pred: torch.Tensor, gt_labels: torch.Tensor
    ) -> torch.Tensor:
        """Calculate focal cls cost.

        Args:
            cls_pred: Predicted classification logits, shape
                (num_query, num_class).
            gt_labels: Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = (
            -(1 - cls_pred + self.eps).log()
            * (1 - self.alpha)
            * cls_pred.pow(self.gamma)
        )
        pos_cost = (
            -(cls_pred + self.eps).log()
            * self.alpha
            * (1 - cls_pred).pow(self.gamma)
        )

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight

    def _mask_focal_loss_cost(
        self, cls_pred: torch.Tensor, gt_labels: torch.Tensor
    ) -> torch.Tensor:
        """Calculate focal cost matrix.

        Args:
            cls_pred: Predicted classfication logits
                in shape (num_query, d1, ..., dn), dtype=torch.float32.
            gt_labels: Ground truth in shape (num_gt, d1, ..., dn),
                dtype=torch.long. Labels should be binary.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_query, num_gt).
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        cls_pred = cls_pred.sigmoid()
        neg_cost = (
            -(1 - cls_pred + self.eps).log()
            * (1 - self.alpha)
            * cls_pred.pow(self.gamma)
        )
        pos_cost = (
            -(cls_pred + self.eps).log()
            * self.alpha
            * (1 - cls_pred).pow(self.gamma)
        )

        cls_cost = torch.einsum(
            "nc,mc->nm", pos_cost, gt_labels
        ) + torch.einsum("nc,mc->nm", neg_cost, (1 - gt_labels))
        return cls_cost / n * self.weight

    def __call__(
        self, cls_pred: torch.Tensor, gt_labels: torch.Tensor
    ) -> torch.Tensor:
        """Calculate focal cls cost.

        Args:
            cls_pred: Predicted classfication logits.
            gt_labels: Labels.

        Returns:
            Focal cost matrix with weight in shape (num_query, num_gt).
        """
        if self.binary_input:
            return self._mask_focal_loss_cost(cls_pred, gt_labels)
        else:
            return self._focal_loss_cost(cls_pred, gt_labels)
