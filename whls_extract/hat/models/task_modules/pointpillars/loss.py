from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import limit_period
from hat.utils.model_helpers import fx_wrap

__all__ = ["PointPillarsLoss"]


@OBJECT_REGISTRY.register
class PointPillarsLoss(nn.Module):
    """PointPillars Loss Module.

    Args:
        num_classes: Number of classes
        loss_cls: Classification loss module.
        loss_bbox: Bbox regression loss module.
        loss_dir: Direction loss module.
        pos_cls_weight: Positive weight. Defaults to 1.0.
        neg_cls_weight: Negative weight. Defaults to 1.0.
        num_direction_bins: Number of direction. Defaults to 2.
        direction_offset: The offset of BEV rotation angles. Defaults to 0.0.
    """

    def __init__(
        self,
        num_classes: int,
        loss_cls: Optional[nn.Module] = None,
        loss_bbox: Optional[nn.Module] = None,
        loss_dir: Optional[nn.Module] = None,
        pos_cls_weight: float = 1.0,
        neg_cls_weight: float = 1.0,
        num_direction_bins: int = 2,
        direction_offset: float = 0.0,
    ):

        super(PointPillarsLoss, self).__init__()

        self.loss_cls = loss_cls
        self.loss_reg = loss_bbox
        self.loss_dir = loss_dir
        self.pos_cls_weight = pos_cls_weight
        self.neg_cls_weight = neg_cls_weight
        self.num_direction_bins = num_direction_bins
        self.direction_offset = direction_offset
        self.num_class = num_classes

    @fx_wrap()
    def forward(
        self,
        anchors: torch.Tensor,
        box_cls_labels: torch.Tensor,
        reg_targets: torch.Tensor,
        box_preds: torch.Tensor,
        cls_preds: torch.Tensor,
        dir_preds: torch.Tensor,
    ):
        """Forward pass, calculate losses.

        Args:
            anchors: Anchors.
            box_cls_labels: Bbox classification label.
            reg_targets: 3D bbox targets.
            box_preds: 3D bbox predictions.
            cls_preds: Classification predictions.
            dir_preds: Direction classification predictions.

        Returns:
            cls_loss: Classification losses.
            loc_loss: Box regression losses.
            dir_loss: Direction classification losses.
        """

        batch_size = int(box_preds.shape[0])
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(
            batch_size, 1, 1
        )

        cls_loss_reduced = self.get_cls_loss(cls_preds, box_cls_labels)
        (loc_loss_reduced, dir_loss_reduced,) = self.get_box_reg_loss(
            batch_anchors=batch_anchors,
            box_cls_labels=box_cls_labels,
            reg_targets=reg_targets,
            box_preds=box_preds,
            dir_cls_preds=dir_preds,
        )

        return {
            "dir_loss": dir_loss_reduced.mean(),
            "cls_loss": cls_loss_reduced.mean(),
            "loc_loss": loc_loss_reduced.mean(),
        }

    def get_cls_loss(
        self,
        cls_preds: torch.Tensor,
        box_cls_labels: torch.Tensor,
    ):
        """Calculate classification loss.

        Args:
            cls_preds: Prediction class.
            box_cls_labels: Bbox classification label.

        Returns:
            cls_loss_reduced: Reduced classification loss.
        """

        batch_size = int(cls_preds.shape[0])
        cls_weights, _, cared = self.prepare_loss_weights(
            labels=box_cls_labels
        )
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        one_hot_targets = self.one_hot_f(
            cls_targets, depth=self.num_class + 1, dtype=cls_preds.dtype
        )
        one_hot_targets = one_hot_targets[..., 1:]

        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        cls_weights = cls_weights.view(batch_size, -1, self.num_class)

        cls_loss = self.loss_cls(
            cls_preds,
            one_hot_targets,
            weight=cls_weights,
        )
        cls_loss_reduced = cls_loss.sum() / batch_size
        return cls_loss_reduced

    def get_box_reg_loss(
        self,
        batch_anchors: torch.Tensor,
        box_cls_labels: torch.Tensor,
        reg_targets: torch.Tensor,
        box_preds: torch.Tensor,
        dir_cls_preds: Optional[torch.Tensor] = None,
    ):
        """Calculate bbox regression and direction classification losses.

        Args:
            batch_anchors: Anchors.
            box_cls_labels: Bbox classification label.
            reg_targets: 3D bbox targets.
            box_preds: 3D bbox predictions.
            dir_cls_preds: Direction classification predictions.

        Returns:
            loc_loss_reduced: Reduced bbox regression loss.
            dir_loss_reduced: Reduced direction classification loss.
        """
        batch_size = int(box_preds.shape[0])

        _, reg_weights, _ = self.prepare_loss_weights(labels=box_cls_labels)
        box_preds = box_preds.view(batch_size, -1, batch_anchors.shape[-1])
        box_preds_sin, reg_targets_sin = self.add_sin_difference(
            box_preds, reg_targets
        )
        loc_loss = self.loss_reg(
            box_preds_sin, reg_targets_sin, weight=reg_weights.unsqueeze(-1)
        )
        loc_loss_reduced = loc_loss.sum() / batch_size

        # direction
        if dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                batch_anchors,
                reg_targets,
                dir_offset=self.direction_offset,
            )
            dir_logits = dir_cls_preds.view(
                batch_size, -1, self.num_direction_bins
            )
            weights = (box_cls_labels > 0).type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.loss_dir(
                dir_logits.reshape(-1, self.num_direction_bins),
                dir_targets.reshape(-1, self.num_direction_bins).max(dim=-1)[
                    1
                ],
                # weight=None,
            )
            # dir_loss * weights
            dir_loss = dir_loss.reshape(weights.shape) * weights

            dir_loss_reduced = dir_loss.sum() / batch_size
        else:
            dir_loss_reduced = torch.tensor(
                0, dtype=loc_loss_reduced.dtype, device=loc_loss_reduced.device
            )

        return loc_loss_reduced, dir_loss_reduced

    def one_hot_f(
        self,
        tensor,
        depth,
        dim: int = -1,
        on_value: float = 1.0,
        dtype=torch.float32,
    ):
        """Encode to one-hot.

        Args:
            tensor: Input tensor to be one-hot encoded.
            depth: Number of classes for one-hot encoding.
            dim: Dimension along which to perform one-hot encoding.
            on_value: Value to fill in the "on" positions.
            dtype: Data type of the resulting tensor.

        Returns:
            tensor_onehot:  one-hot encoded tensor.
        """
        tensor_onehot = torch.zeros(
            *list(tensor.shape), depth, dtype=dtype, device=tensor.device
        )
        tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
        return tensor_onehot

    def add_sin_difference(self, boxes1: torch.Tensor, boxes2: torch.Tensor):
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1: Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2: Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            boxes1: Rotation bbox by sin*cos.
            boxes2: Rotation bbox by cos*sin.
        """
        rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
            boxes2[..., -1:]
        )
        rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(
            boxes2[..., -1:]
        )
        boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
        boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
        return boxes1, boxes2

    def get_pos_neg_loss(self, cls_loss: torch.Tensor, labels: torch.Tensor):
        """Calculate positive and negative object losses.

        Args:
            cls_loss: Classification loss.
            labels: Classification labels.

        Returns:
            cls_pos_loss: Positive classification losses.
            cls_neg_loss: Negative classification losses.
        """
        # cls_loss: [N, num_anchors, num_class]
        # labels: [N, num_anchors]
        batch_size = cls_loss.shape[0]
        if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
            cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
                batch_size, -1
            )
            cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
                batch_size, -1
            )
            cls_pos_loss = cls_pos_loss.sum() / batch_size
            cls_neg_loss = cls_neg_loss.sum() / batch_size
        else:
            cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
            cls_neg_loss = cls_loss[..., 0].sum() / batch_size
        return cls_pos_loss, cls_neg_loss

    def get_direction_target(
        self,
        anchors: torch.Tensor,
        reg_targets: torch.Tensor,
        one_hot: bool = True,
        dir_offset: float = 0.0,
    ):
        """Encode direction to 0 ~ num_bins-1.

        Args:
            anchors: Anchors.
            reg_targets: Bbox regression targets.
            one_hot: Whether to encode as one hot. Default to True.
            dir_offset: Direction offset. Default to 0.

        Returns:
            dir_cls_targets: Encoded direction targets.
        """
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., -1] + anchors[..., -1]
        # dir_cls_targets = ((rot_gt - dir_offset) > 0).long()
        dir_cls_targets = (
            limit_period(rot_gt - dir_offset, 0.5, np.pi * 2) > 0
        ).long()
        if one_hot:
            dir_cls_targets = self.one_hot_f(
                dir_cls_targets, self.num_direction_bins, dtype=anchors.dtype
            )
        return dir_cls_targets

    def prepare_loss_weights(
        self,
        labels: torch.Tensor,
        dtype=torch.float32,
    ):
        """Calculate classification and regression weights.

        Args:
            labels: Classification labels.
            dtype: Data type of the resulting tensor.

        Returns:
            cls_weights: Classification weights.
            reg_weights: Regression weights.
            cared: cared mask.
        """
        pos_cls_weight = self.pos_cls_weight
        neg_cls_weight = self.neg_cls_weight

        cared = labels >= 0
        # cared: [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.type(dtype) * neg_cls_weight
        cls_weights = negative_cls_weights + pos_cls_weight * positives.type(
            dtype
        )
        reg_weights = positives.type(dtype)
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        return cls_weights, reg_weights, cared
