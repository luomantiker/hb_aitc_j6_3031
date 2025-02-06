# Copyright (c) Horizon Robotics. All rights reserved.
# Source code reference to Facebook.

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.core.box_utils import box_center_to_corner, box_corner_to_center
from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import dist_initialized, get_dist_info
from .matcher import HungarianMatcher, generalized_box_iou

__all__ = [
    "DetrCriterion",
    "accuracy",
]


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Compute the precision@k for the specified values of k."""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@OBJECT_REGISTRY.register
class DetrCriterion(nn.Module):
    """This class computes the loss for DETR.

    Args:
        num_classes: number of object categories.
        dec_layers: number of the decoder layers.
        cost_class: weight of the classification error in the matching cost.
        cost_bbox: weight of the L1 error of the bbox in the matching cost.
        cost_giou: weight of the giou loss of the bbox in the matching cost.
        loss_class: weight of the classification loss.
        loss_bbox: weight of the L1 loss of the bbox.
        loss_giou: weight of the giou loss of the bbox.
        eos_coef: classification weight applied to the no-object category.
        losses: list of all the losses to be applied.
        aux_loss: True if auxiliary decoding losses are to be used.
    """

    def __init__(
        self,
        num_classes: int,
        dec_layers: int = 6,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        loss_ce: float = 1.0,
        loss_bbox: float = 5.0,
        loss_giou: float = 2.0,
        eos_coef: float = 0.1,
        losses: Sequence[str] = ("labels", "boxes", "cardinality"),
        aux_loss: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)
        self.weight_dict = {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
        if aux_loss:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in self.weight_dict.items()}
                )
            self.weight_dict.update(aux_weight_dict)
        self.eos_coef = eos_coef
        self.losses = losses
        self.aux_loss = aux_loss
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)."""
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [
                labels[J]
                for labels, (_, J) in zip(targets["gt_classes"], indices)
            ]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o.long()

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}

        if log:
            losses["class_error"] = (
                100 - accuracy(src_logits[idx], target_classes_o)[0]
            )
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute absolute error in the number of predicted non-empty boxes.

        This is not really a loss,
        it is intended for logging purposes only,
        It doesn't propagate gradients.
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(labels) for labels in targets["gt_classes"]], device=device
        )
        # Count the number of predictions that are NOT "no-object"
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(
            1
        )
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes.

        the L1 regression loss and the GIoU loss.
        Targets dicts must contain the key "gt_bboxes",
        which containing a tensor of dim [nb_target_boxes, 4].
        Target boxes are expected in format (center_x, center_y, w, h),
        which normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [boxes[i] for boxes, (_, i) in zip(targets["gt_bboxes"], indices)],
            dim=0,
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes  # normlize?

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_center_to_corner(src_boxes),
                box_center_to_corner(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def prepare_targets(self, targets):
        new_boxes = []
        boxes = targets["gt_bboxes"]
        if "before_pad_shape" in targets:
            shapes = targets["before_pad_shape"]
        else:
            shapes = targets["img_shape"]
        for shape, boxes_per_image in zip(shapes, boxes):
            h, w = shape[-2:]
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=boxes_per_image.device
            )
            gt_boxes = boxes_per_image / image_size_xyxy
            gt_boxes = box_corner_to_center(gt_boxes)
            new_boxes.append(gt_boxes)
        targets["gt_bboxes"] = new_boxes
        return targets

    def forward(self, outs, targets):
        targets = self.prepare_targets(targets)

        outputs_class, outputs_coord = outs
        outputs_coord = outputs_coord.sigmoid()
        outputs = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }
        if self.aux_loss:
            outputs["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord
            )

        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs"
        }

        # Retrieve the matching between the outputs
        # of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes,
        # for normalization purposes
        num_boxes = sum(len(labels) for labels in targets["gt_classes"])
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if dist_initialized():
            torch.distributed.all_reduce(num_boxes)
        _, world_size = get_dist_info()
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes)
            )

        # In case of auxiliary losses, repeat this process
        # with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        num_boxes,
                        **kwargs,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return losses
