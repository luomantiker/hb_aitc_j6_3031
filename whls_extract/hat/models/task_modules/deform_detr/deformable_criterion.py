import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import get_world_size

from hat.models.task_modules.detr import DetrCriterion
from hat.models.task_modules.motr.criterion import sigmoid_focal_loss
from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import dist_initialized

__all__ = ["SetCriterion", "DeformableCriterion"]


class SetCriterion(DetrCriterion):
    """This class computes the loss for Conditional DETR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the
            outputs of the model.
        2) we supervise each pair of matched ground-truth / prediction
            (supervise class and box).
    Args:
        num_classes : The number of object categories, excluding the special
            'no-object' category.
        matcher: A module capable of computing a match between targets and
            proposals.
        weight_dict: A dictionary containing the names of the losses and their
             corresponding weights.
        losses: A list of all the loss types to be applied.
                Default loss types are 'class' and 'boxes'.
        eos_coef: The coefficient for the end-of-sequence class in the
                classification loss.
        loss_class_type: The type of loss used for classification.
            Supported types are 'ce_loss' and 'focal_loss'.
        alpha: The alpha parameter in Focal Loss, if 'focal_loss' is used.
        gamma: The gamma parameter in Focal Loss, if 'focal_loss' is used.

    """

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict[str, float],
        losses: List[str] = None,
        eos_coef: float = 0.1,
        loss_class_type: str = "focal_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super(DetrCriterion, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma
        self.eos_coef = eos_coef
        self.loss_class_type = loss_class_type
        if self.losses is None:
            self.losses = ["class", "boxes"]
        assert loss_class_type in [
            "ce_loss",
            "focal_loss",
        ], "only support ce loss and focal loss for computing cls loss"

        if self.loss_class_type == "ce_loss":
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = eos_coef
            self.register_buffer("empty_weight", empty_weight)

    def loss_labels(
        self,
        outputs: dict,
        targets: dict,
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
    ):
        """Classification loss (Binary focal loss)."""
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["gt_classes"], indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        # Computation classification loss
        if self.loss_class_type == "ce_loss":
            loss_class = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
        elif self.loss_class_type == "focal_loss":
            # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
            # target_classes_one_hot = (2, 300, 80)
            target_classes_onehot = torch.zeros(
                [
                    src_logits.shape[0],
                    src_logits.shape[1],
                    src_logits.shape[2] + 1,
                ],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_class = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                * src_logits.shape[1]
            )

        losses = {"loss_class": loss_class}

        return losses

    def get_loss(
        self,
        loss: str,
        outputs: Dict,
        targets: Dict,
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
        **kwargs,
    ):
        loss_map = {
            "class": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs: Dict, targets: Dict):
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs"
        }

        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes,
        #  for normalization purposes
        num_boxes = sum(len(t) for t in targets["gt_classes"])
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if dist_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes)
            )

        # In case of auxiliary losses, we repeat this process with the output
        # of each intermediate layer.
        weight_dict_all = copy.deepcopy(self.weight_dict)
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)

                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                weight_dict_all.update(
                    {k + f"_{i}": v for k, v in self.weight_dict.items()}
                )
        for k in losses.keys():
            if k in weight_dict_all:
                losses[k] *= weight_dict_all[k]

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__()),
            "losses: {}".format(self.losses),
            "loss_class_type: {}".format(self.loss_class_type),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "focal loss alpha: {}".format(self.alpha),
            "focal loss gamma: {}".format(self.gamma),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


@OBJECT_REGISTRY.register
class DeformableCriterion(SetCriterion):
    """Computes loss for Deformable-DETR and two-stage Deformable-DETR models.

    Args:
        num_classes: Number of object categories,
                 excluding 'no-object' category.
        matcher: Module for matching targets with proposals.
        weight_dict: Loss component names and their weights.
        losses: Types of losses to apply, defaulting to 'class' and 'boxes'.
        eos_coef: End-of-sequence class coefficient in classification loss.
        loss_class_type: Classification loss type ('ce_loss' or 'focal_loss').
        alpha: Alpha parameter for Focal Loss.
        gamma: Gamma parameter for Focal Loss.
        aux_loss: Include auxiliary losses.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict[str, float],
        losses: List[str] = None,
        eos_coef: float = 0.1,
        loss_class_type: str = "focal_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
        aux_loss: bool = True,
    ):
        super(DeformableCriterion, self).__init__(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            eos_coef=eos_coef,
            loss_class_type=loss_class_type,
            alpha=alpha,
            gamma=gamma,
        )

    def forward(self, outputs: Dict, targets: Dict):
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        indices = self.matcher(outputs_without_aux, targets)

        # num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = sum(len(t) for t in targets["gt_classes"])
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if dist_initialized():
            torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        else:
            num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(
                self.get_loss(
                    loss, outputs, targets, indices, num_boxes, **kwargs
                )
            )

        weight_dict_all = copy.deepcopy(self.weight_dict)
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
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
                weight_dict_all.update(
                    {k + f"_{i}": v for k, v in self.weight_dict.items()}
                )

        # Compute losses for two-stage deformable-detr
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for i in range(len(bin_targets["gt_classes"])):
                bin_targets["gt_classes"][i] = torch.zeros_like(
                    bin_targets["gt_classes"][i]
                )
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                l_dict = self.get_loss(
                    loss,
                    enc_outputs,
                    bin_targets,
                    indices,
                    num_boxes,
                    **kwargs,
                )
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)
            weight_dict_all.update(
                {k + "_enc": v for k, v in self.weight_dict.items()}
            )

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= weight_dict_all[k]

        return losses
