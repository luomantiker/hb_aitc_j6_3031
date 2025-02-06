from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

from hat.core.box_utils import bbox_overlaps, box_center_to_corner
from hat.models.task_modules.detr.matcher import (
    HungarianMatcher,
    generalized_box_iou,
)
from hat.models.task_modules.motr.motr_utils import select_instances
from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import dist_initialized, get_dist_info


def sigmoid_focal_loss(
    inputs,
    targets,
    num_boxes,
    alpha: float = 0.25,
    gamma: float = 2,
    mean_in_dim1: bool = True,
):
    """Sigmoid focal Loss used in motr.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs.
            Stores the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_boxes: Num of boxes.
        alpha: Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        mean_in_dim1: Whether to mean in dim.
    """

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if mean_in_dim1:
        return loss.mean(1).sum() / num_boxes
    else:
        return loss.sum() / num_boxes


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
class MotrCriterion(nn.Module):
    """This class computes the loss for Motr.

    Args:
        num_classes: number of object categories.
        num_dec_layers: number of the decoder layers.
        cost_class: weight of the classification error in the matching cost.
        cost_bbox: weight of the L1 error of the bbox in the matching cost.
        cost_giou: weight of the giou loss of the bbox in the matching cost.
        cls_loss_coef: weight of the classification loss.
        bbox_loss_coef: weight of the L1 loss of the bbox.
        giou_loss_coef: weight of the giou loss of the bbox.
        aux_loss: True if auxiliary decoding losses are to be used.
        max_frames_per_seq: The max num frame of seq data.
    """

    def __init__(
        self,
        num_classes,
        num_dec_layers: int = 6,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cls_loss_coef: float = 2,
        bbox_loss_coef: float = 5,
        giou_loss_coef: float = 2,
        aux_loss: bool = True,
        max_frames_per_seq: int = 5,
    ):
        super(MotrCriterion, self).__init__()
        self.num_classes = num_classes

        self.losses = ["labels", "boxes"]
        self.losses_dict = {}
        self._current_frame_idx = 0
        self.aux_loss = aux_loss
        self.matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou, True)

        weight_dict = {}
        for i in range(max_frames_per_seq):
            weight_dict.update(
                {
                    "frame_{}_loss_ce".format(i): cls_loss_coef,
                    "frame_{}_loss_bbox".format(i): bbox_loss_coef,
                    "frame_{}_loss_giou".format(i): giou_loss_coef,
                }
            )

        if aux_loss:
            for i in range(max_frames_per_seq):
                for j in range(num_dec_layers - 1):
                    weight_dict.update(
                        {
                            "frame_{}_aux{}_loss_ce".format(
                                i, j
                            ): cls_loss_coef,
                            "frame_{}_aux{}_loss_bbox".format(
                                i, j
                            ): bbox_loss_coef,
                            "frame_{}_aux{}_loss_giou".format(
                                i, j
                            ): giou_loss_coef,
                        }
                    )

        self.weight_dict = weight_dict

    def initialize_for_single_clip(self):
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(
            num_samples, dtype=torch.float, device=self.sample_device
        )

        if dist_initialized():
            torch.distributed.all_reduce(num_boxes)
        _, world_size = get_dist_info()
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()
        return num_boxes

    def get_loss(
        self, loss, outputs, gt_instances, indices, num_boxes, **kwargs
    ):
        loss_map = {
            "labels": self.loss_labels,
            # 'cardinality': self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](
            outputs, gt_instances, indices, num_boxes, **kwargs
        )

    def loss_boxes(
        self,
        outputs,
        gt_instances: List[Dict],
        indices: List[tuple],
        num_boxes,
    ):
        """Compute the losses related to the bounding boxes.

        the L1 regression loss and the GIoU loss.
        Targets dicts must contain the key "gt_bboxes",
        which containing a tensor of dim [nb_target_boxes, 4].
        Target boxes are expected in format (center_x, center_y, w, h),
        which normalized by the image size.
        """
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [
                gt_per_img.gt_bboxes[i]
                for gt_per_img, (_, i) in zip(gt_instances, indices)
            ],
            dim=0,
        )

        # for pad target, don't calculate regression loss,
        # judged by whether obj_id=-1
        target_obj_ids = torch.cat(
            [
                gt_per_img.gt_ids[i]
                for gt_per_img, (_, i) in zip(gt_instances, indices)
            ],
            dim=0,
        )  # size(16)
        mask = target_obj_ids != -1

        loss_bbox = F.l1_loss(
            src_boxes[mask], target_boxes[mask], reduction="none"
        )
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_center_to_corner(src_boxes[mask]),
                box_center_to_corner(target_boxes[mask]),
            )
        )

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(
        self, outputs, gt_instances: List[Dict], indices, num_boxes, log=False
    ):
        """Classification loss (NLL)."""
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J)
            # set labels of track-appear slots to 0.
            if len(gt_per_img.gt_classes) > 0:
                labels_per_img[J != -1] = gt_per_img.gt_classes[J[J != -1]].to(
                    torch.int64
                )
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o
        gt_labels_target = F.one_hot(
            target_classes, num_classes=self.num_classes + 1
        )[
            :, :, :-1
        ]  # no loss for the last (background) class
        gt_labels_target = gt_labels_target.to(src_logits)
        loss_ce = sigmoid_focal_loss(
            src_logits.flatten(1),
            gt_labels_target.flatten(1),
            alpha=0.25,
            gamma=2,
            num_boxes=num_boxes,
            mean_in_dim1=False,
        )
        loss_ce = loss_ce.sum()
        losses = {"loss_ce": loss_ce}

        if log:
            losses["class_error"] = (
                100 - accuracy(src_logits[idx], target_classes_o)[0]
            )

        return losses

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def match_for_single_frame(
        self,
        combine_instance: dict,
        outputs_classes,
        outputs_coords,
        out_hs,
        gt_instances_i: dict,
    ):

        combine_instance.pred_logits = outputs_classes[-1][0]
        combine_instance.pred_boxes = outputs_coords[-1][0].float().sigmoid()
        combine_instance.pred_boxes_unsigmoid = outputs_coords[-1][0]
        combine_instance.output_embedding = out_hs[0]
        with torch.no_grad():
            combine_instance.scores = (
                outputs_classes[0][0, :].sigmoid().max(dim=-1).values
            )
        keep_idxs = combine_instance.mask_query == 1

        track_instances = select_instances(combine_instance, keep_idxs)

        pred_logits_i = (
            track_instances.pred_logits
        )  # predicted logits of i-th image.
        pred_boxes_i = (
            track_instances.pred_boxes
        )  # predicted boxes of i-th image.

        obj_idxes = gt_instances_i.gt_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {
            obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)
        }
        outputs_i = {
            "pred_logits": pred_logits_i.unsqueeze(0),
            "pred_boxes": pred_boxes_i.unsqueeze(0),
        }

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        for j in range(len(track_instances.query_pos)):
            obj_id = track_instances.obj_idxes[j].item()
            # set new target idx.
            if obj_id >= 0:
                if obj_id in obj_idx_to_gt_idx:
                    track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[
                        obj_id
                    ]
                    # track_instances.matched_gt_idxes[j] = -1
                else:
                    num_disappear_track += 1
                    track_instances.matched_gt_idxes[
                        j
                    ] = -1  # track-disappear case.
            else:
                track_instances.matched_gt_idxes[j] = -1

        full_track_idxes = torch.arange(
            len(track_instances.query_pos), dtype=torch.long
        ).to(pred_logits_i.device)
        matched_track_idxes = track_instances.obj_idxes >= 0  # occu
        prev_matched_indices = torch.stack(
            [
                full_track_idxes[matched_track_idxes],
                track_instances.matched_gt_idxes[matched_track_idxes],
            ],
            dim=1,
        ).to(pred_logits_i.device)

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes
        # are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[
            track_instances.obj_idxes == -1
        ]

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances_i.gt_bboxes)).to(
            pred_logits_i.device
        )
        tgt_state[tgt_indexes] = 1
        untracked_tgt_indexes = torch.arange(len(gt_instances_i.gt_bboxes)).to(
            pred_logits_i.device
        )[tgt_state == 0]

        untracked_gt_instances = EasyDict()
        untracked_gt_instances.gt_bboxes = [
            gt_instances_i.gt_bboxes[untracked_tgt_indexes]
        ]
        untracked_gt_instances.gt_classes = [
            gt_instances_i.gt_classes[untracked_tgt_indexes]
        ]
        untracked_gt_instances.gt_ids = [
            gt_instances_i.gt_ids[untracked_tgt_indexes]
        ]

        def match_for_single_decoder_layer(unmatched_outputs, matcher):
            new_track_indices = matcher(
                unmatched_outputs, untracked_gt_instances
            )  # list[tuple(src_idx, tgt_idx)]
            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt.
            new_matched_indices = torch.stack(
                [
                    unmatched_track_idxes[src_idx],
                    untracked_tgt_indexes[tgt_idx],
                ],
                dim=1,
            ).to(pred_logits_i.device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            "pred_logits": track_instances.pred_logits[
                unmatched_track_idxes
            ].unsqueeze(0),
            "pred_boxes": track_instances.pred_boxes[
                unmatched_track_idxes
            ].unsqueeze(0),
        }
        new_matched_indices = match_for_single_decoder_layer(
            unmatched_outputs, self.matcher
        )

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[
            new_matched_indices[:, 0]
        ] = gt_instances_i.gt_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[
            new_matched_indices[:, 0]
        ] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (
            track_instances.matched_gt_idxes >= 0
        )
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.gt_bboxes[
                track_instances.matched_gt_idxes[active_idxes]
            ]
            active_track_boxes = box_center_to_corner(active_track_boxes)
            gt_boxes = box_center_to_corner(gt_boxes)
            track_instances.iou[active_idxes] = bbox_overlaps(
                active_track_boxes, gt_boxes, is_aligned=True
            )

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat(
            [new_matched_indices, prev_matched_indices], dim=0
        )

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i.gt_bboxes) + num_disappear_track
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            new_track_loss = self.get_loss(
                loss,
                outputs=outputs_i,
                gt_instances=[gt_instances_i],
                indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                num_boxes=1,
            )
            self.losses_dict.update(
                {
                    "frame_{}_{}".format(self._current_frame_idx, key): value
                    for key, value in new_track_loss.items()
                }
            )

        if self.aux_loss:
            aux_outputs = self._set_aux_loss(outputs_classes, outputs_coords)
            for i, aux_output in enumerate(aux_outputs):
                aux_output["pred_boxes"] = (
                    aux_output["pred_boxes"].float().sigmoid()
                )
                unmatched_outputs_layer = {
                    "pred_logits": aux_output["pred_logits"][
                        0, unmatched_track_idxes
                    ].unsqueeze(0),
                    "pred_boxes": aux_output["pred_boxes"][
                        0, unmatched_track_idxes
                    ].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(
                    unmatched_outputs_layer, self.matcher
                )
                matched_indices_layer = torch.cat(
                    [new_matched_indices_layer, prev_matched_indices], dim=0
                )
                for loss in self.losses:
                    if loss == "masks":
                        continue
                    l_dict = self.get_loss(
                        loss,
                        aux_output,
                        gt_instances=[gt_instances_i],
                        indices=[
                            (
                                matched_indices_layer[:, 0],
                                matched_indices_layer[:, 1],
                            )
                        ],
                        num_boxes=1,
                    )
                    self.losses_dict.update(
                        {
                            "frame_{}_aux{}_{}".format(
                                self._current_frame_idx, i, key
                            ): value
                            for key, value in l_dict.items()
                        }
                    )
        self._step()
        return track_instances

    def forward(self):
        losses = {}
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in self.losses_dict.items():
            losses[loss_name] = loss / num_samples
        return losses
