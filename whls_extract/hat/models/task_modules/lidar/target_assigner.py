from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from hat.core.box_utils import bbox_overlaps, center_to_minmax_2d
from hat.models.task_modules.lidar.box_coders import GroundBox3dCoder
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import limit_period
from hat.utils.model_helpers import fx_wrap

__all__ = ["LidarTargetAssigner"]


@OBJECT_REGISTRY.register
class LidarTargetAssigner(nn.Module):
    """TargetAssigner for Lidar.

    Args:
        box_coder: BoxCoder.
        class_names: Class names.
        positive_fraction: Positive fraction.
        sample_size: Sample size.
    """

    def __init__(
        self,
        box_coder: GroundBox3dCoder,
        class_names: List[str],
        positive_fraction: int = None,
        sample_size: int = 512,
    ):

        super(LidarTargetAssigner, self).__init__()
        self._box_coder = box_coder
        self._sample_size = sample_size
        self._class_names = class_names
        if positive_fraction is None or positive_fraction < 0:
            self._positive_fraction = None
        else:
            self._positive_fraction = positive_fraction

    @property
    def box_coder(self):
        """3D boxCoder."""
        return self._box_coder

    @fx_wrap(skip_compile=True)
    def forward(
        self,
        anchors_list: List[torch.Tensor],
        matched_thresholds: List[float],
        unmatched_thresholds: List[float],
        annos: Dict,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """Forward pass, generate targets.

        Args:
            anchors_list: List of anchors.
            match_thresholds: Match thresholds of IoU.
            unmatch_thresholds: Unmatch thresholds of IoU.
            annos: Annotations of ground truth.
            device: The device on which the target will be generated.
        Returns:
            bbox_targets: BBox targets.
            cls_labels: Classification label for bbox.
            reg_weights: Regression weights for each bbox.
        """
        return self.assign_targets(
            anchors_list,
            matched_thresholds,
            unmatched_thresholds,
            annos,
            device,
        )

    def assign_targets(
        self,
        anchors_list: List[torch.Tensor],
        matched_thresholds: List[float],
        unmatched_thresholds: List[float],
        annos: Dict,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """Generate targets.

        Args:
            anchors_list: List of anchors.
            match_thresholds: Match thresholds of IoU.
            unmatch_thresholds: Unmatch thresholds of IoU.
            annos: Annotations of ground truth.
            device: The device on which the target will be generated.
        Returns:
            bbox_targets: BBox targets.
            cls_labels: Classification label for bbox.
            reg_weights: Regression weights for each bbox.
        """
        batch_size = len(annos)

        bbox_targets = []
        cls_labels = []
        reg_weights = []

        for idx in range(batch_size):
            gt_boxes = torch.tensor(
                annos[idx]["gt_boxes"], dtype=torch.float32, device=device
            )
            gt_classes = torch.tensor(annos[idx]["gt_classes"], device=device)
            gt_names = annos[idx]["gt_names"]
            targets_dict = self.assign_per_class(
                classes_names=self._class_names,
                anchors_list=anchors_list,
                matched_thresholds=matched_thresholds,
                unmatched_thresholds=unmatched_thresholds,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                gt_names=gt_names,
            )
            bbox_targets.append(targets_dict["bbox_targets"])
            cls_labels.append(targets_dict["labels"])
            reg_weights.append(targets_dict["reg_weights"])

        bbox_targets = torch.stack(bbox_targets, dim=0)
        cls_labels = torch.stack(cls_labels, dim=0)
        reg_weights = torch.stack(reg_weights, dim=0)

        return bbox_targets, cls_labels, reg_weights

    def assign_per_class(
        self,
        classes_names,
        anchors_list,
        matched_thresholds,
        unmatched_thresholds,
        gt_boxes,
        gt_classes,
        gt_names,
    ):
        """Assign targets for each class.

        Args:
            classes_names: Class names.
            anchors_list: List of anchors.
            match_thresholds: Match thresholds of IoU.
            unmatch_thresholds: Unmatch thresholds of IoU.
            gt_boxes: Ground truth boxes.
            gt_classes: Ground truth classes.
            gt_names: Names of Ground truth.
        Returns:
            labels: Bbox classification label.
            bbox_targets: Bbox.
            reg_weights: Regression weights for each bbox.
        """

        def similarity_fn(anchors, gt_boxes):
            anchors_rbv = anchors[:, [0, 1, 3, 4, -1]]
            gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, -1]]
            return self.nearest_iou_similarity(anchors_rbv, gt_boxes_rbv)

        def box_encoding_fn(boxes, anchors):
            return self._box_coder.encode(boxes, anchors)

        targets_list = []
        for class_name, anchors, match_threshold, unmatch_threshold in zip(
            classes_names,
            anchors_list,
            matched_thresholds,
            unmatched_thresholds,
        ):
            mask = torch.tensor(
                [c == class_name for c in gt_names], dtype=torch.bool
            )
            feature_map_size = anchors.shape[:3]
            # num_loc = anchors.shape[-2]

            targets = self.create_targets_single(
                anchors.reshape(-1, self.box_coder.code_size),
                torch.nan_to_num(gt_boxes[mask]),
                similarity_fn,
                box_encoding_fn,
                gt_classes=gt_classes[mask],
                matched_threshold=match_threshold,
                unmatched_threshold=unmatch_threshold,
                positive_fraction=self._positive_fraction,
                sample_size=self._sample_size,
                norm_by_num_examples=False,
                box_code_size=self.box_coder.code_size,
            )
            targets_list.append(targets)

        targets_dict = {
            "labels": [t["box_cls_labels"] for t in targets_list],
            "bbox_targets": [t["bbox_reg_targets"] for t in targets_list],
            "reg_weights": [t["reg_weights"] for t in targets_list],
        }
        targets_dict["bbox_targets"] = torch.cat(
            [
                v.reshape(*feature_map_size, -1, self.box_coder.code_size)
                for v in targets_dict["bbox_targets"]
            ],
            dim=-2,
        ).view(-1, self.box_coder.code_size)
        targets_dict["labels"] = torch.cat(
            [v.reshape(*feature_map_size, -1) for v in targets_dict["labels"]],
            dim=-1,
        ).view(-1)
        targets_dict["reg_weights"] = torch.cat(
            [
                v.reshape(*feature_map_size, -1)
                for v in targets_dict["reg_weights"]
            ],
            dim=-1,
        ).view(-1)

        return targets_dict

    @property
    def num_anchors_per_location(self):
        """Get number of anchors per location."""
        num = 0
        for a_generator in self._anchor_generators:
            num += a_generator.num_anchors_per_localization
        return num

    @property
    def box_ndim(self):
        """Dimension of box."""
        return self.box_coder.code_size

    def create_targets_single(
        self,
        all_anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
        similarity_fn: Callable,
        box_encoding_fn: Callable,
        gt_classes: Optional[torch.Tensor] = None,
        matched_threshold: float = 0.6,
        unmatched_threshold: float = 0.45,
        positive_fraction: Optional[float] = None,
        sample_size: int = 300,
        norm_by_num_examples: bool = False,
        box_code_size: int = 7,
    ):
        """Create targets.

        Args:
            all_anchors: [num_of_anchors, box_ndim] float tensor.
            gt_boxes: [num_gt_boxes, box_ndim] float tensor.
            similarity_fn: a function, accept anchors and gt_boxes, return
                similarity matrix(such as IoU).
            box_encoding_fn: a function, accept gt_boxes and anchors, return
                box encodings(offsets).
            prune_anchor_fn: a function, accept anchors, return indices that
                indicate valid anchors.
            gt_classes: [num_gt_boxes] int tensor. indicate gt classes, must
                start with 1.
            matched_threshold: float, iou greater than matched_threshold will
                be treated as positives.
            unmatched_threshold: float, iou smaller than unmatched_threshold
                will be treated as negatives.
            positive_fraction: [0-1] float or None. if not None, we will try to
                keep ratio of pos/neg equal to positive_fraction when sample.
                if there is not enough positives, it fills the rest with
                negatives.
            rpn_batch_size: int. sample size.
            norm_by_num_examples: bool. norm box_weight by number of examples.
        Returns:
            box_cls_labels: Bbox classification label.
            bbox_reg_targets: Bbox.
            reg_weights: Regression weights for each bbox.
        """

        total_anchors = all_anchors.shape[0]
        anchors = all_anchors
        num_inside = total_anchors
        # box_ndim = all_anchors.shape[1]
        if gt_classes is None:
            gt_classes = torch.ones(
                [gt_boxes.shape[0]], dtype=torch.int32, device=anchors.device
            )
        # Compute anchor labels:
        # label=1 is positive, 0 is negative, -1 is don't care (ignore)
        labels = torch.ones(
            (num_inside,), dtype=torch.int32, device=anchors.device
        ) * (-1)
        gt_ids = torch.ones(
            (num_inside,), dtype=torch.int32, device=anchors.device
        ) * (-1)

        if len(gt_boxes) > 0:
            # Compute overlaps between the anchors and the gt boxes overlaps
            anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)
            # Map from anchor to gt box that has highest overlap
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)
            # For each anchor, amount of overlap with most overlapping gt box
            anchor_to_gt_max = anchor_by_gt_overlap[
                torch.arange(num_inside, device=anchors.device),
                anchor_to_gt_argmax,
            ]  #
            # Map from gt box to an anchor that has highest overlap
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(dim=0)
            # For each gt box, amount of overlap with most overlapping anchor
            gt_to_anchor_max = anchor_by_gt_overlap[
                gt_to_anchor_argmax,
                torch.arange(
                    anchor_by_gt_overlap.shape[1], device=anchors.device
                ),
            ]
            # must remove gt which doesn't match any anchor.
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1
            # Find all anchors that share the max overlap amount
            # (this includes many ties)
            anchors_with_max_overlap = (
                anchor_by_gt_overlap == gt_to_anchor_max
            ).nonzero()[:, 0]
            # Fg label: for each gt use anchors with highest overlap
            # (including ties)
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()
            # Fg label: above threshold IOU
            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds]
            gt_ids[pos_inds] = gt_inds.int()
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
        else:
            # labels[:] = 0
            bg_inds = torch.arange(num_inside, device=anchors.device)
        fg_inds = (labels > 0).nonzero()[:, 0]
        fg_max_overlap = None  # noqa F841
        if len(gt_boxes) > 0:
            fg_max_overlap = anchor_to_gt_max[fg_inds]  # noqa F841
        gt_pos_ids = gt_ids[fg_inds]  # noqa F841

        # subsample positive labels if we have too many
        if positive_fraction is not None:
            num_fg = int(positive_fraction * sample_size)
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1
                fg_inds = (labels > 0).nonzero()[:, 0]

            num_bg = sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[
                    torch.randint(0, len(bg_inds), size=(num_bg,))
                ]
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:
            if len(gt_boxes) == 0:
                labels[:] = 0
            else:
                labels[bg_inds] = 0
                # re-enable anchors_with_max_overlap
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

        bbox_targets = anchors.new_zeros((num_inside, box_code_size))

        if len(gt_boxes) > 0:

            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]
            bbox_targets[fg_inds, :] = box_encoding_fn(
                fg_gt_boxes.float(), fg_anchors
            )

        reg_weights = anchors.new_zeros((num_inside,))
        # uniform weighting of examples (given non-uniform sampling)
        if norm_by_num_examples:
            num_examples = (labels >= 0).sum()  # neg + pos
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0
        # bbox_outside_weights[labels == 0, :] = 1.0 / num_examples

        ret_dict = {
            "box_cls_labels": labels,
            "bbox_reg_targets": bbox_targets,
            "reg_weights": reg_weights,
            # "assigned_anchors_overlap": fg_max_overlap,
            # "positive_gt_id": gt_pos_ids,
            # "assigned_anchors_inds": fg_inds,
        }

        # ret["assigned_anchors_inds"] = fg_inds
        return ret_dict

    def nearest_iou_similarity(self, boxes1, boxes2):
        """Compute matrix of (negated) sq distances.

        Args:
          boxlist1: BoxList holding N boxes.
          boxlist2: BoxList holding M boxes.

        Returns:
          A tensor with shape [N, M] representing negated pairwise
          squared distance.
        """
        boxes1_bv = self._rbbox2d_to_near_bbox(boxes1)
        boxes2_bv = self._rbbox2d_to_near_bbox(boxes2)
        ret = bbox_overlaps(boxes1_bv, boxes2_bv)
        return ret

    def _rbbox2d_to_near_bbox(
        self,
        in_boxes: torch.Tensor,
        box_mode: str = "wlh",
        rect: bool = False,
    ):
        """Convert rotated bbox to nearest 'standing' or 'lying' bbox.

        Args:
            inboxes: [N, 5(x, y, w, l, ry)] or [N, 7(x,y,z,w,l,h,ry)]
        Returns:
            outboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
        """

        rots = in_boxes[..., -1]
        # limit ry in range np.abs([-np.pi/2., np.pi/2.])
        rots_0_pi_div_2 = torch.abs(limit_period(rots, 0.5, np.pi))
        # this line aims to rotate the box to a vertial or horizonal
        # direction with abs(angle) less than 45'.
        cond = (rots_0_pi_div_2 > np.pi / 4).unsqueeze(-1)
        in_boxes_center = torch.where(
            cond, in_boxes[:, [0, 1, 3, 2]], in_boxes[:, :4]
        )  # if True, change w and l; otherwise keep the same;
        out_boxes = torch.zeros(
            [in_boxes.shape[0], 4],
            dtype=in_boxes.dtype,
            device=in_boxes.device,
        )
        out_boxes[:, :4] = center_to_minmax_2d(
            in_boxes_center[:, :2], in_boxes_center[:, 2:]
        )
        return out_boxes
