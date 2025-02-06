from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.functional import nms
from horizon_plugin_pytorch.functional import sort as stable_sort

from hat.core.box_torch_ops import corner_to_standup_nd
from hat.core.box_utils import center_to_corner_box2d
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["PointPillarsPostProcess"]


@OBJECT_REGISTRY.register
class PointPillarsPostProcess(nn.Module):
    """PointPillars PostProcess Module.

    Args:
        num_classes: Number of classes.
        box_coder: BoxCeder module.
        use_direction_classifier: Whether to use direction.
        num_direction_bins: Number of direction for per anchor. Defaults to 2.
        direction_offset:  Direction offset. Defaults to 0.0.
        use_rotate_nms: Whether to use rotated nms.
        nms_pre_max_size: Max size of nms preprocess.
        nms_post_max_size: Max size of nms postprocess.
        nms_iou_threshold: IoU threshold of nms.
        score_threshold: Score threshold.
        post_center_limit_range: PointCloud range.
        max_per_img: Max number of object per image.
    """

    def __init__(
        self,
        num_classes: int,
        box_coder: int,
        use_direction_classifier: bool = True,
        num_direction_bins: int = 2,
        direction_offset: float = 0.0,
        # nms
        use_rotate_nms: bool = False,
        nms_pre_max_size: int = 1000,
        nms_post_max_size: int = 300,
        nms_iou_threshold: float = 0.5,
        score_threshold: float = 0.05,
        post_center_limit_range: List[float] = [  # noqa B006
            0,
            -39.68,
            -5,
            69.12,
            39.68,
            5,
        ],
        max_per_img: int = 100,
    ):

        super().__init__()

        self.box_coder = box_coder
        self.box_code_size = self.box_coder.code_size
        self.num_classes = num_classes
        self.num_direction_bins = num_direction_bins
        self.direction_offset = direction_offset

        self.num_class_with_bg = self.num_classes

        self.use_rotate_nms = use_rotate_nms
        self.nms_pre_max_size = nms_pre_max_size
        self.nms_post_max_size = nms_post_max_size
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = score_threshold
        self.post_center_limit_range = post_center_limit_range
        self.max_per_img = max_per_img
        self.use_direction_classifier = use_direction_classifier

    @fx_wrap()
    @torch.no_grad()
    def forward(
        self,
        box_preds: torch.Tensor,
        cls_preds: torch.Tensor,
        dir_preds: torch.Tensor,
        anchors: torch.Tensor,
    ):
        """Forward pass.

        Args:
            box_preds: BBox predictions.
            cls_preds: Classification predictions.
            dir_preds: Direction classification predictions.
            anchors: Anchors.

        Returns:
            detections: Batch predictions.
        """

        batch_box_preds = box_preds
        batch_cls_preds = cls_preds
        batch_dir_preds = dir_preds

        batch_size = int(batch_box_preds.shape[0])
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(
            batch_size, 1, 1
        )

        batch_task_anchors = batch_anchors.view(
            batch_size, -1, batch_anchors.shape[-1]
        )

        batch_box_preds = batch_box_preds.view(
            batch_size, -1, self.box_code_size
        )

        batch_cls_preds = batch_cls_preds.view(
            batch_size, -1, self.num_class_with_bg
        )

        batch_reg_preds = self.box_coder.decode(
            batch_box_preds,
            batch_task_anchors,
        )

        if self.use_direction_classifier:
            batch_dir_preds = batch_dir_preds.view(
                batch_size, -1, self.num_direction_bins
            )
        else:
            batch_dir_preds = [None] * batch_size

        detections = self._get_batch_detections(
            batch_cls_preds,
            batch_reg_preds,
            batch_dir_preds,
        )
        return detections

    def _get_batch_detections(
        self,
        batch_cls_preds: torch.Tensor,
        batch_reg_preds: torch.Tensor,
        batch_dir_preds: Optional[torch.Tensor] = None,
    ):
        """Get Batch predictions.

        Args:
            batch_cls_preds: Classification predictions.
            batch_reg_preds: BBox predictions.
            batch_dir_preds: Direction classification predictions.

        Returns:
            batch_predictions: Batch predictions.
        """
        batch_predictions = ()
        post_center_range = self.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds.dtype,
                device=batch_reg_preds.device,
            )

        for box_preds, cls_preds, dir_preds in zip(
            batch_reg_preds,
            batch_cls_preds,
            batch_dir_preds,
        ):
            prediction = self._get_single_detection(
                post_center_range,
                box_preds,
                cls_preds,
                dir_preds,
            )
            batch_predictions += (prediction,)

        return batch_predictions

    def _get_single_detection(
        self,
        post_center_range,
        box_preds,
        cls_preds,
        dir_preds,
    ):
        """Get prediction on single point cloud.

        Args:
            post_center_range: Point cloud range used to filter object.
            box_preds: BBox predictions.
            cls_preds: Classification predictions.
            dir_preds: Direction classification predictions.

        Raises:
            ValueError: Raise ValueError when `use_rotate_nms=True`.

        Returns:
            Box3d predictions in lidar coordinates.
            Classification predictions.
            Score predictions.
        """

        dtype = box_preds.dtype
        device = box_preds.device
        box_preds = box_preds.float()
        cls_preds = cls_preds.float()

        if self.use_direction_classifier:
            dir_labels = torch.max(dir_preds, dim=-1)[1]

        total_scores = torch.sigmoid(cls_preds)

        # Apply NMS in birdeye view
        if self.use_rotate_nms:
            raise ValueError("rotate nms is not implemented yet")
        else:
            nms_func = self.nms

        # get highest score per prediction, than apply nms
        # to remove overlapped box.
        if self.num_class_with_bg == 1:
            top_scores = total_scores.squeeze(-1)
            top_labels = torch.zeros(
                total_scores.shape[0],
                device=total_scores.device,
                dtype=torch.long,
            )

        else:
            top_scores, top_labels = torch.max(total_scores, dim=-1)

        if self.score_threshold > 0.0:
            thresh = torch.tensor(
                [self.score_threshold], device=total_scores.device
            ).type_as(total_scores)
            top_scores_keep = top_scores >= thresh
            top_scores = top_scores.masked_select(top_scores_keep)

        if top_scores.shape[0] != 0:
            if self.score_threshold > 0.0:
                box_preds = box_preds[top_scores_keep]
                if self.use_direction_classifier:
                    dir_labels = dir_labels[top_scores_keep]
                top_labels = top_labels[top_scores_keep]
            # (x,y,z,w,l,h,r) -> (x,y,w,l,r)
            boxes_for_nms = box_preds[:, [0, 1, 3, 4, -1]]
            if not self.use_rotate_nms:
                box_preds_corners = center_to_corner_box2d(
                    boxes_for_nms[:, :2],  # xy
                    boxes_for_nms[:, 2:4],  # wl
                    boxes_for_nms[:, 4],  # r
                )
                boxes_for_nms = corner_to_standup_nd(box_preds_corners)
                # the nms in 3d detection just remove overlap boxes.
            selected = nms_func(
                boxes_for_nms,
                top_scores,
                pre_max_size=self.nms_pre_max_size,
                post_max_size=self.nms_post_max_size,
                iou_threshold=self.nms_iou_threshold,
            )
        else:
            selected = []
        # if selected is not None:
        selected_boxes = box_preds[selected]
        if self.use_direction_classifier:
            selected_dir_labels = dir_labels[selected]
        selected_labels = top_labels[selected]
        selected_scores = top_scores[selected]

        # finally generate predictions.
        if selected_boxes.shape[0] != 0:
            box_preds = selected_boxes
            scores = selected_scores
            label_preds = selected_labels
            if self.use_direction_classifier:
                dir_labels = selected_dir_labels
                opp_labels = (
                    (box_preds[..., -1] - self.direction_offset) > 0
                ) ^ dir_labels.bool()
                box_preds[..., -1] += torch.where(
                    opp_labels,
                    torch.tensor(np.pi).type_as(box_preds),
                    torch.tensor(0.0).type_as(box_preds),
                )

            preds_box3d_lidar = box_preds
            preds_scores = scores
            preds_labels = label_preds

            if post_center_range is not None:
                mask = (preds_box3d_lidar[:, :3] >= post_center_range[:3]).all(
                    1
                )
                mask &= (
                    preds_box3d_lidar[:, :3] <= post_center_range[3:]
                ).all(1)
                preds_box3d_lidar = preds_box3d_lidar[mask]
                preds_scores = preds_scores[mask]
                preds_labels = preds_labels[mask]
        else:

            preds_box3d_lidar = torch.zeros(
                [0, self.box_code_size], dtype=dtype, device=device
            )
            preds_scores = torch.zeros([0], dtype=dtype, device=device)
            preds_labels = torch.zeros(
                [0], dtype=top_labels.dtype, device=device
            )

        return preds_box3d_lidar, preds_labels, preds_scores

    def nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
        pre_max_size: Optional[int] = None,
        post_max_size: Optional[int] = None,
    ):
        """NMS.

        Args:
            boxes: Shape(N, 4), boxes in (x1, y1, x2, y2) format.
            scores: Shape(N), scores.
            iou_threshold: IoU threshold.
            pre_nms_top_n: Get top n boxes by score before nms.
            output_num: Get top n boxes by score after nms.

        Returns:
            Indices.
        """

        if scores.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        if pre_max_size is not None:
            indices = stable_sort(scores, descending=True, stable=True)[1][
                : min(pre_max_size, scores.size(0))
            ]
            scores = scores[indices]
            boxes = boxes[indices]

        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        else:
            keep = nms(boxes, scores, iou_threshold)

        if post_max_size is not None:
            keep = keep[:post_max_size]

        return indices[keep] if pre_max_size is not None else keep
