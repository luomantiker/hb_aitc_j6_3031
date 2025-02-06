from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.core.box3d_utils import xywhr2xyxyr
from hat.core.circle_nms_jit import circle_nms
from hat.core.nms.box3d_nms import nms_bev
from hat.models.task_modules.centerpoint import CenterPointBBoxCoder
from hat.registry import OBJECT_REGISTRY

__all__ = ["CenterPointPostProcess"]


@OBJECT_REGISTRY.register
class CenterPointPostProcess(nn.Module):
    """
    CenterPoint PostProcess Module.

    Args:
        tasks: Task information including class number and class names.
            Default: None.
        norm_bbox: Whether to normalize bounding boxes. Default: True.
        bbox_coder: BoxCoder module. Default: None.
        max_pool_nms: Whether to use max-pooling NMS. Default: False.
        score_threshold: Score threshold for filtering detections.
        post_center_limit_range: Point cloud range. Default: None.
        min_radius: Minimum radius. Default: None.
        out_size_factor: Output size factor. Default: 1.
        nms_type: NMS type, either "rotate" or "circle". Default: "rotate".
        pre_max_size: Maximum size of NMS preprocess. Default: 1000.
        post_max_size: Maximum size of NMS postprocess. Default: 83.
        nms_thr: IoU threshold for NMS. Default: 0.2.
        use_max_pool: Whether to use max-pooling during NMS. Default: False.
        max_pool_kernel: Max-pooling kernel size. Default: 3.
        box_size: Size of bounding boxes. Default: 9.
    """

    def __init__(
        self,
        tasks: Optional[List[dict]] = None,
        norm_bbox: bool = True,
        bbox_coder: Optional[CenterPointBBoxCoder] = None,
        max_pool_nms: bool = False,
        score_threshold: float = 0.0,
        post_center_limit_range: Optional[List[float]] = None,
        min_radius: Optional[List[float]] = None,
        out_size_factor: int = 1,
        nms_type: str = "rotate",
        pre_max_size: int = 1000,
        post_max_size: int = 83,
        nms_thr: float = 0.2,
        use_max_pool: bool = False,
        max_pool_kernel: Optional[int] = 3,
        box_size: Optional[int] = 9,
    ):

        super().__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox
        self.bbox_coder = bbox_coder

        self.nms_type = nms_type
        self.min_radius = min_radius
        self.post_max_size = post_max_size
        self.post_center_limit_range = post_center_limit_range
        self.score_threshold = score_threshold
        self.nms_thr = nms_thr
        self.pre_max_size = pre_max_size
        self.post_max_size = post_max_size
        self.box_size = box_size

        self.max_pool_nms = max_pool_nms
        self.out_size_factor = out_size_factor
        self.use_max_pool = use_max_pool
        self.max_pool_kernel = max_pool_kernel

    @torch.no_grad()
    def forward(self, preds_dicts):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts: Prediction results.

        Returns:
            ret_list: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict["heatmap"].shape[0]
            batch_heatmap = preds_dict["heatmap"].sigmoid()
            if self.use_max_pool:
                kernel = self.max_pool_kernel
                pad = (kernel - 1) // 2
                max_heatmap = F.max_pool2d(
                    batch_heatmap,
                    (kernel, kernel),
                    stride=1,
                    padding=(pad, pad),
                )
                batch_heatmap *= (batch_heatmap == max_heatmap).float()

            batch_reg = preds_dict["reg"]
            batch_hei = preds_dict["height"]

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict["dim"])
            else:
                batch_dim = preds_dict["dim"]

            batch_rots = preds_dict["rot"][:, 0].unsqueeze(1)
            batch_rotc = preds_dict["rot"][:, 1].unsqueeze(1)

            if "vel" in preds_dict:
                batch_vel = preds_dict["vel"]
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id,
            )
            assert self.nms_type in ["circle", "rotate"]
            batch_reg_preds = [box["bboxes"] for box in temp]
            batch_cls_preds = [box["scores"] for box in temp]
            batch_cls_labels = [box["labels"] for box in temp]
            if self.use_max_pool:
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]["bboxes"]
                    scores = temp[i]["scores"]
                    labels = temp[i]["labels"]

                    ret = dict(  # noqa C408
                        bboxes=boxes3d, scores=scores, labels=labels
                    )
                    ret_task.append(ret)
                rets.append(ret_task)
            elif self.nms_type == "circle":
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]["bboxes"]
                    scores = temp[i]["scores"]
                    labels = temp[i]["labels"]
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.min_radius[task_id],
                            post_max_size=self.post_max_size,
                        ),
                        dtype=torch.long,
                        device=boxes.device,
                    )

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(  # noqa C408
                        bboxes=boxes3d, scores=scores, labels=labels
                    )
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(
                        num_class_with_bg,
                        batch_cls_preds,
                        batch_reg_preds,
                        batch_cls_labels,
                    )
                )

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k == "bboxes":
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k == "scores":
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k == "labels":
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append(ret)

        return ret_list

    def get_task_detections(
        self,
        num_class_with_bg: int,
        batch_cls_preds: List[torch.Tensor],
        batch_reg_preds: List[torch.Tensor],
        batch_cls_labels: List[torch.Tensor],
    ):
        """Rotate nms for each task.

        Args:
            num_class_with_bg: Number of classes for the current task.
            batch_cls_preds: Prediction score with the shape of [N].
            batch_reg_preds: Prediction bbox with the shape of [N, 9].
            batch_cls_labels: Prediction label with the shape of [N].

        Returns:
            predictions_dicts: contains the following keys:

                -bboxes: Prediction bboxes after nms with the
                    shape of [N, 9].
                -scores: Prediction scores after nms with the
                    shape of [N].
                -labels: Prediction labels after nms with the
                    shape of [N].
        """
        predictions_dicts = []
        post_center_range = self.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device,
            )

        for _, (box_preds, cls_preds, cls_labels) in enumerate(
            zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)
        ):

            # Apply NMS in bird eye view

            # get the highest score per prediction, then apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long,
                )

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.score_threshold > 0.0:
                thresh = torch.tensor(
                    [self.score_threshold], device=cls_preds.device
                ).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.score_threshold > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                # transform back to bev coordinate
                boxes = box_preds[:, [0, 1, 4, 3, 6]]
                boxes[:, -1] = -boxes[:, -1] - np.pi / 2
                boxes_for_nms = xywhr2xyxyr(boxes)

                # the nms in 3d detection just remove overlap boxes.
                selected = nms_bev(
                    boxes_for_nms,
                    top_scores,
                    thresh=self.nms_thr,
                    pre_max_size=self.pre_max_size,
                    post_max_size=self.post_max_size,
                )
            else:
                selected = []

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (
                        final_box_preds[:, :3] >= post_center_range[:3]
                    ).all(1)
                    mask &= (
                        final_box_preds[:, :3] <= post_center_range[3:]
                    ).all(1)
                    predictions_dict = dict(  # noqa C408
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask],
                    )
                else:
                    predictions_dict = dict(  # noqa C408
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels,
                    )
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(  # noqa C408
                    bboxes=torch.zeros(
                        [0, self.box_size],
                        dtype=dtype,
                        device=device,
                    ),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros(
                        [0], dtype=top_labels.dtype, device=device
                    ),
                )

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
