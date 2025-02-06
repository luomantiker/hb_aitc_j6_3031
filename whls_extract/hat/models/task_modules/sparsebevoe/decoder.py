# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Any, Dict, Optional

import torch

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class SparseBEVOEDecoder(object):
    """
    Sparse BEVOE Decoder for object detection.

    Args:
        num_output: Number of top scoring predictions to output.
                    Defaults to 300.
        score_threshold: Threshold for filtering predictions based on score.
                         Defaults to None.
    """

    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
    ):
        super(SparseBEVOEDecoder, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold

    def __call__(
        self,
        cls_scores: torch.Tensor,
        box_preds: torch.Tensor,
        qulity: torch.Tensor = None,
    ) -> Dict[str, Any]:
        """
        Perform decoding of predictions into a dictionary of detections.

        Args:
            cls_scores: Predicted class scores,
                        shape (bs, num_pred, num_cls).
            box_preds: Predicted box coordinates,
                       shape (bs, num_pred, 4).
            qulity: Predicted quality scores,
                    shape (bs, num_pred). Defaults to None.

        Returns:
            Dictionary containing decoded detections with keys
            'scores', 'boxes', and optionally 'qulity'.
        """
        if isinstance(cls_scores, list):
            cls_scores = cls_scores[-1]
            box_preds = box_preds[-1]
            if qulity is not None:
                qulity = qulity[-1]
        cls_scores = cls_scores.sigmoid()
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output,
            dim=1,
            sorted=True,
        )
        cls_ids = indices % num_cls
        mask = None
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if qulity is not None:
            centerness = qulity[..., 0]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            cls_ids = torch.gather(cls_ids, 1, idx)
            if mask is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        for i in range(bs):
            category_ids = cls_ids[i]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]

            yaw = torch.atan2(box[:, 6], box[:, 7])
            box = torch.cat(
                [
                    box[:, 0:3],
                    box[:, 3:6].exp(),
                    yaw[:, None],
                    box[:, 8:],
                ],
                dim=-1,
            )
            output.append(
                {
                    "bboxes": box.cpu(),
                    "scores": scores.cpu(),
                    "labels": category_ids.cpu(),
                }
            )
        return output
