# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Tuple

import torch
from torch import Tensor, nn

from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


__all__ = ["Detr3dPostProcess"]


def decode_preds(
    bev_range: Tuple[float], cls: Tensor, reg: Tensor, reference_points: Tensor
) -> Tensor:
    """Decode the predicted class probabilities and bounding boxes.

    Args:
        bev_range: The range of the bird's eye view.
        cls: The predicted class probabilities.
        reg: The predicted bounding boxes.
        reference_points: The reference points used for prediction.

    Returns:
        The decoded class probabilities and bounding boxes.
    """
    bs, cls_out_channels, _, _ = cls.shape
    _, reg_out_channels, _, _ = reg.shape

    cls = cls.permute(0, 2, 3, 1).contiguous()
    cls = cls.view(bs, -1, cls_out_channels)

    reg = reg.permute(0, 2, 3, 1).contiguous()

    reg[..., 0:3] = reg[..., 0:3] + reference_points
    reg[..., 0:3] = reg[..., 0:3].sigmoid()
    reg[..., 0] = reg[..., 0] * (bev_range[3] - bev_range[0]) + bev_range[0]
    reg[..., 1] = reg[..., 1] * (bev_range[4] - bev_range[1]) + bev_range[1]
    reg[..., 2] = reg[..., 2] * (bev_range[5] - bev_range[2]) + bev_range[2]
    reg = reg.view(bs, -1, reg_out_channels)
    return cls, reg


@OBJECT_REGISTRY.register
class Detr3dPostProcess(nn.Module):
    """The Detr3d PostProcess.

    Args:
        max_num: Max number of output.
        score_threshold: Score threshold for output.
    """

    def __init__(
        self, bev_range, max_num: int = 100, score_threshold: float = -1.0
    ):
        super(Detr3dPostProcess, self).__init__()
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.bev_range = bev_range

    def _denormalize(self, bbox: Tensor) -> Tensor:
        """Denormalize the bounding box coordinates.

        Args:
            bbox : The normalized bounding box tensor.

        Returns:
            The denormalized bounding box tensor.
        """
        center = bbox[..., 0:3]
        dim = bbox[..., 3:6].exp()
        rot_sine = bbox[..., 6]
        rot_cosine = bbox[..., 7]
        rot = torch.atan2(rot_sine, rot_cosine)
        rot = rot.unsqueeze(-1)
        if bbox.shape[1] == 10:
            vel = bbox[..., 8:10]
            bbox = torch.cat([center, dim, rot, vel], dim=-1)
        else:
            bbox = torch.cat([center, dim, rot], dim=-1)
        return bbox

    def _single_decode(self, cls_pred: Tensor, reg_pred: Tensor) -> Tensor:
        """Decode the predicted outputs for a single example.

        Args:
            cls_pred : The predicted classification tensor.
            reg_pred : The predicted regression tensor.

        Returns:
            The decoded bounding box tensor.
        """

        cls_score = cls_pred.sigmoid()
        num_classes = cls_score.shape[1]
        cls_score = cls_score.view(-1)
        score, indeices = cls_score.topk(self.max_num)
        if self.score_threshold > 0:
            valid_idx = score > self.score_threshold
            score = score[valid_idx]
            indeices = indeices[valid_idx]
        reg_idx = torch.div(indeices, num_classes, rounding_mode="trunc")
        label = indeices % num_classes

        bbox = reg_pred[reg_idx]
        bbox = self._denormalize(bbox)

        score = score.unsqueeze(-1)
        label = label.unsqueeze(-1)
        return torch.cat([bbox, score, label], dim=-1)

    def forward(
        self, cls_preds: Tensor, reg_preds: Tensor, reference_points: Tensor
    ) -> Tensor:
        """Forward pass of the module.

        Args:
            cls_preds : The list of predicted classification tensors.
            reg_preds : The list of predicted regression tensors.

        Returns:
            The list of decoded bounding box tensors.
        """
        rets = []
        cls_preds, reg_preds = decode_preds(
            self.bev_range, cls_preds, reg_preds, reference_points
        )
        for cls_pred, reg_pred in zip(cls_preds, reg_preds):
            ret = self._single_decode(cls_pred, reg_pred)
            rets.append(ret)

        return rets
