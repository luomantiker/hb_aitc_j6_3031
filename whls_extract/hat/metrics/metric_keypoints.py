from typing import Tuple

import torch

from hat.core.heatmap_decoder import decode_heatmap
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY

__all__ = ["MeanKeypointDist", "PCKMetric"]


@OBJECT_REGISTRY.register
class MeanKeypointDist(EvalMetric):
    """
    This metric calculates the mean distance between keypoints.

    Args:
        name: name of the metric
        feat_stride: Stride of the feature map with respect to the input image.
        decode_mode: Mode for decoding the predicted keypoints.
                    "averaged" or "diff_sign"
    """

    def __init__(
        self,
        name: str = "mean_dist",
        feat_stride: int = 4,
        decode_mode: str = "averaged",
    ):
        super(MeanKeypointDist, self).__init__(name)
        self.eval_type = name
        self.feat_stride = feat_stride
        self.decode_mode = decode_mode

    def update(self, data):
        gt_ldmk = data["gt_ldmk"].detach()

        valid_mask = (data["gt_ldmk_attr"].detach() > 0).float()

        B, C, _ = gt_ldmk.shape

        pred_keypoint = data.get("pr_ldmk")
        if pred_keypoint is None:
            pred_heatmap = data["pr_heatmap"].detach()
            pred_keypoint = decode_heatmap(
                pred_heatmap, self.feat_stride, self.decode_mode
            )
        else:
            pred_keypoint = pred_keypoint.detach()

        pred_keypoint = pred_keypoint[:, :, :2]
        dist = torch.norm(pred_keypoint - gt_ldmk, dim=2, p=2)

        self.sum_metric += (dist * valid_mask).sum()
        self.num_inst += valid_mask.sum()


@OBJECT_REGISTRY.register
class PCKMetric(EvalMetric):
    """Compute PCK (Proportion of Correct Keypoints) metric.

    Args:
        alpha: Parameter alpha for defining the PCK threshold as a
                percentage of the object's size.
        feat_stride: Stride of the feature map with respect to the input image.
        img_shape: Shape of the input image in (height, width) format.
        decode_mode: Mode for decoding the predicted keypoints.
                     "averaged" or "diff_sign"

    """

    def __init__(
        self,
        alpha: float,
        feat_stride: int,
        img_shape: Tuple[int],
        decode_mode: str = "diff_sign",
    ):
        name = f"PCK@{alpha}"
        super(PCKMetric, self).__init__(name)
        self.alpha = alpha
        self.feat_stride = feat_stride
        self.img_shape = img_shape
        self.decode_mode = decode_mode

    def update(self, data):
        gt_ldmk = data["gt_ldmk"].detach()
        pred_keypoint = data.get("pr_ldmk")
        if pred_keypoint is None:
            pred_heatmap = data["pr_heatmap"].detach()
            pred_keypoint = decode_heatmap(
                pred_heatmap, self.feat_stride, self.decode_mode
            )
        else:
            pred_keypoint = pred_keypoint.detach()

        pred_keypoint = pred_keypoint[:, :, :2]

        valid_mask = (data["gt_ldmk_attr"].detach() > 0).float()

        Dist = self.get_dist(gt_ldmk, self.img_shape)

        inside_cnt, cnt = self.count_pck(
            pred_keypoint, gt_ldmk, Dist, valid_mask
        )

        self.sum_metric += inside_cnt
        self.num_inst += cnt

    def get_dist(self, gt_ldmk, img_shape):
        min_x = gt_ldmk[:, :, 0].min(axis=1)[0]
        max_x = gt_ldmk[:, :, 0].max(axis=1)[0]
        min_y = gt_ldmk[:, :, 1].min(axis=1)[0]
        max_y = gt_ldmk[:, :, 1].max(axis=1)[0]
        key_width = max_x - min_x
        key_height = max_y - min_y
        max_L = max(img_shape)
        dist = torch.max(key_height, key_width).clamp(0, max_L)
        return dist

    def count_pck(self, pred_keypoint, gt_ldmk, Dist, valid_mask):
        B, C, _ = gt_ldmk.shape
        pred_dist = torch.norm(pred_keypoint - gt_ldmk, dim=2, p=2)

        mask = (
            pred_dist < (Dist.reshape([-1, 1]).expand(B, C) * self.alpha)
        ).float() * valid_mask
        count_inliers = mask.int().sum()
        count = valid_mask.sum()
        return count_inliers, count
