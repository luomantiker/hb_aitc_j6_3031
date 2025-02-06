# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Sequence

import horizon_plugin_pytorch.nn as hnn
from torch import nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["LidarDecoder", "LidarSegDecoder", "LidarDetDecoder"]


logger = logging.getLogger(__name__)


class LidarDecoder(nn.Module):
    """The basic decoder structure of lidar.

    Args:
        head: Head module.
        name: Name for task.
        task_feat_index: Index for the task.
        task_weight: Task weight for the task.
        target: Target module for the task.
        loss: Loss module for the task.
        decoder: Decoder module for the task.
    """

    def __init__(
        self,
        head: nn.Module,
        name: str,
        task_feat_index: int = 0,
        task_weight: float = 1.0,
        target: nn.Module = None,
        loss: nn.Module = None,
        decoder: nn.Module = None,
    ):
        super(LidarDecoder, self).__init__()
        self.head = head
        self.target = target
        self.loss = loss
        self.decoder = decoder
        self.name = name
        self.task_weight = task_weight
        self.task_feat_index = task_feat_index

    def forward(self, feats, meta):
        """
        Forward pass through the LidarDecoder.

        Args:
            feats: Input features or sequence of features.
            meta: Metadata.

        Returns:
            Predictions and additional results.
        """
        if not isinstance(feats, Sequence):
            feats = [feats]
        feat = feats[self.task_feat_index]
        feat = [feat]
        pred = self.head(feat)
        return self._post_process(meta, pred)

    @fx_wrap()
    def _post_process(self, meta, pred):
        """
        Post-processing of predictions.

        Args:
            meta: Metadata.
            pred: Model predictions.

        Returns:
            Predictions and additional results.
        """
        if self.training:
            gts = self._get_gts(meta)
            target = self.target(gts, pred)
            loss = self._loss(target)
            for k, v in loss.items():
                loss[k] = v * self.task_weight
            return [pred], dict(**loss)
        else:
            return self._decode(pred)

    def fuse_model(self):
        """Fuse model operations for quantization."""
        if hasattr(self.head, "fuse_model"):
            self.head.fuse_model()

    def set_qconfig(self):
        """Set quantization configuration for the model."""
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if hasattr(self.head, "set_qconfig"):
            self.head.set_qconfig()

        for module in [self.target, self.loss, self.decoder]:
            if module is not None:
                module.qconfig = None


@OBJECT_REGISTRY.register
class LidarSegDecoder(LidarDecoder):
    """
    Segmentation decoder structure of lidar.

    Args:
        feat_upscale: Feature upscale factor. Defaults to 1.
        **kwargs: Additional keyword arguments passed to the parent class.

    """

    def __init__(self, feat_upscale: int = 1, **kwargs):
        super(LidarSegDecoder, self).__init__(**kwargs)
        self.feat_upscale = feat_upscale
        if self.feat_upscale > 1:
            self.resize = hnn.Interpolate(
                scale_factor=self.feat_upscale,
                align_corners=None,
                recompute_scale_factor=True,
            )

    def forward(self, feats, meta):
        """
        Forward pass through the LidarSegDecoder.

        Args:
            feats: Input features or sequence of features.
            meta: Metadata.

        Returns:
            Predictions and additional results.
        """
        if not isinstance(feats, Sequence):
            feats = [feats]
        feat = feats[self.task_feat_index]
        if self.feat_upscale > 1:
            feat = self.resize(feat)

        feat = [feat]
        pred = self.head(feat)
        return self._post_process(meta, pred)

    def _get_gts(self, meta):
        """
        Get ground truth segmentation labels.

        Args:
            meta: Metadata.

        Returns:
            Ground truth segmentation labels.
        """
        gts = meta["gt_seg_labels"]

        return gts

    def _loss(self, target):
        """
        Compute the loss for segmentation.

        Args:
            target: Segmentation target.

        Returns:
            Loss for segmentation.
        """
        return self.loss(**target)

    def _decode(self, pred):
        """
        Decode segmentation predictions.

        Args:
            pred: Segmentation predictions.

        Returns:
            Predictions and additional results.
        """
        if self.decoder is not None:
            result = self.decoder(pred)
            result = {self.name: result}
            return [pred], result
        else:
            return [pred], None


@OBJECT_REGISTRY.register
class LidarDetDecoder(LidarDecoder):
    """Detection decoder structure of lidar."""

    @fx_wrap()
    def _post_process(self, meta, pred):
        """
        Post-processing of predictions for detection.

        Args:
            meta: Metadata.
            pred: Model predictions.

        Returns:
            Predictions and additional results.
        """
        if self.training:
            heatmaps, anno_boxes, inds, masks = self.target(
                meta["gt_boxes"], meta["gt_classess"]
            )
            loss = self.loss(
                heatmaps,
                anno_boxes,
                inds,
                masks,
                pred,
            )

            for k, v in loss.items():
                loss[k] = v * self.task_weight
            return [pred], dict(**loss)
        else:
            return self._decode(pred)

    def _decode(self, pred):
        """
        Decode detection predictions.

        Args:
            pred: Detection predictions.

        Returns:
            Predictions and additional results.
        """
        if self.decoder is not None:
            result = self.decoder(pred)
            result = {self.name: result}
            return [pred], result
        else:
            ret = []
            for task_pred in pred:
                for _, v in task_pred.items():
                    ret.append(v)
            return ret, None
