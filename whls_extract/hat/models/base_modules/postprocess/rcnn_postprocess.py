# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict
from typing import Dict, Hashable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    from horizon_plugin_pytorch.nn import RcnnPostProcess as RPP_OP
except ImportError:
    RPP_OP = None


@OBJECT_REGISTRY.register
class RCNNPostProcess(nn.Module):
    """Post process for rcnn.

    This operation is used to get refined detection result from
    DetectionPostProcess proposals with the rcnn output
    class score and regress delta. Supported on bernoulli2 and bayes.

    Args:
        input_score_key: Hashable object used to query class score
            from input (rcnn output).
        input_deltas_key: Hashable object used to query regress deltas
            from input (rcnn output).
        num_fg_classes: Number of foreground classes.
        image_hw: Fixed image size in (h, w), set to None if input have
            different sizes. Default to None.
        nms_threshold: IoU threshold for nms. Default to 0.3.
        box_filter_threshold: Score threshold to filter boxes.
            Default to 0.1.
        post_nms_top_k: Maximum number of output bounding boxes in each
            image. Default to 100.
        delta_mean: Mean value to be subtracted from bbox regression
            task in each coordinate.
            Default to (0.0, 0.0, 0.0, 0.0).
        delta_std: Standard deviation value to be divided from
            bbox regression task in each coordinate.
            Default to (1.0, 1.0, 1.0, 1.0).
    """

    @require_packages("horizon_plugin_pytorch>=1.2.3")
    def __init__(
        self,
        input_score_key: Hashable,
        input_deltas_key: Hashable,
        num_fg_classes: int,
        image_hw: Tuple[int, int] = None,
        nms_threshold: float = 0.3,
        box_filter_threshold: float = 0.1,
        post_nms_top_k: int = 100,
        delta_mean: List[float] = (0.0, 0.0, 0.0, 0.0),
        delta_std: List[float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        assert len(delta_mean) == 4, "delta_mean should be a list of size 4."
        assert len(delta_std) == 4, "delta_std should be a list of size 4."
        super().__init__()
        self.input_score_key = input_score_key
        self.input_deltas_key = input_deltas_key
        self.num_fg_classes = num_fg_classes
        self.image_hw = image_hw
        self.nms_threshold = nms_threshold
        self.box_filter_threshold = box_filter_threshold
        self.post_nms_top_k = post_nms_top_k
        self.delta_mean = delta_mean
        self.delta_std = delta_std
        self._rpp = RPP_OP(
            image_size=self.image_hw,
            nms_threshold=self.nms_threshold,
            box_filter_threshold=self.box_filter_threshold,
            num_classes=self.num_fg_classes,
            post_nms_top_k=self.post_nms_top_k,
            delta_mean=self.delta_mean,
            delta_std=self.delta_std,
        )

    def forward(
        self,
        batch_rois: List[Tensor],
        head_out: Dict[str, List[torch.Tensor]],
        im_hw: Optional[Tensor] = None,
    ):
        """Forward of RCNNPostProcess.

        Args:
            batch_rois: List of box of shape
                [num_bbox_per_img, (x1, y1, x2, y2)],
                can be Tensor(float), QTensor(float, int).
            head_out: Dict must include score and deltas:
                bbox_score: shape is
                    [num_batch * num_bbox_per_img, (num_fg_class + 1), 1, 1],
                    dtype is float32
                bbox_deltas: shape is
                    [num_batch * num_bbox_per_img,
                        (num_fg_class + 1) * 4, 1, 1],
                    dtype is float32
            im_hw: shape is [N, 1, 1, 2], dtype is int32, can be None.
                if None, original_img_h and original_img_w
                    must be provided. Defaults to None.

        Returns:
            Tensor[num_batch, nms_top_n, 6]: output data
                in format [x1, y1, x2, y2, score, class_index].
        """
        assert isinstance(batch_rois, List) and (
            batch_rois[0].shape[-1] == 4
        ), (
            "batch_rois must be list of bbox tesnor",
            "each tensor is (num_rois_per_img, 4): x1, y1, x2, y2",
        )
        bbox_score = head_out[self.input_score_key].detach()
        assert bbox_score.shape[1] == (self.num_fg_classes + 1), (
            f"bbox_score shape {bbox_score.shape} must be "
            "[num_batch * num_bbox_per_img, (num_fg_class + 1), 1, 1], "
        )
        bbox_deltas = head_out[self.input_deltas_key].detach()
        assert bbox_deltas.shape[1] == (self.num_fg_classes + 1) * 4, (
            f"bbox_deltas shape {bbox_deltas.shape} must be "
            "[num_batch * num_bbox_per_img, (num_fg_class + 1) * 4, 1, 1]"
        )
        assert (im_hw is not None) or (self.image_hw is not None)
        if im_hw is None:
            assert (
                self.image_hw is not None
            ), "If im_hw is None, self.image_hw must be inited"
            image_sizes = None
        else:
            image_sizes = im_hw.reshape((-1, 1, 1, 2))
        output = self._rpp(
            batch_rois,
            bbox_score,
            bbox_deltas,
            image_sizes,
        )
        return OrderedDict(
            rpp_pred=output,
        )
