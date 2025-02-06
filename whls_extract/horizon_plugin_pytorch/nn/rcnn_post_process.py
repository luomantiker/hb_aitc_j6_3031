from typing import List, Optional, Tuple

import torch
from torch import Tensor

from horizon_plugin_pytorch.nn.quantized.functional import (
    quantize,
    rcnn_post_process,
)
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.typeguard import typechecked


class RcnnPostProcess(torch.nn.Module):
    """Post Process of RCNN output.

    Given bounding boxes and corresponding scores and deltas,
    decodes bounding boxes and performs NMS. In details, it consists of:

    - Argmax on multi-class scores
    - Filter out those belows the given threshold
    - Non-linear Transformation,
      convert box deltas to original image coordinates
    - Bin-sort remaining boxes on score
    - Apply class-aware NMS and return the firstnms_output_box_num of boxes

    Args:
        image_size: a int tuple of (h, w), for fixed image size
        nms_threshold: bounding boxes of IOU greater than nms_threshold
                will be suppressed
        box_filter_threshold: bounding boxes of scores less than
                box_filter_threshold will be discarded
        num_classes: total number of classes
        post_nms_top_k: number of bounding boxes after NMS in each image
        delta_mean: a float list of size 4
        delta_std: a float list of size 4
    """

    @typechecked
    def __init__(
        self,
        image_size: Tuple[int, int] = (1024, 1024),
        nms_threshold: float = 0.3,
        box_filter_threshold: float = 0.1,
        num_classes: int = 1,
        post_nms_top_k: int = 100,
        delta_mean: List[float] = (0.0, 0.0, 0.0, 0.0),
        delta_std: List[float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:

        super().__init__()

        self.fixed_image_h, self.fixed_image_w = image_size
        self.nms_threshold = nms_threshold
        self.box_filter_threshold = box_filter_threshold
        self.num_classes = num_classes
        self.post_nms_top_k = post_nms_top_k
        self.delta_mean = delta_mean
        self.delta_std = delta_std

    @typechecked
    def forward(
        self,
        boxes: List[Tensor],
        scores: Tensor,
        deltas: Tensor,
        image_sizes: Optional[Tensor] = None,
    ):
        """Forward of RcnnPostProcess.

        Args:
            boxes: list of box of shape [box_num, (x1, y1, x2, y2)].
                    can be Tensor(float), QTensor(float, int)
            scores: shape is [num_batch * num_box, num_classes + 1, 1, 1,],
                    dtype is float32
            deltas: shape is [num_batch * num_box, (num_classes + 1) * 4,
                    1, 1,], dtype is float32
            image_sizes: shape is [num_batch, 2], dtype is int32,
                    for dynamic image size, can be None. Defaults to None

        Returns:
            Tensor[num_batch, post_nms_top_k, 6]: output data in format
                    [x1, y1, x2, y2, score, class_index], dtype is float32
                    if the output boxes number is less than `post_nms_top_k`,
                    they are padded with -1.0
        """

        for box in boxes:
            if not isinstance(box, QTensor) and not box.is_floating_point():
                raise TypeError(
                    f"If box is Tensor, it must be float dtype, but get {box.dtype}"  # noqa: E501
                )
        boxes = [
            quantize(
                box,
                torch.tensor([0.25]).to(box.device),
                torch.tensor([0.0]).to(box.device),
                -1,
                "qint16",
            )
            if not isinstance(box, QTensor)
            else box.int_repr()
            for box in boxes
        ]

        _, output_float = rcnn_post_process(
            boxes,
            scores,
            deltas,
            image_sizes,
            self.fixed_image_h,
            self.fixed_image_w,
            self.nms_threshold,
            self.box_filter_threshold,
            self.num_classes,
            self.post_nms_top_k,
            self.delta_mean,
            self.delta_std,
        )

        return output_float
