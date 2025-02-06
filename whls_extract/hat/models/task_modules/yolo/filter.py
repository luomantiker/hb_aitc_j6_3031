# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from hat.models.base_modules.postprocess import FilterModule
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class YOLOv3Filter(nn.Module):  # noqa: D205,D400
    """Filter used for post-processing of YOLOv3

    Args:
        strides: A list contains the strides of feature maps.
        idx_range: The index range of
            values counted in compare of the first input.
            Defaults to None which means use all the values.
        threshold: The lower bound of output.
        last_channels: Last channels.
    """

    def __init__(
        self,
        strides: Sequence[int],
        threshold: float,
        idx_range: Optional[Tuple[int, int]] = None,
        last_channels: float = 75,
    ):
        super(YOLOv3Filter, self).__init__()
        self.strides = strides
        self.num_level = len(strides)
        self.filter_module = FilterModule(
            threshold=threshold,
            idx_range=idx_range,
        )
        self.per_channels = int(last_channels / 3)

    def _filter_forward(self, preds):
        mlvl_outputs = []
        for level in range(self.num_level):
            per_level_preds = preds[level]
            a = per_level_preds[:, : self.per_channels, :, :]
            b = per_level_preds[
                :, self.per_channels : self.per_channels * 2, :, :
            ]
            c = per_level_preds[:, self.per_channels * 2 :, :, :]

            for per_input in [a, b, c]:
                assert len(per_input.shape) == 4, "should be in NCHW layout"
                filter_output = self.filter_module(*[per_input])
                # len(filter_output) equal to batch size
                per_sample_outs = []
                for i in range(len(filter_output)):
                    (
                        _,
                        _,
                        per_img_coord,
                        per_img_score,
                    ) = filter_output[i]
                    per_sample_outs.append(
                        [
                            per_img_coord,
                            per_img_score,
                        ]
                    )
                mlvl_outputs.append(per_sample_outs[0])

        return mlvl_outputs

    def forward(
        self,
        preds: Sequence[torch.Tensor],
        meta_and_label: Optional[Dict] = None,
        **kwargs,
    ) -> Sequence[torch.Tensor]:
        preds = self._filter_forward(preds)
        return (
            preds[0],
            preds[1],
            preds[2],
            preds[3],
            preds[4],
            preds[5],
            preds[6],
            preds[7],
            preds[8],
        )
