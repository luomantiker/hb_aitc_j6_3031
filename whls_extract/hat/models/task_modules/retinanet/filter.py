# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Sequence

import torch.nn as nn

from hat.models.base_modules.postprocess import FilterModule
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class RetinanetMultiStrideFilter(nn.Module):
    def __init__(
        self,
        strides: Sequence[int],
        threshold: float,
    ):
        super(RetinanetMultiStrideFilter, self).__init__()
        self.strides = strides
        self.num_level = len(strides)
        self.filters = nn.ModuleList(
            [
                FilterModule(
                    threshold=threshold,
                    idx_range=None,
                )
                for _ in range(self.num_level)
            ]
        )

    def _filter_forward(self, preds):
        mlvl_outputs = []
        cls_scores, bbox_preds = preds

        for level in range(self.num_level):
            (
                per_level_cls_scores,
                per_level_bbox_preds,
            ) = (cls_scores[level], bbox_preds[level])
            filter_input = [
                per_level_cls_scores,
                per_level_bbox_preds,
            ]
            for per_filter_input in filter_input:
                assert (
                    len(per_filter_input.shape) == 4
                ), "should be in NCHW layout"
            filter_output = self.filters[level](*filter_input)
            # len(filter_output) equal to batch size
            per_sample_outs = []
            for i in range(len(filter_output)):
                (
                    _,
                    _,
                    per_img_coord,
                    per_img_score,
                    per_img_bbox_pred,
                ) = filter_output[i]
                per_sample_outs.append(
                    [
                        per_img_coord,
                        per_img_score,
                        per_img_bbox_pred,
                    ]
                )
            mlvl_outputs.append(per_sample_outs)

        return mlvl_outputs

    def forward(self, cls_scores, bbox_preds):
        preds = [cls_scores, bbox_preds]
        preds = self._filter_forward(preds)
        return preds
