# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from hat.models.base_modules.postprocess import FilterModule
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class FCOSMultiStrideCatFilter(nn.Module):
    """A modified Filter used for post-processing of FCOS.

    In each stride, concatenate the scores of each task as
    the first input of FilterModule, which can reduce latency in BPU.

    Args:
        strides (Sequence[int]): A list contains the strides of feature maps.
        idx_range (Optional[Tuple[int, int]], optional): The index range of
            values counted in compare of the first input.
            Defaults to None which means use all the values.
        threshold (float): The lower bound of output.
        task_strides(Sequence[Sequence[int]]): A list of out_stirdes of each
            task.
    """

    def __init__(
        self,
        strides: Sequence[int],
        threshold: float,
        task_strides: Sequence[Sequence[int]],
        int16_output: bool = False,
        idx_range: Optional[Tuple[int, int]] = None,
    ):
        super(FCOSMultiStrideCatFilter, self).__init__()
        self.cat_op = nn.quantized.FloatFunctional()
        self.strides = strides
        self.num_level = len(self.strides)
        self.task_strides = task_strides
        self.int16_output = int16_output
        for i in range(len(self.strides)):
            setattr(
                self,
                "filter_module_%s" % str(i),
                FilterModule(threshold=threshold, idx_range=idx_range),
            )
        self.set_qconfig()

    def forward(
        self,
        preds: Sequence[torch.Tensor],
        **kwargs,
    ) -> Sequence[torch.Tensor]:

        mlvl_outputs = []

        for stride_ind, stride in enumerate(self.strides):
            score_list = []
            filter_input = []
            for task_ind in range(len(self.task_strides)):
                pred = preds[task_ind]
                if stride in self.task_strides[task_ind]:
                    # bbox
                    filter_input.append(
                        pred[1][self.task_strides[task_ind].index(stride)]
                    )
                    # centerness
                    filter_input.append(
                        pred[2][self.task_strides[task_ind].index(stride)]
                    )
                    # score
                    score_list.append(
                        pred[0][self.task_strides[task_ind].index(stride)]
                    )
            # concatenate the scores of each task as the first input of filter
            if len(score_list) > 1:
                per_level_cls_scores = self.cat_op.cat(score_list, dim=1)
            else:
                per_level_cls_scores = score_list[0]
            filter_input.insert(0, per_level_cls_scores)

            for per_filter_input in filter_input:
                assert (
                    len(per_filter_input.shape) == 4
                ), "should be in NCHW layout"
            filter_output = getattr(
                self, "filter_module_%s" % str(stride_ind)
            )(*filter_input)
            per_sample_outs = []
            for task_ind in range(len(filter_output)):
                per_sample_outs.append(filter_output[task_ind][2:])
            mlvl_outputs.append(per_sample_outs)

        return mlvl_outputs

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if self.int16_output:
            from horizon_plugin_pytorch.dtype import qint16

            from hat.utils.qconfig_manager import QconfigMode

            qconfig_manager.set_qconfig_mode(QconfigMode.QAT)
            self.cat_op.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
            )


@OBJECT_REGISTRY.register
class FCOSMultiStrideCatFilterWithConeInvasion(FCOSMultiStrideCatFilter):
    """A modified Filter used for post-processing of FCOS with cone invasion.

    In each stride, concatenate the scores of each task as
    the first input of FilterModule, which can reduce latency in BPU.

    Args:
        strides: A list contains the strides of feature maps.
        idx_range: The index range of
            values counted in compare of the first input.
            Defaults to None which means use all the values.
        threshold: The lower bound of output.
        task_strides: A list of out_stirdes of each task.
    """

    def __init__(
        self,
        strides: Sequence[int],
        threshold: float,
        task_strides: Sequence[Sequence[int]],
        int16_output: bool = False,
        idx_range: Optional[Tuple[int, int]] = None,
    ):
        super(FCOSMultiStrideCatFilterWithConeInvasion, self).__init__(
            strides, threshold, task_strides, int16_output, idx_range
        )

    def forward(
        self,
        preds: Sequence[torch.Tensor],
        **kwargs,
    ) -> Sequence[torch.Tensor]:

        mlvl_outputs = []

        for stride_ind, stride in enumerate(self.strides):
            score_list = []
            filter_input = []
            cone_invasion_list = []
            for task_ind in range(len(self.task_strides)):
                pred = preds[task_ind]
                if stride in self.task_strides[task_ind]:
                    # bbox
                    filter_input.append(
                        pred[1][self.task_strides[task_ind].index(stride)]
                    )
                    # centerness
                    filter_input.append(
                        pred[2][self.task_strides[task_ind].index(stride)]
                    )
                    # score
                    score_list.append(
                        pred[0][self.task_strides[task_ind].index(stride)]
                    )
                    # cone
                    cone_invasion_list.append(
                        pred[3][self.task_strides[task_ind].index(stride)]
                    )
                    cone_invasion_list.append(
                        pred[4][self.task_strides[task_ind].index(stride)]
                    )
                    cone_invasion_list.append(
                        pred[5][self.task_strides[task_ind].index(stride)]
                    )
            # concatenate the scores of each task as the first input of filter
            if len(score_list) > 1:
                per_level_cls_scores = self.cat_op.cat(score_list, dim=1)
            else:
                per_level_cls_scores = score_list[0]
            filter_input.insert(0, per_level_cls_scores)
            filter_input += cone_invasion_list

            for per_filter_input in filter_input:
                assert (
                    len(per_filter_input.shape) == 4
                ), "should be in NCHW layout"
            filter_output = getattr(
                self, "filter_module_%s" % str(stride_ind)
            )(*filter_input)
            per_sample_outs = []
            for task_ind in range(len(filter_output)):
                per_sample_outs.append(filter_output[task_ind][2:])
            mlvl_outputs.append(per_sample_outs)

        return mlvl_outputs


# TODO(kongtao.hu 0.5): May need to be refactored to become more general.
@OBJECT_REGISTRY.register
class FCOSMultiStrideFilter(nn.Module):  # noqa: D205,D400
    """Filter used for post-processing of
    `FCOS <https://arxiv.org/pdf/1904.01355.pdf>`_.

    Args:
        strides: A list contains the strides of feature maps.
        idx_range: The index range of values counted in compare of the first
            input. Defaults to None which means use all the values.
        threshold: The lower bound of output.
        for_compile: Whether used for compile. if true, should not include
            postprocess.
        decoder: Decoder module.
    """

    def __init__(
        self,
        strides: Sequence[int],
        threshold: float,
        idx_range: Optional[Tuple[int, int]] = None,
        for_compile: bool = False,
        decoder: nn.Module = None,
    ):
        super(FCOSMultiStrideFilter, self).__init__()
        self.strides = strides
        self.num_level = len(strides)
        self.filter_module = FilterModule(
            threshold=threshold,
            idx_range=idx_range,
        )
        self.for_compile = for_compile
        self.decoder = decoder

    def _filter_forward(self, preds):
        mlvl_outputs = []
        cls_scores, bbox_preds, centernesses = preds
        for level in range(self.num_level):
            (
                per_level_cls_scores,
                per_level_bbox_preds,
                per_level_centernesses,
            ) = (cls_scores[level], bbox_preds[level], centernesses[level])
            batch_size = per_level_cls_scores.shape[0]
            if batch_size > 1:
                filter_input_split = [
                    torch.split(per_level_cls_scores, 1, dim=0),
                    torch.split(per_level_bbox_preds, 1, dim=0),
                    torch.split(per_level_centernesses, 1, dim=0),
                ]
            else:
                filter_input_split = [
                    [per_level_cls_scores],
                    [per_level_bbox_preds],
                    [per_level_centernesses],
                ]
            filter_output = []
            for filter_input in zip(*filter_input_split):
                for _, per_filter_input in enumerate(filter_input):
                    assert (
                        len(per_filter_input.shape) == 4
                    ), "should be a 4D Tensor"
                filter_output.extend(self.filter_module(*filter_input))
                # len(filter_output) equal to batch size
            per_sample_outs = []
            for i in range(len(filter_output)):
                (
                    per_img_maxvalue,
                    per_img_maxid,
                    per_img_coord,
                    per_img_score,
                    per_img_bbox_pred,
                    per_img_centerness,
                ) = filter_output[i]
                per_sample_outs.append(
                    [
                        per_img_maxvalue,
                        per_img_maxid,
                        per_img_coord,
                        per_img_score,
                        per_img_bbox_pred,
                        per_img_centerness,
                    ]
                )
            mlvl_outputs.append(per_sample_outs)

        return mlvl_outputs

    def forward(
        self,
        preds: Sequence[torch.Tensor],
        meta_and_label: Optional[Dict] = None,
        **kwargs,
    ) -> Sequence[torch.Tensor]:
        preds = self._filter_forward(preds)

        if self.for_compile:
            preds = [preds[i][0] for i in range(len(preds))]
            return tuple(preds)
        if self.decoder is not None:

            return self.decoder(preds, meta_and_label)
        return preds
