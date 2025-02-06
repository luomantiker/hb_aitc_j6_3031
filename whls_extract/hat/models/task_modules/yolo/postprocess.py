# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Sequence

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.functional import nms

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = ["YOLOV3PostProcess", "YOLOV3HbirPostProcess"]


@OBJECT_REGISTRY.register
class YOLOV3PostProcess(nn.Module):
    """
    The postprocess of YOLOv3.

    Args:
        anchors: The anchors of yolov3.
        num_classes: The num classes of class branch.
        score_thresh: Score thresh of postprocess before nms.
        nms_thresh: Nms thresh.
        topK: The output num of bboxes after postprocess.

    """

    def __init__(
        self,
        anchors: list,
        strides: list,
        num_classes: int,
        score_thresh: float = 0.01,
        nms_thresh: float = 0.45,
        topK: int = 200,
    ):
        super(YOLOV3PostProcess, self).__init__()
        self.anchors = anchors
        self.strides = strides
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.topK = topK

    def forward(self, inputs: Sequence[torch.Tensor]):
        inputs = _as_list(inputs)
        assert len(inputs) == len(self.anchors)
        prediction = []
        for input, anchor, stride in zip(inputs, self.anchors, self.strides):
            self.anchor = anchor
            self.num_anchor = len(anchor)
            prediction.append(self.get_preds_each_level(input, stride))
        prediction = torch.cat(prediction, 1)

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for image_i, image_pred in enumerate(prediction):
            score = (
                image_pred[:, 4]
                .unsqueeze(-1)
                .repeat(1, self.num_classes)
                .reshape(-1)
            )
            bbox = (
                image_pred[:, :4]
                .unsqueeze(-1)
                .repeat(1, 1, self.num_classes)
                .permute(0, 2, 1)
                .reshape(-1, 4)
            )
            class_conf = image_pred[:, 5 : 5 + self.num_classes].reshape(-1)
            class_pred = (
                torch.arange(self.num_classes, device=class_conf.device)
                .unsqueeze(0)
                .repeat(image_pred.shape[0], 1)
                .reshape(-1)
            )
            score = score * class_conf

            bboxes, scores, labels = nms(
                bbox,
                score,
                class_pred,
                self.nms_thresh,
                score_threshold=self.score_thresh,
                pre_nms_top_n=400,
                output_num=self.topK,
            )
            output[image_i] = torch.cat(
                (bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)), -1
            )

        return output

    def get_preds_each_level(self, input, stride):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = stride
        stride_w = stride
        scaled_anchors = [
            (a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchor
        ]
        input = (
            input.view(bs, self.num_anchor, self.num_classes + 5, in_h, in_w)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        x = torch.sigmoid(input[..., 0])
        y = torch.sigmoid(input[..., 1])
        w = input[..., 2]
        h = input[..., 3]
        conf = torch.sigmoid(input[..., 4])
        pred_cls = torch.sigmoid(input[..., 5:])

        if input.is_cuda:
            FloatTensor = torch.cuda.FloatTensor
            LongTensor = torch.cuda.LongTensor
        else:
            FloatTensor = torch.FloatTensor
            LongTensor = torch.LongTensor

        # Calculate offsets for each grid
        grid_x = (
            torch.linspace(0, in_w - 1, in_w)
            .repeat(in_w, 1)
            .repeat(bs * self.num_anchor, 1, 1)
            .view(x.shape)
            .type(FloatTensor)
        )
        grid_y = (
            torch.linspace(0, in_h - 1, in_h)
            .repeat(in_h, 1)
            .t()
            .repeat(bs * self.num_anchor, 1, 1)
            .view(y.shape)
            .type(FloatTensor)
        )

        # Calculate anchor w, h
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = (
            anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        )
        anchor_h = (
            anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        )

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(input[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        out = torch.cat(
            (
                pred_boxes.view(bs, -1, 4) * _scale,
                conf.view(bs, -1, 1),
                pred_cls.view(bs, -1, self.num_classes),
            ),
            -1,
        )
        return out.data


@OBJECT_REGISTRY.register
class YOLOV3HbirPostProcess(YOLOV3PostProcess):
    """
    The postprocess of YOLOv3 Hbir.

    Args:
        anchors: The anchors of yolov3.
        strides: A list of strides.
        num_classes: The num classes of class branch.
        score_thresh: Score thresh of postprocess before nms.
        nms_thresh: Nms thresh.
        topK: The output num of bboxes after postprocess.

    """

    def __init__(
        self,
        anchors: list,
        strides: list,
        num_classes: int,
        score_thresh: float = 0.01,
        nms_thresh: float = 0.45,
        topK: int = 200,
    ):
        super(YOLOV3HbirPostProcess, self).__init__(
            anchors=anchors,
            strides=strides,
            num_classes=num_classes,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            topK=topK,
        )

    def forward(self, inputs: Sequence[torch.Tensor]):
        inputs = _as_list(inputs)

        preds_format = []
        for outputtmp in inputs:
            tmp_result = []
            mask = ~(outputtmp[1] == 0).all(dim=1)
            tmp_result.append(outputtmp[0][mask])
            tmp_result.append(outputtmp[1][mask])
            preds_format.append(tmp_result)
        inputs = preds_format
        bs = 1
        FloatTensor = torch.FloatTensor
        layre_out = []
        for layer in range(len(self.strides)):
            stride_result = []
            stride = self.strides[layer]
            for anchor_id in range(len(self.anchors)):
                anchor = self.anchors[layer][anchor_id]
                branch = layer * 3 + anchor_id
                inputs_tmp = inputs[branch]
                x = torch.sigmoid(inputs_tmp[1][..., 0])
                y = torch.sigmoid(inputs_tmp[1][..., 1])
                w = inputs_tmp[1][..., 2]
                h = inputs_tmp[1][..., 3]
                conf = torch.sigmoid(inputs_tmp[1][..., 4])
                pred_cls = torch.sigmoid(inputs_tmp[1][..., 5:])
                grid = inputs_tmp[0]

                pred_boxes = FloatTensor(inputs_tmp[1][..., :4].shape)
                pred_boxes[..., 0] = x.data + grid[..., 1]
                pred_boxes[..., 1] = y.data + grid[..., 0]

                anchor_w = anchor[0] / stride
                anchor_h = anchor[1] / stride

                pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
                pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

                _scale = stride
                out = torch.cat(
                    (
                        pred_boxes.view(bs, -1, 4) * _scale,
                        conf.view(bs, -1, 1),
                        pred_cls.view(bs, -1, self.num_classes),
                    ),
                    -1,
                )
                stride_result.append(out)
            layre_out.append(torch.cat(stride_result, dim=1))
        prediction = torch.cat(layre_out, 1)

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for image_i, image_pred in enumerate(prediction):
            score = (
                image_pred[:, 4]
                .unsqueeze(-1)
                .repeat(1, self.num_classes)
                .reshape(-1)
            )
            bbox = (
                image_pred[:, :4]
                .unsqueeze(-1)
                .repeat(1, 1, self.num_classes)
                .permute(0, 2, 1)
                .reshape(-1, 4)
            )
            class_conf = image_pred[:, 5 : 5 + self.num_classes].reshape(-1)
            class_pred = (
                torch.arange(self.num_classes, device=class_conf.device)
                .unsqueeze(0)
                .repeat(image_pred.shape[0], 1)
                .reshape(-1)
            )
            score = score * class_conf

            bboxes, scores, labels = nms(
                bbox,
                score,
                class_pred,
                self.nms_thresh,
                score_threshold=self.score_thresh,
                pre_nms_top_n=400,
                output_num=self.topK,
            )
            output[image_i] = torch.cat(
                (bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)), -1
            )

        return output
