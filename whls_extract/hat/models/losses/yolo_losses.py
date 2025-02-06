# Copyright (c) Horizon Robotics. All rights reserved.

import math

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from hat.core.box_utils import bbox_overlaps
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list

__all__ = ["YOLOV3Loss"]


@OBJECT_REGISTRY.register
class YOLOV3Loss(nn.Module):
    """
    The loss module of YOLOv3.

    Args:
        num_classes: Num classes of class branch.
        anchors: The anchors of YOLOv3.
        strides: The strides of feature maps.
        ignore_thresh: Ignore thresh of target.
        loss_xy: Losses of xy.
        loss_wh: Losses of wh.
        loss_conf: Losses of conf.
        loss_cls: Losses of cls.
        lambda_loss: The list of weighted losses.
    """

    def __init__(
        self,
        num_classes: int,
        anchors: list,
        strides: list,
        ignore_thresh: float,
        loss_xy: dict,
        loss_wh: dict,
        loss_conf: dict,
        loss_cls: dict,
        lambda_loss: list,
    ):
        super(YOLOV3Loss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.strides = strides
        self.ignore_thresh = ignore_thresh
        self.loss_xy = loss_xy
        self.loss_wh = loss_wh
        self.loss_conf = loss_conf
        self.loss_cls = loss_cls
        self.lambda_loss = lambda_loss

    @autocast(enabled=False)
    def forward(self, input, target=None):
        xs = _as_list(input)
        losses = 0
        for x, anchors, stride in zip(xs, self.anchors, self.strides):
            self.num_anchors = len(anchors)
            loss = self.forward_each_level(x, target, anchors, stride)
            losses += loss.sum()
        return losses

    def forward_each_level(self, input, target, anchors, stride):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = stride
        stride_w = stride
        num_anchors = len(anchors)
        scaled_anchors = [
            (a_w / stride_w, a_h / stride_h) for a_w, a_h in anchors
        ]

        input = input.float()
        prediction = (
            input.view(bs, num_anchors, 5 + self.num_classes, in_h, in_w)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        if x.is_cuda:
            FloatTensor = torch.cuda.FloatTensor
            LongTensor = torch.cuda.LongTensor
        else:
            FloatTensor = torch.FloatTensor
            LongTensor = torch.LongTensor

        grid_x = (
            torch.linspace(0, in_w - 1, in_w)
            .repeat(in_w, 1)
            .repeat(bs * num_anchors, 1, 1)
            .view(x.shape)
            .type(FloatTensor)
        )
        grid_y = (
            torch.linspace(0, in_h - 1, in_h)
            .repeat(in_h, 1)
            .t()
            .repeat(bs * num_anchors, 1, 1)
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
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        box_corner = pred_boxes.new(pred_boxes.shape)
        # bs * num_anchors * in_w * in_h
        box_corner[..., 0] = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
        box_corner[..., 1] = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
        box_corner[..., 2] = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
        box_corner[..., 3] = pred_boxes[..., 1] + pred_boxes[..., 3] / 2
        pred_boxes[..., :4] = box_corner[..., :4]
        pred_boxes = pred_boxes.reshape(bs, -1, 4)

        (mask, noobj_mask, tx, ty, tw, th, tconf, tcls,) = self._get_target(
            target,
            scaled_anchors,
            bs,
            in_w,
            in_h,
            stride,
            self.ignore_thresh,
            pred_boxes,
        )
        device = input.device
        mask, noobj_mask = mask.to(device=device), noobj_mask.to(device=device)
        tx, ty, tw, th = (
            tx.to(device=device),
            ty.to(device=device),
            tw.to(device=device),
            th.to(device=device),
        )
        tconf, tcls = tconf.to(device=device), tcls.to(device=device)

        loss_x = self.loss_xy(x, tx)[mask > 0]
        loss_y = self.loss_xy(y, ty)[mask > 0]
        loss_w = self.loss_wh(w, tw)[mask > 0]
        loss_h = self.loss_wh(h, th)[mask > 0]
        loss_conf = self.loss_conf(
            conf[mask == 1], tconf[mask == 1]
        ) + self.loss_conf(conf[mask == 0], tconf[mask == 0])
        loss_cls = self.loss_cls(pred_cls[mask == 1], tcls[mask == 1])

        loss_x, loss_y, loss_w, loss_h, loss_conf, loss_cls = (
            loss_x.sum(),
            loss_y.sum(),
            loss_w.sum(),
            loss_h.sum(),
            loss_conf.sum(),
            loss_cls.sum(),
        )

        lambda_xy, lambda_wh, lambda_conf, lambda_cls = self.lambda_loss
        loss = (
            loss_x * lambda_xy
            + loss_y * lambda_xy
            + loss_w * lambda_wh
            + loss_h * lambda_wh
            + loss_conf * lambda_conf
            + loss_cls * lambda_cls
        )
        return loss / bs

    def _get_target(
        self, target, anchors, bs, in_w, in_h, stride, ignore_threshold, preds
    ):
        gt_bboxes, gt_labels = target

        mask = torch.zeros(
            bs, self.num_anchors, in_h, in_w, requires_grad=False
        )
        noobj_mask = torch.ones(
            bs, self.num_anchors, in_h, in_w, requires_grad=False
        )
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(
            bs, self.num_anchors, in_h, in_w, requires_grad=False
        )
        tcls = torch.zeros(
            bs,
            self.num_anchors,
            in_h,
            in_w,
            self.num_classes,
            requires_grad=False,
        )

        for b in range(bs):
            for t in range(gt_bboxes[b].shape[0]):
                if (gt_bboxes[b][t] != -1).sum() == 0:
                    continue
                x1 = gt_bboxes[b][t, 0]
                y1 = gt_bboxes[b][t, 1]
                x2 = gt_bboxes[b][t, 2]
                y2 = gt_bboxes[b][t, 3]
                w = x2 - x1
                h = y2 - y1

                gx = float((x2 + x1) / 2 / stride)
                gy = float((y2 + y1) / 2 / stride)
                gw = float(w / stride)
                gh = float(h / stride)
                gi = int(gx)
                gj = int(gy)
                gt_box = torch.FloatTensor(
                    np.array([-0.5 * gw, -0.5 * gh, 0.5 * gw, 0.5 * gh])
                ).unsqueeze(0)
                anchor_shapes = np.concatenate(
                    (np.zeros((self.num_anchors, 2)), np.array(anchors)), 1
                )
                # anchor from xywh -> x1y1x2y2
                target_anchors = np.zeros_like(anchor_shapes)
                target_anchors[:, 0] = (
                    anchor_shapes[:, 0] - anchor_shapes[:, 2] * 0.5
                )  # yapf:disable
                target_anchors[:, 1] = (
                    anchor_shapes[:, 1] - anchor_shapes[:, 3] * 0.5
                )  # yapf:disable
                target_anchors[:, 2] = (
                    anchor_shapes[:, 0] + anchor_shapes[:, 2] * 0.5
                )  # yapf:disable
                target_anchors[:, 3] = (
                    anchor_shapes[:, 1] + anchor_shapes[:, 3] * 0.5
                )  # yapf:disable

                target_box = np.array(
                    [
                        [
                            float(x1 / stride),
                            float(y1 / stride),
                            float(x2 / stride),
                            float(y2 / stride),
                        ]
                    ]
                )
                dynamic_ious = bbox_overlaps(
                    preds[b].cpu().numpy(), target_box
                ).reshape(self.num_anchors, in_h, in_w)
                mask[b, dynamic_ious > ignore_threshold] = -1

                anch_ious = bbox_overlaps(gt_box.numpy(), target_anchors)[0]
                best_n = np.argmax(anch_ious)

                mask[b, best_n, gj, gi] = 1
                noobj_mask[b, best_n, gj, gi] = 0
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                tw[b, best_n, gj, gi] = math.log(
                    gw / anchors[best_n][0] + 1e-16
                )
                th[b, best_n, gj, gi] = math.log(
                    gh / anchors[best_n][1] + 1e-16
                )
                tconf[b, best_n, gj, gi] = 1
                tcls[b, best_n, gj, gi, int(gt_labels[b][t])] = 1.0
        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls
