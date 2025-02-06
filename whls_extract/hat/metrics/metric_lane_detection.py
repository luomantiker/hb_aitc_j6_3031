# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from hat.registry import OBJECT_REGISTRY
from .metric import EvalMetric

__all__ = ["CulaneF1Score"]


@OBJECT_REGISTRY.register
class CulaneF1Score(EvalMetric):
    """Metric for Lane detection task, using for Culanedataset.

    This metric aligns with the official c++ implementation.

    Args:
        name: Metric name.
        iou_thresh: IOU overlap threshold for TP.
        img_shape: Image shape used when calculating iou.
        width: The width of the line.
        samples: Number of samples between two points.

    """

    def __init__(
        self,
        name: str = "CulaneF1Score",
        iou_thresh: float = 0.5,
        img_shape: Tuple[int, int, int] = (590, 1640, 1),
        width: int = 30,
        samples: int = 50,
    ):
        super(CulaneF1Score, self).__init__(name)

        self.iou_thresh = iou_thresh
        self.img_shape = img_shape

        self.width = width
        self.name = name
        self.samples = samples
        self._init_states()

    def _init_states(self):

        self.add_state(
            "tp",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fp",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fn",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def _draw_lane(self, lane):
        """Draw a line on the image."""
        img = np.zeros(self.img_shape, dtype=np.uint8)
        if len(lane) == 1:
            return img

        num_lane = len(lane)

        lane_array = np.around(lane).astype(np.int32)

        for i in range(num_lane - 1):
            x1 = lane_array[i][0]
            y1 = lane_array[i][1]
            x2 = lane_array[i + 1][0]
            y2 = lane_array[i + 1][1]
            cv2.line(img, (x1, y1), (x2, y2), color=(1,), thickness=self.width)
        return img

    def _compute_lane_cross_iou(self, pred, anno):
        """Calculate iou for each pred and gt."""

        pred = [self._draw_lane(lane) > 0 for lane in pred]
        anno = [self._draw_lane(lane) > 0 for lane in anno]

        ious = np.zeros((len(pred), len(anno)))
        for i, x in enumerate(pred):
            for j, y in enumerate(anno):
                ious[i, j] = (x & y).sum() / (x | y).sum()

        return ious

    def _interp_line(self, points, times=50):
        """Fit the curve and sample times points."""

        num_points = points.shape[0]
        if num_points == 1:
            return points
        elif num_points == 2:
            result = []
            x1 = points[0][0]
            y1 = points[0][1]
            x2 = points[1][0]
            y2 = points[1][1]

            for i in range(times + 1):
                xi = x1 + ((x2 - x1) * i) / times
                yi = y1 + ((y2 - y1) * i) / times
                result.append(np.array([xi, yi]))
            return np.array(result)
        else:

            points_offset = points[1:] - points[:-1]

            Mx = np.array([-1] * num_points, dtype=np.float32)
            My = np.array([-1] * num_points, dtype=np.float32)

            h = np.sqrt(np.power(points_offset, 2).sum(axis=1))

            a_mat = h[:-1].copy()
            c_mat = h[1:].copy()
            b_mat = 2 * (a_mat + c_mat)
            d_mat = 6 * (
                points_offset[1:] / h[1:, None]
                - points_offset[:-1] / h[:-1, None]
            )
            Dx = d_mat[:, 0]
            Dy = d_mat[:, 1]

            c_mat[0] = c_mat[0] / b_mat[0]
            Dx[0] = Dx[0] / b_mat[0]
            Dy[0] = Dy[0] / b_mat[0]

            for i in range(1, num_points - 2):
                tmp = b_mat[i] - a_mat[i] * c_mat[i - 1]
                c_mat[i] = c_mat[i] / tmp
                Dx[i] = (Dx[i] - a_mat[i] * Dx[i - 1]) / tmp
                Dy[i] = (Dy[i] - a_mat[i] * Dy[i - 1]) / tmp

            Mx[-2] = Dx[-1]
            My[-2] = Dy[-1]

            for i in range(num_points - 4, -1, -1):
                Mx[i + 1] = Dx[i] - c_mat[i] * Mx[i + 2]
                My[i + 1] = Dy[i] - c_mat[i] * My[i + 2]

            Mx[0] = 0
            Mx[num_points - 1] = 0
            My[0] = 0
            My[num_points - 1] = 0

            a_x = points[:-1, 0]
            b_x = points_offset[:, 0] / h - (2 * h * Mx[:-1] + h * Mx[1:]) / 6
            c_x = Mx[:-1] / 2
            d_x = (Mx[1:] - Mx[:-1]) / (6 * h)

            a_y = points[:-1, 1]
            b_y = points_offset[:, 1] / h - (2 * h * My[:-1] + h * My[1:]) / 6
            c_y = My[:-1] / 2
            d_y = (My[1:] - My[:-1]) / (6 * h)

            all_deltes = []
            for i in range(num_points - 1):
                deltas = np.arange(0, h[i], h[i] / times)
                if deltas.shape[0] > times:
                    deltas = deltas[:times]
                all_deltes.append(deltas)
            all_deltes = np.array(all_deltes)

            a_x = a_x[:, None]
            b_x = b_x[:, None]
            c_x = c_x[:, None]
            d_x = d_x[:, None]

            a_y = a_y[:, None]
            b_y = b_y[:, None]
            c_y = c_y[:, None]
            d_y = d_y[:, None]

            x1 = (
                a_x
                + b_x * all_deltes
                + c_x * np.power(all_deltes, 2)
                + d_x * np.power(all_deltes, 3)
            )
            y1 = (
                a_y
                + b_y * all_deltes
                + c_y * np.power(all_deltes, 2)
                + d_y * np.power(all_deltes, 3)
            )

            result_interp = np.stack([x1, y1], axis=-1).reshape(-1, 2)
            result = np.concatenate([result_interp, points[None, -1]], axis=0)
        return result

    def update(
        self,
        annos: List[List[np.ndarray]],
        preds: List[List[np.ndarray]],
    ):
        for anno, pred in zip(annos, preds):
            if len(anno) == 0:
                self.fp += len(pred)
            elif len(pred) == 0:
                self.fn += len(anno)
            else:
                interp_pred = []
                interp_anno = []
                for pred_lane in pred:
                    interp_pred.append(
                        self._interp_line(pred_lane, times=self.samples)
                    )
                for anno_lane in anno:
                    interp_anno.append(
                        self._interp_line(anno_lane, times=self.samples)
                    )

                ious = self._compute_lane_cross_iou(interp_pred, interp_anno)
                row_ind, col_ind = linear_sum_assignment(1 - ious)
                tp = int((ious[row_ind, col_ind] > self.iou_thresh).sum())
                fp = len(pred) - tp
                fn = len(anno) - tp
                self.tp += tp
                self.fp += fp
                self.fn += fn

    def compute(self):

        if self.tp == 0:
            return 0.0
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        f1 = 2 * precision * recall / (precision + recall)

        return f1.item()
