# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
from typing import Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt

from hat.core.box_np_ops import rotation_points_single_angle

__all__ = ["lidar_det_visualize"]

logger = logging.getLogger(__name__)


def plot_points(points: np.ndarray, reverse: bool = False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect("equal")
    if reverse:
        ax.plot(points[:, 1], -points[:, 0], "b.", markersize=1)
    else:
        ax.plot(-points[:, 1], points[:, 0], "b.", markersize=1)
    return ax


def xyzwhl2eight(centers):
    """Draw 3d bounding box in image.

    qs: (8,3) array of vertices for the 3d box in following order:
        7 -------- 6
       /|         /|
      4 -------- 5 .
      | |        | |
      . 3 -------- 2
      |/         |/
      0 -------- 1
    """
    corners = []
    for center in centers:
        x, y, z, w, h, p = center[:6]
        corner = np.array(
            [
                [
                    x + w / 2,
                    x + w / 2,
                    x - w / 2,
                    x - w / 2,
                    x + w / 2,
                    x + w / 2,
                    x - w / 2,
                    x - w / 2,
                ],
                [
                    y - h / 2,
                    y + h / 2,
                    y + h / 2,
                    y - h / 2,
                    y - h / 2,
                    y + h / 2,
                    y + h / 2,
                    y - h / 2,
                ],
                [
                    z - p / 2,
                    z - p / 2,
                    z - p / 2,
                    z - p / 2,
                    z + p / 2,
                    z + p / 2,
                    z + p / 2,
                    z + p / 2,
                ],
            ]
        )
        corner = corner.T
        corners.append(corner)
    return corners


def get_boxes(box3d_centers_in_lidar, reverse=False):
    # (N, 7) in (x, y, z, l, h, w, r)
    box3d_corners_in_lidar = xyzwhl2eight(box3d_centers_in_lidar)  # (N, 8, 3)
    box3d_corners_in_lidar_rot = []
    angles = []
    for i, box in enumerate(box3d_corners_in_lidar):  # (8, 3)
        x_mean, y_mean, z_mean = np.mean(box, axis=0)
        # coords relative to origin --> coords relative to box center
        box[:, 0] -= x_mean
        box[:, 1] -= y_mean
        box_rot = []
        if reverse:
            angle = box3d_centers_in_lidar[i, 6]
        else:
            angle = box3d_centers_in_lidar[i, -1]
        angles.append(angle)
        for corner in box:  # (3, )
            rot_corner = rotation_points_single_angle(corner, angle, axis=-1)
            rot_corner[0] = rot_corner[0] + x_mean
            rot_corner[1] = rot_corner[1] + y_mean
            box_rot.append(rot_corner)
        box3d_corners_in_lidar_rot.append(np.array(box_rot))

    return box3d_corners_in_lidar_rot, angles


def get_dir(box, angle, axis_rot, length=4):
    # plane on x,y axis
    box_xy = box[:4, :2]  # [[x,y], [x,y], [x,y], [x,y]]
    x0, y0 = np.mean(box_xy, axis=0)  # [x, y]
    # rotate because the points drew are rotate from xy to yx
    dx, dy = (
        -np.cos(angle + axis_rot) * length,
        -np.sin(angle + axis_rot) * length,
    )
    return x0, y0, dx, dy


def draw_box3d_on_lidar_bev(
    points: np.ndarray,
    bboxes: np.ndarray,
    angles: np.ndarray,
    thresh: float = 0.5,
    labels: np.ndarray = None,
    scores: np.ndarray = None,
    reverse: bool = False,
):
    """Visualize pointcloud and bounding boxes.

    Args:
        points: PointCloud, shape (M, N)
        bboxes: Bounding boxes.
        angles: Rotation angels.
        labels: Prediction labels.
        scores: Prediction scores.
        thresh: Display threshold if `scores` is provided. Defaults to 0.5.
        reverse: Whether to reverse coordinates.
    """

    # draw projected 3d box on lidar bev and show direction
    # direction is represented in (angle)
    if labels is not None and len(bboxes) != len(labels):
        raise ValueError(
            f"The length of labels and bboxes mismatch, {len(labels)} vs {len(bboxes)}"  # noqa E501
        )
    if scores is not None and len(bboxes) != len(scores):
        raise ValueError(
            f"The length of scores and bboxes mismatch, {len(scores)} vs {len(bboxes)}"  # noqa E501
        )

    ax = plot_points(points, reverse)

    for idx, (bbox3d, angle) in enumerate(zip(bboxes, angles)):
        box = bbox3d[:, :2]
        if scores is not None and scores[idx] < thresh:
            continue
        for k in range(4):
            i, j = k, (k + 1) % 4
            if reverse:
                ax.plot((box[i, 1], box[j, 1]), (-box[i, 0], -box[j, 0]), "g-")
            else:
                ax.plot((-box[i, 1], -box[j, 1]), (box[i, 0], box[j, 0]), "g-")
        if reverse:
            x0, y0, dx, dy = get_dir(box, angle, axis_rot=np.pi / 2)
        else:
            x0, y0, dx, dy = get_dir(box, angle, axis_rot=-np.pi / 2)
        if reverse:
            y0 = -y0
            x0 = -x0
        ax.arrow(
            -y0,
            x0,
            dy,
            dx,
            width=0.01,
            length_includes_head=True,
            head_width=0.25,
            head_length=1,
            fc="r",
            ec="r",
        )
        score = "{:.3f}".format(scores.flat[idx]) if scores is not None else ""
        if score:
            ax.text(
                -y0 - 2,
                x0 + 2,
                "{:s}".format(score),
                fontsize=5,
                color="red",
            )


def lidar_det_visualize(
    points: Union[torch.Tensor, np.ndarray],
    predictions: Tuple[Union[torch.Tensor, np.ndarray]],
    reverse: bool = False,
    score_thresh: float = 0.4,
    is_plot: bool = False,
    save_path: str = None,
):
    """Visualize method of lidar det result.

    Args:
        points: PointCloud.
        predictions: Predictions.
        reverse: Whether to reverse coordinates.
        score_thresh: Score thresh for filtering box in plot.
        is_plot: Whether to plot image.
    """

    if isinstance(predictions, dict):
        preds_box3d_lidar = predictions["bboxes"]
        preds_scores = predictions["scores"]
        preds_labels = predictions["labels"]
    else:
        (
            preds_box3d_lidar,
            preds_labels,
            preds_scores,
        ) = predictions

    preds_box3d_lidar = preds_box3d_lidar.cpu().numpy()
    if preds_labels is not None:
        preds_labels = preds_labels.cpu().numpy()
    if preds_scores is not None:
        preds_scores = preds_scores.cpu().numpy()

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    box3d_corners_in_lidar, angles = get_boxes(preds_box3d_lidar, reverse)
    draw_box3d_on_lidar_bev(
        points=points,
        bboxes=box3d_corners_in_lidar,
        angles=angles,
        scores=preds_scores,
        thresh=score_thresh,
        reverse=reverse,
    )

    if is_plot:
        plt.axis("off")
        plt.subplots_adjust(
            top=1, bottom=0, left=0, right=1, hspace=0, wspace=0
        )
        plt.margins(0, 0)
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            result_path = os.path.join(save_path, "lidar_pred.png")
            plt.savefig(result_path)
        else:
            plt.show()
