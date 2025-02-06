# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os

import numpy as np
from matplotlib import pyplot as plt

from hat.registry import OBJECT_REGISTRY

__all__ = ["ArgoverseViz"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ArgoverseViz(object):
    """
    The visiualize method of Argoverse result.

    Args:
        traj_scale: Scale for traj input feature.
        is_plot: Whether to plot image.

    """

    def __init__(self, traj_scale=50, is_plot=True):
        self.is_plot = is_plot
        self.traj_scale = traj_scale

    def draw_traj(self, traj, color):
        x = traj[:, 0]
        y = traj[:, 1]
        plt.plot(x, y, color=color, marker="^", linewidth=24, markersize=64)

    def draw_pred(self, traj, color):
        x = traj[:, 0]
        y = traj[:, 1]
        plt.plot(x, y, color=color, marker="o", linewidth=24, markersize=64)

    def draw_lane(self, lane, color):
        x = lane[:, 0]
        y = lane[:, 1]
        plt.plot(x, y, color=color, marker=".", linewidth=24, markersize=64)

    def _filter(self, trajs, mask):
        valid = mask == 1
        return trajs[valid]

    def __call__(
        self,
        traj_feat_mask,
        traj_feat,
        traj_mask,
        lane_feat,
        lane_mask,
        labels,
        preds=None,
        save_path=None,
    ):
        plt.figure(figsize=(128, 128))
        valid = lane_mask == 0
        lane_feat = lane_feat[valid]

        lanes = np.concatenate(
            [lane_feat[:, :, :2], lane_feat[:, -2:-1, 2:4]], axis=1
        )
        for lane in lanes:
            self.draw_lane(lane, "brown")

        valid = traj_mask == 0
        traj_feat = traj_feat[valid]
        trajs = np.concatenate(
            [traj_feat[:, :, :2], traj_feat[:, -2:-1, 2:4]], axis=1
        )
        trajs = trajs * self.traj_scale

        agent_traj = self._filter(trajs[0], traj_feat_mask[0])

        av_traj = self._filter(trajs[1], traj_feat_mask[1])
        self.draw_traj(agent_traj, "blue")
        self.draw_traj(av_traj, "green")

        other_trajs = trajs[2:]
        for traj, mask in zip(other_trajs, traj_feat_mask[2:]):
            traj = self._filter(traj, mask)
            self.draw_traj(traj, "yellow")
        self.draw_traj(labels, "purple")
        if preds is not None:
            self.draw_pred(preds, "red")

        plt.grid(False)
        if self.is_plot:
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                result_path = os.path.join(save_path, "traj_pred.png")
                plt.savefig(result_path)
            else:
                plt.show()
