# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["DensetntPostprocess"]


@OBJECT_REGISTRY.register
class DensetntPostprocess(nn.Module):
    """postprocess for densetnt.

    Args:
        threshold: threshold for nms.
        pred_steps: steps for traj pred.
        mode_num: number of mode.
    """

    def __init__(self, threshold=2.0, pred_steps=30, mode_num=6):
        super(DensetntPostprocess, self).__init__()
        self.threshold = threshold
        self.mode_num = mode_num
        self.pred_steps = pred_steps

    def select_goals_by_NMS(
        self, goals_scores: Tensor, traj_preds: Tensor, pred_goals: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Perform non-maximum suppression on predicted goals.

        Args:
            goals_scores: Predicted goals scores.
            traj_preds: Predicted trajectories.
            pred_goals: Predicted goals.

        Returns:
            Tuple containing the selected predicted trajectories and scores.
        """

        def get_dis_point_2_points(point, points):
            return np.sqrt(
                np.square(points[:, 0] - point[0])
                + np.square(points[:, 1] - point[1])
            )

        def in_predict(pred_goals, point, threshold):
            return (
                np.min(get_dis_point_2_points(point, pred_goals)) < threshold
            )

        pred_trajs = []
        scores = []
        for pred_scores, traj_pred, goals_2d in zip(
            goals_scores, traj_preds, pred_goals
        ):
            pred_scores = pred_scores.squeeze()
            _, argsort = torch.sort(pred_scores, descending=True)
            goals_2d = goals_2d.squeeze().permute(1, 0).contiguous()
            goals_2d = goals_2d[argsort].cpu().numpy()
            traj_pred = (
                traj_pred.squeeze()
                .permute(1, 0)
                .view(-1, self.pred_steps, 2)
                .contiguous()
            )
            traj_pred = traj_pred[argsort]
            result_goal = []
            result_traj = []
            result_score = []
            for i in range(len(goals_2d)):
                if len(result_goal) > 0 and in_predict(
                    np.array(result_goal), goals_2d[i], self.threshold
                ):
                    continue
                else:
                    result_goal.append(goals_2d[i])
                    result_traj.append(traj_pred[i])
                    result_score.append(pred_scores[i])
                    if len(result_goal) == self.mode_num:
                        break

            while len(result_goal) < self.mode_num:
                i = 0
                result_goal.append(goals_2d[i])
                result_traj.append(traj_pred[i])
                result_score.append(pred_scores[i])
            pred_trajs.append(result_traj)
            scores.append(result_score)
        return pred_trajs, scores

    def forward(
        self,
        goals_scores: Tensor,
        traj_preds: Tensor,
        pred_goals: Tensor,
        data: Dict,
    ) -> Tuple[Tensor, Tensor]:
        """Perform forward pass.

        Args:
            goals_scores: Goals scores.
            traj_preds: Trajectory predictions.
            pred_goals: Predicted goals.
            data: Data dictionary.

        Returns:
            Tuple containing the predicted trajectories and scores.
        """
        return self.select_goals_by_NMS(goals_scores, traj_preds, pred_goals)
