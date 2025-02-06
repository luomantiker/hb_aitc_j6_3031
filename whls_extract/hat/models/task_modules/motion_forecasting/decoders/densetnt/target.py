# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, Tuple

from torch import Tensor, nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["DensetntTarget"]


@OBJECT_REGISTRY.register
class DensetntTarget(nn.Module):
    """Generate densetnt targets."""

    def __init__(self):
        super(DensetntTarget, self).__init__()

    def forward(
        self, goals_preds: Tensor, traj_preds: Tensor, data: Dict
    ) -> Tuple[Tensor, Tensor]:
        """Generate Densetnt targets.

        Args:
            goals_preds: Predicted goals.
            traj_preds: Predicted trajectories.
            data: Data dictionary.

        Returns:
            Tuple containing the goals target and trajectory target.
        """
        goals_2d_labels = data["goals_2d_labels"]
        goals_2d_mask = data["goals_2d_mask"]

        goals_target = {
            "goals_preds": goals_preds,
            "goals_labels": goals_2d_labels,
            "goals_mask": goals_2d_mask,
        }

        traj_labels = data["traj_labels"]
        traj_target = {"traj_preds": traj_preds, "traj_labels": traj_labels}
        return goals_target, traj_target
