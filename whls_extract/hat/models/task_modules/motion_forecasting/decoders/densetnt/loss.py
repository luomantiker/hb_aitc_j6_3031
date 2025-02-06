# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict

import torch.nn.functional as F
from torch import Tensor, nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["DensetntLoss"]


@OBJECT_REGISTRY.register
class DensetntLoss(nn.Module):
    """Generate Densetnt loss."""

    def __init__(self):
        super(DensetntLoss, self).__init__()

    def _goals_loss(self, goals_target: Tensor) -> Tensor:
        goals_preds = goals_target["goals_preds"]
        goals_labels = goals_target["goals_labels"]

        log_pred = F.log_softmax(goals_preds, dim=-1).view(
            goals_preds.shape[0], -1
        )
        return F.nll_loss(log_pred, goals_labels, reduction="mean")

    def _traj_loss(self, traj_target: Tensor) -> Tensor:
        preds = traj_target["traj_preds"].squeeze(-1).squeeze(-1)
        bs = preds.shape[0]
        preds = preds.view(bs, -1, 2).contiguous()
        targets = traj_target["traj_labels"]
        return F.smooth_l1_loss(preds, targets)

    def forward(self, goals_target: Tensor, traj_target: Tensor) -> Dict:
        """Compute the loss.

        Args:
            goals_target: Goals target containing goals_preds and goals_labels.
            traj_target: Trajectory target containing traj_preds
                         and traj_labels.

        Returns:
            Dictionary containing the goals_loss and traj_loss.
        """
        goals_loss = self._goals_loss(goals_target)
        traj_loss = self._traj_loss(traj_target)

        return {"goals_loss": goals_loss, "traj_loss": traj_loss}
