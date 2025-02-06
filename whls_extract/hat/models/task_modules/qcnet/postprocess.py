import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY

__all__ = ["QCNetOEPostprocess"]


@OBJECT_REGISTRY.register
class QCNetOEPostprocess(nn.Module):
    """
    Post-processing module for QCNet output.

    Args:
        num_historical_steps: Number of historical steps used in the input.
        output_dim: Dimension of the output predictions. Default is 2.
        deploy: Flag to indicate deployment mode. Default is False.

    """

    def __init__(
        self,
        num_historical_steps: int,
        output_dim: int = 2,
        deploy: bool = False,
    ):
        super().__init__()
        self.num_historical_steps = num_historical_steps
        self.output_dim = output_dim
        self.deploy = deploy

    def forward(self, pred: dict, data: dict):

        traj_refine = pred["loc_refine_pos"][..., : self.output_dim] * 10.0
        pi = pred["pi"]

        origin_eval = data["agent"]["position"][
            ..., self.num_historical_steps - 1, :
        ]
        B, A = origin_eval.shape[:2]

        theta_eval = data["agent"]["heading"][
            ..., self.num_historical_steps - 1
        ]
        eval_mask = data["agent"]["category"] == 3
        eval_mask = eval_mask.view(B * A)
        origin_eval = origin_eval.view(B * A, 2)[eval_mask]
        theta_eval = theta_eval.view(B * A)[eval_mask]

        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=pi.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(
            traj_refine[eval_mask, :, :, :2], rot_mat.unsqueeze(1)
        ) + origin_eval[:, :2].reshape(-1, 1, 1, 2)

        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        if not self.deploy:
            reg_mask = data["agent"]["predict_mask"][
                :, :, self.num_historical_steps :
            ]
            reg_mask = reg_mask.view(B * A, -1)
            if data["agent"]["position"].shape[2] > self.num_historical_steps:
                pos_ft = data["agent"]["position"][
                    :, :, self.num_historical_steps :, :2
                ]

                gt = pos_ft.view(B * A, -1, 2)[eval_mask]
            else:
                gt = None
            output = {
                "pred": traj_eval[..., : self.output_dim],
                "target": gt,
                "prob": pi_eval,
                "valid_mask": reg_mask[eval_mask],
            }
        else:
            output = {
                "pred": traj_eval[..., : self.output_dim],
                "prob": pi_eval,
            }
        return output
