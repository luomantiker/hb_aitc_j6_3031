import torch
import torch.nn as nn
import torch.nn.functional as F

from hat.models.losses.laplace_loss import LaplaceNLLLoss
from hat.registry import OBJECT_REGISTRY
from .utils import wrap_angle

__all__ = ["QCNetOELoss"]


@OBJECT_REGISTRY.register
class QCNetOELoss(nn.Module):
    """
    QCNet loss module, combining regression and classification losses.

    Args:
        output_dim: The dimensionality of the output.
        num_historical_steps: The number of historical steps used as input
            to the model.
        num_future_steps: The number of future steps to be predicted by
            the model.
    """

    def __init__(
        self, output_dim: int, num_historical_steps: int, num_future_steps: int
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

    def cls_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: torch.Tensor,
        mask: torch.Tensor,
    ):
        nll_loss = LaplaceNLLLoss(eps=1e-6, reduction="none")
        nll = torch.cat(
            [
                nll_loss(
                    pred=pred[..., [i, target.size(-1) + i]],
                    target=target[..., [i]].unsqueeze(1),
                )
                for i in range(target.size(-1))
            ],
            dim=-1,
        )
        nll = (nll * mask.view(-1, 1, target.size(-2), 1)).sum(dim=(-2, -1))

        log_pi = F.log_softmax(prob, dim=-1)
        loss = -torch.logsumexp(log_pi - nll, dim=-1)
        return loss

    def reg_loss(self, pred: torch.Tensor, target: torch.Tensor):
        nll_loss = LaplaceNLLLoss(eps=1e-6, reduction="none")
        nll = torch.cat(
            [
                nll_loss(
                    pred=pred[..., [i, target.size(-1) + i]],
                    target=target[..., [i]],
                )
                for i in range(target.size(-1))
            ],
            dim=-1,
        )
        return nll

    def build_target(self, data: dict):
        origin = data["agent"]["position"][:, :, self.num_historical_steps - 1]
        theta = data["agent"]["heading"][:, :, self.num_historical_steps - 1]
        pos_ft = data["agent"]["position"][
            :, :, self.num_historical_steps :, :2
        ]

        B, A = origin.shape[:2]
        origin = origin.view(B * A, -1)
        pos_ft = pos_ft.view(B * A, -1, 2)
        theta = theta.view(B * A)
        cos = theta.cos()  # .view(B*A)
        sin = theta.sin()  # .view(B*A)
        rot_mat = torch.zeros(B * A, 2, 2).to(theta.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        data["agent"]["target"] = torch.zeros(
            B * A, self.num_future_steps, 4
        ).to(theta.device)
        data["agent"]["target"][..., :2] = torch.bmm(
            pos_ft - origin[..., :2].unsqueeze(1), rot_mat  # [B*A, FT, 2]
        )
        if data["agent"]["position"].size(2) == 3:
            pos3_ft = data["agent"]["position"][
                :, self.num_historical_steps :, 2
            ]
            pos3_ft = pos3_ft.view(B * A, -1)
            data["agent"]["target"][..., 2] = pos3_ft - origin[:, 2].unsqueeze(
                -1
            )
        data["agent"]["target"][..., 3] = wrap_angle(
            data["agent"]["heading"][:, :, self.num_historical_steps :].view(
                B * A, -1
            )
            - theta.unsqueeze(-1)
        )
        return data

    def forward(self, pred: dict, data: dict):
        data = self.build_target(data)
        reg_mask = data["agent"]["predict_mask"][
            :, :, self.num_historical_steps :
        ]
        B, A = reg_mask.shape[:2]
        reg_mask = reg_mask.view(B * A, -1)
        cls_mask = data["agent"]["predict_mask"][:, :, -1].view(-1)

        traj_propose = torch.cat(
            [
                pred["loc_propose_pos"][..., : self.output_dim] * 10.0,
                pred["scale_propose_pos"][..., : self.output_dim],
            ],
            dim=-1,
        )
        traj_refine = torch.cat(
            [
                pred["loc_refine_pos"][..., : self.output_dim] * 10.0,
                pred["scale_refine_pos"][..., : self.output_dim],
            ],
            dim=-1,
        )
        gt = torch.cat(
            [
                data["agent"]["target"][..., : self.output_dim],
                data["agent"]["target"][..., -1:],
            ],
            dim=-1,
        )
        pi = pred["pi"]

        l2_norm = torch.norm(
            traj_propose[..., : self.output_dim]
            - gt[..., : self.output_dim].unsqueeze(1),
            p=2,
            dim=-1,
        )
        l2_norm = (l2_norm * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[
            torch.arange(traj_propose.size(0)), best_mode
        ]
        traj_refine_best = traj_refine[
            torch.arange(traj_refine.size(0)), best_mode
        ]
        reg_loss_propose = (
            self.reg_loss(traj_propose_best, gt[..., : self.output_dim]).sum(
                dim=-1
            )
            * reg_mask
        )
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(
            dim=0
        ).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = (
            self.reg_loss(traj_refine_best, gt[..., : self.output_dim]).sum(
                dim=-1
            )
            * reg_mask
        )
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(
            dim=0
        ).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = (
            self.cls_loss(
                pred=traj_refine[:, :, -1:].detach(),
                target=gt[:, -1:, : self.output_dim],
                prob=pi,
                mask=reg_mask[:, -1:],
            )
            * cls_mask
        )
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        loss_dict = {
            "cls": cls_loss,
            "reg_propose": reg_loss_propose,
            "reg_refine": reg_loss_refine,
        }
        return loss_dict
