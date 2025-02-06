from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from .utils import wrap_angle

__all__ = ["QCNetOEPreprocess"]


@OBJECT_REGISTRY.register
class QCNetOEPreprocess(nn.Module):
    """Preprocess Module for OE version QCNet deploy.

    Args:
        input_dim: Dimension of the input data.
        num_historical_steps: Number of historical time steps.
        time_span: Time span for the model.
        num_t2m_steps: Number of time steps for the decoder's cross attention
            along the time axis.
        num_agent_layers: Number of layers for the agent module.
        num_pl2a: Number of polygons for map polygon to agent cross attention
                (works only when save_memory=True).
        num_a2a: Number of agents for agent to agent cross attention.
                (works only when save_memory=True).
        agent_num: Number of agents in the dataset.
        pl_num: Number of polygons for map polygon to agent cross attention.
        pt_num: Number of points for map polygon to agent cross attention.
        stream: Whether using stream mode for inference.
            Only works when deploy=False.
        save_memory: whether select agent and map neighbors to save memory,
            currently only works for training.
        deploy: Flag to indicate deployment mode. Default is False.
        reuse_agent_rembs: Flag to reuse agent rembs for the decoder.
        quant_infer_cold_start:Indicates whether to use cold start
            for streaming inference.
    """

    def __init__(
        self,
        input_dim: int,
        num_historical_steps: int,
        time_span: int,
        num_t2m_steps: int,
        num_agent_layers: int,
        num_pl2a: int = 32,
        num_a2a: int = 36,
        num_pl2pl: int = -1,
        agent_num: Optional[int] = None,
        pl_num: Optional[int] = None,
        pt_num: Optional[int] = None,
        mask_pl2pl_type: str = "diagonal_valid",
        stream: bool = False,
        save_memory: bool = False,
        reuse_agent_rembs: bool = True,
        deploy: bool = False,
        quant_infer_cold_start: bool = False,
    ):
        super().__init__()
        self.time_span = time_span
        self.input_dim = input_dim
        self.A = agent_num
        self.ht = num_historical_steps
        self.num_t2m_steps = num_t2m_steps
        self.pl_num = pl_num
        self.pt_num = pt_num
        self.num_pl2pl = num_pl2pl
        self.num_pl2a = num_pl2a
        self.num_a2a = num_a2a
        self.num_agent_layers = num_agent_layers
        self.save_memory = save_memory
        self.stream = stream
        self.reuse_agent_rembs = reuse_agent_rembs
        self.deploy = deploy
        self.mask_pl2pl_type = mask_pl2pl_type
        self.quant_infer_cold_start = quant_infer_cold_start

    def build_map_r_inputs(self, data: dict):
        pos_pt = data["map_point"]["position"] / 10.0
        orient_pt = data["map_point"]["orientation"]
        pos_pl = data["map_polygon"]["position"] / 10.0
        orient_pl = data["map_polygon"]["orientation"]

        rel_pos_pt2pl = pos_pt - pos_pl[:, :, None]  # [B, pl, pt, 2]
        rel_orient_pt2pl = wrap_angle(
            orient_pt - orient_pl[:, :, None]
        )  # [B, pl, pt]
        theta_pt2pl = torch.atan2(rel_pos_pt2pl[..., 1], rel_pos_pt2pl[..., 0])
        angle_pt2pl = wrap_angle(
            theta_pt2pl - orient_pl[:, :, None],  # [B, pl, pt] - B, pl]
        )  # [B, pl, pt]
        dist_pt2pl = torch.norm(rel_pos_pt2pl[..., :2], dim=-1, p=2)
        r_pt2pl = [
            dist_pt2pl.unsqueeze(1),  # [B, 1, pl, pt]
            angle_pt2pl.unsqueeze(1),
            rel_orient_pt2pl.unsqueeze(1),
        ]

        B, pl_N = pos_pl.shape[:2]  # [B, pl, 2]
        # mask_pl2pl
        valid_pl = data["map_polygon"]["valid_mask"]  # [B, pl]

        if self.mask_pl2pl_type == "diagonal":
            mask_pl2pl = (
                (torch.ones([B, pl_N, pl_N]) - torch.eye(pl_N).unsqueeze(0))
                .to(pos_pl.device)
                .bool()
            )
        elif self.mask_pl2pl_type == "valid":
            mask_pl2pl = (
                valid_pl[:, :, None] & valid_pl[:, None, :]
            )  # [B, pl, pl]
        elif self.mask_pl2pl_type == "diagonal_valid":
            valid_mask_pl2pl = valid_pl[:, :, None] & valid_pl[:, None, :]
            dia_mask_pl2pl = (
                (torch.ones([B, pl_N, pl_N]) - torch.eye(pl_N).unsqueeze(0))
                .to(pos_pl.device)
                .bool()
            )
            mask_pl2pl = valid_mask_pl2pl & dia_mask_pl2pl

        rel_pos_pl2pl = (
            pos_pl[:, None, :] - pos_pl[:, :, None]
        )  # #[B, pl, pl, 2]
        dist_pl2pl = torch.norm(rel_pos_pl2pl[..., :2], dim=-1, p=2)

        B, pl_N = pos_pl.shape[:2]  # [B, pl, 2]
        pl_K = self.num_pl2pl

        if pl_K > 0 and pl_N > pl_K and self.save_memory:
            # mask_dist_pl2pl= dist_pl2pl.masked_fill(~mask_pl2pl, 1e2)

            # topk_dist_pl2pl, pl2pl_near_idx = torch.topk(
            #    mask_dist_pl2pl,
            #    k=pl_K,
            #    dim=-1,
            #    largest=False,
            # )   # [B, pl_N, pl_K]

            pl2pl_near_idx = (
                (torch.rand([B, pl_N, pl_K]) * pl_K)
                .long()
                .to(dist_pl2pl.device)
            )

            topk_dist_pl2pl = dist_pl2pl.gather(
                -1, pl2pl_near_idx
            )  # [B, pl_N, pl_K]

            valid_pl2pl = mask_pl2pl.gather(-1, pl2pl_near_idx)
            dist_pl2pl = topk_dist_pl2pl.masked_fill(~valid_pl2pl, 0)
            # dist_pl2pl = dist_pl2pl.gather(-1, pl2pl_near_idx)
            rel_orient_pl2pl = wrap_angle(
                orient_pl[:, :, None] - orient_pl[:, None, :]
            ).gather(-1, pl2pl_near_idx)
            rel_pos_pl2pl = rel_pos_pl2pl.gather(
                2, pl2pl_near_idx.unsqueeze(-1).repeat(1, 1, 1, 2)
            )  # [B, pl, pl, 2]-->[B, pl, pl_K, 2]
            theta_pl2pl = torch.atan2(
                rel_pos_pl2pl[..., 1], rel_pos_pl2pl[..., 0]
            )  # [B, pl, pl_K]
            angle_pl2pl = wrap_angle(theta_pl2pl - orient_pl[:, :, None])
            mask_pl2pl = mask_pl2pl.gather(-1, pl2pl_near_idx)  # [B, pl_K]
            type_pl2pl = data["type_pl2pl"]  # [B, pl_N, pl_K]
            data["type_pl2pl"] = type_pl2pl.gather(
                -1, pl2pl_near_idx
            )  # [B, pl_N, pl_K]

        else:
            rel_orient_pl2pl = wrap_angle(
                orient_pl[:, :, None] - orient_pl[:, None, :]
            )
            theta_pl2pl = torch.atan2(
                rel_pos_pl2pl[..., 1], rel_pos_pl2pl[..., 0]
            )
            angle_pl2pl = wrap_angle(theta_pl2pl - orient_pl[:, :, None])
            dist_pl2pl = torch.norm(rel_pos_pl2pl[..., :2], dim=-1, p=2)
            pl2pl_near_idx = None

        r_pl2pl = [
            dist_pl2pl.unsqueeze(1),
            angle_pl2pl.unsqueeze(1),
            rel_orient_pl2pl.unsqueeze(1),
        ]
        if pl2pl_near_idx is not None:
            data["pl2pl_near_idx"] = pl2pl_near_idx

        return {
            "r_pl2pl": r_pl2pl,
            "r_pt2pl": r_pt2pl,
            "mask_pl2pl": mask_pl2pl,
        }

    def build_agent_his_r_inputs(self, data: dict, ht: int, qt: int):
        B, A = data["agent"]["position"].shape[:2]
        ST = self.time_span
        bt = ht - qt - ST  # 此函数用的时间段HT=10-6-2=2 9-5-2=2

        pos_pl = data["map_polygon"]["position"] / 10.0
        orient_pl = data["map_polygon"]["orientation"]
        pos_a = (
            data["agent"]["position"][:, :, bt:ht, :2] / 10.0
        )  # [B, A, HT, 2]
        head_a = data["agent"]["heading"][
            :, :, bt:ht
        ].contiguous()  # [B, A, HT]
        vel = (
            data["agent"]["velocity"][
                :, :, bt:ht, : self.input_dim
            ].contiguous()
            / 10.0
        )
        dist_vel = torch.norm(vel, p=2, dim=-1)

        motion_vector_a = torch.cat(
            [
                torch.zeros(B, A, 1, self.input_dim).to(pos_a.device),
                pos_a[:, :, 1:] - pos_a[:, :, :-1],
            ],
            dim=2,
        )  # [B, A, HT, 2]

        # [B, A, HT, D]
        theta_motion_a = torch.atan2(
            motion_vector_a[..., 1], motion_vector_a[..., 0]
        )
        angle_motion = wrap_angle(theta_motion_a - head_a)  # [B, A, HT]
        theta_vel = torch.atan2(vel[..., 1], vel[..., 0])
        angle_vel = wrap_angle(theta_vel - head_a)  # [B, A, HT]
        dist_motion_a = torch.norm(motion_vector_a[..., :2], dim=-1, p=2)
        x_a = [
            dist_motion_a.unsqueeze(1),
            angle_motion.unsqueeze(1),
            dist_vel.unsqueeze(1),
            angle_vel.unsqueeze(1),
        ]
        pos_t_key = torch.cat(
            [
                pos_a[:, :, ST - i : ST + qt - i].unsqueeze(3)
                for i in range(ST, 0, -1)
            ],
            dim=3,
        ).contiguous()  # [B, A, qt, 2, 2]

        head_t_key = torch.cat(
            [
                head_a[:, :, ST - i : ST + qt - i].unsqueeze(3)
                for i in range(ST, 0, -1)
            ],
            dim=3,
        ).contiguous()

        # [B, D, A, HT]
        pos_a = pos_a[:, :, ST : self.ht, :2].contiguous()  # [B, A, HT, 2]
        head_a = head_a[:, :, ST : self.ht].contiguous()  # [B, A, HT]
        vel = vel[:, :, ST : self.ht, : self.input_dim].contiguous()

        rel_pos_t = pos_a[:, :, :, None, :] - pos_t_key
        # [B, A, QT, ST, 2]
        rel_head_t = wrap_angle(head_a[:, :, :, None] - head_t_key)
        # [B, A, QT, ST,]
        theta_pos_t = torch.atan2(rel_pos_t[..., 1], rel_pos_t[..., 0])
        angle_agent_t = wrap_angle(
            theta_pos_t - head_a[:, :, :, None]
        )  # [B, A, HT, HT]
        diff_t = (
            torch.arange(ST)
            .reshape(1, 1, 1, ST)
            .repeat(B * A, 1, qt, 1)
            .to(pos_a.device)
        )

        dist_pos_t = torch.norm(rel_pos_t[..., :2], dim=-1, p=2)
        r_t = [
            dist_pos_t.view(B * A, 1, qt, ST),
            angle_agent_t.view(B * A, 1, qt, ST),
            rel_head_t.view(B * A, 1, qt, ST),
            diff_t.float().view(B * A, 1, qt, ST),
        ]

        pl_N = pos_pl.shape[1]
        rel_pos_pl2a = pos_pl[:, None, None, :, :] - pos_a[:, :, :, None, :]
        agent_valid = data["agent"]["valid_mask"][:, :, bt:ht][
            :, :, ST : self.ht
        ]  # [B, A, 5]
        pl_valid = data["map_polygon"]["valid_mask"]  # [B, pl]
        pl2a_valid = (
            pl_valid[:, None, None, :] & agent_valid[:, :, :, None]
        )  # [B, A, 5, pl]]

        if (
            self.num_pl2a > 0
            and pl_N > self.num_pl2a + 10
            and self.save_memory
        ):
            pl_K = self.num_pl2a
            pl_N = pl_K
            dist_pl2a = torch.norm(
                rel_pos_pl2a[..., :2], p=2, dim=-1
            )  # [B, A, T, pl]

            # dist_pl2a_valid = dist_pl2a.masked_fill(~pl2a_valid, 1e2)

            # topk_dist_valid, pl_idx = torch.topk(
            #    dist_pl2a_valid,              # 无负号
            #    k=pl_K,
            #    dim=-1,
            #    largest=False,
            # ) # [B, A, qt, pl_k] [B, A, qt, pl_k]

            # rand gather
            pl_idx = (
                (torch.rand(B, A, qt, pl_K) * pl_K).long().to(dist_pl2a.device)
            )  # int64
            topk_dist_valid = dist_pl2a.gather(-1, pl_idx)  # [B, A, qt, pl_k]

            pl2a_valid = pl2a_valid.gather(-1, pl_idx)  # [B, A, 5, pl_k]
            topk_dist = topk_dist_valid.masked_fill(~pl2a_valid, 0)

            select_orient_pl = orient_pl.gather(
                -1, pl_idx.reshape(1, -1)
            ).reshape(B, A, qt, pl_K)
            topk_pos_pl2a = rel_pos_pl2a.gather(
                -2, pl_idx.unsqueeze(-1).repeat(1, 1, 1, 1, 2)
            )
            rel_orient_pl2a = wrap_angle(
                select_orient_pl - head_a[:, :, :, None]
            )
            angle_pl2a = wrap_angle(
                torch.atan2(topk_pos_pl2a[..., 1], topk_pos_pl2a[..., 0])
                - head_a[:, :, :, None]
            )
            r_pl2a = [
                topk_dist.view(B * A, 1, qt, pl_K),
                angle_pl2a.view(B * A, 1, qt, pl_K),
                rel_orient_pl2a.view(B * A, 1, qt, pl_K),
            ]
        else:
            # [B, A, QT, pl]
            rel_orient_pl2a = wrap_angle(
                orient_pl[:, None, None, :] - head_a[:, :, :, None]
            )
            theta_pl2a = torch.atan2(
                rel_pos_pl2a[..., 1], rel_pos_pl2a[..., 0]
            )
            angle_pl2a = wrap_angle(theta_pl2a - head_a[:, :, :, None])
            dist_pl2a = torch.norm(rel_pos_pl2a, p=2, dim=-1)
            r_pl2a = [
                dist_pl2a.view(B * A, 1, qt, pl_N),
                angle_pl2a.view(B * A, 1, qt, pl_N),
                rel_orient_pl2a.view(B * A, 1, qt, pl_N),
            ]
            pl_idx = None

        pos_ta = pos_a.transpose(1, 2)  # [B, T, A, 2]
        head_ta = head_a.transpose(1, 2)  # [B, T, A, 2]
        # [B, T, A, 1] & [B, T, 1, A]
        rel_pos_a2a = (
            pos_ta[:, :, :, None, :] - pos_ta[:, :, None, :, :]
        )  # [B, T, A, A, 2]

        agent_valid = data["agent"]["valid_mask"][:, :, bt:ht][
            :, :, ST : self.ht
        ]  # [B, A, 5]
        agent_valid = agent_valid.transpose(1, 2)  # [B, 5, A]

        if self.mask_pl2pl_type == "diagonal":
            a2a_valid = (
                (
                    torch.ones([B, agent_valid.shape[1], A, A])
                    - torch.eye(A)[None, None, :, :]
                )
                .to(agent_valid.device)
                .bool()
            )
        elif self.mask_pl2pl_type == "valid":
            a2a_valid = (
                agent_valid[:, :, :, None] & agent_valid[:, :, None, :]
            )  # [B, 5, A, A]
        else:
            dia_a2a_valid = (
                (
                    torch.ones([B, agent_valid.shape[1], A, A])
                    - torch.eye(A)[None, None, :, :]
                )
                .to(agent_valid.device)
                .bool()
            )
            val_a2a = (
                agent_valid[:, :, :, None] & agent_valid[:, :, None, :]
            )  # [B, 5, A, A]
            a2a_valid = dia_a2a_valid & val_a2a

        if self.num_a2a > 0 and A > self.num_a2a + 20 and self.save_memory:
            A_K = self.num_a2a
            theta_a2a = torch.atan2(rel_pos_a2a[..., 1], rel_pos_a2a[..., 0])
            dist_a2a = torch.norm(rel_pos_a2a[..., :2], p=2, dim=-1)

            # dist_a2a_valid = dist_a2a.masked_fill(~a2a_valid, 1e2)
            # [B, 5, A, A]

            # topk_dist_a2a_valid, a2a_idx = torch.topk(
            #    dist_a2a_valid,    # 无负号
            #    k=A_K,
            #    dim=-1,
            #    largest=False,
            # )

            # rand gather
            a2a_idx = (
                (torch.rand(B, qt, A, A_K) * A_K).long().to(dist_a2a.device)
            )  # int64
            topk_dist_a2a_valid = dist_a2a.gather(-1, a2a_idx)  # [B, A, A_K]

            a2a_valid = a2a_valid.gather(-1, a2a_idx)  # [B, 5, A, A_K]
            topk_dist_a2a = topk_dist_a2a_valid.masked_fill(~a2a_valid, 0)

            rel_head_a2a = wrap_angle(
                head_ta[:, :, :, None] - head_ta[:, :, None, :]
            )

            angle_a2a = wrap_angle(theta_a2a - head_ta[:, :, :, None])
            rel_head_a2a = rel_head_a2a.gather(3, a2a_idx)
            angle_a2a = angle_a2a.gather(3, a2a_idx)
            r_a2a = [
                topk_dist_a2a.reshape(B * qt, 1, A, A_K),
                angle_a2a.reshape(B * qt, 1, A, A_K),
                rel_head_a2a.reshape(B * qt, 1, A, A_K),
            ]
        else:
            A_K = A
            rel_head_a2a = wrap_angle(
                head_ta[:, :, :, None] - head_ta[:, :, None, :]
            )  # * mask_a2a #[B,T, A, A]
            theta_a2a = torch.atan2(rel_pos_a2a[..., 1], rel_pos_a2a[..., 0])
            angle_a2a = wrap_angle(
                theta_a2a - head_ta[:, :, :, None]
            )  # * mask_a2a
            dist_a2a = torch.norm(rel_pos_a2a[..., :2], p=2, dim=-1)
            r_a2a = [
                dist_a2a.reshape(B * qt, 1, A, A),
                angle_a2a.reshape(B * qt, 1, A, A),
                rel_head_a2a.reshape(B * qt, 1, A, A),
            ]
            a2a_idx = None

        out = {
            "x_a": x_a,  # [B,1, A, HT] *4
            "r_pl2a": r_pl2a,  # [B * A, 1, QT, pl_N] *3
            "r_t": r_t,  # [B*A, 1, QT, ST] *4
            "r_a2a": r_a2a,  # [B*QT, 1, A, A] *3
        }
        if pl_idx is not None:
            out["pl_idx"] = pl_idx
        if a2a_idx is not None:
            out["a2a_idx"] = a2a_idx
        return out

    def build_agent_cur_r_inputs(self, data: dict, cur: int):
        pos_pl = data["map_polygon"]["position"] / 10.0
        orient_pl = data["map_polygon"]["orientation"]
        pos_a = data["agent"]["position"][:, :, :cur] / 10.0  # [B, A, HT, 2]
        head_a = data["agent"]["heading"][:, :, :cur]  # [B, A, HT]
        vel = data["agent"]["velocity"][:, :, :cur, : self.input_dim] / 10.0

        B, A = head_a.shape[:2]
        pos_cur = pos_a[:, :, cur - 1]
        head_cur = data["agent"]["heading"][:, :, cur - 1]
        motion_vector_a = pos_a[:, :, -1] - pos_a[:, :, -2]

        theta_motion_a = torch.atan2(
            motion_vector_a[..., 1], motion_vector_a[..., 0]
        )
        angle_motion = wrap_angle(theta_motion_a - head_cur)  # [B, A,]
        theta_vel = torch.atan2(vel[..., cur - 1, 1], vel[..., cur - 1, 0])
        angle_vel = wrap_angle(theta_vel - head_cur)  # [B, A]
        dist_vel = torch.norm(vel[:, :, cur - 1, :2], p=2, dim=-1)
        dist_motion_a = torch.norm(motion_vector_a[..., :2], p=2, dim=-1)  # BA
        x_a = [
            dist_motion_a.reshape(B, 1, A, 1),
            angle_motion.reshape(B, 1, A, 1),
            dist_vel.reshape(B, 1, A, 1),
            angle_vel.reshape(B, 1, A, 1),
        ]
        if not self.reuse_agent_rembs:
            ST = self.time_span
        else:
            if cur == self.ht:
                ST = self.num_t2m_steps  # get r_t for decoder
            else:
                ST = self.time_span

        pos_t_key = pos_a[:, :, cur - ST - 1 : cur - 1]
        head_t_key = head_a[:, :, cur - ST - 1 : cur - 1]
        rel_pos_t = pos_cur[:, :, None] - pos_t_key
        # [B, A, QT, ST, 2]
        rel_head_t = wrap_angle(head_cur[:, :, None] - head_t_key)
        # [B, A, QT, ST,]
        theta_pos_t = torch.atan2(rel_pos_t[..., 1], rel_pos_t[..., 0])
        angle_agent_t = wrap_angle(
            theta_pos_t - head_cur[:, :, None]
        )  # [B, A, HT, HT]

        diff_t = (
            torch.arange(ST)
            .reshape(1, 1, 1, ST)
            .repeat(B, 1, A, 1)
            .to(pos_a.device)
        )

        dist_pos_t = torch.norm(rel_pos_t[..., :2], p=2, dim=-1)
        r_t = [
            dist_pos_t.view(B, 1, A, ST),
            angle_agent_t.view(B, 1, A, ST),
            rel_head_t.view(B, 1, A, ST),
            diff_t.float().view(B, 1, A, ST),
        ]

        if cur < self.ht and self.quant_infer_cold_start:
            r_t_pedding = torch.zeros(B, 1, A, self.num_t2m_steps - ST).to(
                dist_pos_t.device
            )
            for i in range(len(r_t)):
                r_t[i] = torch.cat((r_t_pedding, r_t[i]), dim=-1)

        pl_N = pos_pl.shape[1]
        rel_pos_pl2a = pos_pl[:, None, :, :] - pos_cur[:, :, None, :]
        # [B, A, pl, 2]
        rel_orient_pl2a = wrap_angle(
            orient_pl[:, None, :] - head_cur[:, :, None]
        )  # [B, A, T, pl]
        theta_pl2a = torch.atan2(rel_pos_pl2a[..., 1], rel_pos_pl2a[..., 0])
        angle_pl2a = wrap_angle(theta_pl2a - head_cur[:, :, None])
        dist_pl2a = torch.norm(rel_pos_pl2a, dim=-1, p=2)
        r_pl2a = [
            dist_pl2a.view(B, 1, A, pl_N),
            angle_pl2a.view(B, 1, A, pl_N),
            rel_orient_pl2a.view(B, 1, A, pl_N),
        ]

        # [B, A, 1] - [B, 1, A]
        rel_pos_a2a = (
            pos_cur[:, :, None, :] - pos_cur[:, None, :, :]
        )  # [B, T, A, A, 2]
        rel_head_a2a = wrap_angle(
            head_cur[:, :, None] - head_cur[:, None, :]
        )  # [B, A, A]
        theta_a2a = torch.atan2(rel_pos_a2a[..., 1], rel_pos_a2a[..., 0])
        angle_a2a = wrap_angle(
            theta_a2a - head_cur[:, :, None]  # 应为 head_cur[:, None, :]
        )
        dist_a2a = torch.norm(rel_pos_a2a[..., :2], dim=-1, p=2)
        r_a2a = [
            dist_a2a.reshape(B, 1, A, A),
            angle_a2a.reshape(B, 1, A, A),
            rel_head_a2a.reshape(B, 1, A, A),
        ]

        mask_a_cur = data["agent"]["valid_mask"][:, :, cur - 1]  # [B, A]
        mask_a2a_cur = data["agent"]["valid_mask_a2a"][
            :, cur - 1, :, :A
        ]  # [B, A, A]
        mask_t_key = data["agent"]["valid_mask"][
            :, :, cur - self.time_span : cur
        ]  # [B, A, 2]

        return {
            "x_a_cur": x_a,
            "r_pl2a_cur": r_pl2a,
            "r_t_cur": r_t,  # [B, 1, A, 6]
            "r_a2a_cur": r_a2a,
            "mask_a_cur": mask_a_cur,
            "mask_a2a_cur": mask_a2a_cur,
            "mask_t_key": mask_t_key,
        }

    def build_decoder_r_inputs(self, data: dict):
        B, A = data["agent"]["position"].shape[:2]
        ht = self.ht
        pt = ht - self.num_t2m_steps

        pos_m = (
            data["agent"]["position"][:, :, self.ht - 1 : self.ht] / 10.0
        )  # [B, A, 1, 2]
        head_m = data["agent"]["heading"][
            :, :, self.ht - 1 : self.ht
        ]  # [B, A, 1, 2]
        pos_t = (
            data["agent"]["position"][:, :, pt : self.ht, : self.input_dim]
            / 10.0
        )  # [B, A, HT, 2]
        head_t = data["agent"]["heading"][:, :, pt : self.ht]  # [B, A, HT]
        pos_pl = data["map_polygon"]["position"][..., : self.input_dim] / 10.0
        orient_pl = data["map_polygon"]["orientation"]

        rel_pos_t2m = pos_t - pos_m  # [B, A, HT, 2]
        rel_head_t2m = wrap_angle(head_t - head_m)  # [B, A, HT]
        theta_t2m = torch.atan2(rel_pos_t2m[..., 1], rel_pos_t2m[..., 0])
        angle_t2m = wrap_angle(theta_t2m - head_m)  # [B, A, HT]
        arange_t = torch.arange(pt, self.ht).reshape(
            [1, 1, self.num_t2m_steps]
        )
        diff_t = (
            (arange_t.to(pos_m.device) - self.ht + 1).float().repeat(B, A, 1)
        )

        dist_t2m = torch.norm(rel_pos_t2m[..., :2], dim=-1, p=2)
        r_t2m = [
            dist_t2m.unsqueeze(1),
            angle_t2m.unsqueeze(1),
            rel_head_t2m.unsqueeze(1),
            diff_t.unsqueeze(1),
        ]  # [B, A, T, 4]

        # [B, pl, 2] - [B, A, 1, 2]
        rel_pos_pl2m = pos_pl[:, None, :, :] - pos_m  # [B, A, pl, 2]
        rel_orient_pl2m = wrap_angle(
            orient_pl[:, None, :] - head_m
        )  # [B, A, pl, 2]
        theta_pl2m = torch.atan2(rel_pos_pl2m[..., 1], rel_pos_pl2m[..., 0])
        angle_pl2m = wrap_angle(theta_pl2m - head_m)
        dist_pl2m = torch.norm(rel_pos_pl2m[..., :2], p=2, dim=-1)
        r_pl2m = [
            dist_pl2m.unsqueeze(1),
            angle_pl2m.unsqueeze(1),
            rel_orient_pl2m.unsqueeze(1),
        ]

        # [B, A, 1, 2]
        rel_pos_a2m = pos_m - pos_m.squeeze(2)[:, None, :]  # [B, A, A, 2]
        rel_head_a2m = wrap_angle(
            head_m - head_m.squeeze(2)[:, None, :]
        )  # [B, A, A]
        theta_a2m = torch.atan2(rel_pos_a2m[..., 1], rel_pos_a2m[..., 0])
        angle_a2m = wrap_angle(theta_a2m - head_m)  # [B, A, A,]
        dist_a2m = torch.norm(rel_pos_a2m[..., :2], p=2, dim=-1)
        r_a2m = [
            dist_a2m.unsqueeze(1),
            angle_a2m.unsqueeze(1),
            rel_head_a2m.unsqueeze(1),
        ]

        return {"r_t2m": r_t2m, "r_pl2m": r_pl2m, "r_a2m": r_a2m}

    def __call__(
        self, collate_data: dict, cur_step: None, his_model_input: bool = False
    ):
        deploy_inputs = OrderedDict()

        map_inputs = self.build_map_r_inputs(collate_data)
        if self.deploy or self.quant_infer_cold_start:
            if not his_model_input:
                if self.quant_infer_cold_start:
                    cur = cur_step  # quant infer and cold start
                else:
                    cur = self.ht  # quant infer and hot start
                agent_inputs = self.build_agent_cur_r_inputs(collate_data, cur)
                if not self.reuse_agent_rembs:
                    decoder_inputs = self.build_decoder_r_inputs(collate_data)
            else:  # quant infer and hot start
                qt = (
                    self.num_t2m_steps
                    + self.time_span * (self.num_agent_layers - 1)
                    - 1
                )
                agent_inputs = self.build_agent_his_r_inputs(
                    collate_data, ht=self.ht - 1, qt=qt
                )
        else:
            if self.stream:  # predict and hot start
                qt = (
                    self.num_t2m_steps
                    + self.time_span * (self.num_agent_layers - 1)
                    - 1
                )
                agent_inputs = self.build_agent_his_r_inputs(
                    collate_data, ht=self.ht - 1, qt=qt
                )
                agent_inputs2 = self.build_agent_cur_r_inputs(
                    collate_data, cur=self.ht
                )
                agent_inputs.update(agent_inputs2)
            else:
                qt = self.num_t2m_steps + self.time_span * (
                    self.num_agent_layers - 1
                )
                agent_inputs = self.build_agent_his_r_inputs(
                    collate_data, ht=self.ht, qt=qt
                )
            if not self.reuse_agent_rembs:
                decoder_inputs = self.build_decoder_r_inputs(collate_data)
        if not self.deploy:
            collate_data["map_point"]["magnitude"] = (
                collate_data["map_point"]["magnitude"] / 10.0
            )
            collate_data["map_polygon"].update(map_inputs)
            collate_data["agent"].update(agent_inputs)
            if not self.reuse_agent_rembs:
                collate_data["decoder"].update(decoder_inputs)
            return collate_data
        elif self.deploy:
            A = self.A
            deploy_inputs["agent"] = OrderedDict()
            deploy_inputs["agent"]["valid_mask"] = collate_data["agent"][
                "valid_mask"
            ][:1, :A, : self.ht]
            deploy_inputs["agent"]["valid_mask_a2a"] = collate_data["agent"][
                "valid_mask_a2a"
            ][:1, :, :A, :A]
            deploy_inputs["agent"]["agent_type"] = collate_data["agent"][
                "agent_type"
            ][:1, :A]

            deploy_inputs["agent"].update(agent_inputs)

            deploy_inputs["map_polygon"] = {
                "pl_type": collate_data["map_polygon"]["pl_type"][:1],
                "is_intersection": collate_data["map_polygon"][
                    "is_intersection"
                ][:1],
            }
            deploy_inputs["map_polygon"].update(map_inputs)
            deploy_inputs["map_point"] = {
                "magnitude": collate_data["map_point"]["magnitude"][:1] / 10.0,
                "pt_type": collate_data["map_point"]["pt_type"][:1],
                "side": collate_data["map_point"]["side"][:1],
                "mask": collate_data["map_point"]["mask"][:1],
            }
            if not his_model_input:
                deploy_inputs["decoder"] = OrderedDict()
                deploy_inputs["decoder"].update(
                    {
                        "mask_a2m": collate_data["decoder"]["mask_a2m"][
                            :1, :A, :A
                        ],
                        "mask_dst": collate_data["decoder"]["mask_dst"][
                            :1, :A
                        ],
                    }
                )
                if not self.reuse_agent_rembs:
                    deploy_inputs["decoder"].update(decoder_inputs)
            deploy_inputs["type_pl2pl"] = collate_data["type_pl2pl"]

            return deploy_inputs
