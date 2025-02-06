# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, Tuple

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import LayerNorm as LayerNorm2d
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor
from torch.quantization import DeQuantStub

from hat.models.base_modules.attention import (
    HorizonMultiheadAttention as MultiheadAttention,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["Densetnt"]


class MLP(nn.Module):
    """MLP module.

    Args:
        in_channels: the input channels of the sub graph layer.
        out_channels: the output channels of the sub graph layer.
        use_layernorm: whether use layernorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_layernorm: bool = False,
    ):
        super(MLP, self).__init__()
        mlp = []
        mlp.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                stride=1,
            )
        )
        if use_layernorm is True:
            mlp.append(
                LayerNorm2d(normalized_shape=[out_channels, 1, 1], dim=1)
            )

        mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.mlp(x)


class GoalSubGraph(nn.Module):
    """Goal subgraph module.

    Args:
        in_channels: the input channels of the sub graph layer.
        graph_channels: the channels of the middle layer.
        out_channels: the output channels of the sub graph layer.
    """

    def __init__(
        self, in_channels: int, graph_channels: int, out_channels: int
    ):
        super(GoalSubGraph, self).__init__()
        self.goal_mlp = MLP(
            in_channels=2,
            out_channels=out_channels,
        )
        self.cat_mlp = MLP(
            in_channels=graph_channels + out_channels,
            out_channels=out_channels,
        )
        self.cat_op = nn.quantized.FloatFunctional()

    def forward(self, goal_feats: Tensor, graph_feats: Tensor) -> Tensor:
        """Perform forward pass.

        Args:
            goal_feats: Goal features.
            graph_feats: Graph features.

        Returns:
            Output tensor.
        """
        feats = self.goal_mlp(goal_feats)
        feats = self.cat_op.cat([feats, graph_feats], dim=1)
        feats = self.cat_mlp(feats)
        return feats


class Decoder(nn.Module):
    """Decoder module.

    Args:
        in_channels: the input channels of the sub graph layer.
        out_channels: the output channels of the sub graph layer.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(Decoder, self).__init__()
        self.fc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.fc(x)
        return x


class TargetSubGraph(nn.Module):
    """Target subgraph module.

    Args:
        in_channels: the input channels of the sub graph layer.
        out_channels: the output channels of the sub graph layer.
        num_layers: the number of sub graph layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
    ):
        super(TargetSubGraph, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                MLP(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                )
            )

    def forward(self, feats: Tensor) -> Tensor:
        """Perform forward pass.

        Args:
            feats: Input tensor.

        Returns:
            Output tensor.
        """
        for layer in self.layers:
            feats = layer(feats)
        return feats


@OBJECT_REGISTRY.register
class Densetnt(nn.Module):
    """Implements the Densetnt head.

    Args:
        in_channels: input channels.
        hidden_size: hidden_size.
        num_traj: number of traj.
        target_graph_depth: depth for traj decoder.
        pred_steps: number of traj pred steps.
        top_k: top k for candidates.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_size: int = 128,
        num_traj: int = 384,
        target_graph_depth: int = 2,
        pred_steps: int = 30,
        top_k: int = 150,
    ):
        super(Densetnt, self).__init__()
        self.hidden_size = hidden_size
        self.goal_sub_graph = GoalSubGraph(2, hidden_size, hidden_size)
        self.goal_traj_fusion = MultiheadAttention(
            embed_dim=hidden_size, num_heads=1
        )
        self.goal_decoder = Decoder(hidden_size, 1)

        self.target_sub_graph = TargetSubGraph(
            2, hidden_size, target_graph_depth
        )

        self.target_traj_fuse = MultiheadAttention(
            embed_dim=hidden_size, num_heads=1
        )

        self.traj_decoder = Decoder(hidden_size * 2, pred_steps * 2)
        self.top_k = top_k
        self.num_traj = num_traj
        self.goal_quant = QuantStub(scale=None)
        self.mask_quant = QuantStub(scale=None)
        self.dequant = DeQuantStub()
        self.cat_op = nn.quantized.FloatFunctional()
        self.add_mask = nn.quantized.FloatFunctional()
        self.add_op = nn.quantized.FloatFunctional()

    @fx_wrap()
    def forward(
        self,
        graph_feats: Tensor,
        gobal_feats: Tensor,
        traj_feats: Tensor,
        lane_feats: Tensor,
        instance_mask: Tensor,
        data: Dict,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform forward pass.

        Args:
            graph_feats: Graph features.
            gobal_feats: Global features.
            traj_feats: Trajectory features.
            lane_feats: Lane features.
            instance_mask: Instance mask.
            data: Data dictionary containing goals and goals mask.

        Returns:
            Tuple containing goals_preds, traj_preds, and pred_goals.
        """
        goals = data["goals_2d"]
        goals_2d_mask = data["goals_2d_mask"]
        mask = self.mask_quant(goals_2d_mask)
        goals = self.goal_quant(goals)
        num_goals = goals.shape[3]
        graph_feats_repeat = graph_feats.repeat(1, 1, 1, num_goals)

        goals_feats = self.goal_sub_graph(goals, graph_feats_repeat)

        instance_mask = instance_mask.squeeze(1)
        instance_mask_repeat = instance_mask.repeat(1, num_goals, 1)
        goals_trajs_fused, _ = self.goal_traj_fusion(
            query=goals_feats,
            key=traj_feats,
            value=traj_feats,
            attn_mask=instance_mask_repeat[..., : self.num_traj],
        )

        goals_preds = self.goal_decoder(goals_trajs_fused)
        goals_preds = self.add_mask.add(goals_preds, mask)

        if self.training:
            end_points = data["end_points"].view(-1, 2, 1, 1)
            end_points = self.goal_quant(end_points)
            target_feats = self.target_sub_graph(end_points)
            target_traj_fuse, _ = self.target_traj_fuse(
                query=target_feats,
                key=traj_feats.detach(),
                value=traj_feats.detach(),
                attn_mask=instance_mask[..., : self.num_traj],
            )
            target_traj_fuse = self.add_op.add(target_traj_fuse, target_feats)
            target_cat_feats = self.cat_op.cat(
                [
                    target_traj_fuse,
                    graph_feats.detach(),
                ],
                dim=1,
            )
            traj_preds = self.traj_decoder(target_cat_feats)

            goals_preds = self.dequant(goals_preds)
            traj_preds = self.dequant(traj_preds)
            return goals_preds, traj_preds
        else:
            traj_preds = []

            goals_preds, idx = torch.topk(goals_preds, dim=3, k=self.top_k)

            idx = idx.repeat((1, 2, 1, 1))
            pred_goals = torch.gather(goals, 3, idx)

            if self.top_k > 1:
                instance_mask_repeat = instance_mask.repeat(1, self.top_k, 1)
            else:
                instance_mask_repeat = instance_mask
            target_feats = self.target_sub_graph(pred_goals)
            target_traj_fuse, _ = self.target_traj_fuse(
                query=target_feats,
                key=traj_feats,
                value=traj_feats,
                attn_mask=instance_mask_repeat[..., : self.num_traj],
            )
            target_traj_fuse = self.add_op.add(target_traj_fuse, target_feats)
            if self.top_k > 1:
                graph_feats_repeat = graph_feats.repeat(1, 1, 1, self.top_k)
            else:
                graph_feats_repeat = graph_feats

            target_cat_feats = self.cat_op.cat(
                [
                    target_traj_fuse,
                    graph_feats_repeat,
                ],
                dim=1,
            )
            traj_preds = self.traj_decoder(target_cat_feats)
            traj_preds = self.dequant(traj_preds)
            goals_preds = self.dequant(goals_preds)
            pred_goals = self.dequant(pred_goals)
            return goals_preds, traj_preds, pred_goals

    def set_qconfig(self) -> None:
        """Set the quantization configuration for the model."""
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.mask_quant.qconfig = horizon.quantization.get_default_qat_qconfig(
            dtype="qint16"
        )
        self.goal_quant.qconfig = horizon.quantization.get_default_qat_qconfig(
            dtype="qint16"
        )

        self.goal_sub_graph.qconfig = (
            horizon.quantization.get_default_qat_qconfig(dtype="qint16")
        )
        self.goal_traj_fusion.matmul.qconfig = (
            horizon.quantization.get_default_qat_qconfig(dtype="qint16")
        )
        self.goal_traj_fusion.mask_add.qconfig = (
            horizon.quantization.get_default_qat_qconfig(dtype="qint16")
        )
        # disable output quantization for last quanti layer.
        self.traj_decoder.fc.qconfig = (
            qconfig_manager.get_default_qat_out_qconfig()
        )
