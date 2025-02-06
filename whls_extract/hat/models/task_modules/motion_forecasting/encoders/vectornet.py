# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Tuple

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

__all__ = ["Vectornet"]


class SubGraphLayer(nn.Module):
    """Implements the vectornet subgraph layer.

    Args:
        in_channels: input channels.
        hidden_size: hidden_size.
        out_channels: output channels.
        num_vec: number of vectors.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: int,
        num_vec: int,
    ):
        super(SubGraphLayer, self).__init__()
        hidden_size = out_channels
        self.mlp = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            LayerNorm2d(normalized_shape=[hidden_size, 1, 1], dim=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.mlp(x)
        return x


class SubGraph(nn.Module):
    """Implements the vectornet subgraph.

    Args:
        in_channels: input channels.
        depth: depth for encoder layer.
        hidden_size: hidden_size.
        num_vec: number of vectors.
    """

    def __init__(self, in_channels, depth=3, hidden_size=64, num_vec=9):
        super(SubGraph, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                SubGraphLayer(
                    in_channels=in_channels if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    out_channels=hidden_size,
                    num_vec=num_vec,
                )
            )
        self.max_pool = nn.MaxPool2d([num_vec, 1], stride=1)

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        x = self.max_pool(x)
        return x


@OBJECT_REGISTRY.register
class Vectornet(nn.Module):
    """Implements the vectornet encoder.

    Args:
        depth: depth for encoder layer.
        traj_in_channels: Traj feat input channels.
        traj_num_vec: Vector number of traj feat.
        lane_in_channels: Lane fat input channels.
        lane_num_vec: Vector number of lane feat.
        hidden_size: hidden_size.
    """

    def __init__(
        self,
        depth: int = 3,
        traj_in_channels: int = 8,
        traj_num_vec: int = 9,
        lane_in_channels: int = 16,
        lane_num_vec: int = 19,
        hidden_size: int = 128,
    ):
        super(Vectornet, self).__init__()
        self.lane_enc = SubGraph(
            in_channels=lane_in_channels,
            num_vec=traj_num_vec,
            depth=depth,
            hidden_size=hidden_size,
        )
        self.traj_enc = SubGraph(
            in_channels=traj_in_channels,
            num_vec=lane_num_vec,
            depth=depth,
            hidden_size=hidden_size,
        )
        self.hidden_size = hidden_size

        self.lane_quant = QuantStub(scale=None)
        self.traj_quant = QuantStub(scale=None)

        self.gobal_graph = MultiheadAttention(
            embed_dim=hidden_size, num_heads=1
        )
        self.fc = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.dequant = DeQuantStub()

    def forward(
        self, traj_feat: Tensor, lane_feat: Tensor, instance_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Perform forward pass.

        Args:
            traj_feat: Trajectory features.
            lane_feat: Lane features.
            instance_mask: Instance mask.

        Returns:
            Tuple containing graph_feat, gobal_feat, traj_feat,
                             lane_feat, instance_mask.
        """
        lane_feat = self.lane_quant(lane_feat)
        lane_feat = self.lane_enc(lane_feat)
        traj_feat = self.traj_quant(traj_feat)
        traj_feat = self.traj_enc(traj_feat)
        gobal_feat = torch.cat([traj_feat, lane_feat], dim=-1)
        graph_feat = gobal_feat[..., 0:1]

        graph_feat, _ = self.gobal_graph(
            query=graph_feat,
            key=gobal_feat,
            value=gobal_feat,
            attn_mask=instance_mask,
        )
        graph_feat = self.fc(graph_feat)
        return graph_feat, gobal_feat, traj_feat, lane_feat, instance_mask

    def set_qconfig(self) -> None:
        """Set the quantization configuration for the model."""
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        self.lane_quant.qconfig = horizon.quantization.get_default_qat_qconfig(
            dtype="qint16"
        )
        self.traj_quant.qconfig = horizon.quantization.get_default_qat_qconfig(
            dtype="qint16"
        )
