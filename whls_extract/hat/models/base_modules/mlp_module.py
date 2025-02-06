# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Optional

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional

__all__ = ["MlpModule2d", "MLP", "FFN"]


class MlpModule2d(nn.Sequential):
    """A mlp block that bundles two fc layers.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Number of hidden channels.
        out_channels: Number of output channels.
        act_layer: Activation layer. Default: None.
        drop_ratio: Dropout ratio of output. Default: 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: Optional[nn.Module] = None,
        drop_ratio: float = 0.0,
    ):
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        fc1 = nn.Linear(in_channels, hidden_channels)
        fc2 = nn.Linear(hidden_channels, out_channels)
        if drop_ratio > 0.0:
            drop1 = nn.Dropout(drop_ratio)
            drop2 = nn.Dropout(drop_ratio)
        else:
            drop1, drop2 = None, None
        layer_list = [fc1, act_layer, drop1, fc2, drop2]
        self.layer_list = [layer for layer in layer_list if layer is not None]
        super(MlpModule2d, self).__init__(*self.layer_list)


class MLP(nn.Module):
    """Multi-layer perceptron layer without dropout and identity connection.

    The feature process order follows Linear -> ReLU -> Linear -> ReLU -> ...

    Args:
        input_dim: The input feature dimension.
        hidden_dim: The hidden dimension of MLPs.
        output_dim: the output feature dimension of MLPs.
        num_layer: The number of FC layer used in MLPs.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> torch.Tensor:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList()
        for i, (n, k) in enumerate(zip([input_dim] + h, h + [output_dim])):
            if i < self.num_layers - 1:
                self.layers.append(nn.Sequential(nn.Linear(n, k), nn.ReLU()))
            else:
                self.layers.append(nn.Linear(n, k))

    def forward(self, x: torch.Tensor):
        """Forward function of `MLP`.

        Args:
            x (torch.Tensor): the input tensor used in `MLP` layers.

        Returns:
            torch.Tensor: the forward results of `MLP` layer
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class FFN(nn.Module):
    """Feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dim: The feature dimension. Same as `MultiheadAttention`.
        feedforward_dim: The hidden dimension of FFNs.
        output_dim: The output feature dimension of FFNs.
            Default: None. If None, the `embed_dim` will be used.
        num_fcs: The number of fully-connected layers in FFNs.
        activation: The activation layer used in FFNs.
        ffn_drop: Probability of an element to be zeroed in FFN.
        add_identity: Whether to add the identity connection.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        feedforward_dim: int = 1024,
        output_dim: Optional[int] = None,
        num_fcs: int = 2,
        activation: Optional[nn.Module] = None,
        ffn_drop: float = 0.0,
        fc_bias: bool = True,
        add_identity: bool = True,
    ):
        super(FFN, self).__init__()
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.num_fcs = num_fcs
        if activation is None:
            activation = nn.ReLU(inplace=True)
        self.activation = activation

        output_dim = embed_dim if output_dim is None else output_dim

        layers = []
        in_channels = embed_dim
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_dim, bias=fc_bias),
                    self.activation,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_dim
        layers.append(nn.Linear(feedforward_dim, output_dim, bias=fc_bias))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity
        self.add_identity_op = FloatFunctional()

    def forward(
        self, x: torch.Tensor, identity: Optional[torch.Tensor] = None
    ):
        """Forward function of `FFN`.

        Args:
            x: the input tensor used in `FFN` layers.
            identity: the tensor with the same shape as `x`,
                which will be used for identity addition. Default: None.
                if None, `x` will be used.

        Returns:
            torch.Tensor: the forward results of `FFN` layer
        """
        out = self.layers(x)
        if not self.add_identity:
            return out
        return self.add_identity_op.add(identity, out)
