import math
from typing import List, Optional

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.layer_norm import LayerNorm
from horizon_plugin_pytorch.quantization import QuantStub

from hat.utils import qconfig_manager
from .utils import weight_init

__all__ = ["FourierEmbedding", "FourierConvEmbedding"]


class FourierEmbedding(nn.Module):
    """
    FourierEmbedding module that embed input data using Fourier features.

    Args:
        input_dim: Dimensionality of the input data.
        hidden_dim: Dimensionality of the hidden layers.
        num_freq_bands: Number of frequency bands for Fourier features.

    """

    def __init__(
        self, input_dim: int, hidden_dim: int, num_freq_bands: int
    ) -> None:
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = (
            nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        )
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=False),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.sin = horizon.nn.Sin()
        self.cos = horizon.nn.Cos()
        self.freqs_quant = QuantStub()
        self.apply(weight_init)

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor] = None,
        categorical_embs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)
            else:
                raise ValueError(
                    "Both continuous_inputs and categorical_embs are None"
                )
        else:
            freqs_weight = self.freqs_quant(self.freqs.weight)
            x = continuous_inputs.unsqueeze(-1) * freqs_weight * (2 * math.pi)
            cos_x = self.cos(x)
            sin_x = self.sin(x)
            x = torch.cat(
                [cos_x, sin_x, continuous_inputs.unsqueeze(-1)], dim=-1
            )
            continuous_embs = 0
            for i in range(self.input_dim):
                if i > 0:
                    continuous_embs = continuous_embs + self.mlps[i](
                        x[..., i, :]
                    )  # [B, A, T, T,  64]
                else:
                    continuous_embs = self.mlps[i](x[..., i, :])
            x = continuous_embs
            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs, dim=0).sum(dim=0)
        return self.to_out(x)

    def set_qconfig(
        self,
    ):
        from horizon_plugin_pytorch.dtype import qint16

        from hat.utils import qconfig_manager

        # set fixscale to cover sin and cos range (-1, 1)
        self.sin.qconfig = qconfig_manager.get_qconfig(
            activation_calibration_observer="fixed_scale",
            activation_qat_observer="fixed_scale",
            activation_qat_qkwargs={"scale": 1.1 / 32768, "dtype": qint16},
            activation_calibration_qkwargs={
                "scale": 1.1 / 32768,
                "dtype": qint16,
            },
        )
        self.cos.qconfig = qconfig_manager.get_qconfig(
            activation_calibration_observer="fixed_scale",
            activation_qat_observer="fixed_scale",
            activation_qat_qkwargs={"scale": 1.1 / 32768, "dtype": qint16},
            activation_calibration_qkwargs={
                "scale": 1.1 / 32768,
                "dtype": qint16,
            },
        )


class FourierConvEmbedding(nn.Module):
    """
    More efficient conv version FourierEmbedding module.

    Args:
        input_dim: Dimensionality of the input data.
        hidden_dim: Dimensionality of the hidden layers.
        num_freq_bands: Number of frequency bands for Fourier features.

    """

    def __init__(
        self, input_dim: int, hidden_dim: int, num_freq_bands: int
    ) -> None:
        super(FourierConvEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.ModuleList(
            [
                nn.Conv2d(1, num_freq_bands, kernel_size=1, bias=False)
                for _ in range(input_dim)
            ]
        )
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        num_freq_bands * 2 + 1, hidden_dim, kernel_size=1
                    ),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            LayerNorm((hidden_dim, 1, 1), dim=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        self.sin = horizon.nn.Sin()
        self.cos = horizon.nn.Cos()
        self.apply(weight_init)
        for m in self.freqs:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        continuous_inputs: Optional[List[torch.Tensor]] = None,
        categorical_embs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)
            else:
                raise ValueError(
                    "Both continuous_inputs and categorical_embs are None"
                )
        else:
            continuous_embs = 0
            for i in range(self.input_dim):
                cx = continuous_inputs[i]
                x = self.freqs[i](cx) * (2 * math.pi)
                cos_x = self.cos(x)
                sin_x = self.sin(x)
                x = torch.cat([cos_x, sin_x, cx], dim=1)
                if i == 0:
                    continuous_embs = self.mlps[i](x)
                else:
                    continuous_embs = continuous_embs + self.mlps[i](x)
            x = continuous_embs
            if categorical_embs is not None:
                x = x + categorical_embs
        return self.to_out(x)

    def set_qconfig(self):
        # set fixscale to cover sin cos range (-1, 1)
        self.sin.qconfig = qconfig_manager.get_qconfig(
            activation_calibration_observer="fixed_scale",
            activation_qat_observer="fixed_scale",
            activation_qat_qkwargs={"scale": 1.1 / 32768, "dtype": qint16},
            activation_calibration_qkwargs={
                "scale": 1.1 / 32768,
                "dtype": qint16,
            },
        )
        self.cos.qconfig = qconfig_manager.get_qconfig(
            activation_calibration_observer="fixed_scale",
            activation_qat_observer="fixed_scale",
            activation_qat_qkwargs={"scale": 1.1 / 32768, "dtype": qint16},
            activation_calibration_qkwargs={
                "scale": 1.1 / 32768,
                "dtype": qint16,
            },
        )
