import copy
from typing import List, Tuple

import numpy as np
import torch
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from torch import nn


def bias_init_with_prob(prior_prob: int) -> float:
    """Initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def get_clone_module(module: nn.Module, N: int) -> nn.ModuleList:
    """Get clone nn modules."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    """Initialize conv/fc bias with constant value."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(
    module: nn.Module,
    gain: float = 1,
    bias: float = 0,
    distribution: str = "normal",
):
    """Initialize conv/fc bias with xavier method."""
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse function of sigmoid."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def get_min_max_coords(
    real_range: List[float],
    grid_resolution: List[float],
) -> Tuple[float]:
    """Get min and max coords."""
    min_x = real_range[0] + grid_resolution[0] / 2
    min_y = real_range[1] + grid_resolution[1] / 2
    max_x = real_range[2] - grid_resolution[0] / 2
    max_y = real_range[3] - grid_resolution[1] / 2
    return min_x, max_x, min_y, max_y


def gen_coords(bev_size: Tuple[int], pc_range: Tuple[float]) -> torch.Tensor:
    """Generate coords."""
    real_w = pc_range[3] - pc_range[0]
    real_h = pc_range[4] - pc_range[1]

    W = bev_size[0]
    H = bev_size[1]

    grid_resolution = (real_w / W, real_h / H)
    real_range = (pc_range[0], pc_range[1], pc_range[3], pc_range[4])

    bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(
        real_range,
        grid_resolution,
    )

    # Generate a tensor for the x-coordinates of the bird's eye view grid
    x = (
        torch.linspace(bev_min_x, bev_max_x, W).reshape((1, W)).repeat(H, 1)
    ).double()
    y = (
        torch.linspace(bev_min_y, bev_max_y, H).reshape((H, 1)).repeat(1, W)
    ).double()
    coords = torch.stack([x, y], dim=-1).unsqueeze(0)
    return coords


class FFN(nn.Module):
    """The basic structure of FFN.

    Args:
        dim: The inputs dim.
        scale: The scale for inputs dim in hidden layers.
        bias: Whether use bias,
        dropout: Probability of an element to be zeroed.
    """

    def __init__(
        self, dim: int, scale: int = 2, bias: bool = True, dropout: float = 0.0
    ):
        super().__init__()
        self.ffn1 = nn.Linear(dim, int(dim * scale), bias=bias)
        self.ffn1_act = nn.ReLU(inplace=True)
        self.ffn_dropout1 = nn.Dropout(dropout)
        self.ffn2 = nn.Linear(int(dim * scale), dim, bias=bias)
        self.ffn_dropout2 = nn.Dropout(dropout)
        self.add = FF()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Foward FFN."""
        x = self.ffn1(x)
        x = self.ffn1_act(x)
        x = self.ffn_dropout1(x)
        x = self.ffn2(x)
        x = self.ffn_dropout2(x)
        x = self.add.add(x, skip)
        return x

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""
        from hat.utils import qconfig_manager

        int16_module = [
            self.add,
            self.ffn2,
        ]
        for m in int16_module:
            m.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={
                    "dtype": qint16,
                },
                activation_calibration_observer="mix",
            )

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""
        from horizon_plugin_pytorch import quantization

        torch.quantization.fuse_modules(
            self,
            ["ffn1", "ffn1_act"],
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )


def normalize_bbox(bboxes: torch.Tensor) -> torch.Tensor:
    """Normalize bbox."""
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    bl = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()
    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, bl, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, bl, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes: torch.Tensor) -> torch.Tensor:
    """Denormalize bbox."""
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    bl = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    bl = bl.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat(
            [cx, cy, cz, w, bl, h, rot, vx, vy], dim=-1
        )
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, bl, h, rot], dim=-1)
    return denormalized_bboxes
