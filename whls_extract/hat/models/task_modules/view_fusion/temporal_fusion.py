# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Any, Dict, Tuple

import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.qat_mode import tricks
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor, nn
from torch.quantization import DeQuantStub

from hat.core.nus_box3d_utils import get_min_max_coords
from hat.models.base_modules.separable_conv_module import SeparableConvModule2d
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

try:
    from hbdk.torch_script.placeholder import placeholder
except ImportError:
    placeholder = None

logger = logging.getLogger(__name__)


class TemporalFusion(nn.Module):
    """Temporal fusion for bev feats.

    Args:
        in_channels: Channels for input.
        out_channels: Channels for ouput.
        num_seq: Number of sequence for multi frames.
        bev_size: Bev size.
        grid_size: Grid size.
        num_encoder: Number of encoder layers.
        num_project: Number of project layers.
        mode: Mode for grid sample.
        padding_mode: Padding mode for grid sample.
        grid_quant_scale: Quanti scale for grid sample.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_seq: int,
        bev_size: Tuple[float],
        grid_size: Tuple[float],
        num_encoder: int = 2,
        num_project: int = 1,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        grid_quant_scale: float = 1 / 512,
    ):
        super(TemporalFusion, self).__init__()
        self.num_seq = num_seq
        self.bev_size = bev_size
        self.grid_size = grid_size
        self.in_channels = in_channels

        encoder = nn.ModuleList()
        for i in range(num_encoder):
            encoder.append(
                SeparableConvModule2d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    pw_norm_layer=nn.BatchNorm2d(out_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
            )
        self.encoder = nn.Sequential(*encoder)

        project = nn.ModuleList()
        for i in range(num_project):
            project.append(
                SeparableConvModule2d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    pw_norm_layer=nn.BatchNorm2d(out_channels),
                    pw_act_layer=nn.ReLU(inplace=True),
                )
            )
        self.project = nn.Sequential(*project)

        self.coords = nn.Parameter(self._gen_coords(), requires_grad=False)

        self.offset = nn.Parameter(self._gen_offset(), requires_grad=False)
        self.grid_sample = hnn.GridSample(
            mode=mode,
            padding_mode=padding_mode,
        )
        self.quant_stub = QuantStub(grid_quant_scale)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    @fx_wrap()
    def _set_scale(self, feat: Tensor, prev_feat) -> Tensor:
        """Set the scale factor of a feature for activation quantization.

        Args:
            feat: The input feature.
            prev_feat: The prev input feature.

        Returns:
            feat: The input feature after setting the scale factor.
        """

        if self.training and isinstance(feat, QTensor):
            self.quant.activation_post_process.scale = prev_feat.scale
        return feat, prev_feat

    def forward(
        self, feats: Tensor, meta: Dict, compile_model: bool, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Perform the forward pass through the modules.

        Args:
            feats: The input features.
            meta: The meta information.
            compile_model: A flag indicating whether to compile the model.

        Returns:
            feat: The output feature.
            prev_feat: The dequantized feature.
        """

        n, c, h, w = feats.shape
        feats = self.encoder(feats)
        fused_feat, prev_feat = self._fusion(feats, meta, compile_model)
        fused_feat, prev_feat = self._set_scale(fused_feat, prev_feat)
        prev_feat = self.dequant(prev_feat)
        return fused_feat, prev_feat

    def _gen_offset(self) -> None:
        """Generate a tensor for the offset values of a bird's eye view grid.

        Returns:
            The tensor containing the offset values.
        """

        W = self.grid_size[0]
        H = self.grid_size[1]

        # Generate a tensor for the x-coordinates of the bird's eye view grid
        bev_x = (
            torch.linspace(0, W - 1, W).reshape((1, W)).repeat(H, 1)
        ).double()
        bev_y = (
            torch.linspace(0, H - 1, H).reshape((H, 1)).repeat(1, W)
        ).double()

        bev_offset = torch.stack([bev_x, bev_y], axis=-1) * -1
        bev_offset = bev_offset.unsqueeze(0)
        return bev_offset

    def _gen_coords(self) -> None:
        """Generate a tensor for the coordinates of a bird's eye view grid.

        Returns:
            The tensor containing the world space coordinates.
        """

        # Get the minimum and maximum x and y coordinates
        # for the bird's eye view grid
        bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(
            self.bev_size
        )

        W = self.grid_size[0]
        H = self.grid_size[1]

        # Generate a tensor for the x-coordinates of the bird's eye view grid
        x = (
            torch.linspace(bev_min_x, bev_max_x, W)
            .reshape((1, W))
            .repeat(H, 1)
        ).double()
        y = (
            torch.linspace(bev_min_y, bev_max_y, H)
            .reshape((H, 1))
            .repeat(1, W)
        ).double()

        coords = torch.stack([x, y], dim=-1).unsqueeze(0)

        return coords

    def _get_matrix(
        self, meta: Dict, idx: int, bev_x: float, bev_y: float
    ) -> Tuple[np.array, np.array]:
        """Compute the transformation matrix.

        Components for warping a point on the bird's eye view grid
        between consecutive frames based on the provided meta information
        and the corresponding index.

        Args:
            meta: The meta information.
            idx: The index corresponding to the frame of interest.
            bev_x: The x-coordinate on the bird's eye view grid.
            bev_y: The y-coordinate on the bird's eye view grid.

        Returns:
            wrap_r: The rotation component of the transformation matrix.
            wrap_t: The translation component of the transformation matrix.
        """

        # Get the ego to global transformation matrix
        ego2global = meta["ego2global"]
        ego2global = np.array(ego2global).astype(np.float64)

        # Get the ego to global transformation matrix for the previous frame
        prev_e2g = ego2global[:, idx + 1]
        # Compute the inverse transformation matrix
        prev_g2e = np.linalg.inv(prev_e2g)
        # Get the ego to global transformation matrix for the current frame
        cur_e2g = ego2global[:, idx]
        # Compute the transformation matrix
        # by multiplying the inverse with the current
        wrap_m = prev_g2e @ cur_e2g
        # Extract the rotation component
        wrap_r = wrap_m[:, :2, :2].transpose((0, 2, 1))
        # Extract the translation component
        wrap_t = wrap_m[:, :2, 3]
        # Adjust the translation component
        # with the bird's eye view grid coordinates
        wrap_t = wrap_t + np.array([bev_x, bev_y])

        wrap_r /= self.bev_size[2]
        wrap_t /= self.bev_size[2]
        return wrap_r, wrap_t

    @fx_wrap()
    def _get_reference_points(
        self, feat: Tensor, meta: Dict, idx: int
    ) -> Tensor:
        """Compute the warped reference points on the bird's eye view grid.

        Args:
            feat: The feature tensor.
            meta: The meta information.
            idx: The index corresponding to the frame of interest.

        Returns:
            new_coords: The warped reference points
                        on the bird's eye view grid.
        """

        bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(
            self.bev_size
        )

        wrap_r, wrap_t = self._get_matrix(meta, idx, bev_max_x, bev_max_y)
        wrap_r = torch.tensor(wrap_r).to(device=feat.device)
        wrap_t = torch.tensor(wrap_t).to(device=feat.device)

        # Compute the transformed coordinates
        new_coords = []
        batch = wrap_r.shape[0]
        for i in range(batch):
            new_coord = torch.matmul(self.coords, wrap_r[i]).float()
            new_coord += wrap_t[i]
            new_coord += self.offset
            new_coords.append(new_coord)
        new_coords = torch.cat(new_coords)

        return new_coords

    def export_reference_points(self, feat, meta):

        prev_point = self._get_reference_points(feat, meta, 0)

        return {"prev_points": prev_point}

    def _transform(self, feat: Tensor, points: Tensor) -> Tensor:
        """Apply a spatial transformation to a feature tensor.

        Args:
            feat: The feature tensor to be transformed.
            points: The reference points for the transformation.

        Returns:
            feat: The transformed feature tensor.
        """

        feat = self.grid_sample(
            feat,
            points,
        )
        return feat

    def fuse_model(self) -> None:
        """Perform model fusion on the modules."""
        for mod_list in [self.encoder, self.project]:
            for mod in mod_list:
                if hasattr(mod, "fuse_model"):
                    mod.fuse_model()

    def set_qconfig(self):
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        tricks.fx_force_duplicate_shared_convbn = False
        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        self.quant_stub.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "saturate": True},
            activation_calibration_qkwargs={"dtype": qint16, "saturate": True},
        )

    @fx_wrap()
    def _process_input(self, inputs: Any) -> Tensor:
        """Process the input data before further operations.

        Args:
            inputs: The input data to be processed.

        Returns:
            inputs: The processed input data.
        """

        if placeholder is not None and isinstance(inputs, placeholder):
            inputs = inputs.sample
        return inputs

    def _fuse(self, cur, prev, new_coords):
        prev = self._transform(prev, new_coords)
        prev = self._fuse_op(prev, cur)
        prev = self.project(prev)
        return prev

    def _fusion(
        self, feats: Tensor, meta: Dict, compile_model: bool
    ) -> Tensor:
        """Perform the fusion operation on the input features.

        Args:
            feats: The input features.
            meta: The meta information.
            compile_model: A flag indicating whether to compile the model.

        Returns:
            fused_feat: The fused features.
        """
        if compile_model is True:
            prev_feats = self._process_input(meta["prev_feats"])
            prev_feats = self.quant(prev_feats)
            prev_points = self._process_input(meta["prev_points"])
            n, c, h, w = feats.shape
            cur_feat = feats
        else:
            prev_points = []
            for i in range(0, self.num_seq - 1):
                new_coords = self._get_reference_points(feats, meta, i)
                prev_points.append(new_coords)
            prev_points = self._warp_stack(prev_points)
            n, c, h, w = feats.shape
            feats = feats.view(-1, self.num_seq, c, h, w)
            prev_feats = (
                feats[:, 1:]
                .permute(1, 0, 2, 3, 4)
                .contiguous()
                .view(-1, c, h, w)
            )
            cur_feat = feats[:, 0]
        bs = cur_feat.shape[0]
        prev = prev_feats[(self.num_seq - 2) * bs :]
        prev_points = self.quant_stub(prev_points)
        for i in reversed(range(0, self.num_seq - 2)):
            cur = prev_feats[i * bs : (i + 1) * bs]
            new_coords = prev_points[(i + 1) * bs : (i + 2) * bs]
            prev = self._fuse(cur, prev, new_coords)
        fused_feat = self._fuse(cur_feat, prev, prev_points[0:bs])
        return fused_feat, cur_feat

    @fx_wrap()
    def _warp_stack(self, prev_points):
        return torch.stack(prev_points, dim=0).flatten(0, 1)


@OBJECT_REGISTRY.register
class AddTemporalFusion(TemporalFusion):
    """Simple Add Temporal fusion for bev feats."""

    def __init__(self, **kwargs):
        super(AddTemporalFusion, self).__init__(**kwargs)
        self.floatFs = FloatFunctional()

    def _fuse_op(self, prev: Tensor, cur: Tensor) -> Tensor:
        """Fuse the previous and the current features.

        Args:
            prev: The previous features.
            cur: The current features.
        Returns:
            fused_features: The fused features.
        """

        return self.floatFs.add(prev, cur)
