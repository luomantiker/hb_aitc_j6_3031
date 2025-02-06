# Copyright (c) Horizon Robotics. All rights reserved.
import collections
import logging
from collections.abc import Sequence
from typing import Any, Dict, List, Tuple

import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor, nn

from hat.core.nus_box3d_utils import adjust_coords, get_min_max_coords
from hat.models.base_modules.conv_module import ConvModule2d
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

try:
    from hbdk.torch_script.placeholder import placeholder
except ImportError:
    placeholder = None

logger = logging.getLogger(__name__)


class ViewTransformer(nn.Module):
    """The view transform structure for converting image view to bev view.

    Args:
        num_views: Number for image views.
        bev_size: Bev size.
        grid_size: Grid size.
        mode: Mode for grid sample.
        padding_mode: Padding mode for grid sample.
        grid_quant_scale: Quanti scale for grid sample.

    """

    def __init__(
        self,
        num_views: int,
        bev_size: Tuple[float],
        grid_size: Tuple[float],
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        grid_quant_scale: float = 1 / 512,
    ):
        super(ViewTransformer, self).__init__()
        self.num_views = num_views
        self.bev_size = bev_size
        self.grid_size = grid_size
        self.floatFs = FloatFunctional()

        self.quant_stub = QuantStub(grid_quant_scale)
        self.grid_sample = hnn.GridSample(
            mode=mode,
            padding_mode=padding_mode,
        )

    def _gen_2d_points(self) -> Tensor:
        """Generate 2D points in the Bird's Eye View (BEV) space."""

        # Get the minimum and maximum x and y coordinates in the BEV space
        bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(
            self.bev_size
        )

        W = self.grid_size[0]
        H = self.grid_size[1]

        # Generate a tensor `x` containing the x-coordinates of the grid
        x = (
            torch.linspace(bev_min_x, bev_max_x, W)
            .reshape((1, W))
            .repeat(H, 1)
        ).double()
        # Generate a tensor `y` containing the y-coordinates of the grid
        y = (
            torch.linspace(bev_min_y, bev_max_y, H)
            .reshape((H, 1))
            .repeat(1, W)
        ).double()

        ones = torch.ones((H, W)).double()
        coords = torch.stack([x, y, ones], dim=-1)
        return coords

    def _gen_3d_points(self, z_range: Tuple[int]) -> Tensor:
        """Generate 3D points in the Bird's Eye View (BEV) space.

        Args:
            z_range: The range of z-coordinates.

        Returns:
            coords: The generated 3D points in BEV space.
        """
        # Get the minimum and maximum x and y coordinates in the BEV space
        bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(
            self.bev_size
        )

        W = self.grid_size[0]
        H = self.grid_size[1]

        Z = int(z_range[1] - z_range[0])

        # Generate a tensor `x` containing the x-coordinates of the grid
        x = (
            torch.linspace(bev_min_x, bev_max_x, W)
            .reshape((1, W, 1))
            .repeat(H, 1, Z)
        ).double()

        # Generate a tensor `y` containing the y-coordinates of the grid
        y = (
            torch.linspace(bev_min_y, bev_max_y, H)
            .reshape((H, 1, 1))
            .repeat(1, W, Z)
        ).double()

        # Generate a tensor `z` containing the z-coordinates of the grid
        # based on the given z_range
        z = (
            torch.linspace(self.z_range[0], self.z_range[1], Z)
            .reshape((1, 1, Z))
            .repeat(H, W, 1)
        ).double()

        # Generate a tensor `ones` containing all ones of shape (H, W, Z)
        ones = torch.ones((H, W, Z)).double()
        coords = torch.stack([x, y, z, ones], dim=-1)
        return coords

    def _convert_p2tensor(self, points: Any) -> Any:
        """Convert points to tensors.

        Args:
            points: The points to be converted.

        Returns:
            points: The converted points as tensors.
        """

        if isinstance(points, collections.abc.Sequence):
            for i in range(len(points)):
                points[i] = self._convert_p2tensor(points[i])
        if placeholder is not None and isinstance(points, placeholder):
            points = points.sample
        return points

    @fx_wrap()
    def forward(
        self, feats: Tensor, meta: Dict, compile_model: bool, **kwargs
    ) -> Any:
        """Perform the forward pass through the modules.

        Processes the input features and meta information to perform
        spatial transformation and generate reference points.

        Args:
            feats: The input features.
            meta: The meta information.
            compile_model: A flag indicating whether to compile the model.

        Returns:
            transformed_feats: The transformed features.
            points: The reference points.
        """

        feats = self._extract(feats)
        if compile_model is True:
            points = self._get_points_from_meta(meta)
        else:
            with torch.no_grad():
                points = self.gen_reference_point(meta, feats)
        return self._spatial_transfom(feats, points), None

    def gen_reference_point(self, meta: Dict, feats: Tensor) -> Any:
        """Generate refrence points.

        Args:
            meta: A dictionary containing the input data.
            feats: The input for reference point generator.

        Returns:
            The Reference points.
        """
        with torch.no_grad():
            homography = self._get_homography(meta, feats.shape[2:])
            points = self._gen_reference_point(homography, feats.shape[2:])
        return points

    def export_reference_points(
        self, meta: Dict, feat_hw: Tuple[int, int]
    ) -> Dict:
        """Export refrence points.

        Args:
            meta: A dictionary containing the input data.
            feat_hw: View transformer input shape
                     for generationg reference points.

        Returns:
            The Reference points.
        """
        homography = self._get_homography(meta, feat_hw)
        points = self._gen_reference_point(homography, feat_hw)
        if not isinstance(points, Sequence):
            points = [points]

        ref_p_dict = {}
        for i, ref_p in enumerate(points):
            ref_p_dict[f"points{i}"] = ref_p
        return ref_p_dict

    def _extract(self, feats: Tensor) -> Tensor:
        """Extract the input features.

        Args:
            feats: The input features.

        Returns:
            feats: The input features.
        """
        return feats

    def _get_homography(self, meta: Dict, feat_hw: Tuple[int, int]) -> Tensor:
        """Compute the homography matrix for mapping coordinates.

        Args:
            meta: The meta information.
            feat_hw: view transformer input shape
                     for generationg reference points.

        Returns:
            homography: The computed homography matrix.
        """

        # Get the ego2img homography matrix and the input
        # and original feature heights and widths
        homography = meta["ego2img"]
        orig_hw = meta["img"][0].shape[1:]
        scales = (feat_hw[0] / orig_hw[0], feat_hw[1] / orig_hw[1])
        view = np.eye(4)
        view[0, 0] = scales[1]
        view[1, 1] = scales[0]
        view = torch.tensor(view).to(device=homography.device).double()
        # Perform the matrix multiplication between
        # the view transformation matrix and the homography matrix
        homography = torch.matmul(view, homography.double())
        return homography

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        self.quant_stub.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "saturate": True},
            activation_calibration_qkwargs={"dtype": qint16, "saturate": True},
        )


@OBJECT_REGISTRY.register
class WrappingTransformer(ViewTransformer):
    """The IPM view transform for converting image view to bev view."""

    def __init__(self, **kwargs):
        super(WrappingTransformer, self).__init__(**kwargs)

    def _get_points_from_meta(self, meta: dict) -> Any:
        """Get the points from the meta information and convert them to tensors.

        Args:
            meta: The meta information.

        Returns:
            points: The points converted to tensors.
        """
        points = meta["points0"]
        return self._convert_p2tensor(points)

    def _gen_reference_point(
        self, homography: Tensor, feat_hw: Tuple[int, int] = None
    ) -> Tensor:
        """Generate and adjust the reference points.

        Args:
            homography: The homography matrix.
            feat_hw: View transformer input shape
                     for generationg reference points.

        Returns:
            new_coords: The generated and adjusted reference points.
        """

        # Generate 2D points coordinates in BEV space
        coords = self._gen_2d_points().to(device=homography.device)

        homography = homography[:, :3, (0, 1, 3)]

        # Perform mapping from BEV space to image space using homography matrix
        new_coords = []
        for homo in homography:
            new_coord = torch.matmul(coords, homo.permute((1, 0))).float()
            new_coords.append(new_coord)
        new_coords = torch.stack(new_coords)
        new_coords[..., 2] = torch.clamp(new_coords[..., 2], min=0.05)
        # Normalize the x and y coordinates by the z-coordinate
        X = new_coords[..., 0] / new_coords[..., 2]
        Y = new_coords[..., 1] / new_coords[..., 2]
        new_coords = torch.stack((X, Y), dim=-1)
        new_coords = adjust_coords(new_coords, self.grid_size)
        return new_coords

    def _spatial_transfom(self, feat: Tensor, points: Tensor) -> Tensor:
        """Apply spatial transformation to the input features.

        Args:
            feat: The input features.
            points: The reference points.

        Returns:
            fused_feats: The output features after spatial transformation.
        """
        if placeholder is not None and isinstance(points, placeholder):
            points = points.sample
        trans_feats = self.grid_sample(
            feat,
            self.quant_stub(points),
        )
        batch_size = int(trans_feats.shape[0] / self.num_views)

        if self.training or batch_size > 1:
            trans_feats = trans_feats.view(
                batch_size,
                self.num_views,
                trans_feats.shape[1],
                trans_feats.shape[2],
                trans_feats.shape[3],
            )
            fused_feats = self.floatFs.sum(trans_feats, keepdim=True, dim=1)
            fused_feats = fused_feats.view(
                batch_size,
                trans_feats.shape[2],
                trans_feats.shape[3],
                trans_feats.shape[4],
            )
        else:
            fused_feats = self.floatFs.sum(trans_feats, keepdim=True, dim=0)

        return fused_feats

    def fuse_model(self) -> None:
        pass


@OBJECT_REGISTRY.register
class LSSTransformer(ViewTransformer):
    """The Lift-Splat-Shoot view transform for converting image view to bev view.

    Args:
        in_channels: In channel of feature.
        feat_channels: Feature channel of lift.
        z_range: The range of Z for bev coordarin.
        num_points: Num points for each voxel.
        depth: Depth value.
        mode: Mode for grid sample.
        padding_mode: Padding mode for grid sample.
        dgrid_quant_scale: Quanti scale for depth grid sample.

    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        z_range: Tuple[float] = (-10.0, 10.0),
        num_points: int = 10,
        depth: int = 60,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        depth_grid_quant_scale: float = 1 / 512,
        **kwargs,
    ):
        super(LSSTransformer, self).__init__(
            mode=mode, padding_mode=padding_mode, **kwargs
        )
        self.depth = depth
        self.z_range = z_range
        self.depth_net = ConvModule2d(
            in_channels=in_channels,
            out_channels=depth,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )
        self.feat_net = ConvModule2d(
            in_channels=in_channels,
            out_channels=feat_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )
        self.softmax = nn.Softmax(dim=1)
        self.dquant_stub = QuantStub(depth_grid_quant_scale)
        self.dgrid_sample = hnn.GridSample(
            mode=mode,
            padding_mode=padding_mode,
        )
        self.num_points = num_points

    def gen_reference_point(self, meta: Dict, feats: List[Tensor]) -> Any:
        """Generate refrence points.

        Args:
            meta: A dictionary containing the input data.
            feats: The input for reference point generator.

        Returns:
            The Reference points.
        """
        return super().gen_reference_point(meta, feats[0])

    def _get_points_from_meta(self, meta: Dict) -> List[Tensor]:
        """Extract points from metadata dictionary.

        Args:
            meta: Metadata dictionary.

        Returns:
            points: List of extracted points as converted tensors.
        """
        points = []
        for k in meta.keys():
            if k.startswith("points"):
                points.append(self._convert_p2tensor(meta[k]))
        assert len(points) == 2
        return points

    def _gen_reference_point(
        self, homography: Tensor, feat_hw: Tuple[int, int]
    ) -> Tuple[Tensor]:
        """Generate reference points using homography matrix and feature tensor.

        Args:
            homography: Homography matrix.
            feat_hw: View transformer input shape
                     for generationg reference points.

        Returns:
            Tuple containing the generated feature points and depth points.
        """

        coords = self._gen_3d_points(self.z_range).to(device=homography.device)
        H, W, Z = coords.shape[:3]
        new_coords = []
        for homo in homography:
            new_coord = torch.matmul(coords, homo.permute((1, 0))).float()
            new_coord = new_coord.permute((2, 0, 1, 3))
            new_coords.append(new_coord)
        new_coords = torch.stack(new_coords, dim=1)
        B = new_coords.shape[1] // self.num_views

        new_coords = (
            new_coords.view(-1, B, self.num_views, H, W, 4)
            .permute(0, 2, 1, 3, 4, 5)
            .contiguous()
        )

        d = torch.clamp(new_coords[..., 2], min=0.05)
        X = (new_coords[..., 0] / d).long()
        Y = (new_coords[..., 1] / d).long()
        D = new_coords[..., 2].long()

        idx = (
            (
                torch.linspace(0, self.num_views - 1, self.num_views)
                .reshape((1, self.num_views, 1, 1, 1))
                .repeat(Z, 1, B, H, W)
            )
            .long()
            .to(device=homography.device)
        )
        new_coords = torch.stack([X, Y, D, idx], dim=-1)

        feat_h, feat_w = feat_hw
        invalid = (
            (new_coords[..., 0] < 0)
            | (new_coords[..., 0] >= feat_w)
            | (new_coords[..., 1] < 0)
            | (new_coords[..., 1] >= feat_h)
            | (new_coords[..., 2] < 0)
            | (new_coords[..., 2] >= self.depth)
        )

        new_coords[invalid] = torch.tensor(
            (feat_w - 1, feat_h - 1, self.depth, self.num_views - 1)
        ).to(device=homography.device)
        new_coords = new_coords.view(-1, B, H, W, 4)
        rank = (
            new_coords[..., 2] * feat_h * feat_w * self.num_views
            + new_coords[..., 1] * feat_w * self.num_views
            + new_coords[..., 0] * self.num_views
            + new_coords[..., 3]
        )
        rank, _ = rank.topk(self.num_points, dim=0, largest=False)
        D = rank // (feat_h * feat_w * self.num_views)
        rank = rank % (feat_h * feat_w * self.num_views)

        Y = rank // (feat_w * self.num_views)
        rank = rank % (feat_w * self.num_views)

        X = rank // self.num_views
        idx = rank % self.num_views

        idx_Y = idx * feat_h + Y
        feat_coords = torch.stack((X, idx_Y), dim=-1)
        feat_points = adjust_coords(feat_coords, self.grid_size)

        X_Y = Y * feat_w + X
        idx_D = idx * self.depth + D
        depth_coords = torch.stack((X_Y, idx_D), dim=-1)
        depth_points = adjust_coords(depth_coords, self.grid_size)
        feat_points = feat_points.view(-1, H, W, 2)
        depth_points = depth_points.view(-1, H, W, 2)
        return (feat_points, depth_points)

    def _extract(
        self, feats: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Extract features and depth using the feature tensor.

        Args:
            feats: Feature tensor.

        Returns:
            Tuple containing the extracted features and depth.
        """

        new_feats = []
        depth = self.softmax(self.depth_net(feats))
        new_feats = self.feat_net(feats)
        return new_feats, depth

    def _spatial_transfom(self, feats: Tensor, points: Tensor) -> Tensor:
        """Apply spatial transformation to the features using the given points.

        Args:
            feats: Tuple of feature tensor and depth tensor.
            points: Tuple of feature points and depth points.

        Returns:
            The transformed feature tensor.
        """
        feat, dfeat = feats
        fpoints, dpoints = points
        fpoints = self.quant_stub(fpoints)
        dpoints = self.dquant_stub(dpoints)

        B = feat.shape[0] // self.num_views
        C, H, W = feat.shape[1:]

        if self.training or B > 1:
            feat = feat.view(B, self.num_views, C, H, W)
            feat = feat.permute(0, 2, 1, 3, 4).contiguous()
        else:
            feat = feat.permute(1, 0, 2, 3).contiguous()

        feat = feat.view(B, C, -1, W)

        dfeat = dfeat.view(B, 1, -1, H * W)
        homo_feats = []
        for i in range(self.num_points):
            homo_feat = self.grid_sample(
                feat,
                fpoints[i * B : (i + 1) * B],
            )

            homo_dfeat = self.dgrid_sample(
                dfeat,
                dpoints[i * B : (i + 1) * B],
            )
            homo_feat = self.floatFs.mul(homo_feat, homo_dfeat)
            homo_feats.append(homo_feat)

        trans_feat = homo_feats[0]
        for f in homo_feats[1:]:
            trans_feat = self.floatFs.add(trans_feat, f)
        return trans_feat

    def fuse_model(self):
        """Perform model fusion on the modules."""
        self.depth_net.fuse_model()
        self.feat_net.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat.utils import qconfig_manager

        self.dquant_stub.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "saturate": True},
            activation_calibration_qkwargs={"dtype": qint16, "saturate": True},
        )
        super().set_qconfig()


class GKTMultiHeadAttention(nn.Module):
    """The GKT multi head attention.

    Args:
        embed_dims: Dims for transformer.
        nhead: number of head.
        dropout: dropout rate.
    """

    def __init__(self, embed_dims: int, nhead: int = 8, dropout: float = 0.0):
        super().__init__()
        self.q = ConvModule2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )

        self.k = ConvModule2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )

        self.v = ConvModule2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )

        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(0)

        self.q_k_mul = FloatFunctional()
        self.q_k_sum = FloatFunctional()
        self.att_v_mul = FloatFunctional()
        self.v_sum = FloatFunctional()

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Forward pass of a self-attention mechanism.

        Args:
            q: Query vectors.
            k: Key vectors.
            v: Value vectors.

        Returns:
            x: The final output tensor after the self-attention operation.
        """
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        attention = self.q_k_mul.mul(q, k)
        attention = self.q_k_sum.sum(attention, dim=1, keepdim=True)
        attention = self.softmax(attention)

        x = self.att_v_mul.mul(attention, v)
        x = self.v_sum.sum(x, dim=0, keepdim=True)

        return x


class GKTTransformerLayer(nn.Module):
    """The GKT transformer layer.

    Args:
        embed_dims: Dims for transformer.
        dropout: dropout rate.
    """

    def __init__(self, embed_dims: int, dropout: float = 0.1):
        super().__init__()

        self.multihead_attn = GKTMultiHeadAttention(embed_dims)

        self.linear1 = ConvModule2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )
        self.linear2 = ConvModule2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = hnn.LayerNorm(normalized_shape=(embed_dims, 1, 1), dim=1)
        self.norm2 = hnn.LayerNorm(normalized_shape=(embed_dims, 1, 1), dim=1)
        self.add1 = FloatFunctional()
        self.add2 = FloatFunctional()
        self.act = nn.ReLU(inplace=True)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Forward pass through the transformer block.

        Args:
            q: The query tensor.
            k: The key tensor.
            v: The value tensor.

        Returns:
            tgt: The output tensor after passing through the transformer block.
        """
        norm_q = self.norm1(q)
        tgt = self.multihead_attn(norm_q, k, v)

        tgt = self.add1.add(self.dropout1(tgt), norm_q)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout2(self.act(self.linear1(tgt))))

        tgt = self.add2.add(tgt, self.dropout3(tgt2))
        return tgt


@OBJECT_REGISTRY.register
class GKTTransformer(ViewTransformer):
    """The GKT view transform for converting image view to bev view.

    Args:
        kernel_size: Kernel size for points.
        embed_dims: Dims for transformer.
    """

    def __init__(
        self,
        kernel_size: Tuple[float] = (3, 3),
        embed_dims: int = 160,
        grid_size: Tuple[float] = None,
        **kwargs,
    ):
        super(GKTTransformer, self).__init__(grid_size=grid_size, **kwargs)
        self.kernel_size = kernel_size
        if grid_size is None:
            grid_size = (64, 64)

        self.gkt_layer = GKTTransformerLayer(embed_dims)
        query_pos_embed = torch.zeros((1, embed_dims, *grid_size))
        self.query_pos_embed = nn.Parameter(
            query_pos_embed, requires_grad=True
        )
        self.floatFs = FloatFunctional()
        self.floatFs2 = FloatFunctional()
        self.query_quant_stub = QuantStub()

    def _get_points_from_meta(self, meta: Dict) -> List[Tensor]:
        """Get points from the metadata dictionary and convert them to tensors.

        Args:
            meta: The metadata dictionary containing the points as values.

        Returns:
            points: A list of tensors representing the points.
        """
        points = []
        for k in meta.keys():
            if k.startswith("points"):
                points.append(self._convert_p2tensor(meta[k]))
        return points

    def _gen_coords_from_kernel(self, coords: Tensor) -> Tensor:
        """Generate new coordinates.

        Args:
            coords: The input coordinates.

        Returns:
            kernel_coords: The new tensor of coordinates
                           with kernel offsets applied.
        """
        h = self.kernel_size[0] - 2
        w = self.kernel_size[1] - 2
        kernel_coords = []
        for i in range(-h, h + 1):
            for j in range(-w, w + 1):
                new_coords = coords.clone()
                new_coords[..., 0] += j
                new_coords[..., 1] += i
                kernel_coords.append(new_coords)
        kernel_coords = torch.stack(kernel_coords)
        return kernel_coords

    def _gen_reference_point(
        self, homography: Tensor, feat_hw: Tuple[int, int] = None
    ) -> Tensor:
        """Generate and adjust the reference points.

         Args:
             homography: The homography matrix.
             feat_hw: View transformer input shape
                      for generationg reference points.

        Returns:
             new_coords: The generated and adjusted reference points.
        """

        # Generate 2D points coordinates in BEV space
        coords = self._gen_2d_points().to(device=homography.device)
        homography = homography[:, :3, (0, 1, 3)]
        new_coords = []
        for homo in homography:
            new_coord = torch.matmul(coords, homo.permute((1, 0))).float()
            new_coords.append(new_coord)
        new_coords = torch.stack(new_coords, dim=0)
        new_coords[..., 2] = torch.clamp(new_coords[..., 2], min=0.05)
        X = new_coords[..., 0] / new_coords[..., 2]
        Y = new_coords[..., 1] / new_coords[..., 2]
        new_coords = torch.stack((X, Y), dim=-1)
        new_coords = self._gen_coords_from_kernel(new_coords)
        new_coords = adjust_coords(new_coords, self.grid_size)
        new_coords = torch.unbind(new_coords, dim=0)
        return new_coords

    def _spatial_transfom(self, feats: Tensor, points: Tensor) -> Tensor:
        """Apply spatial transformation to the input features.

        Using the provided reference points and return the fused features
        after applying a Graph Kernel Transformer (GKT) layer.

        Args:
            feats: The input features.
            points: The reference points.

        Returns:
            fused_feats: The fused features after spatial
                         transformation and GKT layer.
        """
        num_points = self.kernel_size[0] * self.kernel_size[1]
        N, C, _, _ = feats.shape
        H, W = self.grid_size
        B = N // self.num_views
        trans_feats = []
        for i in range(num_points):
            trans_feat = self.grid_sample(
                feats,
                self.quant_stub(points[i]),
            )
            if B > 1:
                trans_feat = trans_feat.view(B, self.num_views, C, H, W)
                trans_feat = self.floatFs.sum(
                    trans_feat, dim=1, keepdim=True
                ).squeeze()
            else:
                trans_feat = self.floatFs.sum(trans_feat, dim=0, keepdim=True)
            trans_feats.append(trans_feat)
        trans_feats = self.floatFs.cat(trans_feats)

        query_pos_embed = self.query_quant_stub(self.query_pos_embed)
        if B > 1:
            fused_feats = []
            trans_feats = trans_feats.view(num_points, B, C, H, W).permute(
                1, 0, 2, 3, 4
            )
            for i in range(B):
                fused_feat = self.gkt_layer(
                    query_pos_embed, trans_feats[i], trans_feats[i]
                )
                fused_feats.append(fused_feat)
            fused_feats = self.floatFs2.cat(fused_feats)
        else:
            fused_feats = self.gkt_layer(
                query_pos_embed, trans_feats, trans_feats
            )
        return fused_feats

    def fuse_model(self) -> None:
        pass
