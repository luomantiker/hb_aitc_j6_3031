# Copyright (c) Horizon Robotics. All rights reserved.

# PointPillars point cloud encoder
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.functional import point_pillars_scatter
from horizon_plugin_pytorch.quantization import QuantStub
from torch.nn import functional as F

from hat.models.utils import _get_paddings_indicator
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["PillarFeatureNet", "PointPillarScatter"]


class PFNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_kwargs: dict = None,
        last_layer: bool = False,
        use_conv: bool = True,
        pool_size: Tuple[int, int] = (1, 1),
        hw_reverse: bool = False,
    ):
        """Pillar Feature Net Layer.

        This layer is used to convert point cloud into pseudo-image.
        Can stack multiple to form Pillar Feature Net.
        The original PointPillars paper uses only a single PFNLayer.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            bn_kwrags (dict): batch normalization arguments. Defaults to None.
            last_layer (bool, optional): if True, there is no concatenation of
                layers. Defaults to False.
        """
        super().__init__()
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if bn_kwargs is None:
            bn_kwargs = dict(eps=1e-3, momentum=0.01)  # noqa C408

        self.bn_kwargs = bn_kwargs
        self.use_conv = use_conv
        self.hw_reverse = hw_reverse

        if not self.use_conv:
            self.linear = nn.Linear(in_channels, self.units, bias=False)
            self.norm = nn.BatchNorm1d(self.units, **self.bn_kwargs)
        else:
            self.linear = nn.Conv2d(
                in_channels, self.units, kernel_size=1, bias=False
            )
            self.norm = nn.BatchNorm2d(self.units, **bn_kwargs)
            self.relu = nn.ReLU(inplace=True)
            self.max_pool = nn.MaxPool2d(
                kernel_size=pool_size, stride=pool_size
            )

    def forward(self, inputs: torch.Tensor):
        if not self.use_conv:
            x = self.linear(inputs)
            x = (
                self.norm(x.permute(0, 2, 1).contiguous())
                .permute(0, 2, 1)
                .contiguous()
            )
            x = F.relu(x)

            x_max = torch.max(x, dim=1, keepdim=True)[0]

            if self.last_vfe:
                return x_max
            else:
                x_repeat = x_max.rpeat(1, inputs.shape[1], 1)
                x_concatenated = torch.cat([x, x_repeat], dim=2)
                return x_concatenated
        else:
            x = self.linear(inputs)
            x = self.norm(x)
            x = self.relu(x)
            x_max = self.max_pool(x)
            if self.hw_reverse:
                x_max = x_max.permute(0, 2, 3, 1).contiguous()
            else:
                x_max = x_max.permute(0, 3, 2, 1).contiguous()
            return x_max

    def fuse_model(self):
        if self.use_conv:
            try:
                from horizon_plugin_pytorch import quantization

                fuser_func = quantization.fuse_known_modules
            except Warning:
                logging.warning(
                    "Please install horizon_plugin_pytorch first, "
                    "otherwise use pytorch official quantification"
                )
                from torch.quantization.fuse_modules import fuse_known_modules

                fuser_func = fuse_known_modules

            fuse_list = ["linear", "norm", "relu"]
            torch.quantization.fuse_modules(
                self,
                fuse_list,
                inplace=True,
                fuser_func=fuser_func,
            )

    def set_qconfig(
        self,
    ):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class PillarFeatureNet(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        num_filters: Tuple[int, ...] = (64,),
        with_distance: bool = False,
        voxel_size: Tuple[float, float, int] = (0.2, 0.2, 4),
        pc_range: Tuple[float, ...] = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
        bn_kwargs: dict = None,
        quantize: bool = False,
        use_4dim: bool = False,
        use_conv: bool = False,
        pool_size: Tuple[int, int] = (1, 1),
        normalize_xyz: bool = False,
        hw_reverse: bool = False,
    ):
        """Pillar Feature Net.

        The network prepares the pillar features and performs forward pass
        through PFNLayers.

        Args:
            num_input_features: number of input features.
            num_filters: number of filters in each
                of the N PFNLayers.
            with_distance: whether to include Eulidean
                distance to points.
            voxel_size: size of voxels.
            pc_range: point cloud range.
            bn_kwargs: batch norm parameters.
            quantize: Whether to quantize PillarFeatureNet.
            use_4dim: Whether to use 4-dim of points.
            use_conv: Whether to use conv in PFNLayer.
            pool_size: Max_pool size of PFNLayer.
            normalize_xyz: Whether to normalize xyz dims feature.
        """
        super().__init__()
        assert len(num_filters) > 0

        self.use_4dim = use_4dim
        self.pool_size = pool_size
        self.use_conv = use_conv
        self.normalize_xyz = normalize_xyz

        self.num_input = num_input_features
        if not self.use_4dim:
            num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters, out_filters = num_filters[i], num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    bn_kwargs,
                    last_layer,
                    use_conv=self.use_conv,
                    pool_size=self.pool_size,
                    hw_reverse=hw_reverse,
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate
        # pillar offset
        self.vx, self.vy = voxel_size[:2]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

        self.quantize = quantize
        if self.quantize:
            self.quant = QuantStub()
        self.pc_range = torch.tensor(pc_range)

    def forward(
        self,
        features: torch.Tensor,
        num_voxels: Optional[torch.Tensor] = None,
        coors: Optional[torch.Tensor] = None,
        horizon_preprocess: bool = False,
    ):
        if horizon_preprocess:
            # used horizon preprocess(which support quantize),
            # skip default preprocess here.
            features = self._extract_feature(features)
        else:
            # default preprocess
            assert num_voxels is not None, "`num_voxels` can not be None."
            features = self._extend_dim(features, num_voxels, coors)
            features = self._extract_feature(features)
        return features

    def _extract_feature(self, features):

        if self.quantize:
            features = self.quant(features)

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)
        if not self.use_conv:
            features = features.squeeze()
        return features

    @fx_wrap()
    def _extend_dim(self, features, num_voxels, coors):
        dtype = features.dtype
        device = features.device

        if not self.use_4dim:
            # (P, N, 4) --> (P, N, 9)
            # Find distance of x, y and z from cluster center
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True
            ) / num_voxels.type_as(features).view(-1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean

            # Find distance of x, y and z from pillar center
            f_center = torch.zeros_like(features[:, :, :2])
            f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset
            )
            f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
            )

            if self.normalize_xyz:
                self.pc_range = self.pc_range.to(device)
                mean = (self.pc_range[:3] + self.pc_range[3:6]) / 2
                norm = (self.pc_range[3:6] - self.pc_range[:3]) / 2
                features[:, :, :3] = features[:, :, :3] - mean
                features[:, :, :3] = features[:, :, :3] / norm

            # # Combine together feature decorations
            features_ls = [features, f_cluster, f_center]
        else:
            # use 4 dim
            self.pc_range = self.pc_range.to(device)
            features[:, :, :3] = features[:, :, :3] - self.pc_range[:3]
            features[:, :, :3] = features[:, :, :3] / (
                self.pc_range[3:6] - self.pc_range[:3]
            )
            features_ls = [features]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that empty pillars remain set to
        # zeros.
        voxel_count = features.shape[1]
        mask = _get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        if self.use_conv:
            # (1, C, P, N)
            features = features.unsqueeze(0).permute(0, 3, 1, 2).contiguous()

        return features

    def fuse_model(self):
        if self.quantize:
            if self.pfn_layers is not None:
                for m in self.pfn_layers:
                    m.fuse_model()

    def set_qconfig(self):
        if not self.quantize:
            self.qconfig = None
        else:
            for m in self.pfn_layers:
                if hasattr(m, "set_qconfig"):
                    m.set_qconfig()


@OBJECT_REGISTRY.register
class PointPillarScatter(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        use_horizon_pillar_scatter: bool = False,
        quantize=False,
        **kwargs,
    ):
        """Point Pillar's Scatter.

        Scatter the features back to the canvas to form a pseudo-image.
        The output pseudo-image has a shape of NCHW, where the H & W is
        determined by the point cloud's range and each voxel's size.

        Args:
            num_input_features (int): number of input features.
            use_horizon_pillar_scatter: Whether to use horizon pillar scatter,
                which is same with origin PillarScatter but support quantize.
        """
        super().__init__()
        self.nchannels = num_input_features
        self.nx = 0
        self.ny = 0
        self.use_horizon_pillar_scatter = use_horizon_pillar_scatter
        self.quantize = quantize

    @fx_wrap()
    def forward(
        self,
        voxel_features: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int,
        input_shape: torch.Tensor,
    ):
        """Forward pass of the scatter module.

        Note: batch_size has to be passed in additionally, because voxel
        features are concatenated on the M-channel since the number of voxels
        in each frame differs and there is no easy way we concat them same as
        image (CHW -> NCHW). M-channel concatenation would require another
        tensor to record number of voxels per frame, which indicates batch_size
        consequently.

        Args:
            voxel_features (torch.Tensor): MxC tensor of pillar features, where
                M is number of pillars, C is each pillar's feature dim.
            coords (torch.Tensor): each pillar's original BEV coordinate.
            batch_size (int): batch size of the feature.
            input_shape (torch.Tensor): shape of the expected BEC map. Derived
                from point-cloud range and voxel size.

        Returns:
            [torch.Tensor]: a BEV view feature tensor with point features
                scattered on it.
        """
        self.nx = input_shape[0]
        self.ny = input_shape[1]

        P, C = voxel_features.size(-2), voxel_features.size(-1)
        voxel_features = voxel_features.reshape(P, C)

        if self.use_horizon_pillar_scatter:
            out_shape = (batch_size, self.nchannels, self.ny, self.nx)
            batch_canvas = point_pillars_scatter(
                voxel_features, coords, out_shape
            )
        else:
            # batch_canvas will be the final output.
            batch_canvas = []
            for batch_id in range(batch_size):
                # Create a canvas for this sample
                canvas = torch.zeros(
                    self.nchannels,
                    self.nx * self.ny,  # This is P. p = nx * ny.
                    dtype=voxel_features.dtype,
                    device=voxel_features.device,
                )

                # Only include non-empty pillars
                batch_mask = coords[:, 0] == batch_id

                this_coords = coords[batch_mask, :]
                indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
                indices = indices.type(torch.long)
                voxels = voxel_features[batch_mask, :]
                voxels = voxels.t()

                # Scatter the blob back to teh canvas
                canvas[:, indices] = voxels

                # Append to a list for later stacking
                batch_canvas.append(canvas)

            # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
            batch_canvas = torch.stack(batch_canvas, 0)

            # Undo the column stacking to final 4-dim tensor
            batch_canvas = batch_canvas.view(
                batch_size, self.nchannels, self.ny, self.nx
            )
            # canvas = torch.zeros(
            #     self.nchannels,
            #     self.nx * self.ny,
            #     dtype=voxel_features.dtype,
            #     device=voxel_features.device)

            # indices = coords[:, 2] * float(self.nx) + coords[:, 3]
            # indices = indices.long()
            # voxels = voxel_features.t()
            # # Now scatter the blob back to the canvas.
            # canvas[:, indices] = voxels
            # # Undo the column stacking to final 4-dim tensor
            # canvas = canvas.view(1, self.nchannels, self.ny, self.nx)
            # return canvas

        return batch_canvas

    def set_qconfig(self):
        if self.quantize:
            from hat.utils import qconfig_manager

            self.qconfig = qconfig_manager.get_default_qat_qconfig()
        else:
            self.qconfig = None
