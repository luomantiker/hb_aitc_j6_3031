# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Any, Dict, List, Optional

import horizon_plugin_pytorch.nn as hnn
import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.quantization import (
    FixedScaleObserver,
    QuantStub,
    get_default_qat_qconfig,
)
from torch.cuda.amp import autocast
from torch.quantization import DeQuantStub

from hat.models.weight_init import constant_init, xavier_init
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "Scale",
    "DecoupledMultiheadAttention",
    "DeformableFeatureAggregationOE",
    "AsymmetricFFNOE",
    "DenseDepthNetOE",
    "DeformableFeatureAggregationOEv2",
]


QINT16_MAX = 32767.5


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale: Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
        self.scale_quant_stub = QuantStub()
        self.mul = FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass through the Scale module.

        Args:
            x (torch.Tensor): Input tensor to be scaled.

        Returns:
            torch.Tensor: Scaled output tensor.
        """
        scale = self.scale_quant_stub(self.scale)
        return self.mul.mul(x, scale)


class DecoupledMultiheadAttention(nn.Module):
    """
    Decoupled Multihead Attention module.

    Args:
        embed_dim: Dimension of embedding. Default is 256.
        num_heads: Number of attention heads. Default is 8.
        batch_first: If True, input and output tensors are
                     provided as (batch, seq_len, feature).
                     Default is True.
        dropout: Dropout probability. Default is 0.1.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        batch_first: bool = True,
        dropout: float = 0.1,
    ):
        super(DecoupledMultiheadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim * 2,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.query_cat = FloatFunctional()
        self.key_cat = FloatFunctional()
        self.add = FloatFunctional()

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Decoupled Multihead Attention module.

        Args:
            query: Query tensor of shape (seq_len, batch_size, embed_dim).
            key: Key tensor of shape (seq_len, batch_size, embed_dim).
            value: Value tensor of shape (seq_len, batch_size, embed_dim).
            query_pos: Positional encoding for query tensor of shape
                       (seq_len, batch_size, embed_dim).
            key_pos: Positional encoding for key tensor of shape
                     (seq_len, batch_size, embed_dim).
            attn_mask: Attention mask tensor of shape
                       (batch_size, seq_len, seq_len).

        Returns:
            Output tensor of shape (seq_len, batch_size, embed_dim).
        """

        if key is None:
            key = query
        if query_pos is not None:
            query = self.query_cat.cat([query, query_pos], dim=-1)
        if key_pos is not None:
            key = self.key_cat.cat([key, key_pos], dim=-1)
        elif query_pos is not None:
            key = self.key_cat.cat([key, query_pos], dim=-1)

        identity = query
        if value is None:
            value = key
        out, _ = self.attn(query, key, value, attn_mask=attn_mask)
        out = self.add.add(identity, self.dropout(out))
        return out


def linear_relu_ln(
    embed_dims: int,
    in_loops: int,
    out_loops: int,
    input_dims: Optional[int] = None,
) -> List[nn.Module]:
    """
    Create a sequence of linear, ReLU, and Layer Normalization layers.

    Args:
        embed_dims: Dimensionality of the embedding.
        in_loops: Number of linear and ReLU layers to stack inside each loop.
        out_loops: Number of times to repeat the sequence of layers.
        input_dims: Dimensionality of the input.
                    Defaults to None, which sets it to embed_dims.

    Returns:
        List containing the created layers.
    """
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims, eps=1e-5))
    return layers


@OBJECT_REGISTRY.register
class DeformableFeatureAggregationOE(nn.Module):
    """
    Deformable Feature Aggregation module for multi-view object embedding.

    Args:
        kps_generator: Key points generator function.
        embed_dims: Dimensionality of the embedding.
                    Default is 256.
        num_groups: Number of groups for group-wise linear layers.
                    Default is 8.
        num_levels: Number of levels for multi-view fusion.
                    Default is 4.
        num_cams: Number of cameras/views.
                  Default is 6.
        attn_drop: Dropout probability for attention weights. Default is 0.0.
        proj_drop: Dropout probability for projection layer. Default is 0.0.
        use_camera_embed: Whether to use camera embeddings. Default is False.
        residual_mode: Residual connection mode ('add' or 'cat').
                  Default is 'add'.
    """

    def __init__(
        self,
        kps_generator: nn.Module,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_camera_embed: bool = False,
        residual_mode: str = "add",
    ):
        super(DeformableFeatureAggregationOE, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )

        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups

        self.num_cams = num_cams
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.residual_mode = residual_mode

        self.kps_generator = kps_generator
        self.num_pts = self.kps_generator.num_pts
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

        self.weight_add = FloatFunctional()
        self.cam_add = FloatFunctional()
        self.weight_softmax = nn.Softmax(dim=-2)
        self.reciprocal_op = hnn.Reciprocal()

        self.point_quant_stub = QuantStub()
        self.weights_quant_stub = QuantStub()
        self.dequant_weights = DeQuantStub()

        self.point_cat = FloatFunctional()
        self.point_matmul = FloatFunctional()
        self.point_mul = FloatFunctional()
        self.point_sum = FloatFunctional()

        self.feat_cat = FloatFunctional()
        self.feat_mul = FloatFunctional()
        self.feat_sum = FloatFunctional()
        self.residual_op = FloatFunctional()

    def init_weight(self) -> None:
        """Initialize weights of weights_fc and output_proj layers."""
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: torch.Tensor,
        projection_mat: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform forward pass through the DeformableFeatureAggregationOE module.

        Args:
            instance_feature : Instance features tensor of shape
                               (bs, num_anchor, embed_dims).
            anchor : Anchor tensor of shape
                     (bs, num_anchor, embed_dims).
            anchor_embed : Anchor embedding tensor of shape
                           (bs, num_anchor, embed_dims).
            feature_maps : Feature maps tensor.
            projection_mat : Projection matrix tensor.
            **kwargs: Additional keyword arguments.
        Returns:
            Output tensor after feature aggregation and projection.
        """

        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)

        weights = self._get_weights(
            instance_feature, anchor_embed, projection_mat
        )
        features = self.feature_sampling(
            feature_maps,
            key_points,
            projection_mat,
        )
        features = self.multi_view_level_fusion(features, weights)

        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = self.residual_op.add(output, instance_feature)
        elif self.residual_mode == "cat":
            output = self.residual_op.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(
        self,
        instance_feature: torch.Tensor,
        anchor_embed: torch.Tensor,
        projection_mat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weights for feature fusion.

        Args:
            instance_feature: Instance features tensor.
            anchor_embed: Anchor embedding tensor.
            projection_mat: Projection matrix tensor.

        Returns:
            Computed weights tensor.
        """

        bs, num_anchor = instance_feature.shape[:2]
        feature = self.weight_add.add(instance_feature, anchor_embed)
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                projection_mat[:, :, :3].reshape(bs, self.num_cams, -1)
                # .float()
            )
            feature = self.cam_add.add(
                feature[:, :, None], camera_embed[:, None]
            )
        weights = self.weights_fc(feature).reshape(
            bs, num_anchor, -1, self.num_groups
        )
        weights = self.weight_softmax(weights)

        if self.training and self.attn_drop > 0:
            if isinstance(weights, QTensor):
                self.weights_quant_stub.activation_post_process.reset_dtype(
                    weights.dtype, False
                )
                self.weights_quant_stub.activation_post_process.set_qparams(
                    weights.q_scale()
                )
            weights = self.dequant_weights(weights)
            weights = weights.reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
            weights = weights.reshape(bs, num_anchor, -1, self.num_groups)
            weights = self.weights_quant_stub(weights)
        return weights

    def project_points(
        self, key_points: torch.Tensor, projection_mat: torch.Tensor
    ) -> torch.Tensor:
        """
        Project key points onto 2D space using projection matrix.

        Args:
            key_points: Key points tensor of shape
                        (bs, num_anchor, num_pts, dim).
            projection_mat: Projection matrix tensor of shape
                        (bs, num_cams, dim, 3).

        Returns:
            Projected points tensor of shape
                 (bs, num_anchor, num_pts, 2).
        """

        bs, num_anchor, num_pts = key_points.shape[:3]

        # Extend key_points with ones for homogeneous coordinates
        ones = self.point_quant_stub(torch.ones_like(key_points[..., :1]))
        pts_extend = self.point_cat.cat([key_points, ones], dim=-1)

        # Perform projection onto 2D space
        # points_2d = self.point_matmul.matmul(
        #     projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        # ).squeeze(-1)

        points_2d = self.point_matmul.mul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None, :]
        )
        points_2d = self.point_sum.sum(points_2d, dim=-1)

        # Depth normalization and final projection
        depth = self.reciprocal_op(torch.clamp(points_2d[..., 2:3], min=1e-5))
        xy = points_2d[..., :2]
        points_2d = self.point_mul.mul(xy, depth)
        return points_2d

    def feature_sampling(
        self,
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample features from feature maps using projected key points.

        Args:
            feature_maps: List of feature map tensors of different levels.
            key_points: Key points tensor of shape
                        (bs, num_anchor, num_pts, dim).
            projection_mat: Projection matrix tensor of shape
                        (bs, num_cams, dim, 3).

        Returns:
            Sampled features tensor of shape.
        """
        num_levels = len(feature_maps)
        bs, num_anchor, num_pts = key_points.shape[:3]
        num_cams = self.num_cams

        points_2d = self.project_points(
            key_points,
            projection_mat,
        )
        points_2d = points_2d.flatten(end_dim=1)
        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm, points_2d, align_corners=False
                )
            )

        features = self.feat_cat.cat(features, dim=3)
        features = (
            features.view(bs, num_cams, -1, num_anchor, num_levels * num_pts)
            .permute(0, 3, 1, 4, 2)
            .contiguous()
            .view(bs, num_anchor, num_cams * num_levels * num_pts, -1)
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dimo
        return features

    def multi_view_level_fusion(
        self, features: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform multi-view level fusion of features using learned weights.

        Args:
            features: Input features tensor.
            weights: Weights tensor.

        Returns:
            Fused features tensor of shape
            (bs, num_anchor, embed_dims).
        """

        bs, num_anchor = weights.shape[:2]
        # Multiply features by weights
        features = self.feat_mul.mul(
            weights[..., None],
            features.reshape(
                features.shape[:-1] + (self.num_groups, self.group_dims)
            ),
        )
        features = features.view(bs, num_anchor, -1, self.embed_dims)
        features = self.feat_sum.sum(features, dim=2, keepdim=False)
        return features

    def set_qconfig(self) -> None:
        """Set the qconfig."""
        from horizon_plugin_pytorch.dtype import qint16

        from hat.utils import qconfig_manager

        self.point_quant_stub.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "averaging_constant": 0},
            weight_qat_qkwargs={
                "averaging_constant": 1,
            },
            activation_calibration_qkwargs={
                "dtype": qint16,
            },
        )
        self.point_cat.qconfig = get_default_qat_qconfig(
            dtype="qint16",
            activation_qkwargs={
                "observer": FixedScaleObserver,
                "scale": 60 / QINT16_MAX,
            },
        )
        # self.point_matmul.qconfig = get_default_qat_qconfig(
        #     dtype="qint16",
        #     activation_qkwargs={
        #         "observer": FixedScaleObserver,
        #         "scale": 60 / QINT16_MAX,
        #     },
        # )
        self.point_matmul.qconfig = get_default_qat_qconfig(
            dtype="qint16",
            # activation_qkwargs={
            #     "observer": FixedScaleObserver,
            #     "scale": 60 / QINT16_MAX,
            # },
        )
        self.point_sum.qconfig = get_default_qat_qconfig(
            dtype="qint16",
        )
        self.reciprocal_op.qconfig = get_default_qat_qconfig(
            dtype="qint16",
            activation_qkwargs={
                "observer": FixedScaleObserver,
                "scale": 11 / QINT16_MAX,
            },
        )
        self.point_mul.qconfig = get_default_qat_qconfig(
            dtype="qint16",
            activation_qkwargs={
                "observer": FixedScaleObserver,
                "scale": 1.1 / QINT16_MAX,
            },
        )
        self.kps_generator.set_qconfig()


class SE_Block(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        return x * self.att(x)


@OBJECT_REGISTRY.register
class InstanceFuseModule(nn.Module):
    def __init__(self, input_channels: List[int], fuse_channel: int):
        super().__init__()
        self.reduce_linear = nn.Linear(sum(input_channels), fuse_channel)
        self.norm = nn.LayerNorm(fuse_channel)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x = torch.cat((x1, x2), dim=-1)
        x = self.reduce_linear(x)
        x = self.norm(x)
        return x


@OBJECT_REGISTRY.register
class DeformableFeatureAggregationLiFOE(DeformableFeatureAggregationOE):
    def __init__(
        self,
        kps_generator: nn.Module,
        fuse_module: nn.Module,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_camera_embed: bool = False,
        residual_mode: str = "add",
        point_cloud_range=None,
        lidar_only: bool = False,
    ):
        super(DeformableFeatureAggregationLiFOE, self).__init__(
            kps_generator,
            embed_dims,
            num_groups,
            num_levels,
            num_cams,
            attn_drop,
            proj_drop,
            use_camera_embed,
            residual_mode,
        )
        self.fuse_module = fuse_module
        self.point_cloud_range = point_cloud_range
        self.lidar_only = lidar_only
        self.lidar_weight_sum = FloatFunctional()
        self.lidar_weight_fc = nn.Linear(
            embed_dims, num_groups * 2 * self.num_pts
        )

    def gen_lidar_weights(self, instance_feature, anchor_embed):
        bs, num_anchor = instance_feature.shape[:2]
        feature = self.weight_add.add(instance_feature, anchor_embed)
        weights = self.lidar_weight_fc(feature).reshape(
            bs, num_anchor, -1, self.num_groups
        )
        weights = self.weight_softmax(weights)
        return weights

    def lidar_feature_sampling(self, lidar_feature, key_points):
        bs, num_anchor, num_pts = key_points.shape[:3]
        bev_width = self.point_cloud_range[3] - self.point_cloud_range[0]
        bev_height = self.point_cloud_range[4] - self.point_cloud_range[1]
        pts_x = (
            2.0 * (key_points[..., 0] - self.point_cloud_range[0]) / bev_width
            - 1.0
        )
        pts_y = (
            2.0 * (key_points[..., 1] - self.point_cloud_range[1]) / bev_height
            - 1.0
        )
        grid = torch.stack([pts_x, pts_y], dim=-1)
        if isinstance(lidar_feature, List):
            num_levels = len(lidar_feature)
            features = []
            for lf in lidar_feature:
                features.append(
                    torch.nn.functional.grid_sample(
                        lf.float(), grid, align_corners=False
                    )
                )
            features = torch.cat(features, dim=-1)
            feature = (
                features.view(bs, -1, num_anchor, num_levels * num_pts)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(bs, num_anchor, num_levels * num_pts, -1)
            )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dimo
        else:
            features = torch.nn.functional.grid_sample(
                lidar_feature.float(), grid, align_corners=False
            )  # bsembed_dims, 16, 44

            feature = (
                features.view(bs, -1, num_anchor, num_pts)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(bs, num_anchor, num_pts, -1)
            )  # bs, num_anchor, num_pts, embed_dim
        return feature

    def lidar_weight_fuse(self, lidar_feature, weights):
        bs, num_anchor = weights.shape[:2]
        weights = weights.reshape(
            bs,
            num_anchor,
            self.num_cams * self.num_levels,
            self.num_pts,
            self.num_groups,
        )
        lidar_feature = lidar_feature.reshape(
            bs, num_anchor, self.num_pts, self.num_groups, self.group_dims
        )

        lidar_weights = self.lidar_weight_sum.sum(
            weights[..., None], dim=2
        )  # .sum(dim=2)
        lidar_feature = lidar_feature * lidar_weights
        lidar_feature = lidar_feature.sum(dim=-3) / (
            self.num_cams * self.num_levels
        )
        lidar_feature = lidar_feature.reshape(bs, num_anchor, -1)
        return lidar_feature

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: torch.Tensor,
        lidar_feature: torch.Tensor,
        projection_mat: torch.Tensor,
        **kwargs,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)

        weights = self._get_weights(
            instance_feature, anchor_embed, projection_mat
        )
        if not self.lidar_only:
            features = self.feature_sampling(
                feature_maps,
                key_points,
                projection_mat,
            )
            features = self.multi_view_level_fusion(features, weights)
        # print("lidar_feature", lidar_feature.shape)
        sampling_lidar_feats = self.lidar_feature_sampling(
            lidar_feature,
            key_points,
        )
        if isinstance(lidar_feature, List):
            lidar_weights = self.gen_lidar_weights(
                instance_feature, anchor_embed
            )
            lidar_feature = self.multi_view_level_fusion(
                sampling_lidar_feats, lidar_weights
            )
        else:
            lidar_feature = self.lidar_weight_fuse(
                sampling_lidar_feats, weights
            )
        if not self.lidar_only:
            features = self.fuse_module(lidar_feature, features)
        else:
            features = lidar_feature
        features = self.proj_drop(self.output_proj(features))

        if self.residual_mode == "add":
            output = self.residual_op.add(features, instance_feature)
        elif self.residual_mode == "cat":
            output = self.residual_op.cat([features, instance_feature], dim=-1)

        return output


@OBJECT_REGISTRY.register
class DenseDepthNetOE(nn.Module):
    """
    Dense Depth Network for depth estimation using feature maps.

    Args:
        embed_dims: Dimension of input feature maps.
        num_depth_layers: Number of depth estimation layers.
        equal_focal: Default focal length for depth calculation.
        max_depth: Maximum depth value.
        loss_weight: Weight for depth estimation loss.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_depth_layers: int = 1,
        equal_focal: int = 100,
        max_depth: int = 60,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight
        self.dequant = DeQuantStub()
        self.depth_layers = nn.ModuleList()
        for _ in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def _get_focal(self, metas: Dict[str, Any]) -> torch.Tensor:
        """
        Retrieve focal length from metadata.

        Args:
            metas: Metadata dictionary containing camera intrinsic parameters.

        Returns:
            Focal length tensor.
        """
        return metas["camera_intrinsic"][..., 0, 0]

    @autocast(enabled=False)
    def forward(
        self, feature_maps: List[torch.Tensor], metas: Dict[str, Any]
    ) -> List[torch.Tensor]:
        """
        Forward pass through the DenseDepthNetOE model.

        Args:
            feature_maps: List of input feature maps.
            metas: Metadata dictionary containing camera intrinsic parameters.

        Returns:
            List of estimated depth maps.
        """
        focal = self._get_focal(metas)
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](self.dequant(feat).float()).exp()
            depth = depth.transpose(0, -1) * focal / self.equal_focal
            depth = depth.transpose(0, -1)
            depths.append(depth)
        return depths

    @autocast(enabled=False)
    def _get_gt(
        self, depth_preds: List[torch.Tensor], metas: Dict[str, Any]
    ) -> List[torch.Tensor]:
        """
        Compute ground truth depth maps based on predicted and metadata.

        Args:
            depth_preds: List of predicted depth maps.
            metas: Metadata dictionary containing relevant information.

        Returns:
            List of computed ground truth depth maps.
        """
        img_hw = torch.tensor(
            metas["img"].shape[2:], device=depth_preds[0].device
        )  # Get height and width of image
        bs = len(metas["points"])
        lidar2img = metas["lidar2img"].view(bs, -1, 4, 4)
        points = metas["points"]
        gt_depth = [[] for _ in range(len(depth_preds))]
        for point, l2i_list in zip(points, lidar2img):
            point = torch.tensor(
                point, device=depth_preds[0].device, dtype=torch.float64
            )[..., :3]
            H, W = img_hw
            pts_ones = torch.ones((point.shape[0], 1)).to(
                device=depth_preds[0].device
            )
            pts_3d = torch.cat([point, pts_ones], dim=1).double()
            for l2i in l2i_list:
                pts_2d = pts_3d @ l2i.T.double()
                pts_2d[:, :2] /= pts_2d[:, 2:3]
                U = torch.round(pts_2d[:, 0]).long()
                V = torch.round(pts_2d[:, 1]).long()
                depths = pts_2d[:, 2]
                mask = (
                    (V >= 0) & (V < H) & (U >= 0) & (U < W) & (depths >= 0.1)
                )
                V, U, depths = V[mask], U[mask], depths[mask]
                sort_idx = torch.argsort(depths, descending=True)
                V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
                depths = torch.clip(depths, 0.1, self.max_depth)
                for j, pred in enumerate(depth_preds):
                    pred_hw = torch.tensor(
                        pred.shape[2:], device=depth_preds[0].device
                    )
                    scale = img_hw / pred_hw
                    u = torch.floor(U / scale[1]).long()
                    v = torch.floor(V / scale[0]).long()
                    depth_map = (
                        torch.ones(
                            (pred.shape[2], pred.shape[3]),
                            device=depth_preds[0].device,
                        )
                        * -1
                    )
                    depth_map[v, u] = depths.float()
                    gt_depth[j].append(depth_map)
        for i in range(len(gt_depth)):
            gt_depth[i] = torch.stack(gt_depth[i])
        return gt_depth

    def loss(
        self, depth_preds: List[torch.Tensor], metas: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute the loss function between predicted depth and ground truth.

        Args:
            depth_preds: List of predicted depth maps.
            metas: Metadata dictionary containing relevant information.

        Returns:
            Total loss value.
        """
        loss = 0.0
        with torch.no_grad():
            gt_depths = self._get_gt(depth_preds, metas)
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depth_preds))
                    * self.loss_weight
                )
            loss = loss + _loss
        return loss


@OBJECT_REGISTRY.register
class AsymmetricFFNOE(nn.Module):
    """
    Asymmetric Feed-Forward Neural Network with Optional Identity Connection.

    Args:
        in_channels: Number of input channels. Defaults to None.
        pre_norm: Whether to apply Layer Normalization before each FC layer.
                  Defaults to False.
        embed_dims: Dimensionality of the embedding space. Defaults to 256.
        feedforward_channels: Number of channels in the feedforward layers.
                              Defaults to 1024.
        num_fcs: Number of fully connected layers (should be >= 2).
                 Defaults to 2.
        ffn_drop: Dropout probability in the feedforward layers.
                  Defaults to 0.0.
        add_identity: Whether to add an identity connection.
                  Defaults to True.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels: int,
        pre_norm: bool = False,
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        ffn_drop: float = 0.0,
        add_identity: bool = True,
        **kwargs,
    ):
        super(AsymmetricFFNOE, self).__init__()
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = nn.ReLU(inplace=True)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm:
            self.pre_norm = nn.LayerNorm(in_channels)
        else:
            self.pre_norm = None

        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity
        self.short_add = FloatFunctional()

        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if self.in_channels == embed_dims
                else nn.Linear(self.in_channels, embed_dims)
            )

    def forward(
        self, x: torch.Tensor, identity: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the AsymmetricFFNOE module.

        Args:
            x: Input tensor.
            identity: Input tensor.

        Returns:
            Output tensor.
        """

        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return self.short_add.add(identity, out)


@OBJECT_REGISTRY.register
class DeformableFeatureAggregationOEv2(DeformableFeatureAggregationOE):
    def __init__(
        self,
        grid_align_num: int = 8,
        **kwargs,
    ):
        super(DeformableFeatureAggregationOEv2, self).__init__(**kwargs)

        self.attn_drop = nn.Dropout(kwargs["attn_drop"])
        self.grid_align_num = grid_align_num
        self.reduce_sum = nn.Linear(
            self.grid_align_num * self.num_pts * self.num_levels,
            self.grid_align_num,
            bias=False,
        )

    def init_weight(self) -> None:
        """Initialize weights of weights_fc and output_proj layers."""
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

        # init reduce_sum weight
        reduce_sum_weight = torch.zeros(
            self.reduce_sum.weight.size(), dtype=self.reduce_sum.weight.dtype
        )
        for i in range(self.grid_align_num):
            for j in range(self.num_levels):
                for k in range(self.num_pts):
                    index = (
                        j * self.grid_align_num * self.num_pts
                        + i * self.num_pts
                        + k
                    )
                    reduce_sum_weight[i, index] = 1

        self.reduce_sum.weight = torch.nn.Parameter(
            reduce_sum_weight, requires_grad=False
        )

    def _get_weights(
        self,
        instance_feature: torch.Tensor,
        anchor_embed: torch.Tensor,
        projection_mat: torch.Tensor,
    ) -> torch.Tensor:

        bs, num_anchor = instance_feature.shape[:2]
        feature = self.weight_add.add(instance_feature, anchor_embed)
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                projection_mat[:, :, :3].reshape(bs, self.num_cams, -1)
                # .float()
            )
            feature = self.cam_add.add(
                feature[:, :, None], camera_embed[:, None]
            )
        weights = self.weights_fc(feature).reshape(
            bs, num_anchor, -1, self.num_groups
        )
        weights = self.weight_softmax(weights)
        weights = (
            weights.view(
                bs,
                -1,
                self.grid_align_num,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
                1,
            )
            .permute(0, 3, 6, 7, 1, 4, 2, 5)
            .contiguous()
            .flatten(0, 1)
            .flatten(-3)
            .repeat(1, 1, self.group_dims, 1, 1)
            .view(
                bs * self.num_cams,
                self.embed_dims,
                -1,
                self.num_levels * self.grid_align_num * self.num_pts,
            )
        )

        weights = self.attn_drop(weights)
        return weights

    def project_points(
        self, key_points: torch.Tensor, projection_mat: torch.Tensor
    ) -> torch.Tensor:
        bs, num_anchor, num_pts = key_points.shape[:3]

        # Extend key_points with ones for homogeneous coordinates
        ones = self.point_quant_stub(torch.ones_like(key_points[..., :1]))
        pts_extend = self.point_cat.cat([key_points, ones], dim=-1)

        # Perform projection onto 2D space
        points_2d = self.point_matmul.mul(
            projection_mat[:, :, None, None],
            pts_extend[:, None, ..., None, :],
        )
        points_2d = self.point_sum.sum(points_2d, dim=-1)

        # Depth normalization and final projection
        depth = self.reciprocal_op(torch.clamp(points_2d[..., 2:3], min=0.1))
        xy = points_2d[..., :2]
        points_2d = self.point_mul.mul(xy, depth)
        points_2d = torch.clamp(points_2d, -1.1, 1.1)
        return points_2d

    def feature_sampling(
        self,
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
    ) -> torch.Tensor:
        bs, num_anchor, num_pts = key_points.shape[:3]
        num_cams = self.num_cams

        points_2d = self.project_points(
            key_points,
            projection_mat,
        )
        points_2d = points_2d.flatten(end_dim=1).view(
            bs * num_cams, -1, self.grid_align_num * num_pts, 2
        )
        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm, points_2d, align_corners=False
                )
            )
        if len(features) > 1:
            features = self.feat_cat.cat(features, dim=3)
        else:
            # bs*num_cams, embed_dimo, num_anchor/grid, num_levels*grid*num_pts
            features = features[0]
        return features

    def multi_view_level_fusion(
        self, features: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:

        bs = features.shape[0] // self.num_cams
        features = self.feat_mul.mul(weights, features)
        features = self.reduce_sum(features)
        features = features.view(bs, self.num_cams, self.embed_dims, -1)
        features = self.feat_sum.sum(features, dim=1, keepdim=False)
        features = features.transpose(1, 2)
        return features

    def fix_weight_qscale(self) -> None:
        """Fix the qscale of conv weight when calibration or qat stage."""

        self.reduce_sum.weight_fake_quant.disable_observer()
        weight_scale_reduce_sum = torch.ones(
            self.reduce_sum.weight.shape[0],
            device=self.reduce_sum.weight.device,
        )
        weight_scale_reduce_sum[...] = 1.0
        self.reduce_sum.weight_fake_quant.set_qparams(weight_scale_reduce_sum)
