# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional

from hat.models.weight_init import xavier_init
from hat.registry import OBJECT_REGISTRY
from .blocks import Scale, linear_relu_ln

__all__ = [
    "SparseBEVOEKeyPointsGenerator",
    "SparseBEVOERefinementModule",
    "SparseBEVOEEncoder",
]


@OBJECT_REGISTRY.register
class SparseBEVOERefinementModule(nn.Module):
    """
    Refinement Module for refining outputs based on embedded features.

    Args:
        embed_dims: Dimension of embedded features.
        output_dim: Dimension of output.
        num_cls: Number of classes for classification branch.
        refine_yaw: Whether to refine yaw.
        with_cls_branch: Whether to include classification branch.
        with_quality_estimation: Whether to include quality estimation branch.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        output_dim: int = 11,
        num_cls: int = 10,
        refine_yaw: bool = False,
        with_cls_branch: bool = True,
        with_quality_estimation: bool = True,
    ):
        super(SparseBEVOERefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.refine_yaw = refine_yaw

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )
        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                nn.Linear(self.embed_dims, self.num_cls),
            )
        self.with_quality_estimation = with_quality_estimation
        if with_quality_estimation:
            self.quality_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                nn.Linear(self.embed_dims, 2),
            )

        self.add1 = FloatFunctional()
        self.add2 = FloatFunctional()

    def init_weight(self) -> None:
        """Initialize weights."""
        if self.with_cls_branch:
            prior_prob = 0.01
            bias_init = float(-np.log((1 - prior_prob) / prior_prob))
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        return_cls: bool = True,
    ) -> Union[
        torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]
    ]:
        """
        Forward pass of the Sparse BEVOE Refinement Module.

        Args:
            instance_feature : Input tensor of shape
                               (batch_size, embed_dims).
            anchor : Anchor tensor of shape
                     (batch_size, anchor_dims).
            anchor_embed : Embedded anchor tensor of shape
                           (batch_size, embed_dims).
            return_cls : Whether to return classification output.
                         Defaults to True.
        """
        feature = self.add1.add(instance_feature, anchor_embed)
        output = self.layers(feature)

        output = self.add2.add(output, anchor)
        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)
        else:
            cls = None
        if return_cls and self.with_quality_estimation:
            quality = self.quality_layers(feature)
        else:
            quality = None
        return output, cls, quality

    def set_qconfig(self) -> None:
        """Set the qconfig."""
        from horizon_plugin_pytorch.dtype import qint16

        from hat.utils import qconfig_manager

        self.layers[-1].qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "averaging_constant": 0},
            activation_calibration_qkwargs={
                "dtype": qint16,
            },
            weight_qat_qkwargs={
                "averaging_constant": 1,
            },
        )
        self.add2.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "averaging_constant": 0},
            activation_calibration_qkwargs={
                "dtype": qint16,
            },
        )
        if self.with_cls_branch is True:
            self.cls_layers[
                -1
            ].qconfig = qconfig_manager.get_default_qat_out_qconfig()
        if self.with_quality_estimation is True:
            self.quality_layers[
                -1
            ].qconfig = qconfig_manager.get_default_qat_out_qconfig()


@OBJECT_REGISTRY.register
class SparseBEVOEKeyPointsGenerator(nn.Module):
    """
    Module for generating keypoints based on embedded features.

    Args:
        embed_dims: Dimension of embedded features. Defaults to 256.
        num_pts: Number of keypoints. Defaults to 13.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_pts: int = 13,
    ):
        super(SparseBEVOEKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims

        self.num_pts = num_pts

        self.offset = nn.Linear(self.embed_dims, self.num_pts * 3)

        self.keypoints_add = FloatFunctional()

    def init_weight(self) -> None:
        """Initialize weights."""
        xavier_init(self.offset, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor: torch.Tensor,
        instance_feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the SparseBEVOEKeyPointsGenerator module.

        Args:
            anchor: Anchor tensor of shape
                    (batch_size, num_anchor, anchor_dims).
            instance_feature: Instance feature tensor of shape
                              (batch_size, num_anchor, embed_dims).

        Returns:
            Generated keypoints tensor of shape
            (batch_size, num_anchor, num_pts, 3).
        """
        bs, num_anchor = anchor.shape[:2]
        key_points = self.offset(instance_feature).view(
            bs, num_anchor, self.num_pts, 3
        )
        key_points = self.keypoints_add.add(key_points, anchor[..., None, 0:3])
        return key_points

    def set_qconfig(self) -> None:
        """Set the qconfig."""
        from horizon_plugin_pytorch.dtype import qint16

        from hat.utils import qconfig_manager

        self.offset.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "averaging_constant": 0},
            weight_qat_qkwargs={
                "averaging_constant": 1,
            },
            activation_calibration_qkwargs={
                "dtype": qint16,
            },
        )
        self.keypoints_add.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "averaging_constant": 0},
            weight_qat_qkwargs={
                "averaging_constant": 1,
            },
            activation_calibration_qkwargs={
                "dtype": qint16,
            },
        )


@OBJECT_REGISTRY.register
class SparseBEVOEEncoder(nn.Module):
    """
    Module for encoding 3D box features into embeddings.

    Args:
        pos_embed_dims: Dimension of position embeddings. Defaults to 128.
        size_embed_dims: Dimension of size embeddings. Defaults to 32.
        yaw_embed_dims: Dimension of yaw embeddings. Defaults to 32.
        vel_embed_dims: Dimension of velocity embeddings. Defaults to 64.
        vel_dims: Number of velocity dimensions. Defaults to 3.
    """

    def __init__(
        self,
        pos_embed_dims: int = 128,
        size_embed_dims: int = 32,
        yaw_embed_dims: int = 32,
        vel_embed_dims: int = 64,
        vel_dims: int = 3,
    ):
        super().__init__()
        self.vel_dims = vel_dims
        self.pos_embed_dims = pos_embed_dims
        self.size_embed_dims = size_embed_dims
        self.yaw_embed_dims = yaw_embed_dims
        self.vel_embed_dims = vel_embed_dims

        def embedding_layer(input_dims, out_dims):
            return nn.Sequential(*linear_relu_ln(out_dims, 1, 4, input_dims))

        self.pos_fc = embedding_layer(3, self.pos_embed_dims)
        self.size_fc = embedding_layer(3, self.size_embed_dims)
        self.yaw_fc = embedding_layer(2, self.yaw_embed_dims)
        if vel_dims > 0:
            self.vel_fc = embedding_layer(self.vel_dims, self.vel_embed_dims)
        self.cat = FloatFunctional()

    def forward(self, box_3d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SparseBEVOEEncoder module.

        Args:
            box_3d : Input tensor of shape
                     (batch_size, ..., 8 + vel_dims).
                     Contains 3D box coordinates and
                     optionally velocity information.

        Returns:
            Encoded feature tensor.
        """
        pos_feat = self.pos_fc(box_3d[..., 0:3])
        size_feat = self.size_fc(box_3d[..., 3:6])
        yaw_feat = self.yaw_fc(box_3d[..., 6:8])
        if self.vel_dims > 0:
            vel_feat = self.vel_fc(box_3d[..., 8 : 8 + self.vel_dims])
            output = self.cat.cat(
                [pos_feat, size_feat, yaw_feat, vel_feat], dim=-1
            )
        else:
            output = self.cat.cat([pos_feat, size_feat, yaw_feat], dim=-1)
        return output
