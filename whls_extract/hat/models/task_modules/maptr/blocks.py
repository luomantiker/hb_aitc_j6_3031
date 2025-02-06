# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub

from hat.models.task_modules.sparsebevoe.blocks import Scale, linear_relu_ln
from hat.models.weight_init import xavier_init
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "SparseOEPoint3DEncoder",
    "SparsePoint3DKeyPointsGenerator",
    "SparsePoint3DRefinementModule",
]


@OBJECT_REGISTRY.register
class SparseOEPoint3DEncoder(nn.Module):
    """
    Point3DEncoder for sparse maptroe.

    Args:
        embed_dims: Dimension of the embedding. Defaults to 256.
        input_dim: Dimension of the input. Defaults to 2.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        input_dim: int = 2,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.input_dim = input_dim

        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.pos_fc = embedding_layer(input_dim)

    def forward(self, points: torch.Tensor):
        point_feat = self.pos_fc(points)
        return point_feat


@OBJECT_REGISTRY.register
class SparsePoint3DKeyPointsGenerator(nn.Module):
    """
    SparsePoint3DKeyPointsGenerator class.

    Args:
        embed_dims: Dimension of the embedding. Defaults to 256.
        num_sample: Number of samples. Defaults to 20.
        num_learnable_pts: Number of learnable points. Defaults to 0.
        fix_height: Fixed height values. Defaults to (0,).
        ground_height: Ground height value. Defaults to 0.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        num_learnable_pts: int = 0,
        fix_height: Tuple = (0,),
        ground_height: int = 0,
    ):
        super(SparsePoint3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.num_learnable_pts = num_learnable_pts
        self.num_pts = num_sample * len(fix_height) * num_learnable_pts
        if self.num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, self.num_pts * 2)

        self.fix_height = np.array(fix_height)
        self.ground_height = ground_height
        self.quant_fix_height = QuantStub()

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        assert self.num_learnable_pts > 0, "No learnable pts"
        bs, num_anchor, _ = anchor.shape
        key_points = anchor.view(bs, num_anchor, self.num_sample, -1)
        offset = self.learnable_fc(instance_feature).reshape(
            bs,
            num_anchor,
            self.num_sample,
            len(self.fix_height),
            self.num_learnable_pts,
            2,
        )
        key_points = offset + key_points[..., None, None, :]
        key_points = torch.cat(
            [
                key_points,
                key_points.new_ones(key_points.shape[:-1] + (1,))
                * self.ground_height,
            ],
            dim=-1,
        )
        fix_height = key_points.new_tensor(self.fix_height)
        fix_height = self.quant_fix_height(fix_height)
        height_offset = key_points.new_zeros([len(fix_height), 2])
        height_offset = torch.cat([height_offset, fix_height[:, None]], dim=-1)
        key_points = key_points + height_offset[None, None, None, :, None]
        key_points = key_points.flatten(2, 4)
        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        temp_key_points_list = []
        for i, _ in enumerate(temp_timestamps):
            temp_key_points = key_points
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list

    # @staticmethod
    def anchor_projection(
        self,
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            dst_anchor = anchor.clone()
            bs, num_anchor, _ = anchor.shape
            dst_anchor = dst_anchor.reshape(
                bs, num_anchor, self.num_sample, -1
            ).flatten(1, 2)
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            )

            dst_anchor = (
                torch.matmul(
                    T_src2dst[..., :2, :2], dst_anchor[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., :2, 3]
            )

            dst_anchor = dst_anchor.reshape(
                bs, num_anchor, self.num_sample, -1
            ).flatten(2, 3)
            dst_anchors.append(dst_anchor)
        return dst_anchors


@OBJECT_REGISTRY.register
class SparsePoint3DRefinementModule(nn.Module):
    """
    SparsePoint3DRefinementModule class.

    Args:
        embed_dims: Dimension of the embedding. Defaults to 256.
        num_sample: Number of samples. Defaults to 20.
        coords_dim: Dimension of the coordinates. Defaults to 2.
        num_cls: Number of classes. Defaults to 3.
        with_cls_branch: Whether to include the classification branch.
            Defaults to True.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
        num_cls: int = 3,
        with_cls_branch: bool = True,
    ):
        super(SparsePoint3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.output_dim = num_sample * coords_dim
        self.num_cls = num_cls

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

    def init_weight(self):
        if self.with_cls_branch:
            prior_prob = 0.01
            bias_init = float(-np.log((1 - prior_prob) / prior_prob))
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        output = self.layers(instance_feature + anchor_embed)
        output = output + anchor
        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)
        else:
            cls = None
        return output, cls
