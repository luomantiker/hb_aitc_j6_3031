import torch
from torch import Tensor
from torch.overrides import handle_torch_function, has_torch_function

from horizon_plugin_pytorch.fx import fx_helper


def bev_pool_v2(
    depth: Tensor,
    feat: Tensor,
    ranks_depth: Tensor,
    ranks_feat: Tensor,
    ranks_bev: Tensor,
    interval_starts: Tensor,
    interval_lengths: Tensor,
    bev_feat_shape,
):
    """BEVPoolv2 implementation for Lift-Splat-Shoot view transformation.

    This impl is same as following **except** the layout of inout feature:
    https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/mmdet3d/ops/bev_pool_v2/bev_pool.py

    Args:
        depth (Tensor[b, n, d, h, w]): Input depth.
        feat (Tensor[b, n, c, h, w]): Input features.
        ranks_depth (Tensor[n_points]): Depth index of points.
        ranks_feat (Tensor[n_points]): Feat index of points.
        ranks_bev (Tensor[n_points]): Output index of points.
        interval_starts (Tensor[n_pillars]): Starting position in ranks_xxx for each pooled point.  # noqa: E501
        interval_lengths (Tensor[n_pillars]): How many points in each pooled point.  # noqa: E501
        bev_feat_shape: Output shape in [b, z_out, h_out, w_out, c] or
            [z_out, h_out, w_out] or [h_out, w_out] format.
            When z_out is not given, its value will be 1 by default.

    Returns:
        Tensor[b, c, z_out, h_out, w_out]: Output features.
    """
    if len(bev_feat_shape) not in (2, 3, 5):
        raise ValueError("Illegal bev_feat_shape length")

    if len(bev_feat_shape) < 5:
        bev_feat_shape = tuple(bev_feat_shape)
        if len(bev_feat_shape) == 2:
            bev_feat_shape = (1,) + bev_feat_shape
        b = feat.size(0)
        c = feat.size(2)
        bev_feat_shape = (b,) + tuple(bev_feat_shape) + (c,)

    if has_torch_function((depth, feat)):
        return handle_torch_function(
            bev_pool_v2,
            (depth, feat),
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
            bev_feat_shape,
        )

    x = torch.ops.horizon.bev_pool_v2(
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
        bev_feat_shape,
    )
    return x


@fx_helper.replace_torch_op("bev_pool_v2")
class BevPoolV2(torch.nn.Module):
    """Module impl of bev_pool_v2.

    Please refer to the docstring of bev_pool_v2 for more info.
    """

    def __init__(self, bev_feat_shape=None) -> None:
        super().__init__()
        self.bev_feat_shape = bev_feat_shape

    def forward(
        self,
        depth: Tensor,
        feat: Tensor,
        ranks_depth: Tensor,
        ranks_feat: Tensor,
        ranks_bev: Tensor,
        interval_starts: Tensor,
        interval_lengths: Tensor,
        bev_feat_shape=None,
    ):
        if bev_feat_shape is None:
            bev_feat_shape = self.bev_feat_shape
        return bev_pool_v2(
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
            bev_feat_shape,
        )
