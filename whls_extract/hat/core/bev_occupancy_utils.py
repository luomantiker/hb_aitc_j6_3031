from typing import Tuple

import numpy as np

from hat.core.bev_elevation_utils import (
    crop_roi_vcs_range,
    get_area_threshod_mask,
    get_center_coord,
    get_max_contour_mask,
    max_freespace_contain_ego,
)

__all__ = [
    "crop_roi_vcs_range",
    "get_area_threshod_mask",
    "generate_bev_weight_map",
    "get_max_contour_mask",
    "max_freespace_contain_ego",
]


def generate_bev_weight_map(
    target_size: Tuple,
    vcs_range: Tuple,
    bev_mask_scale: float = 2.0,
    bev_mask_bias: float = 0.1,
) -> np.ndarray:
    """Generate depth-wise bev weight map.

    This function is used to reweight model's outputs and enhance
    the attention to nearby targets.

    Args:
        target_size: bev gt size, (h, w)
        vcs_range: roi vcs range in (bottom, right, top, left) order.
        bev_mask_scale: hyper-parameter to scale bev_weight_map
        bev_mask_bias: hyper-parameter to bias bev_weight_map

    Returns:
        ndarray: bev weight map, (h, w)
    """
    target_size_h, target_size_w = target_size
    bev_weight_map = np.ones(target_size, dtype=np.float32)
    vcs_origin_coord = get_center_coord(vcs_range, target_size)
    vcs_origin_coord_v, vcs_origin_coord_u = vcs_origin_coord

    assert 0 <= vcs_origin_coord_v < target_size_h
    assert 0 <= vcs_origin_coord_u < target_size_w
    mesh_u, mesh_v = np.meshgrid(
        np.arange(target_size_w),
        np.arange(target_size_h),
    )
    bev_weight_map[:vcs_origin_coord_v, :] = (
        bev_mask_scale * mesh_v[:vcs_origin_coord_v, :] / vcs_origin_coord_v
        + bev_mask_bias
    )
    bev_weight_map[vcs_origin_coord_v:, :] = (
        bev_mask_scale
        * (target_size_h - mesh_v[vcs_origin_coord_v:, :])
        / (target_size_h - vcs_origin_coord_v)
        + bev_mask_bias
    )

    return bev_weight_map
