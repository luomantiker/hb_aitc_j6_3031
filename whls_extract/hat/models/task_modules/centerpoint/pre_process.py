import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from hat.models.task_modules.pointpillars.preprocess import BatchVoxelization
from hat.models.utils import _get_paddings_indicator
from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["CenterPointPreProcess"]

logger = logging.getLogger(__name__)


@fx_wrap()
def _voxel_feature_encoder(
    norm_range: Tensor,
    norm_dims: List[int],
    features: Tensor,
    num_points_in_voxel: Tensor,
) -> Tensor:
    # normolize features
    if norm_range is not None and norm_dims is not None:
        for idx, dim in enumerate(norm_dims):
            start = norm_range[idx]
            norm = norm_range[idx + len(norm_range) // 2] - norm_range[idx]
            features[:, :, dim] = features[:, :, dim] - start
            features[:, :, dim] = features[:, :, dim] / norm
    else:
        logger.warning(
            "norm_range and norm_dims are not specific, "
            "voxel feature will skip normalization, "
            "quantization accuracy may be affected."
        )

    # The feature decorations were calculated without regard to whether
    # pillar was empty. Need to ensure that empty pillars remain set to
    # zeros.
    voxel_count = features.shape[1]
    mask = _get_paddings_indicator(num_points_in_voxel, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, -1).type_as(features)
    features *= mask

    features = features.unsqueeze(0).permute(0, 3, 2, 1).contiguous()

    return features


@OBJECT_REGISTRY.register
class CenterPointPreProcess(nn.Module):
    """Centerpoint preprocess, include voxelization and features encoder.

    Args:
        pc_range: Point cloud range.
        voxel_size: voxel size, (x, y, z) scale.
        max_voxels_num: Max voxel number to use. Defaults to 30000.
        max_points_in_voxel: Number of points in per voxel. Defaults to 20.
        norm_range: Feature range, like
            [x_min, y_min, z_min, ..., x_max, y_max, z_max, ...].
        norm_dims: Dims to do normalize.
    """

    def __init__(
        self,
        pc_range: List[float],
        voxel_size: List[float],
        max_voxels_num: Union[tuple, int] = 30000,
        max_points_in_voxel: int = 20,
        norm_range: Optional[List] = None,
        norm_dims: Optional[List] = None,
    ):
        super().__init__()
        self.norm_range = norm_range
        self.norm_dims = norm_dims
        self.voxel_generator = BatchVoxelization(
            pc_range=pc_range,
            voxel_size=voxel_size,
            max_voxels_num=max_voxels_num,
            max_points_in_voxel=max_points_in_voxel,
        )

    def forward(
        self,
        points_lst: List[torch.Tensor],
        is_deploy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of Centerpoint preprocess.

        Args:
            points_lst: List of input point clouds.
            is_deploy: Flag indicating whether the model is in
                deployment mode. Default is False.

        Returns:
            A tuple containing the following elements:
                - features: Voxel-encoded feature map.
                - coors_batch: Voxel coordinates for the batch.
        """
        (
            voxel_feature,
            coors_batch,
            num_points_per_voxel,
        ) = self.voxel_generator(points_lst, is_deploy)

        features = _voxel_feature_encoder(
            norm_range=self.norm_range,
            norm_dims=self.norm_dims,
            features=voxel_feature,
            num_points_in_voxel=num_points_per_voxel,
        )
        return features, coors_batch
