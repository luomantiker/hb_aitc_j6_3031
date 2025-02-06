from typing import Any, List

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["Anchor3DGeneratorStride"]


@OBJECT_REGISTRY.register
class Anchor3DGeneratorStride(nn.Module):
    """Lidar 3D Anchor Generator by stride.

    Args:
        anchor_sizes: 3D sizes of anchors.
        anchor_strides: Strides of anchors.
        anchor_offsets: Offsets of anchors.
        rotations: Rotations of anchors in a feature grid.
        class_names: Class names of data.
        match_thresholds: Match thresholds of IoU.
        unmatch_thresholds: Unmatch thresholds of IoU.
    """

    def __init__(
        self,
        class_names: List[str],
        anchor_sizes: List[List[float]],
        anchor_strides: List[List[float]],
        anchor_offsets: List[List[float]],
        rotations: List[List[float]],
        match_thresholds: List[float],
        unmatch_thresholds: List[float],
        dtype: Any = torch.float32,
    ):
        super(Anchor3DGeneratorStride, self).__init__()

        assert (
            len(anchor_sizes)
            == len(anchor_offsets)
            == len(anchor_offsets)
            == len(class_names)
            == len(match_thresholds)
            == len(unmatch_thresholds)
        )
        self._anchor_sizes = anchor_sizes
        self._anchor_strides = anchor_strides
        self._anchor_offsets = anchor_offsets
        self._anchor_rotations = rotations
        self._dtype = dtype
        self._class_names = class_names
        self._match_thresholds = match_thresholds
        self._unmatch_thresholds = unmatch_thresholds

        self._num_of_anchor_sets = len(self._anchor_sizes)

    @property
    def class_name(self):
        """Class names of data."""
        return self._class_names

    @property
    def match_thresholds(self):
        """Match thresholds of IoU."""
        return self._match_thresholds

    @property
    def unmatch_thresholds(self):
        """Unmatch thresholds of IoU."""
        return self._unmatch_thresholds

    @property
    def num_of_anchor_sets(self):
        """Get number of anchor settings."""
        return self._num_of_anchor_sets

    @property
    def num_anchors_per_localization(self):
        """Get number of anchors on per location."""
        num_anchors_per_localization = []
        for rot, size in zip(self._anchor_rotations, self._anchor_sizes):
            num_rot = len(rot)
            num_size = torch.tensor(size).reshape([-1, 3]).shape[0]
            num_anchors = num_rot * num_size
            num_anchors_per_localization.append(num_anchors)
        return num_anchors_per_localization

    @fx_wrap(skip_compile=True)
    def forward(self, feature_map_size, device):
        """Forward pass, generate anchors.

        Args:
            feature_map_size: Feature map size, (1, H, W).
            device: device.

        Returns:
            Anchor list.
            Match thresholds of IoU.
            Unmatch thresholds of IoU.
        """
        anchors_list = self.generate_anchors(feature_map_size, device)
        anchor_dict = {
            "anchors": anchors_list,
            "matched_thresholds": self.match_thresholds,
            "unmatched_thresholds": self.unmatch_thresholds,
        }
        return anchor_dict

    def generate_anchors(self, feature_map_size, device=None):
        """Generate anchors.

        Args:
            feature_map_size: Feature map size, (1, H, W).
            device: device.

        Returns:
            List of Anchors.
        """
        feature_map_size = [feature_map_size] * self._num_of_anchor_sets

        self._anchors_list = []
        for i in range(self._num_of_anchor_sets):
            anchors = self._create_anchor_3d_stride(
                feature_size=feature_map_size[i],  # (1, H, W)
                sizes=self._anchor_sizes[i],
                anchor_strides=self._anchor_strides[i],
                anchor_offsets=self._anchor_offsets[i],
                rotations=self._anchor_rotations[i],
                dtype=self._dtype,
                device=device,
            )
            anchors = anchors.reshape(
                [*anchors.shape[:3], -1, anchors.shape[-1]]
            )
            self._anchors_list.append(anchors)

        return self._anchors_list

    def _create_anchor_3d_stride(
        self,
        feature_size: List[int],
        sizes: List[List[float]],
        anchor_strides: List[List[float]],
        anchor_offsets: List[List[float]],
        rotations: List[List[float]],
        dtype=torch.float32,
        device="cuda",
    ):
        """Create anchors.

        Args:
            feature_size: list [D, H, W](zyx)
            sizes: [N, 3] size of anchors, xyz
            anchor_sizes: 3D sizes of anchors.
            anchor_strides: Strides of anchors.
            anchor_offsets: Offsets of anchors.
            rotations: Rotations of anchors in a feature grid.

        Returns:
            anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
        """
        x_stride, y_stride, z_stride = anchor_strides
        x_offset, y_offset, z_offset = anchor_offsets

        z_centers = torch.arange(feature_size[0], dtype=dtype, device=device)
        y_centers = torch.arange(feature_size[1], dtype=dtype, device=device)
        x_centers = torch.arange(feature_size[2], dtype=dtype, device=device)

        z_centers = z_centers * z_stride + z_offset
        y_centers = y_centers * y_stride + y_offset
        x_centers = x_centers * x_stride + x_offset
        sizes = torch.tensor(sizes, dtype=dtype, device=device).reshape(
            [-1, 3]
        )
        rotations = torch.tensor(rotations, dtype=dtype, device=device)
        # torch.meshgrid default behavior is 'id', np's default is 'xy'
        rets = torch.meshgrid(x_centers, y_centers, z_centers, rotations)
        # torch.meshgrid returns a tuple rather than list
        rets = list(rets)
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)
        sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)
        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])

        return ret
