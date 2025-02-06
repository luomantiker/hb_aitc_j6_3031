# Copyright (c) Horizon Robotics. All rights reserved.

# Voxel generation utilities
from typing import Tuple

import numba
import numpy as np


@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(
    points: np.ndarray,
    voxel_size: np.ndarray,
    coors_range: np.ndarray,
    num_points_per_voxel: int,
    coor_to_voxelidx: np.ndarray,
    voxels: np.ndarray,
    coors: np.ndarray,
    max_points: int = 35,
    max_voxels: int = 20000,
) -> int:
    """Kernel of the voxel generation function.

    Generates voxels and assign points to each voxel.

    Args:
        points : an [N, >=3] array containing point clouds.
        voxel_size : xyz dimension of each voxel.
        coors_range : xyzxyz range of the entire point cloud.
        num_points_per_voxel : a 1-d array that is used to record
            number of points in each voxel.
        coor_to_voxelidx : mapping from coordinate to voxel id.
        voxels : a 3-d tensor of voxels. Each voxel is a collection
            of points. The max number of points in each voxel is defined by
            max_points.
            For more details, can refer to PointPillars paper.
        coors : a [P, 3] matrix recording each voxel's coordinate.
        max_points : max number of points in a voxel.
            Defaults to 35.
        max_voxels : max number of voxels. Defaults to 20000.

    Returns:
        int: number of non-empty voxels. Additionally, voxels will be created
            and points will be assigned to voxels.
    """
    # Put all computations to one loop.
    # Shouldn't create large array in main jit code, causing performance drop
    N = points.shape[0]  # number of ponits
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros((3,), dtype=np.int32)
    voxel_num = 0

    failed = False

    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


@numba.jit(nopython=True)
def _points_to_voxel_kernel(
    points: np.ndarray,
    voxel_size: np.ndarray,
    coors_range: np.ndarray,
    num_points_per_voxel: np.ndarray,
    coor_to_voxelidx: np.ndarray,
    voxels: np.ndarray,
    coors: np.ndarray,
    max_points: int = 35,
    max_voxels: int = 20000,
) -> int:
    """Kernel of the voxel generation function.

    Generates voxels and assign points to each voxel.

    Args:
        points : an [N, >=3] array containing point clouds.
        voxel_size : xyz dimension of each voxel.
        coors_range : xyzxyz range of the entire point cloud.
        num_points_per_voxel : a 1-d array that is used to record
            number of points in each voxel.
        coor_to_voxelidx : mapping from coordinate to voxel id.
        voxels : a 3-d tensor of voxels. Each voxel is a collection
            of points. The max number of points in each voxel is defined by
            max_points.
            For more details, can refer to PointPillars paper.
        coors : a [P, 3] matrix recording each voxel's coordinate.
        max_points : max number of points in a voxel.
            Defaults to 35.
        max_voxels : max number of voxels. Defaults to 20000.

    Returns:
        int: number of non-empty voxels. Additionally, voxels will be created
            and points will be assigned to voxels.
    """
    # NOTE: need mutex if written in cuda, but numba.cuda doesn't support
    # mutex. Also, pytorch doesn't support cuda in dataloader(tf supports this)

    # Put all computations in one loop.
    # Should not create large array in main JIT code, causing performance drop
    N = points.shape[0]  # number of points
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    # Temp var to store info for each point
    # 1. coord of the voxel that point is in
    # 2. voxel count
    coor = np.zeros((3,), dtype=np.int32)
    voxel_num = 0

    failed = False

    for i in range(N):
        failed = False
        # go through each dimension
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def points_to_voxel(
    points: np.ndarray,
    voxel_size: np.ndarray,
    coors_range: np.ndarray,
    max_points: int = 35,
    reverse_index: bool = True,
    max_voxels: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert lidar points [N, >=3] to voxels.

    With the help of JIT
    this function runs fast. Note that it seems to run faster under Ubuntu
    than under Windows 10.

    Args:
        points : [N, >=3] float array or tensor. The first 3
            columns contain xyz points, and the rest columns contain other
            information.
        voxel_size : xyz dimension of the voxel.
        coors_range : xyzxyz dimension of the point cloud.
        max_points : maximum number of points in a voxel.
            Defaults to 35.
        reverse_index : whether to return reversed coords.
            If points has xyz format and reverse_index=True, then output
            coords will be zyx format, but points in features always have
            xyz format.
            Defaults to True.
        max_voxels : maximum number of voxels to create.
            for SECOND, 20000 is a good choice. You should shuffle points
            before call this function because max_voxels may drop some points.
            Defaults to 20000.

    Returns:
        voxels: [M, max_points, ndim] a float tensor of voxels.
        coordinates: [M, 3] int 32 tensor recording each voxel's coordinate.
        num_points_per_voxel: [M] int32 tensor recording number of points in
            each voxel.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    # get grid_size
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32))
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # NOTE: do not create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    # P, N, D in PP
    # P: num_voxels  N: num_points per voxel D: dim of each point
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype
    )
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel(
            points,
            voxel_size,
            coors_range,
            num_points_per_voxel,
            coor_to_voxelidx,
            voxels,
            coors,
            max_points,
            max_voxels,
        )

    else:
        voxel_num = _points_to_voxel_kernel(
            points,
            voxel_size,
            coors_range,
            num_points_per_voxel,
            coor_to_voxelidx,
            voxels,
            coors,
            max_points,
            max_voxels,
        )

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    return voxels, coors, num_points_per_voxel


class VoxelGenerator(object):
    def __init__(
        self,
        voxel_size: Tuple[float, float, float],
        point_cloud_range: Tuple[float, ...],
        max_num_points: int,
        max_voxels: int = 20000,
    ):
        """Generate voxels from point cloud.

        Args:
            voxel_size : xyz dimension
                of each voxel.
            point_cloud_range : xyzxyz of point
                cloud range. Taking ego car as origin.
            max_num_points : maximum number of points in a voxel.
            max_voxels : maximum number of voxels.
                Defaults to 20000.
        """
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = np.divide(
            (point_cloud_range[3:] - point_cloud_range[:3]), voxel_size
        )
        grid_size = np.round(grid_size).astype(np.int64)
        # NOTE: grid_size has to be integers. We don't do a dtype check here,
        # so the user has to manually calculate point_cloud_range and
        # voxel_size to satisfy both dtype requirement and downsampling
        # requirement.
        # TODO: grid_size is not used anywhere. The kernels compute grid_size
        # by themselves. Can consider removing it.
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points: np.ndarray) -> np.ndarray:
        return points_to_voxel(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._max_num_points,
            True,
            self._max_voxels,
        )

    @property
    def voxel_size(self) -> np.ndarray:
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self) -> int:
        return self._max_num_points

    @property
    def point_cloud_range(self) -> np.ndarray:
        return self._point_cloud_range

    @property
    def grid_size(self) -> np.ndarray:
        return self._grid_size
