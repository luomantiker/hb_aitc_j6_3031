# Copyright (c) Horizon Robotics. All rights reserved.

# Bounding box operations written in numpy.
from typing import List, Tuple

import numba
import numpy as np

from hat.core.box_utils import corners_nd
from hat.core.point_geometry import (
    _points_to_truncate_points_all,
    dropout_points_in_box,
    points_count_convex_polygon_3d_jit,
    points_in_convex_polygon_3d_jit,
    points_to_truncate_points,
)

corner_idxes = np.array(
    [
        0,
        1,
        2,
        3,
        7,
        6,
        5,
        4,
        0,
        3,
        7,
        4,
        1,
        5,
        6,
        2,
        0,
        4,
        5,
        1,
        3,
        2,
        6,
        7,
    ]
).reshape(6, 4)


def points_count_rbbox(
    points: np.ndarray,
    rbbox: np.ndarray,
    z_axis: int = 2,
    origin: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """Count number of points in each groundtruth box.

    Args:
        points : [num_points, >=3] points array.
        rbbox : [num_boxes, 8] groundtruth boxes array.
        z_axis : dimension index where z-axis is lcoated.
            Defaults to 2.
        origin : Origin point.
            Defaults to (0.5, 0.5, 0.5).

    Returns:
        number of points each box has.
    """
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin, axis=z_axis
    )
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    return points_count_convex_polygon_3d_jit(points[:, :3], surfaces)


def make_truncate_points(
    points: np.ndarray,
    rbbox: np.ndarray,
    mode: str = "all",
    angle_range: Tuple[float, float] = (0, 180),
    center: Tuple[float, float, float] = (0, 0, 0),
    z_axis: int = 2,
    origin: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    axis_rotat: float = 90,
) -> np.ndarray:
    """Keep points,which in truncate area.

    Args:
        points : [num_points, >=3] points array.
        rbbox : [num_boxes, 8] groundtruth boxes array.
        mode : all or other, other means only compute point in GT box.
        angle_range : truncate angle.
        center: trancate center point.
        z_axis : dimension index where z-axis is lcoated.
            Defaults to 2.
        origin : Origin point.
            Defaults to (0.5, 0.5, 0.5).
        axis_rotat: rotat of axis.

    Returns:
        points in truncate area.
    """
    if mode == "all":
        keep_list = _points_to_truncate_points_all(
            points,
            angle_range,
            center=center,
            axis_rotat=axis_rotat,
        )
        return points[keep_list]
    else:
        # The center point information of the
        # three body frame becomes the information of eight corners [n,8,3]
        rbbox_corners = center_to_corner_box3d(
            rbbox[:, :3],
            rbbox[:, 3:6],
            rbbox[:, -1],
            origin=origin,
            axis=z_axis,
        )
        # Convert eight corners into six faces
        # Four corners per face [N, 6, 4, 3]
        surfaces = corner_to_surfaces_3d(rbbox_corners)
        return points_to_truncate_points(points, surfaces, angle_range)


def dropout_points_in_gt(
    points: np.ndarray,
    rbbox: np.ndarray,
    prob: float = 0.2,
    z_axis: int = 2,
    origin: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """Random drop points in each groundtruth box.

    Args:
        points : [num_points, >=3] points array.
        rbbox : [num_boxes, 8] groundtruth boxes array.
        prob: drop probility.
        z_axis : dimension index where z-axis is lcoated.
            Defaults to 2.
        origin : Origin point.
            Defaults to (0.5, 0.5, 0.5).

    Returns:
        points keeped.
    """
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin, axis=z_axis
    )  # [n,8,3]
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    # [N, 6, 4, 3]
    points = dropout_points_in_box(points, surfaces, prob)
    return points


def points_in_rbbox(
    points: np.ndarray,
    rbbox: np.ndarray,
    z_axis: int = 2,
    origin: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """Find points in each rubberband box.

    Args:
        points : [num_points, >=3] points array.
        rbbox : [num_boxes, 8] rubberband boxes array.
        z_axis : dimension index where z-axis is lcoated.
            Defaults to 2.
        origin : Origin point.
            Defaults to (0.5, 0.5, 0.5).

    Returns:
        indices of points in rubberband box.
    """
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin, axis=z_axis
    )
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


def corner_to_surfaces_3d(corners: np.ndarray) -> np.ndarray:
    """Convert 3d box corners.

    From corner function above to surfaces with normal vectors all directing
    to internal.

    Args:
        corners : [N, 8, 3], must from corner functions in this module.
    Returns:
        surfaces : surce array.
    """

    surfaces = np.array(
        [
            [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
            [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
            [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
            [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
            [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
            [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
        ]
    ).transpose([2, 0, 1, 3])
    return surfaces


@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners: np.ndarray) -> np.ndarray:
    """Convert 3d box corners.

    From corner function above to surfaces that normal vectors all direct
    to internal.

    Args:
        corners : [N, 8, 3], must from corner functions in this module.
    Returns:
        surfaces : surfaces that normal vectors all direct to internal.
    """

    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)

    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces


def rotation_3d_in_axis(
    points: np.ndarray, angles: np.ndarray, axis: int = 0
) -> np.ndarray:
    """Rotate a set of points according to individual angles.

    Args:
        points: [N, ndim, 3] tensor of boxes. if 3d boxes, ndim
            should be 3.
        angles: [N] tensor of angles. Each box has its own
            rotation angle.
        axis : rotation axis. Defaults to 0.

    Raises:
        ValueError: raised when axis num is not in [-1, 2].

    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                [rot_cos, zeros, -rot_sin],
                [zeros, ones, zeros],
                [rot_sin, zeros, rot_cos],
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                [rot_cos, -rot_sin, zeros],
                [rot_sin, rot_cos, zeros],
                [zeros, zeros, ones],
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                [zeros, rot_cos, -rot_sin],
                [zeros, rot_sin, rot_cos],
                [ones, zeros, zeros],
            ]
        )
    else:
        raise ValueError("axis should in range")

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def center_to_corner_box3d(
    centers: np.ndarray,
    dims: np.ndarray,
    angles: np.ndarray = None,
    origin: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    axis: int = 2,
) -> np.ndarray:
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers : locations in kitti label file.
        dims : dimensions in kitti label file.
        angles : rotation_y in kitti label file.
        origin : origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis : rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3], relative to object center
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def rotation_points_single_angle(
    points: np.ndarray, angle: float, axis: int = 0
) -> np.ndarray:
    """Rotate points around an axis.

    Args:
        points : [N, 3] point array.
        angle : rotation angle.
        axis : axis index. Defaults to 0.

    Raises:
        ValueError: if axis is not in [-1, 0, 1, 2].

    Returns:
        rotated points.
    """
    # points: [N, 3]
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array(
            [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
            dtype=points.dtype,
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.array(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
            dtype=points.dtype,
        )
    elif axis == 0:
        rot_mat_T = np.array(
            [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
            dtype=points.dtype,
        )
    else:
        raise ValueError("axis should in range -1, 0, 1, 2")

    return points @ rot_mat_T


def project_to_image(
    points_3d: np.ndarray, proj_mat: np.ndarray
) -> np.ndarray:
    """Convert vehicle world 3dpoints to image 2d points .

    Args:
        points_3d :[N, 3].
        proj_mat: [4, 4].

    Returns:
        points_2d_res :[N, 2].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    points_4 = np.concatenate([points_3d, np.ones(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    return point_2d_res


def box3d_to_bbox(
    box3d: np.ndarray, rect: np.ndarray, Trv2c: np.ndarray, P2: np.ndarray
) -> np.ndarray:
    """Convert vehicle world box3d to image 2d box .

    Args:
        box3d:[N, 7].
        rect:[4, 4].
        Trv2c[N, 3] tranlate matix vehicle.
        P2:[N, 3] project matrix.

    Returns:
        bbox:(float array, shape=[N, 8,2]).
    """
    # box3d_to_cam = box_lidar_to_camera(box3d, rect, Trv2c)
    box_corners = center_to_corner_box3d(
        box3d[:, :3], box3d[:, 3:6], box3d[:, 6], [0.5, 1.0, 0.5], axis=1
    )
    box_corners_in_image = project_to_image(box_corners, P2)
    # box_corners_in_image: [N, 8, 2]
    minxy = np.min(box_corners_in_image, axis=1)
    maxxy = np.max(box_corners_in_image, axis=1)
    bbox = np.concatenate([minxy, maxxy], axis=1)
    return bbox


def limit_period(
    val: np.ndarray, offset: float = 0.5, period: float = np.pi
) -> np.ndarray:
    """Limit period.

    Args:
        val:[N, 6] value of period.
        offset:period offset.

    Returns:
        val:[N, 6] value of period.
    """
    return val - np.floor(val / period + offset) * period


def projection_matrix_to_CRT_kitti(
    proj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert projection_matrix to CRT.

    Args:
        proj:(shape=[4, 4]).

    Returns:
        C:(shape=[3, 3]).
        R:(shape=[3, 3]).
        T:(shape=[3, 1]).
    """
    # P = C @ [R|T]
    # C is upper triangular matrix, so we need to inverse CR and use QR
    # stable for all kitti camera projection matrix
    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


def minmax_to_corner_2d_v2(minmax_box: np.ndarray) -> np.ndarray:
    """Box's min max type to center type.

    Convert kitti locations imensions and angles to corners.

    format: center(xy)

    Args:
        minmax_box :(shape=[N, 4]).

    Returns:
        [type]: [description]
    """
    # N, 4 -> N 4 2
    return minmax_box[..., [0, 1, 0, 3, 2, 3, 2, 1]].reshape(-1, 4, 2)


def get_frustum(
    bboxes: np.ndarray,
    C: np.ndarray,
    near_clip: float = 0.001,
    far_clip: float = 100,
) -> np.ndarray:
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    num_box = bboxes.shape[0]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[
        np.newaxis, :, np.newaxis
    ]
    z_points = np.tile(z_points, [num_box, 1, 1])
    box_corners = minmax_to_corner_2d_v2(bboxes)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype
    )
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype
    )
    ret_xy = np.concatenate(
        [near_box_corners, far_box_corners], axis=1
    )  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=-1)
    return ret_xyz


def camera_to_lidar(
    points: np.ndarray, r_rect: np.ndarray, velo2cam: np.ndarray
) -> np.ndarray:
    """Convert camera world points to lidar world points .

    Args:
        points :shape=[N, 3] or shape=[N, 4].
        r_rect: shape=[4, 4] matrix rect.
        velo2cam:shape=[4, 4] translate matrix vel2cam.
    Returns:
        points_2d_res : shape=[N, 2].
    """
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


@numba.jit(nopython=True)
def box2d_to_corner_jit(boxes: np.ndarray) -> np.ndarray:
    """Box's center type to corner type.

    Args:
        boxes: shape=[N, 5].

    Returns:
        boxes: shape=[N, 5].
    """
    num_box = boxes.shape[0]
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * corners_norm.reshape(
        1, 4, 2
    )
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
        box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]
    return box_corners


@numba.njit
def corner_to_standup_nd_jit(boxes_corner: np.ndarray) -> np.ndarray:
    """Box's corner type to standup corner.

    Args:
        boxes_corner: shape=[N, ndim].

    Returns:
        result: shape=[N, ndim*2].
    """
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


def remove_outside_points(
    points: np.ndarray,
    rect: np.ndarray,
    Trv2c: np.ndarray,
    P2: np.ndarray,
    image_shape: List[int],
) -> np.ndarray:
    """Remove point clouds that are distributed outside the image range.

    Args:
        points: Point cloud, shape=[N, 3] or shape=[N, 4].
        rect: matrix rect, shape=[4, 4].
        Trv2c: Translate matrix vel2cam, shape=[4, 4].
        P2: Project matrix, shape=[4, 4].
        image_shape: Image shape, (H, W, ...) format.

    Returns: Point clouds are distributed inside the image range.
    """
    # 5x faster than remove_outside_points_v1(2ms vs 10ms)
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = np.array(
        [[0, 0, image_shape[1], image_shape[0]]], dtype=np.int32
    )

    frustum = get_frustum(image_bbox, C)
    if len(frustum.shape) == 3:
        frustum = frustum.squeeze()
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, Trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
    points = points[indices.reshape([-1])]
    return points


def box_camera_to_lidar(
    camera_box: np.ndarray, r_rect: np.ndarray, velo2cam: np.ndarray
) -> np.ndarray:
    """Convert camera 3d box to lidar 3d box.

    Args:
        camera_box: Camera box, shape=[N, 7], (x, y, z, l, h, w, rot_y).
        r_rect: Matrix rect, shape=[4, 4].
        velo2cam: Translate matrix vel2cam, shape=[4, 4].

    Returns:
        np.ndarray: Lidar 3d box, shape=[N, 7], (x, y, z, h, w, l, rot_y).
    """
    xyz = camera_box[:, 0:3]
    l, h, w = camera_box[:, 3:4], camera_box[:, 4:5], camera_box[:, 5:6]
    r = camera_box[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)


def change_box3d_center_(box3d, src, dst):
    dst = np.array(dst, dtype=box3d.dtype)
    src = np.array(src, dtype=box3d.dtype)
    # bo3d: [x_liar, y_lidar, z_lidar, w, l, h, rot_y]
    box3d[..., :3] += box3d[..., 3:6] * (dst - src)


def lidar_to_camera(
    points: np.ndarray,
    r_rect: np.ndarray,
    velo2cam: np.ndarray,
) -> np.ndarray:
    """Convert lidar world points to camera world points .

    Args:
        points :shape=[N, 3] or shape=[N, 4].
        r_rect: shape=[4, 4] matrix rect.
        velo2cam:shape=[4, 4] translate matrix vel2cam.
    Returns:
        points_2d_res : shape=[N, 2].
    """
    points_shape = list(points.shape[:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    camera_points = points @ (r_rect @ velo2cam).T
    return camera_points[..., :3]


def box_lidar_to_camera(
    box3d_lidar: np.ndarray,
    r_rect: np.ndarray,
    velo2cam: np.ndarray,
) -> np.ndarray:
    """Convert Lidar3D boxes to Camera3D boxes.

    Args:
        box3d_lidar: Lidar3D boxes, shape=(N, 7).
        r_rect: Matrix rect, shape=[4, 4].
        velo2cam: Translate matrix vel2cam, shape=[4, 4].

    Returns:
        np.ndarray: Camera3D boxes, shape=(N, 7).
    """
    xyz_lidar = box3d_lidar[:, 0:3]
    w, l, h = box3d_lidar[:, 3:4], box3d_lidar[:, 4:5], box3d_lidar[:, 5:6]
    r = box3d_lidar[:, 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return np.concatenate([xyz, l, h, w, r], axis=1)
