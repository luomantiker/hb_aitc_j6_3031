# Copyright (c) Horizon Robotics. All rights reserved.

import json
import os
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from shapely.geometry import Polygon

from hat.core.virtual_camera import FisheyeCamera, PinholeCamera
from .rotate_box_utils import let_nms, rotate_iou

VALID_Z = 0.1


def load_calib(
    attri_file: str, cameras: List[str]
) -> Tuple[Dict, np.ndarray, Dict, Dict]:
    """Load calibration for bev task in SD project.

    Args:
        attri_file: the json path, could be attribute.json or calibration.json
        cameras: camera view names

    Returns:
        lidar_calibs: cam intrinsics and lidar2cam extrinsics
        lidar2chassis: lidar2vcs matrix
        cam_per_view_shape: cam original image shape, [img_h, img_w]
        cameras_inst: camera instance
    """
    lidar_calibs = defaultdict(dict)
    cam_per_view_shape = defaultdict(list)
    cameras_inst = OrderedDict()
    with_lidar_calib = False
    if attri_file.endswith("attribute.json"):
        attr = json.load(open(attri_file, "r"))
        calibration = attr["calibration"]
        for view in cameras:
            if view in calibration:
                calibs = {}
                calibs["P2"] = np.array(calibration[view]["K"])
                calibs["disCoeffs"] = np.array(calibration[view]["d"])
                if f"lidar_top_2_{view}" in calibration:
                    with_lidar_calib = True
                    Tr_vel2cam = np.array(calibration[f"lidar_top_2_{view}"])
                    calibs["lidar_to_cam_R"] = Tr_vel2cam[:3, :3]
                    calibs["lidar_to_cam_T"] = Tr_vel2cam[:3, 3].reshape(
                        (-1, 1)
                    )
                    calibs["Tr_vel2cam"] = Tr_vel2cam
                lidar_calibs[view] = calibs
                img_h = calibration[view]["image_height"]
                img_w = calibration[view]["image_width"]
                cam_per_view_shape[view] = [img_h, img_w]

                if "fisheye" in view:
                    cameras_inst[view] = FisheyeCamera.init_cam_param_by_dict(
                        calibration[view]
                    )
                else:
                    cameras_inst[view] = PinholeCamera.init_cam_param_by_dict(
                        calibration[view]
                    )
                if with_lidar_calib:
                    cameras_inst[view].poseMat_lidar2cam = np.array(
                        calibration[f"lidar_top_2_{view}"]
                    )
                    cameras_inst[view].poseMat_lidar2vcs = np.array(
                        calibration["lidar_top_2_chassis"]
                    )
        if with_lidar_calib:
            lidar2chassis = np.array(
                calibration["lidar_top_2_chassis"], dtype=np.float64
            )
        else:
            lidar2chassis = None
    elif attri_file.endswith("calibration.json"):
        calibration = json.load(open(attri_file, "r"))
        camera_view_rename = {
            "camera_front_left": "camera_frontleft",
            "camera_front": "camera_front",
            "camera_front_right": "camera_frontright",
            "camera_rear_left": "camera_rearleft",
            "camera_rear": "camera_rear",
            "camera_rear_right": "camera_rearright",
            "fisheye_front": "camera_fisheye_front",
            "fisheye_rear": "camera_fisheye_rear",
            "fisheye_left": "camera_fisheye_left",
            "fisheye_right": "camera_fisheye_right",
            "camera_front_30fov": "camera_front_30fov",
        }
        with_lidar_calib = False
        for view in cameras:
            rename_view = camera_view_rename[view]
            if rename_view in calibration:
                calibs = {}
                calibs["P2"] = np.array(
                    calibration[rename_view]["K"], dtype=np.float64
                )
                calibs["disCoeffs"] = np.array(
                    calibration[rename_view]["d"], dtype=np.float64
                )
                if f"lidar_top_2_{rename_view}" in calibration:
                    with_lidar_calib = True
                    Tr_vel2cam = np.array(
                        calibration[f"lidar_top_2_{rename_view}"]
                    )
                    calibs["lidar_to_cam_R"] = Tr_vel2cam[:3, :3]
                    calibs["lidar_to_cam_T"] = Tr_vel2cam[:3, 3].reshape(
                        (-1, 1)
                    )
                    calibs["Tr_vel2cam"] = Tr_vel2cam
                lidar_calibs[view] = calibs
                img_h = calibration[rename_view + "_json"]["image_height"]
                img_w = calibration[rename_view + "_json"]["image_width"]
                cam_per_view_shape[view] = [img_h, img_w]

                if "fisheye" in view:
                    cameras_inst[view] = FisheyeCamera.init_cam_param_by_dict(
                        calibration[f"{rename_view}_json"]
                    )
                else:
                    cameras_inst[view] = PinholeCamera.init_cam_param_by_dict(
                        calibration[f"{rename_view}_json"]
                    )
                if with_lidar_calib:
                    cameras_inst[view].poseMat_lidar2cam = np.array(
                        calibration[f"lidar_top_2_{rename_view}"]
                    )
                    cameras_inst[view].poseMat_lidar2vcs = np.array(
                        calibration["lidar_top_2_vcs"]["T"]
                    )
        if with_lidar_calib:
            lidar2chassis = np.array(
                calibration["lidar_top_2_vcs"]["T"], dtype=np.float64
            )
        else:
            lidar2chassis = None
    else:
        assert os.path.exists(
            attri_file
        ), f"Please check the attribute file: {attri_file}"

    return lidar_calibs, lidar2chassis, cam_per_view_shape, cameras_inst


def get_3dbox_corners(
    loc: Sequence[np.ndarray],
    dim: Sequence[np.ndarray],
    heading_angle: float,
    coord_system: str = "vcs",
    with_line: bool = False,
    pts_per_line: int = 20,
) -> np.ndarray:
    """Get 3D box eight corners.

    In standard coordinate system, such as ``vcs``, ``lidar`` and ``local``,
    generate eight corners like:
                         z(h)
                          ^
                          |
                 5 -------|-4
                /|        |/|
               6 -------- 7 .  ^x(l)
               | |        | | /
               . 1 -------- 0
               |/         |/
       y(w)<---2 -------- 3

    In opencv camera coordinate system, i.e., ``cv_camera``,
    generate eight corners like:
                    ^z(w)
                   /
                  /
                 6 ---------5
                /|         /|
               7 -------- 4---->x(l)
               | |        | |
               . 2 -------- 1
               |/         |/
               3 -------- 0
               |
               |
               Vy(h)

    Args:
        loc : In standard coordinate, like [x, y, bottom_surface_cz]
            In 'cv_camera' coordinate, like [x, bottom_suface_cy, z]
        dim : [length, w, h]
        heading_angle : in radian, heading angle
            In standard coordinate, starting from x-axis towards
                counter-colockwise
            In 'cv_camera' coordinate, starting from x-axis towards colockwise
        coord_system : Type of coordinate system, candidates are
            ["cv_camera", "local", "vcs", "lidar"]
        with_line : Whether to return sample points of edges of bbox
            between corners.
        pts_per_line: Number of points per line.

    Returns
        corners : ndarray, shape [8, 3]
    """
    assert coord_system in ["cv_camera", "local", "vcs", "lidar"], (
        "Currently coord_system only supports"
        "['cv_camera', 'local', 'vcs', 'lidar']"
    )
    length, width, height = dim
    xx = [
        length / 2,
        length / 2,
        -length / 2,
        -length / 2,
        length / 2,
        length / 2,
        -length / 2,
        -length / 2,
    ]
    c, s = np.cos(heading_angle), np.sin(heading_angle)
    width_vector = [
        -width / 2,
        width / 2,
        width / 2,
        -width / 2,
        -width / 2,
        width / 2,
        width / 2,
        -width / 2,
    ]
    if coord_system == "cv_camera":
        zz = width_vector
        yy = [0, 0, 0, 0, -height, -height, -height, -height]
        rot_mat = np.array(
            [[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32
        )
    else:
        yy = width_vector
        zz = [0, 0, 0, 0, height, height, height, height]
        rot_mat = np.array(
            [[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32
        )
    corners = [xx, yy, zz]
    if with_line:
        pts = []
        for m in range(2):
            for i in range(4):
                j = 0 if i == 3 else i + 1
                one_line = [
                    np.linspace(
                        corners[k][m * 4 + i],
                        corners[k][m * 4 + j],
                        num=pts_per_line,
                        endpoint=False,
                    )
                    for k in range(3)
                ]
                pts.append(one_line)

        for x in [length / 2, -length / 2]:
            for y in [width / 2, -width / 2]:
                if coord_system == "cv_camera":
                    one_line = [
                        np.full((pts_per_line,), x),
                        np.linspace(
                            0, -height, num=pts_per_line, endpoint=False
                        ),
                        np.full((pts_per_line,), y),
                    ]
                else:
                    one_line = [
                        np.full((pts_per_line,), x),
                        np.full((pts_per_line,), y),
                        np.linspace(
                            0, height, num=pts_per_line, endpoint=False
                        ),
                    ]
                pts.append(one_line)
        corners = np.array(pts).transpose((0, 2, 1)).reshape((-1, 3)).T
    else:
        corners = np.array(corners, dtype=np.float32)

    corners = np.dot(rot_mat, corners)
    corners = corners + np.array(loc, dtype=np.float32).reshape(3, 1)
    return corners.T


def compute_box_3d(
    dim: Union[List, np.ndarray],
    location: Union[List, np.ndarray],
    rotation_y: float,
    pitch: float = 0.0,
) -> np.ndarray:
    """Compute 3d box by dimensions location and rotation_y.

    Args:
        dim: Shape (3), dimensions in (height, width, length) format.
        location: Shape (3), location in (x, y, z) format.
        rotation_y: yaw angle.
        pitch: pitch angle.

    Returns:
        corners_3d: Shape (8, 3).
    """
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)  # y

    cx, sx = np.cos(pitch), np.sin(pitch)
    Rx = np.array(
        [[1, 0, 0], [0, cx, sx], [0, -sx, cx]], dtype=np.float32
    )  # x

    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(Ry, corners)
    corners_3d = np.dot(Rx, corners_3d)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(
        3, 1
    )
    return corners_3d.transpose(1, 0)


def project_to_image(
    image_sise: Tuple[int],
    pts_3d: np.ndarray,
    P: np.ndarray,
    dist_coeff: Optional[np.ndarray] = None,
    fisheye: bool = False,
) -> np.ndarray:
    """Project 3d points to image plane.

    Args:
        image_size: Image shape.
        pts_3d: (N, 3), N x 3 matrix
        P:      (3, 4), 3 x 4 projection matrix
        dist_coeff: Distortion coefficient.
        fisheye: Whether to use fisheye. Defaults to False.

    Returns:
        pts_2d: (N, 2)
    """

    P = np.array(P)
    if fisheye:
        camera = FisheyeCamera(
            image_size=image_sise,
            camera_matrix=P[:3, :3],
            distcoeffs=dist_coeff[:4],
        )
    else:
        camera = PinholeCamera(
            image_size=image_sise,
            camera_matrix=P[:3, :3],
            distcoeffs=dist_coeff,
        )

    return camera.project_cam2pixel(pts_3d)


def project_3d_to_bird(
    pt: np.ndarray, out_size: np.ndarray, world_size: np.ndarray
) -> np.ndarray:
    pt[:, 0] += world_size / 2
    pt[:, 1] = world_size - pt[:, 1]
    pt = pt * out_size / world_size
    return pt.astype(np.int32)


def cam2img_pinhole_linear_kernel(
    points_cam: np.ndarray, P2: np.ndarray
) -> np.ndarray:
    """Convert points from camera coordinate to image."""
    x = points_cam[:, 0] / points_cam[:, 2]
    y = points_cam[:, 1] / points_cam[:, 2]

    fx, fy = P2[0, 0], P2[1, 1]
    cx, cy = P2[0, 2], P2[1, 2]
    xp = x * fx + cx
    yp = y * fy + cy
    image_pts = np.hstack((xp.reshape(-1, 1), yp.reshape(-1, 1)))
    return image_pts


def camera2image_pinhole(
    points_cam_input: np.ndarray,
    P2: np.ndarray,
    distCoeffs: np.ndarray,
    img_w: int = 1920,
    img_h: int = 1080,
    ratio: float = 0.5,
):
    """Project points from camera coordinate to image coordinate, \
        using pinhole model.

    http://wiki.hobot.cc/pages/viewpage.action?pageId=220151807
    """

    # Get valid point index
    # print("img_w: %d, img_h: %d"%(img_w, img_h))
    # print(points_cam.shape)
    wl = -ratio * img_w
    hl = -ratio * img_h
    wh = (1 + ratio) * img_w
    hh = (1 + ratio) * img_h
    points_cam = points_cam_input.copy()
    box_h = np.max(points_cam[:, 1]) - np.min(points_cam[:, 1])
    z_valid_index = np.squeeze(np.argwhere(points_cam[:, 2] > VALID_Z))
    if z_valid_index.size <= 2:
        return None
    z_valid_points = points_cam[z_valid_index, :]
    P2 = np.array(P2)
    z_valid_points_image = cam2img_pinhole_linear_kernel(z_valid_points, P2)
    valid_point_index = np.squeeze(
        np.argwhere(
            np.logical_and(
                np.logical_and(
                    z_valid_points_image[:, 0] > wl,
                    z_valid_points_image[:, 0] < wh,
                ),
                np.logical_and(
                    z_valid_points_image[:, 1] > hl,
                    z_valid_points_image[:, 1] < hh,
                ),
            )
        )
    )
    if valid_point_index.size <= 2:
        return None
    valid_point_original = z_valid_index[valid_point_index]
    valid_point_cam = points_cam[valid_point_original, :]

    valid_points_indx_set = set(valid_point_original)
    for i in range(points_cam.shape[0]):
        pp = points_cam[i, :]
        if i not in valid_points_indx_set:
            candi_anchors = valid_point_cam[
                np.abs(pp[1] - valid_point_cam[:, 1]) < box_h * 0.4, :
            ]
            if candi_anchors.shape[0] == 0:
                print("Error, there should be at least one valid point")
                return None
            elif candi_anchors.shape[0] == 1:
                anchor = candi_anchors[0, :]
            else:
                dist = np.sum(
                    np.square(candi_anchors[:, [0, 2]] - pp[[0, 2]]), axis=-1
                )
                anchor = candi_anchors[np.argsort(dist)[-2]]
            l = anchor[0]  # noqa E741
            r = pp[0]
            edge_p = np.zeros((1, 3))
            edge_p[0, 0] = pp[0]
            edge_p[0, 1] = pp[1]
            edge_p[0, 2] = pp[2]
            while np.abs(l - r) > 1e-6:
                mid = (l + r) / 2.0
                z = (pp[2] - anchor[2]) / (pp[0] - anchor[0]) * (
                    mid - pp[0]
                ) + pp[2]
                if z < VALID_Z:
                    r = mid
                    continue
                edge_p[0, 0] = mid
                edge_p[0, 2] = z
                edge_p_image = cam2img_pinhole_linear_kernel(edge_p, P2)
                if (
                    edge_p_image[0, 0] < wl
                    or edge_p_image[0, 0] > wh
                    or edge_p_image[0, 1] < hl
                    or edge_p_image[0, 1] > hh
                ):
                    r = mid
                else:
                    l = mid  # noqa E741
            points_cam[i, :] = edge_p[0, :]

    image_pts = project_to_image([img_w, img_h], points_cam, P2, distCoeffs)
    return image_pts


def project_velo_to_camera(vel_data: np.ndarray, Tr: np.ndarray) -> np.ndarray:
    # vel_data_c: col 0: back -> front
    #             col 1: down -> up
    #             col 2: left -> right
    homo_vel_data = np.hstack(
        (vel_data[:, :3], np.ones((vel_data.shape[0], 1), dtype="float32"))
    )
    vel_data_c = np.dot(homo_vel_data, Tr.T)
    vel_data_c /= vel_data_c[:, -1].reshape((-1, 1))
    vel_data_c = np.hstack(
        (vel_data_c[:, :3], vel_data[:, -1].reshape((-1, 1)))
    )
    return vel_data_c


def compute_corners3d_lidar(
    corners3d: np.ndarray, lidar_to_camera: Union[np.ndarray, List]
) -> np.ndarray:
    # compute lidar_corners3d
    assert lidar_to_camera is not None
    if isinstance(lidar_to_camera, list):
        lidar_to_camera = np.array(lidar_to_camera)
    homo_cam_corners3d = np.hstack(
        (corners3d[:, :3], np.ones((corners3d.shape[0], 1), dtype="float32"))
    )
    lidar_corners3d = project_velo_to_camera(
        homo_cam_corners3d, np.linalg.inv(lidar_to_camera["Tr_vel2cam"])
    )
    lidar_corners3d = lidar_corners3d[:, :3]
    return lidar_corners3d


def compute_yaw_lidar(lidar_corners3d: np.ndarray) -> np.ndarray:
    assert lidar_corners3d.shape[0] == 8
    bev_bbox = lidar_corners3d[:4, :].reshape(-1, 3)
    bev_bbox = bev_bbox[:, :2]
    # cal yaw
    p1 = (
        bev_bbox[0, :] + bev_bbox[1, :] + bev_bbox[2, :] + bev_bbox[3, :]
    ) / 4
    p2 = (bev_bbox[0, :] + bev_bbox[1, :]) / 2
    lidar_yaw = np.arctan2(p2[0] - p1[0], p2[1] - p1[1])
    return lidar_yaw


def compute_location_lidar(
    location: np.ndarray, lidar_to_camera: Union[np.ndarray, List]
) -> np.ndarray:
    if isinstance(location, list):
        location = np.array(location).reshape(-1, 3)
    if isinstance(lidar_to_camera, list):
        lidar_to_camera = np.array(lidar_to_camera)
    homo_cam_location3d = np.hstack(
        (location, np.ones((location.shape[0], 1), dtype="float32"))
    )
    lidar_location = project_velo_to_camera(
        homo_cam_location3d,
        np.linalg.inv(np.array(lidar_to_camera["Tr_vel2cam"])),
    )  # noqa
    lidar_location = lidar_location[:, :3]
    return lidar_location


def compute_box_3d_lidar(
    dim: Union[np.ndarray, List],
    location: Union[np.ndarray, List],
    yaw: np.ndarray,
    with_size: bool = False,
) -> np.ndarray:
    x, y, z = location[0], location[1], location[2]
    h, w, l = dim[0], dim[1], dim[2]  # noqa E741
    corner = np.array(
        [
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )
    rotMat = np.array(
        [
            [np.cos(np.pi / 2 + yaw), np.sin(np.pi / 2 + yaw), 0.0],
            [-np.sin(np.pi / 2 + yaw), np.cos(np.pi / 2 + yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # rotMat = np.array([[np.cos(yaw), np.sin(yaw), 0.0],
    #                    [-np.sin(yaw), np.cos(yaw), 0.0],
    #                    [0.0, 0.0, 1.0]])
    cornerPosInVelo = (
        np.dot(rotMat, corner) + np.tile(np.array([x, y, z]), (8, 1)).T
    )
    box3d = cornerPosInVelo.transpose()
    if with_size:
        return box3d, h, w, l
    return box3d


def compute_corners3d_camera(
    lidar_corner3d: np.ndarray, lidar_to_camera: Union[np.ndarray, List]
) -> np.ndarray:
    # compute camera_corners3d
    assert lidar_to_camera is not None
    if isinstance(lidar_to_camera, list):
        lidar_to_camera = np.array(lidar_to_camera)
    homo_lidar_corners3d = np.hstack(
        (
            lidar_corner3d[:, :3],
            np.ones((lidar_corner3d.shape[0], 1), dtype="float32"),
        )
    )  # noqa
    cam_corners3d = project_velo_to_camera(
        homo_lidar_corners3d, np.array(lidar_to_camera["Tr_vel2cam"])
    )
    cam_corners3d = cam_corners3d[:, :3]
    return cam_corners3d


def inside_rot_cube(points: np.ndarray, cube3d: np.ndarray) -> np.ndarray:
    # return points which inside cube.
    assert cube3d.shape[0] == 8
    assert cube3d.shape[1] == 3
    b1, b2, b3, b4, t1, t2, t3, t4 = cube3d

    dir1 = t1 - b1
    size1 = np.linalg.norm(dir1)
    dir1 = dir1 / size1

    dir2 = b2 - b1
    size2 = np.linalg.norm(dir2)
    dir2 = dir2 / size2

    dir3 = b4 - b1
    size3 = np.linalg.norm(dir3)
    dir3 = dir3 / size3

    cube3d_center = (b1 + t3) / 2.0

    dir_vec = points - cube3d_center

    res1 = np.where((np.absolute(np.dot(dir_vec, dir1)) * 2) <= size1)[0]
    res2 = np.where((np.absolute(np.dot(dir_vec, dir2)) * 2) <= size2)[0]
    res3 = np.where((np.absolute(np.dot(dir_vec, dir3)) * 2) <= size3)[0]
    inside_index = list(set(res1) & set(res2) & set(res3))

    return points[inside_index]


def inside_rect(
    points: np.ndarray, rect: Union[np.ndarray, List]
) -> np.ndarray:
    # return points which inside rect.
    assert points.shape[1] == 2
    x1, y1, x2, y2 = rect

    res1 = np.where(points[:, 0] < x2)[0]
    res2 = np.where(points[:, 0] >= x1)[0]
    res3 = np.where(points[:, 1] < y2)[0]
    res4 = np.where(points[:, 1] >= y1)[0]
    inside_index = list(set(res1) & set(res2) & set(res3) & set(res4))
    if len(inside_index) == 0:
        print(points)
        return None
    return points[inside_index]


def indicies_of_inliers(points: np.ndarray) -> np.ndarray:
    index = np.abs(points - np.mean(points, axis=0)) <= (
        3 * np.std(points, axis=0)
    )  # noqa
    index = np.where(index[:, 0] & index[:, 1])[0]
    return points[index]


def project_to_image_ex(
    corners3d: np.ndarray,
    calib: np.ndarray,
    dist_coeff: np.ndarray,
    fisheye: bool,
    img_wh: Tuple[int],
    bin_num: int = 10,
    z_thresh: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    # project corners3d to image and get 2d circumscribed rectangle.
    # pay attention if z_min < 0
    assert corners3d.shape[0] == 8
    assert corners3d.shape[1] == 3

    x_min = np.min(corners3d[:, 0])
    x_max = np.max(corners3d[:, 0])
    y_min = np.min(corners3d[:, 1])
    y_max = np.max(corners3d[:, 1])
    z_min = np.min(corners3d[:, 2])
    z_max = np.max(corners3d[:, 2])
    rect = (0, 0, img_wh[0], img_wh[1])

    if z_min > z_thresh:
        corners3d_proj = project_to_image(
            corners3d, calib, dist_coeff=dist_coeff, fisheye=fisheye
        )
        corners3d_proj = corners3d_proj.reshape(-1, 2).astype(np.int32)
        bbox2d = np.concatenate(
            [np.min(corners3d_proj, axis=0), np.max(corners3d_proj, axis=0)]
        )
    else:
        x_ = np.linspace(x_min, x_max, bin_num * 2)
        y_ = np.linspace(y_min, y_max, bin_num * 2)
        z_ = np.linspace(max(0.5, z_min), max(1.0, z_max), bin_num)
        x, y, z = np.meshgrid(x_, y_, z_, indexing="ij")
        discrete_points = np.stack((x, y, z), axis=-1)
        discrete_points = discrete_points.reshape(-1, 3)
        discrete_points = inside_rot_cube(discrete_points, corners3d)
        discrete_points = np.concatenate((discrete_points, corners3d), axis=0)
        corners3d_proj = project_to_image(
            discrete_points, calib, dist_coeff=dist_coeff, fisheye=fisheye
        )
        corners3d_proj = corners3d_proj.reshape(-1, 2)
        corners3d_proj = inside_rect(corners3d_proj, rect)
        if corners3d_proj is None:
            return None, None
        corners3d_proj = indicies_of_inliers(corners3d_proj)
        corners3d_proj = corners3d_proj.astype(np.int32)
        bbox2d = np.concatenate(
            [np.min(corners3d_proj, axis=0), np.max(corners3d_proj, axis=0)]
        )
    return bbox2d, corners3d_proj


def compute_2d_box(
    dim: Union[np.ndarray, List],
    location: Union[np.ndarray, List],
    yaw: np.ndarray,
    calib: np.ndarray,
    img_wh: Tuple[int],
    z_thresh: float,
    dist_coeff: Optional[np.ndarray] = None,
    fisheye: bool = False,
    lidar_to_camera: Optional[Union[np.ndarray, List]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if lidar_to_camera is not None:
        corners3d_lidar = compute_box_3d_lidar(dim, location, yaw)
        # corners3d in lidar-> corners3d in cam
        corners3d = compute_corners3d_camera(corners3d_lidar, lidar_to_camera)
    else:
        # default camera, corners3d in camera
        corners3d = compute_box_3d(dim, location, yaw)
    # proj to image
    if fisheye:
        bbox2d, corners3d_proj = project_to_image_ex(
            corners3d, calib, dist_coeff, fisheye, img_wh, z_thresh=z_thresh
        )
    else:
        corners3d_proj = camera2image_pinhole(
            corners3d, calib, dist_coeff, img_w=img_wh[0], img_h=img_wh[1]
        )
        if corners3d_proj is None:
            bbox2d = None
        else:
            corners3d_proj = corners3d_proj.reshape(-1, 2).astype(np.int32)
            bbox2d = np.concatenate(
                [
                    np.min(corners3d_proj, axis=0),
                    np.max(corners3d_proj, axis=0),
                ]
            )
    return bbox2d, corners3d


def bev3d_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    thresh: float,
    cls_ids: Optional[torch.Tensor] = None,
    agnostic: bool = True,
) -> np.ndarray:
    """Nms function for bev3d on vcs plane.

    Args:
        boxes: Input boxes could be of two kinds of shape.
            If the shape is [N, 7], 7 means [x, y, z, h, w, l, rot_z].
            If the shape is [N, 5], 5 means [x, y, l, w, rot_z].
        scores: Scores of boxes with the shape of [N].
        thresh: Threshold.
        cls_ids: Class id of boxes. It must be specified when not do
            class-agnostic NMS.
        agnostic: whether do class-agnostic NMS or not, default is True.
    Returns:
        Indexes to keep after nms.
    """
    assert len(boxes) == len(scores), "len(scores) must be equal to len(boxes)"
    if not agnostic:
        assert cls_ids is not None and len(cls_ids) == len(scores)
        cls_ids = cls_ids.cpu().numpy()

    boxes = boxes.cpu().numpy()
    if boxes.shape[1] == 7:
        # change the boxes [N,7] -> [N,5] for rotate_2d iou calculation
        loc_xy = boxes[:, :2]
        dim_lw = boxes[:, 5:3:-1]
        yaw = boxes[:, -1:]
        det_bbox3d = np.concatenate([loc_xy, dim_lw, -yaw], axis=1)
    else:
        assert boxes.shape[1] == 5
        det_bbox3d = boxes

    scores = scores.cpu().numpy()
    order = np.argsort(scores)
    keep = np.zeros(det_bbox3d.shape[0])

    # picked_boxes = []
    while order.size > 0:
        index = order[-1]
        keep[index] = 1
        # picked_boxes.append(det_bbox3d[index])
        current_box = np.expand_dims(det_bbox3d[index], axis=0)
        left_boxes = det_bbox3d[order[:-1]]
        rotate_3d_ious = rotate_iou(current_box, left_boxes)
        rotate_3d_ious = np.squeeze(rotate_3d_ious)
        left = rotate_3d_ious < thresh
        if agnostic:
            left = np.where(left)
        else:
            current_id = cls_ids[index]
            left_ids = cls_ids[order[:-1]]
            left = np.where(np.logical_or(left, left_ids != current_id))
        order = order[left]
    return keep.astype("bool")


def bev3d_let_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    let_nms_param: List[dict],
    cls_ids: torch.Tensor,
    agnostic: bool = True,
) -> np.ndarray:
    """Letnms function for bev3d on vcs plane.

    Args:
        boxes: Input boxes could be of two kinds of shape.
            If the shape is [N, 7], 7 means [x, y, z, h, w, l, rot_z].
            If the shape is [N, 5], 5 means [x, y, l, w, rot_z].
        scores: Scores of boxes with the shape of [N].
        cls_ids: Class id of boxes.
        let_nms_param: list of Dict specify the letnms parameters
            of each category.
            for instance
                [
                    {
                        "p_t": 0.15,
                        "min_t": 4.0,
                        "max_t": 6.0,
                        "radius": 0.4,
                        "e_loc_threshold": 0.1,
                        "angle_threshold": 1.0,
                        "area_threshold": 0.8,
                    },
                ]
        agnostic: whether do class-agnostic NMS or not, default is True.
    Returns:
        Indexes to keep after nms.
    """
    assert len(boxes) == len(scores), "len(scores) must be equal to len(boxes)"
    assert len(cls_ids) == len(
        scores
    ), "len(cls_ids) must be equal to len(scores)"
    with torch.no_grad():
        scores = scores.cpu().numpy()
        boxes = boxes.cpu().numpy()
    if boxes.shape[1] == 7:
        # keep consistent with rotate_2d iou calculation format
        loc_xy = boxes[:, :2]
        dim_lw = boxes[:, 5:3:-1]
        yaw = boxes[:, -1:]
        det_bbox3d = np.concatenate([loc_xy, dim_lw, yaw], axis=1)
    else:
        assert boxes.shape[1] == 5
        det_bbox3d = boxes

    cls_ids = cls_ids.cpu().numpy()
    order = np.argsort(scores, kind="stable")
    keep = np.zeros(det_bbox3d.shape[0])

    # picked_boxes = []
    while order.size > 0:
        index = order[-1]
        keep[index] = 1
        # picked_boxes.append(det_bbox3d[index])
        current_box = np.expand_dims(det_bbox3d[index], axis=0)
        left_boxes = det_bbox3d[order[:-1]]
        left = let_nms(
            current_box, left_boxes, **let_nms_param[cls_ids[index]]
        )
        left = np.squeeze(left)
        if agnostic:
            left = np.where(left)
        else:
            current_id = cls_ids[index]
            left_ids = cls_ids[order[:-1]]
            left = np.where(np.logical_or(left, left_ids != current_id))
        order = order[left]
    return keep.astype("bool")


def boxes_iou3d_vcs(
    box_a: Sequence, box_b: Sequence
) -> Tuple[np.ndarray, np.ndarray]:
    """Return iou3d results, boxes in vcs coord.

    Args:
        box_a: [x, y, z, l, w, h, yaw]
        box_b: [x, y, z, l, w, h, yaw]
    return:
        iou_bev
        iou3d
    """
    corner_3d_a = get_3dbox_corners(box_a[:3], box_a[3:6], box_a[6])
    corner_3d_b = get_3dbox_corners(box_b[:3], box_b[3:6], box_b[6])
    bbox_bev_a = corner_3d_a[:4, :2].tolist()
    bbox_bev_b = corner_3d_b[:4, :2].tolist()
    ground_bbox_polygon_a = Polygon(bbox_bev_a)
    ground_bbox_polygon_b = Polygon(bbox_bev_b)
    area_intersection = ground_bbox_polygon_a.intersection(
        ground_bbox_polygon_b
    ).area
    iou_bev = area_intersection / (
        ground_bbox_polygon_a.area
        + ground_bbox_polygon_b.area
        - area_intersection
    )

    min_line = max(corner_3d_a[:, 2].min(), corner_3d_b[:, 2].min())
    max_line = min(corner_3d_a[:, 2].max(), corner_3d_b[:, 2].max())
    height_intersection = max(0, max_line - min_line)
    intersection = area_intersection * height_intersection
    union = np.prod(box_a[3:6]) + np.prod(box_b[3:6]) - intersection
    iou3d = np.clip(intersection / union, 0, 1)
    return iou_bev, iou3d


def bev3d_multiclass_nms(
    boxes: torch.Tensor,
    boxes_for_nms: torch.Tensor,
    scores: torch.Tensor,
    score_thresh: float,
    nms_thresh: float,
    max_num: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multiclass nms function for bev3d on vcs plane.

    Args:
        boxes: boxes with shape (N, M). M is the dimensions of boxes.
        boxes_for_nms: Input boxes could be of two kinds of shape.
            If the shape is [N, 7], 7 means [x, y, z, h, w, l, rot_z].
            If the shape is [N, 5], 5 means [x, y, l, w, rot_z].
        scores: Scores of boxes with the shape of [N, C].
        score_thresh: Threshold of box scores.
        nms_thresh: Threshold of nms.
        max_num: Maximum number of boxes will be kept.
    Returns:
        Results after nms, including 3D
            bounding boxes, scores, labels.
    """

    num_classes = scores.shape[1]
    keep_bboxes = []
    keep_scores = []
    keep_labels = []

    for cls_i in range(0, num_classes):
        cls_inds = scores[:, cls_i] > score_thresh
        if not cls_inds.any():
            continue

        _scores = scores[cls_inds, cls_i]
        _boxes_for_nms = boxes_for_nms[cls_inds, :]
        order = _scores.sort(0, descending=True)[1]
        _boxes_for_nms = _boxes_for_nms[order].contiguous()
        _scores = _scores[order]

        _selected = bev3d_nms(_boxes_for_nms, _scores, nms_thresh)

        _selected = torch.from_numpy(_selected)

        keep_scores.append(_scores[_selected])

        _selected = order[_selected]
        cls_label = boxes.new_full((len(_selected),), cls_i, dtype=torch.long)
        keep_labels.append(cls_label)
        _boxes = boxes[cls_inds, :]
        keep_bboxes.append(_boxes[_selected])

    if keep_bboxes:
        keep_bboxes = torch.cat(keep_bboxes, dim=0)
        keep_scores = torch.cat(keep_scores, dim=0)
        keep_labels = torch.cat(keep_labels, dim=0)

        if keep_bboxes.shape[0] > max_num:
            _, inds = keep_scores.sort(descending=True)
            inds = inds[:max_num]
            keep_bboxes = keep_bboxes[inds, :]
            keep_labels = keep_labels[inds]
            keep_scores = keep_scores[inds]

    else:
        keep_bboxes = boxes.new_zeros((0, boxes.size(-1)))
        keep_scores = boxes.new_zeros((0,))
        keep_labels = boxes.new_zeros((0,), dtype=torch.long)

    results = (keep_bboxes, keep_scores, keep_labels)

    return results


def points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d: Points in shape (N, 3)
        proj_mat: Transformation matrix between coordinates.
        with_depth: Whether to keep depth in the output.
            Defaults to False.

    Returns:
        Points in image coordinates,
            with shape [N, 2] if `with_depth=False`, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, (
        "The dimension of the projection"
        f" matrix should be 2 instead of {len(proj_mat.shape)}."
    )
    d1, d2 = proj_mat.shape[:2]
    assert (
        (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (d1 == 4 and d2 == 4)
    ), ("The shape of the projection matrix" f" ({d1}*{d2}) is not supported.")
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype
        )
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

    return point_2d_res


def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes: Bounding boxes with shape (N, 5).
        labels: Labels with shape (N, ).
        scores: Scores with shape (N, ).
        attrs: Attributes with shape (N, ).
            Defaults to None.

    Returns:
        Bounding box results in cpu mode.

            - boxes_3d: 3D boxes.
            - scores: Prediction scores.
            - labels_3d: Box labels.
            - attrs_3d: Box attributes.
    """
    result_dict = {
        "boxes_3d": bboxes.to("cpu"),
        "scores_3d": scores.cpu(),
        "labels_3d": labels.cpu(),
    }

    if attrs is not None:
        result_dict["attrs_3d"] = attrs.cpu()

    return result_dict


def points_img2cam(points, cam2img):
    """Project points in image coordinates to camera coordinates.

    Args:
        points: 2.5D points in 2D images, [N, 3],
            3 corresponds with x, y in the image and depth.
        cam2img: Camera intrinsic matrix. The shape can be
            [3, 3], [3, 4] or [4, 4].

    Returns:
        points in 3D space. [N, 3],
            3 corresponds with x, y, z in 3D space.
    """
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.tensor(cam2img)
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    pad_cam2img[: cam2img.shape[0], : cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]

    return points3D


def xywhr2xyxyr(boxes_xywhr: torch.Tensor):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr: Rotated boxes in XYWHR format.

    Returns:
        Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[..., 2] / 2
    half_h = boxes_xywhr[..., 3] / 2

    boxes[..., 0] = boxes_xywhr[..., 0] - half_w
    boxes[..., 1] = boxes_xywhr[..., 1] - half_h
    boxes[..., 2] = boxes_xywhr[..., 0] + half_w
    boxes[..., 3] = boxes_xywhr[..., 1] + half_h
    boxes[..., 4] = boxes_xywhr[..., 4]
    return boxes


def rotation_3d_in_axis(
    points: torch.Tensor,
    angles: torch.Tensor,
    axis: int = 0,
    return_mat: bool = False,
    clockwise: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """Rotate points by angles according to axis.

    Args:
        points:
            Points of shape (N, M, 3).
        angles:
            Vector of angles in shape (N,)
        axis: The axis to be rotated. Defaults to 0.
        return_mat: Whether or not return the rotation matrix (transposed).
            Defaults to False.
        clockwise: Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will
            raise value error.

    Returns:
        Rotated points in shape (N, M, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles)

    assert (
        len(points.shape) == 3
        and len(angles.shape) == 1
        and points.shape[0] == angles.shape[0]
    ), (
        f"Incorrect shape of points " f"angles: {points.shape}, {angles.shape}"
    )

    assert points.shape[-1] in [
        2,
        3,
    ], f"Points size should be 2 or 3 instead of {points.shape[-1]}"

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = torch.stack(
                [
                    torch.stack([rot_cos, zeros, -rot_sin]),
                    torch.stack([zeros, ones, zeros]),
                    torch.stack([rot_sin, zeros, rot_cos]),
                ]
            )
        elif axis == 2 or axis == -1:
            rot_mat_T = torch.stack(
                [
                    torch.stack([rot_cos, rot_sin, zeros]),
                    torch.stack([-rot_sin, rot_cos, zeros]),
                    torch.stack([zeros, zeros, ones]),
                ]
            )
        elif axis == 0 or axis == -3:
            rot_mat_T = torch.stack(
                [
                    torch.stack([ones, zeros, zeros]),
                    torch.stack([zeros, rot_cos, rot_sin]),
                    torch.stack([zeros, -rot_sin, rot_cos]),
                ]
            )
        else:
            raise ValueError(
                f"axis should in range " f"[-3, -2, -1, 0, 1, 2], got {axis}"
            )
    else:
        rot_mat_T = torch.stack(
            [torch.stack([rot_cos, rot_sin]), torch.stack([-rot_sin, rot_cos])]
        )

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum("aij,jka->aik", points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum("jka->ajk", rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new


def yaw2local(yaw: torch.Tensor, loc: torch.Tensor):
    """Transform global yaw to local yaw (alpha in kitti) in camera \
    coordinates, ranges from -pi to pi.

    Args:
        yaw: A vector with local yaw of each box.
            shape: (N, )
        loc: gravity center of each box.
            shape: (N, 3)

    Returns:
        local yaw (alpha in kitti).
    """
    local_yaw = yaw - torch.atan2(loc[:, 0], loc[:, 2])
    larger_idx = (local_yaw > np.pi).nonzero(as_tuple=False)
    small_idx = (local_yaw < -np.pi).nonzero(as_tuple=False)
    if len(larger_idx) != 0:
        local_yaw[larger_idx] -= 2 * np.pi
    if len(small_idx) != 0:
        local_yaw[small_idx] += 2 * np.pi

    return local_yaw


def corners_to_local_rot_y(corners):
    x, y = 0.5 * (
        corners[0, [2, 0]]
        + corners[3, [2, 0]]
        - corners[1, [2, 0]]
        - corners[2, [2, 0]]
    )
    rot_y = np.arctan2(y, x) - np.pi / 2
    if rot_y < -np.pi:
        rot_y += 2 * np.pi
    elif rot_y > np.pi:
        rot_y -= 2 * np.pi
    return rot_y


def get_3dboxcorner_in_velo(box3d, with_size=False):
    """Convert from KITTI label to 3dbox in velo."""
    x, y, z, w, l, h, yaw = box3d
    corner = np.array(
        [
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )
    rotMat = np.array(
        [
            [np.cos(np.pi / 2 + yaw), np.sin(np.pi / 2 + yaw), 0.0],
            [-np.sin(np.pi / 2 + yaw), np.cos(np.pi / 2 + yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    cornerPosInVelo = (
        np.dot(rotMat, corner) + np.tile(np.array([x, y, z]), (8, 1)).T
    )
    box3d = cornerPosInVelo.transpose()
    if with_size:
        return box3d, h, w, l
    return box3d


def cal_rot_y_from_3dbox_corners_in_camera(corners):
    """Calculate the rotation angle of 3d box in camera coordinate.

    Args:
        corners: 8 x 3 array like 3d box corners in camera coordinate.
    Returns:
        rot_y: rotation angle of 3d box in camera coordinate.
    """
    x, y = 0.5 * (
        corners[0, [0, 2]]
        - corners[3, [0, 2]]
        + corners[1, [0, 2]]
        - corners[2, [0, 2]]
    )
    rot_y = -np.arctan2(y, x)
    if rot_y < -np.pi:
        rot_y += 2 * np.pi
    elif rot_y > np.pi:
        rot_y -= 2 * np.pi
    return rot_y
