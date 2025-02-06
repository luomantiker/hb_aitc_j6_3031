# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import math

import cv2
import numpy as np

PI = 3.14
FLT_EPSILON = 0.00001
DEFAULT_HEIGHT = 720
DEFAULT_WIDTH = 1280
DEFAULT_DISTORT = [0, 0, 0, 0]
DEFAULT_CAMERA_TYPE = 0
DEFAULT_PINHOLE_FOV = 58.0
DEFAULT_FISHEYE_FOV = 192.0


def pt_distort(pt, mat_distort, mat_camera):
    x, y = pt
    try:
        assert len(mat_distort) >= 4
    except AssertionError:
        mat_distort = [0.0, 0.0, 0.0, 0.0]

    k1, k2, p1, p2 = mat_distort[:4]
    k3, k4, k5, k6 = 0.0, 0.0, 0.0, 0.0
    if len(mat_distort) > 4:
        k3 = mat_distort[4]
    if len(mat_distort) > 5:
        k4 = mat_distort[5]
    if len(mat_distort) > 6:
        k5 = mat_distort[6]
    if len(mat_distort) > 7:
        k6 = mat_distort[7]

    fu = mat_camera[0][0]
    fv = mat_camera[1][1]
    cu = mat_camera[0][2]
    cv = mat_camera[1][2]

    u = (x - cu) / fu
    v = (y - cv) / fv
    r2 = u ** 2 + v ** 2

    coef = (1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (
        1.0 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
    )  # noqa
    dist_u = u * coef + 2 * p1 * u * v + p2 * (r2 + 2 * u ** 2)
    dist_v = v * coef + p1 * (r2 + 2 * v ** 2) + 2 * p2 * u * v

    dist_x = dist_u * fu + cu
    dist_y = dist_v * fv + cv

    return [dist_x, dist_y]


def distort_bbox(bbox, mat_distort, mat_camera):
    top = [0.5 * (bbox[0] + bbox[2]), bbox[1]]
    left = [bbox[0], 0.5 * (bbox[1] + bbox[3])]
    bottom = [0.5 * (bbox[0] + bbox[2]), bbox[3]]
    right = [bbox[2], 0.5 * (bbox[1] + bbox[3])]

    dist_top = pt_distort(top, mat_distort, mat_camera)
    dist_left = pt_distort(left, mat_distort, mat_camera)
    dist_bottom = pt_distort(bottom, mat_distort, mat_camera)
    dist_right = pt_distort(right, mat_distort, mat_camera)

    x1 = dist_left[0]
    y1 = dist_top[1]
    x2 = dist_right[0]
    y2 = dist_bottom[1]

    return [x1, y1, x2, y2]


def undistort_bbox(bbox, lut):
    left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    return box_undistort(left, top, right, bottom, lut)


def get_normal_lut(disort, camera_mat, fov2, h, w):
    f = [
        camera_mat[0][0],
        camera_mat[1][1],
        camera_mat[0][2],
        camera_mat[1][2],
    ]
    undistort_lut = np.zeros((h // 2, w // 2, 2))

    temp_map = np.zeros(
        (undistort_lut.shape[0] * undistort_lut.shape[1], 1, 2)
    )
    for t in range(h // 2):
        y = t * w // 2
        ty = t * 2
        for x in range(w // 2):
            tx = x * 2
            temp_map[y + x][0][0] = tx
            temp_map[y + x][0][1] = ty
    temp_map.tolist()
    temp_map2 = cv2.undistortPoints(temp_map, camera_mat, disort)
    for t in range(undistort_lut.shape[0]):
        y = t * undistort_lut.shape[1]
        for x in range(undistort_lut.shape[1]):
            x0 = temp_map[y + x][0][0]
            y0 = temp_map[y + x][0][1]
            x1 = temp_map2[y + x][0][0]
            y1 = temp_map2[y + x][0][1]

            undistort_lut[t][x][0] = x1 * f[0] + f[2] - x0
            undistort_lut[t][x][1] = y1 * f[1] + f[3] - y0
    undistort_lut.tolist()
    return undistort_lut


def get_fisheye_lut(disort, camera_mat, fov2, h, w):
    f = [
        camera_mat[0][0],
        camera_mat[1][1],
        camera_mat[0][2],
        camera_mat[1][2],
    ]
    theta_thres = 85.0 * PI / 180.0
    theta_thres2 = theta_thres * theta_thres
    theta_thres4 = theta_thres2 * theta_thres2
    theta_thres6 = theta_thres4 * theta_thres2
    theta_thres8 = theta_thres4 * theta_thres4
    thetad_thres = theta_thres * (
        1.0
        + disort[0] * theta_thres2
        + disort[1] * theta_thres4
        + disort[2] * theta_thres6
        + disort[3] * theta_thres8
    )
    undistort_lut = np.zeros(((h // 2) + 1, (w // 2) + 1, 2))
    for y in range(h // 2):
        dst = undistort_lut[y]
        ty = y * 2
        delta_x = 0.0
        delta_y = 0.0
        ddx = 0.0
        ddy = 0.0
        flag = True
        start_x = f[2] / 2.0
        for x in range(int(start_x), w // 2, 1):
            tx = x * 2
            pw = [(tx - f[2]) / f[0], (ty - f[3]) / f[1]]
            r = math.sqrt(pw[0] * pw[0] + pw[1] * pw[1])
            theta_d = copy.deepcopy(r)
            scale = 1.0
            if theta_d > FLT_EPSILON:
                if theta_d > thetad_thres:
                    delta_x = dst[(x - 1)][0] - dst[(x - 2)][0]
                    delta_y = dst[(x - 1)][1] - dst[(x - 2)][1]
                    if flag:
                        ddx = delta_x - (dst[(x - 2)][0] - dst[(x - 3)][0])
                        ddy = delta_y - (dst[(x - 2)][1] - dst[(x - 3)][1])
                        flag = False
                    dst[x][0] = dst[(x - 1)][0] + delta_x + ddx
                    dst[x][1] = dst[(x - 1)][1] + delta_y + ddy
                    continue
                theta = copy.deepcopy(theta_d)
                for _ in range(10):
                    theta2 = theta * theta
                    theta4 = theta2 * theta2
                    theta6 = theta4 * theta2
                    theta8 = theta6 * theta2
                    k0_theta2 = disort[0] * theta2
                    k1_theta4 = disort[1] * theta4
                    k2_theta6 = disort[2] * theta6
                    k3_theta8 = disort[3] * theta8
                    theta_fix = (
                        theta
                        * (1.0 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8)
                        - theta_d
                    ) / (
                        1.0
                        + 3.0 * k0_theta2
                        + 5.0 * k1_theta4
                        + 7.0 * k2_theta6
                        + 9.0 * k3_theta8
                    )
                    theta = theta - theta_fix
                    if math.fabs(theta_fix) < FLT_EPSILON:
                        break
                scale = math.tan(theta) / theta_d
            dx = pw[0] * scale * f[0] + f[2] - tx
            dy = pw[1] * scale * f[1] + f[3] - ty
            dst[x][0] = dx
            dst[x][1] = dy
        flag = True
        start_x = f[2] / 2.0
        for x in range((int(start_x) - 1), -1, -1):
            tx = x * 2
            pw[0] = (tx - f[2]) / f[0]
            pw[1] = (ty - f[3]) / f[1]
            r = math.sqrt(pw[0] * pw[0] + pw[1] * pw[1])
            theta_d = copy.deepcopy(r)
            scale = 1.0
            if theta_d > FLT_EPSILON:
                if theta_d > thetad_thres:
                    delta_x = dst[(x + 1)][0] - dst[(x + 2)][0]
                    delta_y = dst[(x + 1)][1] - dst[(x + 2)][1]
                    if flag:
                        ddx = delta_x - (dst[(x + 2)][0] - dst[(x + 3)][0])
                        ddy = delta_y - (dst[(x + 2)][1] - dst[(x + 3)][1])
                        flag = False
                    dst[x][0] = dst[(x + 1)][0] + delta_x + ddx
                    dst[x][1] = dst[(x + 1)][1] + delta_y + ddy
                    continue
                theta = theta_d
                for _ in range(10):
                    theta2 = theta * theta
                    theta4 = theta2 * theta2
                    theta6 = theta4 * theta2
                    theta8 = theta6 * theta2
                    k0_theta2 = disort[0] * theta2
                    k1_theta4 = disort[1] * theta4
                    k2_theta6 = disort[2] * theta6
                    k3_theta8 = disort[3] * theta8
                    theta_fix = (
                        theta
                        * (1.0 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8)
                        - theta_d
                    ) / (
                        1.0
                        + 3.0 * k0_theta2
                        + 5.0 * k1_theta4
                        + 7.0 * k2_theta6
                        + 9.0 * k3_theta8
                    )
                    theta = theta - theta_fix
                    if math.fabs(theta_fix) < FLT_EPSILON:
                        break
                scale = math.tan(theta) / theta_d

            dx = pw[0] * scale * f[0] + f[2] - tx
            dy = pw[1] * scale * f[1] + f[3] - ty
            dst[x][0] = dx
            dst[x][1] = dy
    undistort_lut[h // 2] = undistort_lut[h // 2 - 1]
    undistort_lut[h // 2][w // 2] = undistort_lut[h // 2 - 1][w // 2 - 1]
    return undistort_lut


def ptr_undistort(pt_x, pt_y, undistort_lut):
    ix = int(pt_x / 2)
    iy = int(pt_y / 2)
    margin_x = 2
    margin_y = 2
    if ix < margin_x:
        ix = margin_x
    if ix > undistort_lut.shape[1] - margin_x:
        ix = undistort_lut.shape[1] - margin_x
    if iy < margin_y:
        iy = margin_y
    if iy > undistort_lut.shape[0] - margin_y:
        iy = undistort_lut.shape[0] - margin_y

    return pt_x + undistort_lut[iy][ix][0], pt_y + undistort_lut[iy][ix][1]


def box_undistort(left, top, right, bottom, undistort_lut):
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    w = (right - left) / 2.0
    h = (bottom - top) / 2.0

    pt_top = (cx, cy - h)
    pt_bottom = (cx, cy + h)
    pt_left = (cx - w, cy)
    pt_right = (cx + w, cy)
    pt_top = ptr_undistort(pt_top[0], pt_top[1], undistort_lut)
    pt_bottom = ptr_undistort(pt_bottom[0], pt_bottom[1], undistort_lut)
    pt_left = ptr_undistort(pt_left[0], pt_left[1], undistort_lut)
    pt_right = ptr_undistort(pt_right[0], pt_right[1], undistort_lut)
    left = pt_left[0]
    top = pt_top[1]
    right = pt_right[0]
    bottom = pt_bottom[1]
    return [float(left), float(top), float(right), float(bottom)]


def img_undistort(
    image, camera_type, mat_distort, mat_camera, img_height, img_width
):
    if camera_type == 0:
        undisort_fun = cv2.initUndistortRectifyMap
    else:
        undisort_fun = cv2.fisheye.initUndistortRectifyMap
    mapx, mapy = undisort_fun(
        mat_camera,
        mat_distort,
        np.eye(3, 3),
        mat_camera,
        (img_width, img_height),
        cv2.CV_16SC2,
    )
    image_undistort = cv2.remap(
        image, mapx, mapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
    )
    return image_undistort


def get_undistort_lut(
    camera_type, camera_fov, mat_distort, mat_camera, img_height, img_width
):
    if camera_type == 0:
        f_get_lut = get_normal_lut
    else:
        f_get_lut = get_fisheye_lut
    fov2 = float(camera_fov) / 2
    undistort_lut = f_get_lut(
        mat_distort, mat_camera, fov2, img_height, img_width
    )
    return undistort_lut


def get_camera_info(camera_info):
    camera_type = camera_info.get("type", DEFAULT_CAMERA_TYPE)
    if camera_type == 0:
        camera_fov = float(camera_info.get("fov", DEFAULT_PINHOLE_FOV))
    else:
        camera_fov = float(camera_info.get("fov", DEFAULT_FISHEYE_FOV))
    # if pack not contain distort, then return raw result
    distort = (
        camera_info["distort"]["param"]
        if "distort" in camera_info
        else DEFAULT_DISTORT
    )
    mat_distort = np.array(distort)
    mat_camera = np.array(
        [
            [camera_info["focalU"], 0.0, camera_info["centerU"]],
            [0.0, camera_info["focalV"], camera_info["centerV"]],
            [0, 0, 1],
        ]
    )
    cam_hash = "%s_%s_%s_%s" % (
        str(camera_type),
        str(camera_fov),
        str(np.sum(mat_distort) * 1000),
        str(np.sum(mat_camera) * 1000),
    )
    img_height = camera_info.get("height", DEFAULT_HEIGHT)
    img_width = camera_info.get("width", DEFAULT_WIDTH)
    return (
        camera_type,
        camera_fov,
        mat_distort,
        mat_camera,
        cam_hash,
        img_height,
        img_width,
    )
