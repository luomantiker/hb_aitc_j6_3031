from collections import defaultdict
from typing import List, Tuple, Union

import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline

from .undistort_lut import get_undistort_points

global_eq_fu_map = defaultdict()
global_eq_fv_map = defaultdict()


def cal_equivalent_focal_length(
    uv_points: np.ndarray,
    calib: Union[np.ndarray, List],
    dist: Union[np.ndarray, List],
    img_wh: Tuple[int, int],
    undistort_by_cv: bool = False,
):
    """Calculate equivalent focal length.

    Args:
        uv_points: Distorted uv points, in shape (num_points, 1, 2)
        calib: Calibration mat
        dist: Distort mat
        img_wh: Image width and height
        undistort_by_cv: Whether use opencv to undistort
    """
    if isinstance(calib, list):
        calib = np.array(calib, dtype=np.float32)
    if calib.shape[0] > 3:
        calib = calib[:3]
    if calib.shape[1] > 3:
        calib = calib[:, :3]
    if isinstance(dist, list):
        dist = np.array(dist, dtype=np.float32)

    fu = calib[0, 0]
    fv = calib[1, 1]
    f = (fu + fv) / 2
    cu = calib[0, 2]
    cv = calib[1, 2]
    dist_u = uv_points[:, 0, 0]
    dist_v = uv_points[:, 0, 1]

    if undistort_by_cv:
        undist_uv = cv2.undistortPointsIter(
            src=uv_points,
            cameraMatrix=calib,
            distCoeffs=dist,
            R=None,
            P=calib,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER, 4, 0.1),
        )
        undist_u = undist_uv[:, 0, 0]
        undist_v = undist_uv[:, 0, 1]
    else:
        undist_uv = get_undistort_points(
            points=uv_points[:, 0, :],
            calib=calib,
            distCoeffs=dist,
            img_wh=img_wh,
        )
        undist_u = undist_uv[:, 0]
        undist_v = undist_uv[:, 1]
    x = (undist_u - cu) / fu
    y = (undist_v - cv) / fv
    e_x = (dist_u - cu) / fu
    e_y = (dist_v - cv) / fv

    r = np.sqrt(x * x + y * y)
    e_r = np.sqrt(e_x * e_x + e_y * e_y)

    e_f = e_r.clip(min=1e-4) * f / r.clip(min=1e-4)

    # image center points is not stable
    e_f[e_r < 1e-3] = f
    e_f = e_f.clip(min=f * 0.1, max=f)

    return e_f


def cal_equivalent_focal_length_uv_mat(
    width,
    height,
    calib,
    dist,
    downsample=4,
    undistort_by_cv=False,
):
    if isinstance(dist, list):
        dist = np.array(dist, dtype=np.float32)
    hash_distcoeffs = dist.tobytes()
    if hash_distcoeffs in global_eq_fu_map.keys():
        eq_fu = global_eq_fu_map[hash_distcoeffs]
        eq_fv = global_eq_fv_map[hash_distcoeffs]
    else:
        ori_width = width
        ori_height = height
        width //= downsample
        height //= downsample

        u_pos = np.arange(0, ori_width, downsample, dtype=np.float32)
        u_pos = np.tile(u_pos, (height, 1))
        v_pos = np.arange(0, ori_height, downsample, dtype=np.float32)
        v_pos = np.tile(v_pos, (width, 1)).transpose()

        uv_points = np.stack([u_pos, v_pos]).reshape((2, -1)).transpose()
        uv_points = uv_points.reshape(-1, 1, 2)
        e_f = cal_equivalent_focal_length(
            uv_points, calib, dist, (ori_width, ori_height), undistort_by_cv
        )
        e_f_mat = e_f.reshape(height, width)

        cu = calib[0, 2]
        cv = calib[1, 2]
        dfu = np.zeros_like(e_f_mat)
        dfv = np.zeros_like(e_f_mat)
        offset_u = u_pos - cu
        offset_v = v_pos - cv

        dfu[:, 1:-1] = (e_f_mat[:, 2:] - e_f_mat[:, :-2]) / (2 * downsample)
        dfu[:, 0] = (e_f_mat[:, 1] - e_f_mat[:, 0]) / downsample
        dfu[:, -1] = (e_f_mat[:, -1] - e_f_mat[:, -2]) / downsample
        eq_fu = e_f_mat / (1 - dfu * offset_u / e_f_mat).clip(min=1)
        eq_fu.clip(min=1)

        dfv[1:-1] = (e_f_mat[2:] - e_f_mat[:-2]) / (2 * downsample)
        dfv[0] = (e_f_mat[1] - e_f_mat[0]) / downsample
        dfv[-1] = (e_f_mat[-1] - e_f_mat[-2]) / downsample
        eq_fv = e_f_mat / (1 - dfv * offset_v / e_f_mat).clip(min=1)
        eq_fv.clip(min=1)

        interp_eq_fu = RectBivariateSpline(
            np.arange(0, ori_width, downsample),
            np.arange(0, ori_height, downsample),
            eq_fu.T,
        )

        interp_eq_fv = RectBivariateSpline(
            np.arange(0, ori_width, downsample),
            np.arange(0, ori_height, downsample),
            eq_fv.T,
        )
        eq_fu = interp_eq_fu(np.arange(ori_width), np.arange(ori_height)).T
        eq_fv = interp_eq_fv(np.arange(ori_width), np.arange(ori_height)).T

        global_eq_fu_map[hash_distcoeffs] = eq_fu
        global_eq_fv_map[hash_distcoeffs] = eq_fv

    return eq_fu, eq_fv
