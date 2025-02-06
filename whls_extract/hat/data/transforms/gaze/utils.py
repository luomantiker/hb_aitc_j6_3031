# Copyright (c) Horizon Robotics. All rights reserved.
import math
import random

import cv2
import numpy as np

__all__ = ["eye_ldmk_mirror"]


EPS = 10 ** -8

FLIP_EYE_ORD = [
    4,
    3,
    2,
    1,
    0,
    7,
    6,
    5,
    11,
    10,
    9,
    8,
    15,
    14,
    13,
    12,
    16,
    19,
    18,
    17,
    20,
]


def convert_channel_to_420sp(image):
    """Transfer image to align with YUV420SP.

    Image will be downsampled to YUV420 resolution (0.5 x 0.5)
    and upsampled to original size
    """
    img_h, img_w = image.shape[:2]
    downsampled_w = int(math.ceil(img_w / 2.0))
    downsmapled_h = int(math.ceil(img_h / 2.0))
    img = cv2.resize(image, (downsampled_w, downsmapled_h))
    img = np.repeat(img, 2, axis=0)
    img = np.repeat(img, 2, axis=1)
    return img[:img_h, :img_w]


def degree2rad(degree):
    return degree / 180.0 * np.pi


def rad2degree(rad):
    return rad * 180.0 / np.pi


def euler_to_unit_vector(pitch, yaw):
    pitch = degree2rad(pitch)
    yaw = degree2rad(yaw)
    x = np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = -np.cos(pitch) * np.cos(yaw)
    return [x, y, z]


def unit_vector_to_euler(x, y, z):
    unit_parm = np.sqrt(x * x + y * y + z * z)
    x = x / unit_parm
    y = y / unit_parm
    z = z / unit_parm
    pitch = np.arcsin(y)
    yaw = np.arcsin(x / np.sqrt(1 - y * y))
    return rad2degree(pitch), rad2degree(yaw)


def transform_ldmk(ldmks, mat_transpose):
    assert ldmks.shape[1] == 2
    assert len(ldmks.shape) == 2
    assert mat_transpose.shape == (3, 3)
    if np.abs(mat_transpose - np.eye(3)).max() > EPS:
        ldmks_h = np.concatenate((ldmks, np.ones_like(ldmks)[:, :1]), axis=1)
        # pid = os.getpid()
        # with cp.cuda.Device(pid % 4):
        #     ldmks_h = cp.asnumpy(cp.array(ldmks_h) @ cp.array(mat_transpose))
        # ldmks_h = ldmks_h @ mat_transpose
        ldmks_h = np.einsum("ij,jk->ik", ldmks_h, mat_transpose)
        return ldmks_h[:, :2] / ldmks_h[:, 2:]
    return ldmks


def calc_roi_with_ldmk(
    face_ldmks, cropping_ratio, crop_size, rand_crop_cropper_border, is_train
):
    x_min, y_min = np.min(face_ldmks, axis=0)
    x_max, y_max = np.max(face_ldmks, axis=0)
    left = x_min - 0.5 * cropping_ratio * (x_max - x_min)
    right = x_max + 0.5 * cropping_ratio * (x_max - x_min)
    expended_width = right - left
    expended_height = expended_width / crop_size[0] * crop_size[1]
    top = (y_max + y_min - expended_height) / 2
    bottom = (y_max + y_min + expended_height) / 2
    if is_train:
        left -= rand_crop_cropper_border
        top -= rand_crop_cropper_border
        right += rand_crop_cropper_border
        bottom += rand_crop_cropper_border
    return [int(_) for _ in [left, top, right, bottom]]


def warp_crop_image(
    image, eye_bbox_ori, eye_bbox_new, K_inv, R, K, rand_inter_type, is_train
):
    left, top, right, bottom = eye_bbox_new
    w, h = int(right - left), int(bottom - top)
    if is_train:
        range_x = np.linspace(left, right, w, endpoint=False, dtype=np.float32)
        range_y = np.linspace(top, bottom, h, endpoint=False, dtype=np.float32)
        map_ori = np.transpose(
            np.meshgrid(range_x, range_y), (1, 2, 0)
        ).reshape(-1, 2)
        map = transform_ldmk(map_ori, K_inv.T @ R @ K.T)
        map = map - np.array([eye_bbox_ori[0], eye_bbox_ori[1]]).reshape(-1, 2)
        map = map.reshape(h, w, 2).astype(np.float32)
        inter = random.randint(0, 2) if rand_inter_type else 1
        image_crop = cv2.remap(image, map, None, inter, cv2.BORDER_REFLECT)
    else:
        x1, y1 = [int(_) for _ in eye_bbox_ori[:2]]
        assert left - x1 >= 0 and top - y1 >= 0
        image_crop = image[top - y1 : bottom - y1, left - x1 : right - x1]
    h, w = np.array(image_crop.shape[:2]) // 2 * 2
    return image_crop[:h, :w]


def generate_pos_map(eye_bbox_new, shape, K_inv, std_intrin, to_yuv420sp):
    left, top, right, bottom = eye_bbox_new
    h, w = shape[:2]
    range_x = np.linspace(left, right, w, endpoint=False)
    range_y = np.linspace(top, bottom, h, endpoint=False)
    pos_map_ori = np.transpose(
        np.meshgrid(range_x, range_y), (1, 2, 0)
    ).reshape(-1, 2)
    pos_map = transform_ldmk(pos_map_ori, K_inv.T @ std_intrin.T)
    pos_map *= np.array([255 / 1280, 255 / 960])
    pos_map = pos_map.reshape(h, w, 2).astype(np.uint8)
    horizon_pos_map, vertical_pos_map = pos_map[..., 0], pos_map[..., 1]
    if to_yuv420sp:
        horizon_pos_map = convert_channel_to_420sp(horizon_pos_map)
        vertical_pos_map = convert_channel_to_420sp(vertical_pos_map)
    return horizon_pos_map, vertical_pos_map


def rotate_gaze(gaze, R):
    left_vec = euler_to_unit_vector(*gaze[:2])
    right_vec = euler_to_unit_vector(*gaze[2:])
    left_vec = R @ np.array(left_vec)
    right_vec = R @ np.array(right_vec)
    left_pitch, left_yaw = unit_vector_to_euler(*left_vec)
    right_pitch, right_yaw = unit_vector_to_euler(*right_vec)
    return np.array([left_pitch, left_yaw, right_pitch, right_yaw])


def eyeldmk_transform(eye_ldmks, eye_bbox, resize_shape=None):
    """Convert raw eye landmarks from camera frame to ratio in input region.

    Output is ratio, value in [0, 1].

    Args:
        eye_ldmks: Raw eye landmarks coordinates in camera frame
            (-1000 for invalid eye landmarks).
        eye_bbox: Original eye bounding box.
        resize_shape: Resize shape.

    Returns:
        eye_ldmks: Eye_ldmks ratio of resize_shape
    """
    bbox_w = eye_bbox[2] - eye_bbox[0]
    bbox_h = eye_bbox[3] - eye_bbox[1]

    eye_bbox_upperleft = np.array([[eye_bbox[0], eye_bbox[1]]])
    eye_bbox_shape = np.array([[bbox_w, bbox_h]])

    # eye_ldmks ratio
    eye_ldmks = (eye_ldmks - eye_bbox_upperleft) / eye_bbox_shape

    # filter invalid points
    mask = (1 - (eye_ldmks > 0) * (eye_ldmks < 1)).astype("bool")
    eye_ldmks[mask] = -1000
    return eye_ldmks.astype(np.float32)


def fixed_crop(src, x0, y0, w, h):
    """Crop src at fixed location, and (optionally) resize it to size.

    Args:
        src: Input image
        x0: Left boundary of the cropping area
        y0: Top boundary of the cropping area
        w: Width of the cropping area
        h: Height of the cropping area
    """
    out = src[y0 : y0 + h, x0 : x0 + w, :]
    return out


def eye_ldmk_mirror(eye_ldmk, normd=True):
    """Flip eye landmarks.

    Eye landmarks(21 points) here are already computed as ratio within
    final input image.
    """

    T_flip = np.array([[-1, 0], [0, 1], [1, 0]])
    if not normd:
        return eye_ldmk[[_ + 21 for _ in FLIP_EYE_ORD] + FLIP_EYE_ORD]
    single_eye_ldmk_num = len(eye_ldmk) // 2
    left_eye_ldmk = eye_ldmk[:single_eye_ldmk_num]
    right_eye_ldmk = eye_ldmk[single_eye_ldmk_num : 2 * single_eye_ldmk_num]
    flip_eye_ord = FLIP_EYE_ORD[:single_eye_ldmk_num]
    flip_left_eye_ldmk = left_eye_ldmk[flip_eye_ord]
    flip_right_eye_ldmk = right_eye_ldmk[flip_eye_ord]
    flip_eye_ldmks = np.concatenate(
        (flip_right_eye_ldmk, flip_left_eye_ldmk), axis=0
    )
    flip_eye_ldmks_mask = flip_eye_ldmks > -1000
    flip_eye_ldmks_tmp = np.c_[
        flip_eye_ldmks, np.ones(flip_eye_ldmks.shape[0])
    ]
    flip_eye_ldmks = (
        flip_eye_ldmks_tmp @ T_flip * flip_eye_ldmks_mask
        - 1000 * (1 - flip_eye_ldmks_mask).astype("bool")
    )
    return flip_eye_ldmks
