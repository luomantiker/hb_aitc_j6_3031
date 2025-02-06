# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import math
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera(object):
    # We assume that camera version is 1
    # in this case rx->roll, ry->pitch, rz->yaw
    # Note: for old coordinate camera, please select version as 0
    def __init__(self, rx, ry, rz, tx, ty, tz, fu, fv, cu, cv, version=1):
        self._rx = rx  # roll
        self._ry = ry  # pitch
        self._rz = rz  # yaw
        self._tx = tx
        self._ty = ty
        self._tz = tz
        self._fu = fu
        self._fv = fv
        self._cu = cu
        self._cv = cv
        self.version = version
        self.K = None
        self.R = None
        self.T = None
        self.local2img = None
        self.gnd2img = None
        self.img2gnd = None
        if version == 0:
            self.cvt_camera()
            self.version = 1
        self.set_k()
        self.set_r()
        self.set_t()
        self.update_camera()

    def set_k(self):
        self.K = [[self._fu, 0, self._cu], [0, self._fv, self._cv], [0, 0, 1]]
        self.K = np.array(self.K)

    def set_r(self):
        self.R = [
            [
                math.cos(self._ry) * math.cos(self._rz),
                math.cos(self._rz) * math.sin(self._rx) * math.sin(self._ry)
                + math.cos(self._rx) * math.sin(self._rz),
                math.sin(self._rx) * math.sin(self._rz)
                - math.cos(self._rx) * math.cos(self._rz) * math.sin(self._ry),
            ],
            [
                -math.cos(self._ry) * math.sin(self._rz),
                -math.sin(self._rx) * math.sin(self._ry) * math.sin(self._rz)
                + math.cos(self._rx) * math.cos(self._rz),
                math.cos(self._rx) * math.sin(self._ry) * math.sin(self._rz)
                + math.cos(self._rz) * math.sin(self._rx),
            ],
            [
                math.sin(self._ry),
                -math.cos(self._ry) * math.sin(self._rx),
                math.cos(self._rx) * math.cos(self._ry),
            ],
        ]
        self.R = np.array(self.R)

    def set_t(self):
        t = [[self._tx], [self._ty], [self._tz]]
        t = np.array(t)
        if self.R is not None:
            self.T = -np.dot(self.R, t)
        else:
            logger.info("setR first")

    def update_camera(self):
        if self.R is not None and self.T is not None:
            self.R = np.hstack((self.R, self.T))
        else:
            logger.info("R or T is None")
            return
        if self.R is not None and self.K is not None:
            axis = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
            axis = np.array(axis)
            tmp = np.dot(self.K, axis)
            self.local2img = np.dot(tmp, self.R)
            self.gnd2img = self.local2img[:, [0, 1, 3]]
            self.img2gnd = np.linalg.inv(self.gnd2img)
        else:
            logger.info("R or K is None")
            return

    def cvt_camera(self):
        rot = [
            [
                math.cos(self._ry) * math.cos(self._rz),
                math.cos(self._rz) * math.sin(self._rx) * math.sin(self._ry)
                + math.cos(self._rx) * math.sin(self._rz),
                math.sin(self._rx) * math.sin(self._rz)
                - math.cos(self._rx) * math.cos(self._rz) * math.sin(self._ry),
            ],
            [
                -math.cos(self._ry) * math.sin(self._rz),
                -math.sin(self._rx) * math.sin(self._ry) * math.sin(self._rz)
                + math.cos(self._rx) * math.cos(self._rz),
                math.cos(self._rx) * math.sin(self._ry) * math.sin(self._rz)
                + math.cos(self._rz) * math.sin(self._rx),
            ],
            [
                math.sin(self._ry),
                -math.cos(self._ry) * math.sin(self._rx),
                math.cos(self._rx) * math.cos(self._ry),
            ],
        ]
        rot = np.array(rot)
        R_axis = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        R_axis = np.array(R_axis)
        R_axis_inv = np.linalg.inv(R_axis)
        rot_new = np.dot(R_axis, rot)
        rot_new = np.dot(rot_new, R_axis_inv)
        tmp = (rot_new[1][0] ** 2 + rot_new[1][1] ** 2) ** 0.5
        self._rx = math.atan2(-rot_new[1][2], tmp)
        self._ry = math.atan2(rot_new[0][2], rot_new[2][2])
        self._rz = math.atan2(rot_new[1][0], rot_new[1][1])
        temp = self._tx
        self._tx = self._tz
        self._tz = -self._ty
        self._ty = -temp

    def pt_img2gnd(self, pt):
        pt_img = np.array([pt[0], pt[1], 1])
        pt_gnd = np.dot(self.img2gnd, pt_img)
        r = []
        r.append(pt_gnd[0] / pt_gnd[2])
        r.append(pt_gnd[1] / pt_gnd[2])
        return r

    # new gnd x:forward, y:left
    def get_bbox_distance(self, bbox):
        pt = ((bbox[0] + bbox[2]) / 2.0, bbox[3])
        r = self.pt_img2gnd(pt)
        distance_forward = r[0]
        distance_lateral = r[1]
        return distance_forward, distance_lateral

    def draw_camera_grid(
        self,
        image,
        forward_list: Optional[Union[List, Tuple]] = None,
        lateral_list: Optional[Union[List, Tuple]] = None,
        line_color: Optional[Union[List, Tuple]] = None,
        thickness=1,
        type_=cv2.LINE_AA,
    ):
        if forward_list is None:
            forward_list = range(20, 101, 10)
        if lateral_list is None:
            lateral_list = range(-8, 9, 4)
        if line_color is None:
            line_color = (0, 255, 0)

        def _pt_img(mat, pt):
            pt_img = np.dot(mat, np.array(pt))
            return (pt_img[0] / pt_img[2], pt_img[1] / pt_img[2])

        horizon_lines = [
            [
                _pt_img(self.gnd2img, [forward, lateral_list[0], 1]),
                _pt_img(self.gnd2img, [forward, lateral_list[-1], 1]),
                (forward, lateral_list[0]),
                (forward, lateral_list[-1]),
            ]
            for forward in forward_list
        ]
        vertical_lines = [
            [
                _pt_img(self.gnd2img, [forward_list[0], lateral, 1]),
                _pt_img(self.gnd2img, [forward_list[-1], lateral, 1]),
                (forward_list[0], lateral),
                (forward_list[-1], lateral),
            ]
            for lateral in lateral_list
        ]
        lines = horizon_lines + vertical_lines
        for line in lines:
            cv2.line(
                image,
                (int(line[0][0]), int(line[0][1])),
                (int(line[1][0]), int(line[1][1])),
                color=line_color,
                lineType=type_,
                thickness=thickness,
            )
            cv2.putText(
                image,
                "{}".format(line[2]),
                (int(line[0][0]), int(line[0][1])),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                color=line_color,
            )
            cv2.putText(
                image,
                "{}".format(line[3]),
                (int(line[1][0]), int(line[1][1])),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                color=line_color,
            )

        fp_x = int(self.gnd2img[0][0] / self.gnd2img[2][0])
        fp_y = int(self.gnd2img[1][0] / self.gnd2img[2][0])
        cv2.line(
            image,
            (fp_x - 10, fp_y),
            (fp_x + 10, fp_y),
            color=line_color,
            lineType=type_,
            thickness=thickness,
        )
        cv2.line(
            image,
            (fp_x, fp_y - 10),
            (fp_x, fp_y + 10),
            color=line_color,
            lineType=type_,
            thickness=thickness,
        )
        return image


def pt_img2gnd(p, mat):
    r = []
    gs = p[0] * mat[6] + p[1] * mat[7] + mat[8]
    r.append((p[0] * mat[0] + p[1] * mat[1] + mat[2]) / gs)
    r.append((p[0] * mat[3] + p[1] * mat[4] + mat[5]) / gs)
    return r
