# Copyright (c) Horizon Robotics. All rights reserved.

import json
from logging import warning
from typing import Tuple, Union

import numpy as np

from .camera_base import CameraBase
from .projection import (
    CylindricalProjection,
    FisheyeProjection,
    PinholeProjection,
    SphericalProjection,
)


class CylindricalCamera(CameraBase):
    """Cylindrical camera method."""

    def __init__(
        self,
        image_size=None,
        camera_matrix=None,
        distcoeffs=None,
        poseMat_vcs2cam=None,
        poseMat_lidar2cam=None,
        poseMat_lidar2vcs=None,
        is_virtual=False,
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            camera_matrix=camera_matrix,
            distcoeffs=distcoeffs,
            poseMat_vcs2cam=poseMat_vcs2cam,
            poseMat_lidar2cam=poseMat_lidar2cam,
            poseMat_lidar2vcs=poseMat_lidar2vcs,
            is_virtual=is_virtual,
            lens=CylindricalProjection(),
            **kwargs,
        )

    def calculate_fov(self):
        self.vfov_up = (
            np.arctan2(np.array(self.principle_point[1]), np.array(self.fy))
            * 180
            / np.pi
        )
        self.vfov_down = (
            np.arctan2(
                np.array(self.image_size[1])
                - np.array(self.principle_point[1]),
                np.array(self.fy),
            )
            * 180
            / np.pi
        )
        vfov = self.vfov_up + self.vfov_down  # noqa F841
        hfov = (self.principle_point[0] / self.fx) * 180 / np.pi * 2

        return hfov, vfov

    def calculate_blind_spot(self, calib_path):
        with open(calib_path) as f:
            config = json.load(f)
            camera_height = config["camera_z"]
        ground_bsd = (
            np.array(self.fy)
            / (
                np.array(self.image_size[1])
                - np.array(self.principle_point[1])
            )
            * camera_height
        )
        visionable_height = (
            1 * (np.tan(self.vfov_up * np.pi / 180)) + camera_height
        )

        return ground_bsd, visionable_height


class SphericalCamera(CameraBase):
    """Spherical camera method."""

    def __init__(
        self,
        image_size=None,
        camera_matrix=None,
        distcoeffs=None,
        poseMat_vcs2cam=None,
        poseMat_lidar2cam=None,
        poseMat_lidar2vcs=None,
        is_virtual=False,
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            camera_matrix=camera_matrix,
            distcoeffs=distcoeffs,
            poseMat_vcs2cam=poseMat_vcs2cam,
            poseMat_lidar2cam=poseMat_lidar2cam,
            poseMat_lidar2vcs=poseMat_lidar2vcs,
            is_virtual=is_virtual,
            lens=SphericalProjection(),
            **kwargs,
        )


class FisheyeCamera(CameraBase):
    """Fisheye camera method."""

    def __init__(
        self,
        image_size=None,
        camera_matrix=None,
        distcoeffs=None,
        poseMat_vcs2cam=None,
        poseMat_lidar2cam=None,
        poseMat_lidar2vcs=None,
        is_virtual=False,
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            camera_matrix=camera_matrix,
            distcoeffs=distcoeffs,
            poseMat_vcs2cam=poseMat_vcs2cam,
            poseMat_lidar2cam=poseMat_lidar2cam,
            poseMat_lidar2vcs=poseMat_lidar2vcs,
            is_virtual=is_virtual,
            lens=FisheyeProjection(),
            **kwargs,
        )

    def set_lens_param(
        self,
    ):
        self.lens.distcoeffs = self.distcoeffs


class PinholeCamera(CameraBase):
    """Pinhole camera method."""

    def __init__(
        self,
        image_size=None,
        camera_matrix=None,
        distcoeffs=None,
        poseMat_vcs2cam=None,
        poseMat_lidar2cam=None,
        poseMat_lidar2vcs=None,
        is_virtual=False,
        num_fov_points=8,
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            camera_matrix=camera_matrix,
            distcoeffs=distcoeffs,
            poseMat_vcs2cam=poseMat_vcs2cam,
            poseMat_lidar2cam=poseMat_lidar2cam,
            poseMat_lidar2vcs=poseMat_lidar2vcs,
            is_virtual=is_virtual,
            lens=PinholeProjection(),
            num_fov_points=num_fov_points,
            **kwargs,
        )
        self.api_mode = "opencv"

    def set_lens_param(
        self,
    ):
        self.lens.camera_matrix = self.camera_matrix
        self.lens.distcoeffs = self.distcoeffs


class IPMCamera(PinholeCamera):
    """
    IPMCamera method.

    Args:
        image_size: in order W*H, can dynamic changed.
        vcs_range: vcs coord visual range(m).
            in order:(bottom, right, top, left).
    Notes:
        resolution: BEV imgae physics resolution(m/px).
            (image coord x axis-width, image coord y axis-height).
        right-handed coordinate system:
        coord defined as:
                  x               z                   y
                  ^               ^     extr matrix   ^
                  |   vcs         | cam ----------->  |  IPMCamera
                  |               |                   |
         y <------. z           y .------> x x <------. z

    """

    def __init__(
        self,
        image_size: Union[Tuple[int], np.ndarray],
        vcs_range: Union[Tuple[float], np.ndarray],
    ):
        self._vcs_range = vcs_range
        self._resolution = np.array(
            [
                (abs(vcs_range[3] - vcs_range[1]) / image_size[0]),
                (abs(vcs_range[2] - vcs_range[0]) / image_size[1]),
            ]
        )
        principle_point = (
            np.array([vcs_range[3], vcs_range[2]]) / self.resolution
        )
        super().__init__(
            image_size=image_size,
            camera_matrix=np.array(
                [
                    [1 / self._resolution[0], 0, principle_point[0]],
                    [0, 1 / self._resolution[1], principle_point[1]],
                    [0, 0, 1],
                ]
            ),
            distcoeffs=np.zeros(8),
            poseMat_vcs2cam=np.array(
                [
                    [0, -1, 0, 0],
                    [-1, 0, 0, 0],
                    [0, 0, -1, 1],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            ),
        )
        self._is_valid_warning = True
        self._valid_check()

    @property
    def image_size(self):
        return self._image_size

    @image_size.setter
    def image_size(self, image_size):
        self._image_size = image_size
        self._update_intrinsic()
        self._valid_check()

    @property
    def resolution(self):
        if hasattr(self, "_fx") and hasattr(self, "_fy"):
            self._resolution[0] = 1 / self.fx
            self._resolution[1] = 1 / self.fy
        return self._resolution

    @property
    def vcs_range(self):
        return self._vcs_range

    @vcs_range.setter
    def vcs_range(self, vcs_range):
        self._vcs_range = vcs_range
        self._update_intrinsic()
        self._valid_check()

    def _valid_check(self):
        if abs(self._resolution[0] - self._resolution[1]) >= 1e-5:
            if self._is_valid_warning:
                warning("x axis resolution not equal y axis resolution!!!")
                self._is_valid_warning = False
            else:
                pass

    def _update_intrinsic(self):
        self._resolution = np.array(
            [
                (
                    abs(self.vcs_range[3] - self.vcs_range[1])
                    / self.image_size[0]
                ),
                (
                    abs(self.vcs_range[2] - self.vcs_range[0])
                    / self.image_size[1]
                ),
            ]
        )
        self.fx = 1 / self._resolution[0]
        self.fy = 1 / self._resolution[1]

    @staticmethod
    def get_poseMat_vcsgnd2pixel(cam):
        T_vcs2pixel = np.dot(cam.camera_matrix, cam.poseMat_vcs2cam[:3, :])
        return T_vcs2pixel[:, (0, 1, 3)]

    def get_homoMat_ipm2img(self, dst_cam):
        """Get homoMat of ipm plane to image plane."""
        T_vcsgnd2pixel = self.get_poseMat_vcsgnd2pixel(dst_cam)
        T_ipm2vcsgnd = self.get_poseMat_vcsgnd2pixel(self)
        homoMat_ipm2img = np.dot(T_vcsgnd2pixel, T_ipm2vcsgnd)
        return homoMat_ipm2img

    def get_homoMat_img2ipm(self, dst_cam):
        """Get homoMat of image plane to ipm plane."""
        homoMat_ipm2img = self.get_homoMat_ipm2img(dst_cam)
        homoMat_img2ipm = np.linalg.inv(homoMat_ipm2img)
        return homoMat_img2ipm
