# Copyright (c) Horizon Robotics. All rights reserved.

import json
import os
import uuid
from copy import deepcopy

try:
    from types import NoneType
except ImportError:
    NoneType = type(None)

from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .projection import Projection
from .utils import (
    ensure_point_list,
    parse_extrinsicParam,
    transform_euler2rotMat,
)

EPS = 1e-8


class SetCameraParam(object):
    """Camera parameters member setter center.

    Extrinsic matrix naming Conventions:
        rotMat:    rotation matrix, shape: (3, 3)
        transMat:  translation matrix, shape: (3,)
        poseMat:   Trans+rot matrix, shape: (4, 4)
    Args:
        image_size: image size for projection plane. in order W*H. \
            Default None.
        camera_matrix: intrinsic param, format:
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0,  1 ]
            ]
            fx (float): Focal length for x axis, must be setted. \
                Default: None.
            fy (float): Focal length for y axis, must be setted. \
                Default: None.
        principle_point: Light axis in image plane, \
                also called light center.
        distcoeffs: Distort coeffs of lens, Pinhole 8 params and
            Fisheye 4 params on Horizon. Default: None.
        ExtrinsicParam:
        poseMat_lidar2vcs : pose Matrix for lidar coordinate to
            vcs coordinate, left dot. Default: np.identity(4)
        poseMat_vcs2cam : pose Matrix for world/vcs coordinate to
            cammera coordinate, left dot. Default: np.identity(4)
        poseMat_lidar2cam : pose Matrix for lidar coordinate to
            cammera coordinate, left dot. Default: np.identity(4)
        is_virtual : is virual camera flag, only when camera init,
            process camera param, else is just a flag. Default: False
        lens: lens type. Default: Projection()
        param_dict: Horizon Camera Calib result content in dict.

    coord system:
        cam: Opencv camera coord system.
        adascam: Horizon ADAS System privated Camera coord system.
        local: project adascam coord to ground plane, rectify roll/pitch.
        vcs: center of chassis projected to ground plane.
        coord system:
                  x(front)                         z(front)
                  ^                                ^
                  |   vcs/adascam/local/lidar      | cam(opencv)
                  |                                |
         y <------. z                            y .------> x

    """

    def __init__(
        self,
        image_size: Union[List[int], np.ndarray, NoneType] = None,
        camera_matrix: np.ndarray = None,
        distcoeffs: Union[List, np.ndarray] = None,
        poseMat_vcs2cam: Union[List, np.ndarray] = None,
        poseMat_lidar2cam: Optional[Union[List, np.ndarray]] = None,
        poseMat_lidar2vcs: Optional[Union[List, np.ndarray]] = None,
        is_virtual: bool = False,
        lens: Projection = None,
        param_dict: Optional[Union[Dict, NoneType]] = None,
        num_fov_points: int = 4,
    ):
        self._image_size = np.array(image_size)
        self._height = None
        self._width = None
        self._camera_matrix = camera_matrix
        self._principle_point = None
        self._fx = None
        self._fy = None
        self._distcoeffs = distcoeffs
        self.identity_4 = np.identity(4)
        self.identity_3 = np.identity(3)
        self.zeros_3 = np.zeros(3)
        self.affine_matrix = self.identity_3
        self.num_fov_points = num_fov_points
        self._rotMat_vcs2cam = self.identity_3
        self._transMat_vcs2cam = self.zeros_3
        self._poseMat_vcs2cam = poseMat_vcs2cam
        self._rotMat_lidar2cam = self.identity_3
        self._transMat_lidar2cam = self.zeros_3
        self._poseMat_lidar2cam = poseMat_lidar2cam
        self._rotMat_lidar2vcs = self.identity_3
        self._transMat_lidar2vcs = self.zeros_3
        self._poseMat_lidar2vcs = poseMat_lidar2vcs

        # horizon adas private coord system.
        self.poseMat_cam2adascam = np.array(
            [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        self._poseMat_local2vcs = self.identity_4
        self._poseMat_vcs2local = self.identity_4
        self._poseMat_adascam2local = self.identity_4
        self._poseMat_vcs2adascam = self.identity_4

        self._dist_mode = "default"
        self._is_virtual = is_virtual
        self._lens = lens
        # CameraClass init inner api: horizon or "opencv"
        self.api_mode = "horizon"

        # calib json dict
        self.param_dict = param_dict
        # reused pose matrix
        self.poseMat = None
        self.reset_lens_param_flag = False
        self.set_lens_param()
        self._uuid = None
        self._image_grid = None

    @property
    def lens(self):
        if self._lens is None:
            self._lens = Projection()
        return self._lens

    @lens.setter
    def lens(self, lens):
        self._lens = lens

    @property
    def uuid(self):
        distcoeffs = str(self.distcoeffs)
        poseMat_vcs2cam = str(self.poseMat_vcs2cam)
        camera_matrix = str(self.camera_matrix)
        camera_param_str = distcoeffs + poseMat_vcs2cam + camera_matrix
        self._uuid = uuid.uuid3(uuid.NAMESPACE_DNS, camera_param_str)
        return self._uuid

    @property
    def is_virtual(self):
        return self._is_virtual

    @is_virtual.setter
    def is_virtual(self, is_virtual):
        self._is_virtual = is_virtual
        return self._is_virtual

    @property
    def image_size(self):
        return np.array([int(size) for size in self._image_size])

    @image_size.setter
    def image_size(self, image_size):
        self._image_size = image_size

    @property
    def height(self):
        self._height = self._image_size[1]
        return int(self._height)

    @height.setter
    def height(self, height):
        self._height = height
        self._image_size[1] = height

    @property
    def width(self):
        self._width = self._image_size[0]
        return int(self._width)

    @width.setter
    def width(self, width):
        self._width = width
        self._image_size[0] = width

    @property
    def camera_matrix(self):
        if self._camera_matrix is None:
            self._camera_matrix = self.identity_3
        if not isinstance(self._camera_matrix, np.ndarray):
            self._camera_matrix = np.array(self._camera_matrix)
        return self._camera_matrix

    @camera_matrix.setter
    def camera_matrix(self, camera_matrix):
        assert np.shape(camera_matrix) == (
            3,
            3,
        ), "camera_matrix shape must be (3, 3)"
        self._camera_matrix = camera_matrix
        self._principle_point = camera_matrix[:2, 2]
        self._fx = camera_matrix[0, 0]
        self._fy = camera_matrix[1, 1]
        # must update lens camera_matrix
        self.lens.camera_matrix = camera_matrix

    @property
    def principle_point(self):
        if self._principle_point is None:
            self._principle_point = self.camera_matrix[:2, 2]
        if not isinstance(self._principle_point, np.ndarray):
            self._principle_point = np.array(self._principle_point)
        return self._principle_point

    @principle_point.setter
    def principle_point(self, principle_point):
        self._principle_point = np.array(principle_point)
        self._camera_matrix[:2, 2] = self._principle_point

    @property
    def cx(self):
        return self.principle_point[0]

    @cx.setter
    def cx(self, cx):
        self.principle_point[0] = cx

    @property
    def cy(self):
        return self.principle_point[1]

    @cy.setter
    def cy(self, cy):
        self.principle_point[1] = cy

    @property
    def fx(self):
        if self._fx is None:
            self._fx = self.camera_matrix[0, 0]
        return self._fx

    @fx.setter
    def fx(self, fx):
        self._camera_matrix[0, 0] = fx
        self._fx = fx

    @property
    def fy(self):
        if self._fy is None:
            self._fy = self.camera_matrix[1, 1]
        return self._fy

    @fy.setter
    def fy(self, fy):
        self._camera_matrix[1, 1] = fy
        self._fy = fy

    @property
    def distcoeffs(self):
        if self._distcoeffs is None:
            self._distcoeffs = np.zeros(4)
        if not isinstance(self._distcoeffs, np.ndarray):
            self._distcoeffs = np.array(self._distcoeffs)
        return self._distcoeffs

    @distcoeffs.setter
    def distcoeffs(self, distcoeffs):
        self._distcoeffs = distcoeffs
        # must update lens distcoeffs
        self.lens.distcoeffs = distcoeffs

    @property
    def rotMat_vcs2cam(self):
        if self.poseMat_vcs2cam is not None:
            self._rotMat_vcs2cam = self.poseMat_vcs2cam[0:3, 0:3]
        return self._rotMat_vcs2cam

    @rotMat_vcs2cam.setter
    def rotMat_vcs2cam(self, rotMat_vcs2cam):
        self._rotMat_vcs2cam = rotMat_vcs2cam
        self._poseMat_vcs2cam = self.update_poseMat(
            self._rotMat_vcs2cam, self._transMat_vcs2cam
        )

    @property
    def transMat_vcs2cam(self):
        if self.poseMat_vcs2cam is not None:
            self._transMat_vcs2cam = self.poseMat_vcs2cam[0:3, 3]
        return self._transMat_vcs2cam

    @transMat_vcs2cam.setter
    def transMat_vcs2cam(self, transMat_vcs2cam):
        self._transMat_vcs2cam = transMat_vcs2cam
        self._poseMat_vcs2cam = self.update_poseMat(
            self._rotMat_vcs2cam, self._transMat_vcs2cam
        )

    @property
    def poseMat_vcs2cam(self):
        if self._poseMat_vcs2cam is None:
            self._poseMat_vcs2cam = self.identity_4
        return self._poseMat_vcs2cam

    @poseMat_vcs2cam.setter
    def poseMat_vcs2cam(self, poseMat_vcs2cam):
        assert poseMat_vcs2cam.shape == (
            4,
            4,
        ), "poseMat shape should be (4, 4)\n"
        self._poseMat_vcs2cam = poseMat_vcs2cam
        self._rotMat_vcs2cam = poseMat_vcs2cam[0:3, 0:3]
        self._transMat_vcs2cam = poseMat_vcs2cam[0:3, 3]

    @property
    def rotMat_lidar2cam(self):
        return self._rotMat_lidar2cam

    @rotMat_lidar2cam.setter
    def rotMat_lidar2cam(self, rotMat_lidar2cam):
        self._rotMat_lidar2cam = rotMat_lidar2cam
        self._poseMat_lidar2cam = self.update_poseMat(
            self._rotMat_lidar2cam, self._transMat_lidar2cam
        )

    @property
    def transMat_lidar2cam(self):
        return self._transMat_lidar2cam

    @transMat_lidar2cam.setter
    def transMat_lidar2cam(self, transMat_lidar2cam):
        self._transMat_lidar2cam = transMat_lidar2cam
        self._poseMat_lidar2cam = self.update_poseMat(
            self._rotMat_lidar2cam, self._transMat_lidar2cam
        )

    @property
    def poseMat_lidar2cam(self):
        if self._poseMat_lidar2cam is None:
            self._poseMat_lidar2cam = self.identity_4
        return self._poseMat_lidar2cam

    @poseMat_lidar2cam.setter
    def poseMat_lidar2cam(self, poseMat_lidar2cam):
        assert poseMat_lidar2cam.shape == (
            4,
            4,
        ), "poseMat shape should be (4, 4)\n"
        self._poseMat_lidar2cam = poseMat_lidar2cam
        self._rotMat_lidar2cam = poseMat_lidar2cam[0:3, 0:3]
        self._transMat_lidar2cam = poseMat_lidar2cam[0:3, 3]

    @property
    def transMat_lidar2vcs(self):
        return self._transMat_lidar2vcs

    @transMat_lidar2vcs.setter
    def transMat_lidar2vcs(self, transMat_lidar2vcs):
        self._transMat_lidar2vcs = transMat_lidar2vcs
        self._poseMat_lidar2vcs = self.update_poseMat(
            self._rotMat_lidar2vcs, self._transMat_lidar2vcs
        )

    @property
    def rotMat_lidar2vcs(self):
        return self._rotMat_lidar2vcs

    @rotMat_lidar2vcs.setter
    def rotMat_lidar2vcs(self, rotMat_lidar2vcs):
        self._rotMat_lidar2vcs = rotMat_lidar2vcs
        self._poseMat_lidar2vcs = self.update_poseMat(
            self._rotMat_lidar2vcs, self._transMat_lidar2vcs
        )

    @property
    def poseMat_lidar2vcs(self):
        if self._poseMat_lidar2vcs is None:
            self._poseMat_lidar2vcs = self.identity_4
        return self._poseMat_lidar2vcs

    @poseMat_lidar2vcs.setter
    def poseMat_lidar2vcs(self, poseMat_lidar2vcs):
        assert poseMat_lidar2vcs.shape == (
            4,
            4,
        ), "poseMat shape should be (4, 4)\n"
        self._poseMat_lidar2vcs = poseMat_lidar2vcs
        self._rotMat_lidar2vcs = poseMat_lidar2vcs[0:3, 0:3]
        self._transMat_lidar2vcs = poseMat_lidar2vcs[0:3, 3]

    @property
    def poseMat_vcs2adascam(self):
        self._poseMat_vcs2adascam = np.dot(
            self.poseMat_cam2adascam, self._poseMat_vcs2cam
        )
        return self._poseMat_vcs2adascam

    @property
    def poseMat_adascam2vcs(self):
        return np.linalg.inv(self.poseMat_vcs2adascam)

    @property
    def poseMat_local2vcs(self):
        if self.param_dict is None:
            rotMat_adascam2vcs = Rotation.from_matrix(
                self.poseMat_adascam2vcs[:3, :3]
            )
            # absolute angle for local coord z axis
            # has a eps value < 1 deg
            ypr_adascam2vcs = rotMat_adascam2vcs.as_euler("zyx")
            translation_local2vcs = self.poseMat_adascam2vcs[:3, 3]
            # local coord in the same xy plane as vcs coord system.
            translation_local2vcs[2] = 0
            # local yaw estimate camera direction
            rpy_local2vcs = np.array([0, 0, ypr_adascam2vcs[0]])
        else:
            rpy_local2vcs = self.param_dict["vcs"]["rotation"]
            translation_local2vcs = self.param_dict["vcs"]["translation"]

        # local to vcs coord
        # vcs_pt = poseMat_local2vcs * local_pt
        rotMat_local2vcs = transform_euler2rotMat(rpy_local2vcs)
        self._poseMat_local2vcs[:3, :3] = rotMat_local2vcs
        self._poseMat_local2vcs[0:3, 3] = np.array(translation_local2vcs).T
        return self._poseMat_local2vcs

    @property
    def poseMat_vcs2local(self):
        self._poseMat_vcs2local = np.linalg.inv(self.poseMat_local2vcs)
        return self._poseMat_vcs2local

    @property
    def poseMat_adascam2local(self):
        self._poseMat_adascam2local = np.dot(
            self._poseMat_vcs2local, self.poseMat_adascam2vcs
        )
        return self._poseMat_adascam2local

    @property
    def dist_mode(self):
        return self._dist_mode

    @dist_mode.setter
    def dist_mode(self, dist_mode):
        self._dist_mode = dist_mode

    @staticmethod
    def update_poseMat(rotMat, transMat):
        poseMat = np.eye(4)
        poseMat[0:3, 3] = transMat.reshape(-1)
        poseMat[0:3, 0:3] = rotMat
        poseMat = np.asarray(poseMat, dtype=float)
        return poseMat

    def set_lens_param(self, **kwargs):
        """If need to set lens_param."""
        pass

    def resize(
        self,
        image_size: Optional[Sequence[int]] = None,
        scale_x: Optional[float] = 1.0,
        scale_y: Optional[float] = 1.0,
    ):
        """Resize image plane of camera.

        Args:
            image_size: zoom to new image size. Defaults to None.
            scale_x: width scale factor. Defaults to 1.0.
            scale_y: height scale factor. Defaults to 1.0.
        """

        if image_size is not None:
            scale_x = image_size[0] / self.width
            scale_y = image_size[1] / self.height
            self._image_size = np.array(image_size)
        else:
            self.width = self.width * scale_x
            self.height = self.height * scale_y

        self.camera_matrix = self.camera_matrix * np.array(
            [
                [scale_x, 1, scale_x],
                [0, scale_y, scale_y],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        return self

    def crop(
        self,
        image_size: Sequence[int] = None,
        crop_left: Optional[int] = 0,
        crop_top: Optional[int] = 0,
    ):
        """Crop image plane of camera.

        Args:
            image_size: crop to new image size. Defaults to None.
            left: left crop point. Defaults to 0.0.
            top: top crop pont. Defaults to 0.0.
        """
        assert image_size is not None, "Crop method need pre-cal image size!"
        self.image_size = image_size
        self.principle_point -= np.array([crop_left, crop_top])
        return self

    def pad(
        self,
        pad: Optional[Sequence[int]] = (0, 0, 0, 0),
    ):
        """Pad image plane of camera.

        Args:
            pad: pad params for image plane.
                in order (pad_left, pad_right, pad_top, pad_bottom)
        """
        self.image_size += np.array([sum(pad[:2]), sum(pad[2:])])
        self.principle_point += np.array([pad[0], pad[2]])
        return self

    def hflip(
        self,
    ):
        """Horizontial flip camera by intrinsic param."""
        self.fx = -self.fx
        self.cx = self.width - self.cx
        return self

    def vflip(
        self,
    ):
        """Vertial flip camera by intrinsic param."""
        self.fy = -self.fy
        self.cy = self.height - self.cy
        return self

    def set_cam_param(
        self,
        image_size,
        principle_point,
        fx,
        fy,
    ):
        """Visualize simple set api."""

        self.image_size = image_size  # [w,h]
        self.principle_point = principle_point  # cx,cy
        self.fx = fx
        self.fy = fy

    @property
    def image_grid(
        self,
    ):
        if self._image_grid is not None:
            reset = self._image_grid.shape[:-1] != (self.width, self.height)
        else:
            reset = True
        if reset:
            x_range = np.arange(0, self.width, dtype=int)
            y_range = np.arange(0, self.height, dtype=int)
            y, x = np.meshgrid(y_range, x_range)
            self._image_grid = np.stack((x, y), axis=-1)
        return self._image_grid


class CameraParam(SetCameraParam):
    @classmethod
    def init_cam_param_by_dict(
        camera, param_dict, is_virtual=False, return_extra=False
    ):
        """Init Camera object from a Horizon Camera Calib dict.

        Args:
            params reference to
            https://horizonrobotics.feishu.cn/wiki/wikcnItt8QffFBJ9zRIinOuYDOc#R71aRI
            return_extra: Return local2vcs and local2cam for visual.
        """

        principle_point = [param_dict["center_u"], param_dict["center_v"]]
        image_size = [param_dict["image_width"], param_dict["image_height"]]
        fx = param_dict["focal_u"]
        fy = param_dict["focal_v"]
        camera_matrix = np.identity(3)
        camera_matrix[0, 0] = fx
        camera_matrix[1, 1] = fy
        camera_matrix[:2, 2] = principle_point
        if "param" in param_dict["distort"]:
            distcoeffs = list(param_dict["distort"]["param"])
        else:
            distcoeffs = list(param_dict["distort"])
        if is_virtual:
            distcoeffs = np.zeros(4)

        if return_extra:
            poseMat_vcs2cam, local2vcs, local2cam = parse_extrinsicParam(
                param_dict, is_virtual, return_extra=True
            )
            return (
                camera(
                    image_size=image_size,
                    camera_matrix=camera_matrix,
                    distcoeffs=distcoeffs,
                    poseMat_vcs2cam=poseMat_vcs2cam,
                    is_virtual=is_virtual,
                    param_dict=param_dict,
                ),
                local2vcs,
                local2cam,
            )
        else:
            poseMat_vcs2cam = parse_extrinsicParam(
                param_dict, is_virtual, return_extra=False
            )
            return camera(
                image_size=image_size,
                camera_matrix=camera_matrix,
                distcoeffs=distcoeffs,
                poseMat_vcs2cam=poseMat_vcs2cam,
                is_virtual=is_virtual,
                param_dict=param_dict,
            )

    @classmethod
    def init_cam_param_by_file(camera, calib_path, is_virtual=False):
        """Init Camera object from a Horizon Camera Calib json file."""
        assert os.path.exists(
            calib_path
        ), f"{calib_path} not exist, please check!!"
        with open(calib_path) as f:
            config = json.load(f)
        return camera.init_cam_param_by_dict(config, is_virtual)

    @classmethod
    def init_cam_param_by_matrix(
        camera,
        image_size: List[int],
        camera_matrix: np.ndarray,
        poseMat_vcs2cam: Optional[float] = None,
        distcoeffs: Union[List, np.ndarray] = None,
        is_virtual: bool = True,
    ):
        """Matrix style parameter for init camera.

        Args:
            image_size list: image plane size
            camera_matrix: [0, 0]fx, [1, 1]fy, [0, 2]cx, [1, 2]cy
                shape: 3*3
            poseMat_vcs2cam: pose matrix from rotation and translation
            distcoeffs: length must be 4 for pinhole or 8 for fisheye
            is_virtual: whether need virtualize
        """
        if poseMat_vcs2cam is None:
            poseMat_vcs2cam = np.eye(4)

        if distcoeffs is not None:
            assert len(distcoeffs) in [
                4,
                8,
            ], "distcoeffs only support pinhole 8 params or fisheye 4 params"
            if is_virtual:
                distcoeffs = np.zeros(4)

        # only real position camera can be virtualize
        if (
            is_virtual
            and not (poseMat_vcs2cam == np.eye(4)).all()
            and not camera().is_virtual
        ):
            poseMat_vcs2adascam = np.dot(
                camera().poseMat_cam2adascam, poseMat_vcs2cam
            )
            poseMat_adascam2vcs = np.linalg.inv(poseMat_vcs2adascam)
            rotMat_adascam2vcs = Rotation.from_matrix(
                poseMat_adascam2vcs[:3, :3]
            )
            # absolute angle for local coord z axis
            # has a eps value < 1 deg
            ypr_adascam2vcs = rotMat_adascam2vcs.as_euler("zyx")
            # translation not changed.
            translation_local2vcs = poseMat_adascam2vcs[:3, 3]
            # local yaw estimate camera direction
            rpy_local2vcs = np.array([0, 0, ypr_adascam2vcs[0]])
            # local to vcs coord
            # vcs_pt = poseMat_local2vcs * local_pt
            rotMat_local2vcs = transform_euler2rotMat(rpy_local2vcs)
            poseMat_local2vcs = np.eye(4)
            poseMat_local2vcs[:3, :3] = rotMat_local2vcs
            poseMat_local2vcs[0:3, 3] = np.array(translation_local2vcs).T
            # rectify local rpy
            poseMat_adascam2local = np.eye(4)
            poseMat_cam2vcs = np.dot(
                np.dot(poseMat_local2vcs, poseMat_adascam2local),
                camera().poseMat_cam2adascam,
            )
            poseMat_vcs2cam = np.linalg.inv(poseMat_cam2vcs)
        return camera(
            image_size=image_size,
            camera_matrix=camera_matrix,
            distcoeffs=distcoeffs,
            poseMat_vcs2cam=poseMat_vcs2cam,
            is_virtual=is_virtual,
        )

    def init_lidar2cam_param(self, **kwargs):
        raise NotImplementedError


class CameraBase(CameraParam):
    @property
    def hfov(
        self,
    ):
        self.calculate_fov_range()
        self._hfov = np.rad2deg(sum(np.abs(self.hfov_range)))
        return self._hfov

    @property
    def vfov(
        self,
    ):
        self.calculate_fov_range()
        self._vfov = np.rad2deg(sum(np.abs(self.vfov_range)))
        return self._vfov

    def _project_3d_to_pixel(
        self,
        points_3d: np.ndarray,
    ) -> np.ndarray:
        """Project 3d(lidar/cam/vcs) coordinate point to pixel coordinate.

        Args:
            points_3d: 3d point in any coordinate, shape:(n, 3)
        Note: inner api
        Return:
            pixel_points array in pixel coordinate
        """
        assert self.poseMat is not None, "Please set poseMat!"
        points_3d = ensure_point_list(points_3d, dim=4)

        # cam(4*n).T = (poseMat @ 3d coord(4*n)).T
        # --> cam(n*4) = 3d coord(n*4) @ poseMat.T
        camera_points = np.dot(points_3d, self.poseMat.T)
        lens_points = self.lens.project_3d_to_2d(camera_points[:, 0:3])
        if self.api_mode != "opencv":
            image_points = [self.fx, self.fy] * lens_points
            pixel_points = image_points + self.principle_point
        else:
            pixel_points = lens_points
        return pixel_points

    def _project_pixel_to_3d(
        self,
        pixel_points: np.ndarray,
        r_plane: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Project pixel point to 3d(lidar/cam/vcs) coordinate.

        Args:
            pixel_points: Shape:(n, 2)
            r_plane: Shape:(n, 1)/(1). Default:array([1])
        Note: inner api
        Return:
            3d(lidar/cam/vcs) coordinate pionts in pixel coordinate
        """
        assert self.poseMat is not None, "Please set poseMat!"
        pixel_points = ensure_point_list(
            pixel_points, dim=2, concatenate=False, crop=False
        )
        if r_plane is None:
            r_plane = np.array([1.0])
        r_plane = ensure_point_list(
            r_plane[:, np.newaxis], dim=1, concatenate=False, crop=False
        )
        if self.api_mode != "opencv":
            image_points = pixel_points - self.principle_point
            # image_points transform to lens_points
            lens_points = image_points / [self.fx, self.fy]
        else:
            lens_points = pixel_points
        camera_points = self.lens.project_2d_to_3d(lens_points, r_plane)
        camera_points = ensure_point_list(camera_points, dim=4)

        # 3d coord(4*n).T = (inv_poseMat @ cam(4*n)).T
        # --> 3d coord(n*4) = cam(n*4) @ inv_poseMat.T
        points_3d = np.dot(camera_points, self.inv_poseMat.T)
        points_3d = points_3d[:, :3]
        return points_3d

    def _line_fitting_3d(
        self,
        pts_pair: Union[Tuple[np.ndarray, np.ndarray], List[np.ndarray]],
    ):
        """Fitting 3d line by 2 points.

        pts_pair: paired ref points in 3d space.

        """
        # fit a line between the 3D points
        # return a function to get [y,z]
        x1, y1, z1 = pts_pair[0].T
        x2, y2, z2 = pts_pair[1].T

        ay = (y1 - y2) * 1.0 / (x1 - x2 + EPS)
        by = (x1 * y2 - x2 * y1) * 1.0 / (x1 - x2 + EPS)

        az = (z1 - z2) * 1.0 / (x1 - x2 + EPS)
        bz = (x1 * z2 - x2 * z1) * 1.0 / (x1 - x2 + EPS)

        def line_func(pts_x):
            pts_y = ay * pts_x + by
            pts_z = az * pts_x + bz
            return [pts_y, pts_z]

        return line_func

    def _update_pts_by_ref_x(self, pts_in, ref_x_coords, poseMat_cam23d):
        cam_in_ref_coord_system = poseMat_cam23d[:3, 3]
        cam_in_lidar_coord = np.tile(
            cam_in_ref_coord_system[None, ...], pts_in.shape[0]
        )

        line_func = self._line_fitting_3d((cam_in_lidar_coord, pts_in))
        y, z = line_func(ref_x_coords)
        pts_out = np.stack((ref_x_coords, y, z), axis=-1)
        return pts_out

    def _get_homoMat_pixel_to_3d(self):
        """Calculate homography Matrix from pixel to 3d coordinate."""
        raise NotImplementedError

    def _get_homoMat_3d_to_pixel(self):
        """Calculate homography Matrix from 3d to pixel coordinate."""
        raise NotImplementedError

    def project_lidar2pixel(self, lidar_points: np.ndarray) -> np.ndarray:
        """Lidar coordinate to pixel coordinate.

        Args:
            lidar_points: shape:(n,3).
        """
        self.poseMat = self.poseMat_lidar2cam
        self.inv_poseMat = np.linalg.inv(self.poseMat)
        pixel_points = self._project_3d_to_pixel(lidar_points)
        return pixel_points

    def project_pixel2lidar(
        self,
        pixel_points: Union[List, np.ndarray],
        ref_x_coords: np.ndarray = None,
    ):
        """Pixel coordinate to lidar coordinate.

        Args:
            pixel_points: shape:(n,3).
            ref_x_coords: defined x coords value in lidar coord.
                Shape: (n,)
        """
        assert len(ref_x_coords) != len(pixel_points), "ref_x_coords"
        self.poseMat = self.poseMat_lidar2cam
        self.inv_poseMat = np.linalg.inv(self.poseMat)
        lidar_points = self._project_pixel_to_3d(pixel_points)
        if ref_x_coords is not None:
            lidar_points = self._update_pts_by_ref_x(
                lidar_points, ref_x_coords, self.inv_poseMat
            )
        return lidar_points

    def project_vcs2pixel(self, world_points):
        self.poseMat = self.poseMat_vcs2cam
        self.inv_poseMat = np.linalg.inv(self.poseMat)
        pixel_points = self._project_3d_to_pixel(world_points)
        return pixel_points

    def project_pixel2vcs(
        self,
        pixel_points: Union[List, np.ndarray],
        ref_x_coords: np.ndarray = None,
    ):
        self.poseMat = self.poseMat_vcs2cam
        self.inv_poseMat = np.linalg.inv(self.poseMat)
        world_points = self._project_pixel_to_3d(pixel_points)
        if ref_x_coords is not None:
            world_points = self._update_pts_by_ref_x(
                world_points, ref_x_coords, self.inv_poseMat
            )
        return world_points

    def project_cam2pixel(self, cam_points):
        self.poseMat = np.eye(4)
        self.inv_poseMat = np.eye(4)
        pixel_points = self._project_3d_to_pixel(cam_points)
        return pixel_points

    def project_pixel2cam(self, pixel_points, r_plane=None, depth=None):
        self.poseMat = np.eye(4)
        self.inv_poseMat = np.eye(4)
        cam_points = self._project_pixel_to_3d(pixel_points, r_plane)
        # only support z axis now
        if depth is not None:
            assert len(depth) == len(
                cam_points
            ), "depth nums must equal to input point nums"
            scale = np.expand_dims(
                cam_points[:, 2] / (depth.reshape(-1) + EPS), axis=1
            )
            cam_points = cam_points / (scale + EPS)
        return cam_points

    def project_roty2dstcam(self, dst_cam, roty):
        rot_vector = np.stack(
            [np.cos(roty), np.zeros_like(roty), -np.sin(roty)],
            axis=-1,
        )
        rot_vector = self.project_cam2dstCam(dst_cam, rot_vector)
        roty = np.arctan2(-rot_vector[..., 2], rot_vector[..., 0])
        return roty

    def project_cam2vcs(self, cam_points):
        self.inv_poseMat = np.linalg.inv(self.poseMat_vcs2cam)
        cam_points = ensure_point_list(cam_points, dim=4)
        world_points = np.dot(cam_points, self.inv_poseMat.T)
        return world_points[:, :3]

    def project_vcs2cam(self, world_points):
        world_points = ensure_point_list(world_points, dim=4)
        cam_points = np.dot(world_points, self.poseMat_vcs2cam.T)
        return cam_points[:, :3]

    def project_cam2lidar(self, cam_points):
        self.inv_poseMat = np.linalg.inv(self.poseMat_lidar2cam)
        cam_points = ensure_point_list(cam_points, dim=4)
        lidar_points = np.dot(cam_points, self.inv_poseMat.T)
        return lidar_points[:, :3]

    def project_lidar2cam(self, lidar_points):
        lidar_points = ensure_point_list(lidar_points, dim=4)
        cam_points = np.dot(lidar_points, self.poseMat_lidar2cam.T)
        return cam_points[:, :3]

    def project_cam2dstCam(self, dst_cam, cam_points):
        world_points = self.project_cam2vcs(cam_points)
        cam_points_dst = dst_cam.project_vcs2cam(world_points)
        return cam_points_dst

    def project_pixel2dstCam(
        self, dst_cam: "CameraBase", pixel_point: List[Tuple]  # noqa
    ):
        """Project pixel_point to another Camera.

        Args:
            dst_cam: target camera instance
            pixel_point: stylelike [(x1,y1),(x2,y2)...,(xn,yn)]
        """
        points_in = np.array(pixel_point)
        world_point = self.project_pixel2vcs(points_in)
        pjd_points = dst_cam.project_vcs2pixel(world_point)
        return pjd_points

    @staticmethod
    def project_SO3(poseMat_A2B, point_A):
        point_A = ensure_point_list(point_A, dim=4)
        point_B = np.dot(point_A, poseMat_A2B.T)
        return point_B[:, :3]

    def project_bbox2d2dstcam(
        self,
        dst_cam=None,
        bboxes: Union[np.ndarray, List[List[float]]] = None,
        use_bbox_edge_center: bool = True,
    ):
        """Project 2d bbox to another Camera.

        Args:
            dst_cam: target camera instance
            bboxes: stylelike [[x1,y1,x2,y2]...,],shape=(N,4)
            use_bbox_edge_center: whether use bbox edge
                center point instead.
        Return:
            bboxes_prj: array([[x1,y1,x2,y2]...,]),shape=(N,4)
        """

        if isinstance(bboxes, np.ndarray):
            assert bboxes.shape[1] == 4
            bboxes = bboxes.tolist()
        elif isinstance(bboxes, list):
            assert np.array(bboxes).shape[1] == 4
        else:
            raise NotImplementedError("bboxes only support list/np.narray")

        bboxes_prj = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            if not use_bbox_edge_center:
                points_in = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            else:
                points_in = [
                    (x1, (y2 + y1) // 2),
                    ((x1 + x2) // 2, y1),
                    (x2, (y2 + y1) // 2),
                    ((x1 + x2) // 2, y2),
                ]

            virtual_points = self.project_pixel2dstCam(dst_cam, points_in)
            x1 = np.min(virtual_points[:, 0])
            y1 = np.min(virtual_points[:, 1])
            x2 = np.max(virtual_points[:, 0])
            y2 = np.max(virtual_points[:, 1])
            bbox = np.array([x1, y1, x2, y2])
            bboxes_prj.append(bbox)
        return np.array(bboxes_prj).reshape(-1, 4)

    def generate_mapping(
        self, dst_cam, concatenate=False, return_offset=False
    ):
        """Generate mapping from dst_cam to src_camera."""
        points = dst_cam.image_grid.reshape((-1, 2))
        projected_points = dst_cam.project_pixel2vcs(points)
        _, valid_index = self.filter_points_by_fov(
            projected_points, coord_system="vcs"
        )
        source_points = self.project_vcs2pixel(projected_points)
        source_points[~valid_index] = -1
        source_points = source_points.reshape(dst_cam.image_grid.shape)
        if return_offset:
            source_points -= dst_cam.image_grid
        if concatenate:
            # cv2.remap not support float64
            return source_points.astype(np.float32)
        u_map = np.expand_dims(source_points.T[0], axis=2).astype(np.float32)
        v_map = np.expand_dims(source_points.T[1], axis=2).astype(np.float32)

        return u_map, v_map

    def project_image2dstCam(
        self, dst_cam, src_im, uv_map=None, interpolation=cv2.INTER_LINEAR
    ):
        """Warp src_img to dst_cam."""

        if uv_map is None:
            u_map, v_map = self.generate_mapping(dst_cam)
        elif isinstance(uv_map, tuple):
            u_map, v_map = uv_map[0], uv_map[1]
        elif isinstance(uv_map, np.ndarray):
            u_map, v_map = uv_map[..., 0], uv_map[..., 1]
        else:
            raise NotImplementedError

        warp_image = cv2.remap(
            src_im,
            u_map.astype(np.float32),
            v_map.astype(np.float32),
            interpolation,
        )
        return warp_image

    def imresize(self, src_im: np.ndarray, scale: float):
        """Resize image to defined scale.

        This is a example for image scale by camera class

        Args:
            src_im: image input
            scale: scale coeffs
        """
        resize_cam = deepcopy(self)
        dst_size = np.array(self.image_size) * scale
        dst_size = dst_size.astype(int)
        resize_cam.image_size = dst_size
        dst_cxy = np.array(self.principle_point) * scale
        resize_cam.principle_point = dst_cxy
        resize_cam.fx = self.fx * scale
        resize_cam.fy = self.fy * scale
        resized_image = self.project_image2dstCam(resize_cam, src_im)
        return resized_image

    def _apply_clip(self, points, clip_source) -> np.ndarray:
        if self.image_size[0] == 0 or self.image_size[1] == 0:
            raise RuntimeError("clipping without a size is not possible")
        mask = (
            (clip_source[:, 0] < 0)
            | (clip_source[:, 0] >= self.image_size[0])
            | (clip_source[:, 1] < 0)
            | (clip_source[:, 1] >= self.image_size[1])
        )

        points[mask] = [-1]
        return points

    def calculate_fov_range(self):
        """Calculate fov by intrinsic params."""

        assert self.image_size is not None
        w, h = self.image_size
        if self.num_fov_points == 4:
            points = np.array(
                [
                    [0, h // 2],
                    [w - 1, h // 2],
                    [w // 2, 0],
                    [w // 2, h - 1],
                ],
            )
        elif self.num_fov_points == 8:
            points = np.array(
                [
                    [0, h // 2],
                    [w - 1, h // 2],
                    [w // 2, 0],
                    [w // 2, h - 1],
                    [0, 0],
                    [w - 1, 0],
                    [0, h - 1],
                    [w - 1, h - 1],
                ]
            )
        else:
            raise NotImplementedError

        cam_points = self.project_pixel2cam(points)
        azimuth_x = np.arctan2(cam_points[:, 0], cam_points[:, 2])
        # limit abnormal fov
        azimuth_x[np.abs(azimuth_x) > 3.14] = 0
        azimuth_y = np.arctan2(cam_points[:, 1], cam_points[:, 2])
        # limit abnormal fov
        azimuth_y[np.abs(azimuth_y) > 3.14] = 0
        self.hfov_range = np.array([azimuth_x.min(), azimuth_x.max()])
        self.vfov_range = np.array([azimuth_y.min(), azimuth_y.max()])

    def is_points_in_fov(self, points_3d, axis=0):
        """Verify 3d points whether in camera fov range.

        Args:
            points_3d: 3D point in camera coord. Shape:(N, 3).
            axis: verify axis.
                0: x axis
                1: y axis
                2: both x and y axis
        Return:
            is_valid: flag of valid. Shape:(N,)
        """

        assert isinstance(axis, int), "axis type shoule be int."
        if isinstance(points_3d, Sequence):
            points_3d = np.array(points_3d)
        assert isinstance(points_3d, np.ndarray)
        assert points_3d.ndim == 2, "only support shape=(N, 3)"

        if not hasattr(self, "hfov_range"):
            self.calculate_fov_range()

        if axis in [0, 2]:
            azimuth_x = np.arctan2(points_3d[:, 0], points_3d[:, 2])
            valid_x = np.full(azimuth_x.shape, False)
            valid_x[
                np.logical_and(
                    azimuth_x > self.hfov_range[0],
                    azimuth_x < self.hfov_range[1],
                )
            ] = True

        if axis in [1, 2]:
            azimuth_y = np.arctan2(points_3d[:, 1], points_3d[:, 2])
            valid_y = np.full(azimuth_y.shape, False)
            valid_y[
                np.logical_and(
                    azimuth_y > self.vfov_range[0],
                    azimuth_y < self.vfov_range[1],
                )
            ] = True

        if axis == 0:
            return valid_x
        elif axis == 1:
            return valid_y
        elif axis == 2:
            valid_xy = np.full(valid_x.shape, False)
            valid_xy[np.logical_and(valid_x, valid_y)] = True
            return valid_xy
        else:
            raise NotImplementedError("axis only support 0,1,2")

    def filter_points_by_fov(self, points_3d, axis=0, coord_system="camera"):
        """Filter 3d points out of camera fov range.

        Args:
            points_3d: 3D point in camera coord. Shape:(N, 3).
            axis: verify axis.
                0: x axis
                1: y axis
                2: both x and y axis
            coord_system: input pts coord, camera/lidar/vcs.
        Return:
            points_3d: filtered pts in fov.
        """

        if coord_system == "camera":
            points_3d_cam = points_3d
        elif coord_system == "lidar":
            points_3d_cam = self.project_lidar2cam(points_3d)
        elif coord_system == "vcs":
            points_3d_cam = self.project_vcs2cam(points_3d)
        else:
            raise ValueError(
                f"pts_coord only support camera/lidar/vcs. Got {coord_system}."
            )

        valid_index = self.is_points_in_fov(points_3d_cam, axis)
        return points_3d[valid_index], valid_index

    @staticmethod
    def convert_matrix_to_homogeneous(matrix: np.ndarray, N=4):
        """
        Convert matrix shape to homogeneous.

            matrix: general matrix shape is 4x4/3x3.

        """
        if (
            not matrix.ndim == 2
            and matrix.shape[0] == matrix.shape[1]
            and not matrix.shape[0] <= N
        ):
            raise ValueError(
                f"Input matrix must be KxK (K<=N) array. Got {matrix.shape}"
            )
        new_matrix = np.eye(N)
        index = matrix
        new_matrix[:index, :index] = matrix
        return new_matrix
