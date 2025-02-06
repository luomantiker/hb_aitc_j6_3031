# Copyright (c) Horizon Robotics. All rights reserved.

import uuid
from copy import deepcopy
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from hat.core.virtual_camera.camera_base import CameraBase
from hat.registry import OBJECT_REGISTRY
from .conversions import (
    convert_affinematrix_to_homography3d,
    convert_points_to_homogeneous,
    create_meshgrid,
    pad_reshape_length,
)
from .lens_module import (
    CylindricalProjModule,
    FisheyeProjModule,
    LensBaseModule,
    PinholeProjModule,
)
from .warping_module import WarpingModule

__all__ = [
    "DifferentiableCameraBase",
    "DifferentiablePinholeCamera",
    "DifferentiableFisheyeCamera",
    "DifferentiableCylindricalCamera",
]

HOMO_PAD_VALUE = 65535


@OBJECT_REGISTRY.register
class DifferentiableCameraBase(nn.Module):
    """Differentiable camera base class for all type of cameras.

    Coord system details:
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

    For different cameras should set different LensBaseModule.

    Args:
        num_cameras: number of cameras on batch axis.
        image_size: image size only one set, in order WxH. Shape: 2.
        camera_matrix: intrinsic param. Shape: B*3*3. Format:
            [
                [
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0,  1 ],
                ]
            ]
            fx: focal length for x axis.
            fy: focal length for y axis.
            cx: x coord light axis in image piexl coord system.
            cy: y coord light axis in image piexl coord system.

        distcoeffs: Distort coeffs of lens. Shape: B*K.
            Pinhole 8 params and Fisheye 4 params on Horizon.

        ExtrinsicParam:
        poseMat_vcs2cam : pose Matrix for world/vcs coordinate to
            cammera coordinate, left dot. Shape: B*4*4
        poseMat_lidar2vcs : pose Matrix for lidar coordinate to
            vcs coordinate, left dot. Shape: B*4*4
        poseMat_lidar2cam : pose Matrix for lidar coordinate to
            cammera coordinate, left dot. Shape: B*4*4
        lens_model: lens projection model. Default: LensBaseModule.
        affine_matrix: affine projection matrix. Shape: B*4*4.
        parameter_type: camera param type one of Tensor/Buffer/Parameter.
            Default: Buffer
        num_iters: number of undistortion iterations. Default: 10.
        optimizer: torch.optim.Optimizer of solve undist points.
        lr: Learning rate. Default: 1e-1.
        early_stop_iter: early stop strategy max iter. Default: 5.
        epsilon: tolorance max error to stop iter. Default:1e-5 m
        num_fov_points: num of points to cal fov. Default: 8.

    """

    _FIELDS = (
        "camera_matrix",
        "distcoeffs",
        "poseMat_vcs2cam",
        "poseMat_lidar2vcs",
        "poseMat_lidar2cam",
        "affine_matrix",
    )

    _SHARED_FIELDS = ("image_size",)

    def __init__(
        self,
        num_cameras: int = None,
        image_size: torch.Tensor = None,
        camera_matrix: torch.Tensor = None,
        distcoeffs: torch.Tensor = None,
        poseMat_vcs2cam: torch.Tensor = None,
        cameras: Optional[List[CameraBase]] = None,
        poseMat_lidar2vcs: Optional[torch.Tensor] = None,
        poseMat_lidar2cam: Optional[torch.Tensor] = None,
        affine_matrix: Optional[torch.Tensor] = None,
        parameter_type: str = "Buffer",
        num_iters: int = 10,
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,
        lr: float = 1e-1,
        early_stop_iter: int = 5,
        epsilon: float = 1e-5,
        num_fov_points: int = 8,
    ):
        super(DifferentiableCameraBase, self).__init__()
        self._num_cameras = num_cameras
        self.cameras = cameras
        self._image_size = image_size
        self._camera_matrix = camera_matrix
        self._distcoeffs = distcoeffs
        self._poseMat_vcs2cam = poseMat_vcs2cam
        self._poseMat_lidar2vcs = poseMat_lidar2vcs
        self._poseMat_lidar2cam = poseMat_lidar2cam
        self._image_grid = None
        self._lens_model = LensBaseModule()
        self._affine_matrix = affine_matrix

        # reused pose matrix
        self.poseMat = None
        self._uuid = None

        self._dtype = torch.float32
        self._device = None
        self._identity_matrix_3 = torch.eye(3).repeat(self.num_cameras, 1, 1)
        self._identity_matrix_4 = torch.eye(4).repeat(self.num_cameras, 1, 1)

        assert parameter_type in ["Tensor", "Buffer", "Parameter"]
        self._parameter_type = parameter_type
        self.num_iters = num_iters
        self.optimizer = optimizer
        self.lr = lr
        self.early_stop_iter = early_stop_iter
        self.epsilon = epsilon

        self._register_camera_param(DifferentiableCameraBase._SHARED_FIELDS)
        self._register_camera_param(DifferentiableCameraBase._FIELDS)
        self.image_size = self.image_size.int()
        self.num_fov_points = num_fov_points

    @property
    def parameter_type(
        self,
    ):
        return self._parameter_type

    def _register_camera_param(self, filed_domain):
        for filed in filed_domain:
            if self.parameter_type == "Buffer":
                self.register_buffer(filed, getattr(self, f"_{filed}"))
            elif self.parameter_type == "Parameter":
                self.register_buffer(
                    filed,
                    nn.Parameter(
                        getattr(self, f"_{filed}"), requires_grad=False
                    ),
                )
            else:
                setattr(self, filed, getattr(self, f"_{filed}"))

    @property
    def device(
        self,
    ):
        self._device = self.camera_matrix.device
        return self._device

    @property
    def image_grid(
        self,
    ):
        if self._image_grid is not None:
            reset = self._image_grid.shape != (
                self.num_cameras,
                self.image_size[1],
                self.image_size[0],
            )
        else:
            reset = True
        if reset:
            self._image_grid = create_meshgrid(
                self.num_cameras,
                self.image_size[1],
                self.image_size[0],
            )
        if self._image_grid.device != self.device:
            self._image_grid = self._image_grid.to(self.device)
        if self._image_grid.dtype != self.dtype:
            self._image_grid = self._image_grid.to(self.dtype)
        return self._image_grid

    @property
    def lens_model(self):
        if self._lens_model is None:
            self._lens_model = LensBaseModule()
        return self._lens_model

    @lens_model.setter
    def lens_model(self, lens_model):
        self._lens_model = lens_model

    @property
    def num_cameras(self):
        assert not isinstance(
            self._num_cameras, type(None)
        ), "Please init num_cameras!"
        return self._num_cameras

    @property
    def uuid(self):
        distcoeffs = str(self.distcoeffs)
        poseMat_vcs2cam = str(self.poseMat_vcs2cam)
        camera_matrix = str(self.camera_matrix)
        camera_param_str = distcoeffs + poseMat_vcs2cam + camera_matrix
        self._uuid = uuid.uuid3(uuid.NAMESPACE_DNS, camera_param_str)
        return self._uuid

    @property
    def dtype(
        self,
    ):
        self._dtype = self.camera_matrix.dtype
        return self._dtype

    @property
    def identity_matrix_3(
        self,
    ):
        return self._identity_matrix_3.to(self.camera_matrix)

    @property
    def identity_matrix_4(
        self,
    ):
        return self._identity_matrix_4.to(self.camera_matrix)

    def clone(
        self,
    ):
        return deepcopy(self)

    @classmethod
    def init_from_cameras_list(
        differentiablecamera,
        cameras_list: Sequence[CameraBase],
        parameter_type: str = "Buffer",
    ):
        """Create a differentiable camera from cameras_list.

        Args:
            differentiablecamera: spycify camera instance.
            cameras_list: list of cameras has the same type and image size.
            parameter_type: camera param type one of Tensor/Buffer/Parameter.
                Default: Buffer
        Returns:
            differentiablecamera: inited differentiable camera in batch.

        Note:
            class初始化方式为：classname(有括号),带括号则先初始化类，需要所有参数输入
            classemethod调用classmethod初始化类的话需要用函数式调用，如：
            classname.init_class_method_function()  无括号
        """

        assert (
            isinstance(cameras_list, Sequence) and len(cameras_list) > 0
        ), "Make sure list of cameras is not None!"

        camera_0 = cameras_list[0]

        if not all(isinstance(c, CameraBase) for c in cameras_list):
            raise ValueError(
                "cameras in cameras_list must inherit from CameraBase"
            )

        if not all(type(c) is type(camera_0) for c in cameras_list[1:]):
            raise ValueError("all cameras must be of the same type")

        fields = differentiablecamera._FIELDS
        share_fields = differentiablecamera._SHARED_FIELDS
        # Concat the fields to make a batched tensor
        kwargs = {}
        for field in share_fields:
            attri = getattr(camera_0, field)
            if not np.all(
                [attri == getattr(c, field) for c in cameras_list[1:]]
            ):
                raise ValueError("all cameras must be of the same share field")
            kwargs[field] = torch.tensor(attri).float()

        for field in fields:
            field_type = [
                isinstance(getattr(c, field), np.ndarray) for c in cameras_list
            ]
            if not all(field_type):
                raise ValueError("field only support np.ndarray!")
            attrs = np.array([getattr(c, field) for c in cameras_list])
            kwargs[field] = torch.tensor(attrs).float()
        kwargs["cameras"] = cameras_list
        kwargs["num_cameras"] = len(cameras_list)
        kwargs["parameter_type"] = parameter_type
        return differentiablecamera(**kwargs)

    def join_cameras(self, dst_cam):
        assert type(dst_cam) == type(
            self
        ), "join cameras should be the same classmethod!"
        dst_cam = dst_cam.to(self.device)
        kwargs = {}
        kwargs["num_cameras"] = self.num_cameras + dst_cam.num_cameras

        if self.cameras and dst_cam.cameras:
            kwargs["cameras"] = self.cameras + dst_cam.cameras

        kwargs["num_iters"] = self.num_iters
        kwargs["optimizer"] = self.optimizer
        kwargs["lr"] = self.lr
        kwargs["early_stop_iter"] = self.early_stop_iter
        kwargs["epsilon"] = self.epsilon

        assert (
            self.parameter_type == dst_cam.parameter_type
        ), "Only support join the same parameter_type"
        kwargs["parameter_type"] = self.parameter_type
        for share_filed in DifferentiableCameraBase._SHARED_FIELDS:
            assert all(
                getattr(self, share_filed) == getattr(dst_cam, share_filed)
            ), f"{share_filed} must be the same!"
            kwargs[share_filed] = getattr(self, share_filed)
        for filed in DifferentiableCameraBase._FIELDS:
            if torch.is_tensor(getattr(self, filed)):
                kwargs[filed] = torch.cat(
                    [getattr(self, filed), getattr(dst_cam, filed)], dim=0
                )
            elif isinstance(getattr(self, filed), Sequence):
                kwargs[filed] = getattr(self, filed).append(
                    getattr(dst_cam, filed)
                )
            else:
                NotImplementedError(f"Not Implemented filed:{filed}")
        return self.__class__(**kwargs)

    def resize(
        self,
        image_size: Optional[Sequence[int]] = None,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ):
        """Resize image plane of camera.

        Args:
            image_size: zoom to new image size. Defaults to None.
            scale_x: width scale factor. Defaults to 1.0.
            scale_y: height scale factor. Defaults to 1.0.
        """

        if image_size is not None:
            if isinstance(image_size, Sequence):
                image_size = torch.tensor(
                    image_size, dtype=torch.int, device=self.device
                )
            elif isinstance(image_size, np.ndarray):
                image_size = torch.from_numpy(image_size).to(self.device)
            if not isinstance(image_size, torch.Tensor):
                raise NotImplementedError("Please check image size type!")

            scale_x, scale_y = torch.div(image_size, self.image_size)
            self.image_size = image_size.int()
        else:
            scale_xy = torch.tensor(
                [scale_x, scale_y], dtype=self.dtype, device=self.device
            )
            self.image_size = (self.image_size * scale_xy).int()

        affine_matrix = torch.tensor(
            [
                [
                    [scale_x, 1, scale_x],
                    [0, scale_y, scale_y],
                    [0, 0, 1],
                ]
            ],
            dtype=self.dtype,
            device=self.device,
        ).repeat(self.num_cameras, 1, 1)

        self.camera_matrix = self.camera_matrix * affine_matrix
        return self

    @property
    def inv_camera_matrix(self):
        """Invert camera matrices."""
        invert_camera_matrix = self.camera_matrix.clone()
        invert_camera_matrix[..., 0, 0] = 1.0 / self.camera_matrix[..., 0, 0]
        invert_camera_matrix[..., 1, 1] = 1.0 / self.camera_matrix[..., 1, 1]
        invert_camera_matrix[..., 0, 2] = (
            -1.0
            * self.camera_matrix[..., 0, 2]
            / self.camera_matrix[..., 0, 0]
        )
        invert_camera_matrix[..., 1, 2] = (
            -1.0
            * self.camera_matrix[..., 1, 2]
            / self.camera_matrix[..., 1, 1]
        )

        return invert_camera_matrix

    def invert_poseMat(self, poseMat: torch.Tensor):
        """Invert poseMat use BPU support api.

        Args:
            poseMat: [B, 4, 4] batch of pose matrices

        Note:
            Specify to cal poseMat, not universal method.
        """
        assert len(poseMat.shape) == 3 and poseMat.shape[1:] == (
            4,
            4,
        ), "Only works for batch of pose matrices."

        transposed_rotation = torch.transpose(poseMat[..., :3, :3], -2, -1)
        translation = poseMat[..., :3, 3:]

        inverse_mat = torch.cat(
            [
                transposed_rotation,
                -torch.matmul(
                    transposed_rotation.double(), translation.double()
                ).to(self.dtype),
            ],
            dim=-1,
        )  # [B,3,4]
        inverse_mat = convert_affinematrix_to_homography3d(
            inverse_mat
        )  # [B,4,4]
        return inverse_mat

    def _project_3d_to_pixel(
        self,
        points_3d: torch.Tensor,
    ) -> torch.Tensor:
        """Project pts3d(in any 3d coord system) to pixel coord system.

        Args:
            points_3d: 3d point in any 3d coord system, Shape: B*xxx*3.

        Return:
            pts in pixel coord system. Shape: B*xxx*2.
        """
        assert self.poseMat is not None, "Please set poseMat!"

        assert points_3d.shape[-1] == 3, "points_3d shape[-1] should be 3"
        assert points_3d.ndim >= 3, "points_3d ndim should be greater than 3"
        assert (
            points_3d.device == self.device
        ), "points_3d device should be same as camera device"

        points_3d = convert_points_to_homogeneous(points_3d)

        axis_idx = torch.arange(points_3d.ndim)
        pmt_order = (*axis_idx[:-2], axis_idx[-1], axis_idx[-2])
        param_shape = pad_reshape_length(self.poseMat.shape, points_3d.ndim)

        camera_points = (
            torch.matmul(
                self.poseMat.double().reshape(param_shape),
                points_3d.permute(pmt_order).double(),
            )
            .permute(pmt_order)
            .to(self.dtype)
        )

        lens_points = self.lens_model.project_3d_to_2d(
            camera_points[..., :3], self.distcoeffs
        )

        lens_points = convert_points_to_homogeneous(lens_points)

        param_shape = pad_reshape_length(
            self.camera_matrix.shape, points_3d.ndim
        )

        pixel_points = (
            torch.matmul(
                self.camera_matrix.double().reshape(param_shape),
                lens_points.permute(pmt_order).double(),
            )
            .permute(pmt_order)
            .to(self.dtype)
        )
        return pixel_points[..., :2]

    def _project_pixel_to_3d(
        self,
        pixel_points: torch.Tensor,
        norm_plane_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project pixel point to any 3d coord system.

        Args:
            pixel_points: pts in pixel coord system. Shape: B*xxx*2.
            norm_plane_z: dynamic depth plane in camera coord system.
                Shape: B*xxx*1. Defaults to None.

        Return:
            pts in any 3d coord system. Shape: B*xxx*3.
        """
        assert self.poseMat is not None, "Please set poseMat!"
        assert (
            pixel_points.shape[-1] == 2
        ), "pixel_points shape[-1] should be 2"
        assert (
            pixel_points.ndim >= 3
        ), "pixel_points axis should be greater than 3"
        assert (
            pixel_points.device == self.device
        ), "pixel_points device should be same as camera device"

        pixel_points = convert_points_to_homogeneous(pixel_points)

        axis_idx = torch.arange(pixel_points.ndim)
        pmt_order = (*axis_idx[:-2], axis_idx[-1], axis_idx[-2])
        param_shape = pad_reshape_length(
            self.inv_camera_matrix.shape, pixel_points.ndim
        )

        lens_points: torch.Tensor = (
            torch.matmul(
                self.inv_camera_matrix.double().reshape(param_shape),
                pixel_points.double().permute(pmt_order),
            )
            .permute(pmt_order)
            .to(self.dtype)
        )

        camera_points = self.lens_model.project_2d_to_3d(
            points=lens_points[..., :2],
            distcoeffs=self.distcoeffs,
            norm_plane_z=norm_plane_z,
            num_iters=self.num_iters,
            optimizer=self.optimizer,
            lr=self.lr,
            early_stop_iter=self.early_stop_iter,
            epsilon=self.epsilon,
        )

        camera_points = convert_points_to_homogeneous(camera_points)
        param_shape = pad_reshape_length(
            self.inv_poseMat.shape, pixel_points.ndim
        )

        points_3d = (
            torch.matmul(
                self.inv_poseMat.double().reshape(param_shape),
                camera_points.double().permute(pmt_order),
            )
            .permute(pmt_order)
            .to(self.dtype)
        )
        return points_3d[..., :3]

    def project_cam2pixel(self, cam_points: torch.Tensor) -> torch.Tensor:
        """Project camera pts to pixel coord system.

        Args:
            cam_points: pts in camera coord system. Shape: B*xxx*3.

        Returns:
            pts in pixel coord system. Shape: B*xxx*2.
        """
        self.poseMat = self.identity_matrix_4
        self.inv_poseMat = self.identity_matrix_4
        pixel_points = self._project_3d_to_pixel(cam_points)
        return pixel_points

    def project_pixel2cam(
        self,
        pixel_points: torch.Tensor,
        norm_plane_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project image pixel pts to camera coord system.

        Args:
            pixel_points: pts in pixel coord system. Shape: B*xxx*2.
            norm_plane_z: dynamic depth plane in camera coord system.
                Shape: B*xxx*1. Defaults to None.

        Returns:
            pts in camera system coord. Shape: B*xxx*3.
        """
        self.poseMat = self.identity_matrix_4
        self.inv_poseMat = self.identity_matrix_4
        cam_points = self._project_pixel_to_3d(pixel_points, norm_plane_z)
        return cam_points

    def project_vcs2pixel(self, vcs_points: torch.Tensor) -> torch.Tensor:
        """Project vcs pts to pixel coord system.

        Args:
            vcs_points: pts in vcs coord system.

        Returns:
            pts in pixel coord system. Shape: B*xxx*2.
        """
        self.poseMat = self.poseMat_vcs2cam
        self.inv_poseMat = self.invert_poseMat(self.poseMat)
        pixel_points = self._project_3d_to_pixel(vcs_points)
        return pixel_points

    def project_pixel2vcs(
        self,
        pixel_points: torch.Tensor,
        norm_plane_z: torch.Tensor = None,
    ) -> torch.Tensor:
        """Project pixel pts to vcs coord system.

        Args:
            pixel_points: pts in pixel coord system. Shape: B*xxx*2.
            norm_plane_z: dynamic depth plane in camera coord system.
                Shape: B*xxx*1. Defaults to None.

        Returns:
            pts in vcs system coord. Shape: B*xxx*3.
        """
        self.poseMat = self.poseMat_vcs2cam
        self.inv_poseMat = self.invert_poseMat(self.poseMat)
        vcs_points = self._project_pixel_to_3d(pixel_points, norm_plane_z)
        return vcs_points

    def project_cam2vcs(self, cam_points: torch.Tensor):
        assert (
            cam_points.device == self.device
        ), "cam_points device should be same as camera device"
        self.inv_poseMat = self.invert_poseMat(self.poseMat_vcs2cam)
        cam_points = convert_points_to_homogeneous(cam_points)
        axis_idx = torch.arange(cam_points.ndim)
        pmt_order = (*axis_idx[:-2], axis_idx[-1], axis_idx[-2])
        param_shape = pad_reshape_length(
            self.inv_poseMat.shape, cam_points.ndim
        )

        world_points = (
            torch.matmul(
                self.inv_poseMat.double().reshape(param_shape),
                cam_points.double().permute(pmt_order),
            )
            .permute(pmt_order)
            .to(self.dtype)
        )
        return world_points[..., :3]

    def project_vcs2cam(self, vcs_points: torch.Tensor):
        assert (
            vcs_points.device == self.device
        ), "vcs_points device should be same as camera device"
        vcs_points = convert_points_to_homogeneous(vcs_points)
        axis_idx = torch.arange(vcs_points.ndim)
        pmt_order = (*axis_idx[:-2], axis_idx[-1], axis_idx[-2])
        param_shape = pad_reshape_length(
            self.poseMat_vcs2cam.shape, vcs_points.ndim
        )

        cam_points = (
            torch.matmul(
                self.poseMat_vcs2cam.double().reshape(param_shape),
                vcs_points.double().permute(pmt_order),
            )
            .permute(pmt_order)
            .to(self.dtype)
        )
        return cam_points[..., :3]

    def project_cam2dstCam(self, dst_cam, cam_points: torch.Tensor):
        world_points = self.project_cam2vcs(cam_points)
        cam_points_dst = dst_cam.project_vcs2cam(world_points)
        return cam_points_dst

    def project_pixel2dstCam(self, dst_cam, pixel_point: torch.Tensor):
        """Project pixel_point to another Camera.

        Args:
            dst_cam CameraBase: target camera instance
            point list(tuple) stylelike tensor([[(x1,y1),(x2,y2)...,(xn,yn)]])
                Shape: BxNx2.
        """
        assert isinstance(
            pixel_point, torch.Tensor
        ), "only support torch.Tensor!"
        world_point = self.project_pixel2vcs(pixel_point)
        prj_points = dst_cam.project_vcs2pixel(world_point)
        return prj_points

    def generate_mapping(
        self,
        dst_cam,
        return_offset: bool = True,
    ):
        """Generate mapping from dst_cam to src_camera."""
        dst_image_grid: torch.Tensor = dst_cam.image_grid
        assert (
            dst_cam.device == self.device
        ), "src and dst cam should be on the same device!"
        assert (
            dst_cam.dtype == self.dtype
        ), "src and dst cam should have the same dtype!"
        projected_points = dst_cam.project_pixel2vcs(dst_image_grid)
        num_fov_points_buffer = self.num_fov_points
        if type(dst_cam.lens_model) == LensBaseModule:
            # align with auto3dv
            self.num_fov_points = 4
        valid_index = self.judge_points_in_fov(
            projected_points, axis=0, coord_system="vcs"
        )
        source_points = self.project_vcs2pixel(projected_points)
        if return_offset:
            source_points -= dst_image_grid
        source_points = torch.clip(
            source_points, min=-HOMO_PAD_VALUE, max=HOMO_PAD_VALUE
        )
        source_points[~valid_index] = HOMO_PAD_VALUE
        self.num_fov_points = num_fov_points_buffer
        return source_points

    def project_image2dstCam(
        self,
        dst_cam: nn.Module,
        src_im: torch.Tensor,
        uv_map: torch.Tensor = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ) -> torch.Tensor:
        """Warping image to dst camera.

        Args:
            dst_cam: dst camera.
            src_im: src image.
            uv_map: mapping for warping. Defaults to None.
            mode: interpolation mode to calculate output values.
                Only "bilinear"/"nearest" is supported now.
                Defaults to "bilinear".
            padding_mode: padding mode for outside grid values.
                Only "zeros" and "border" is supported now.
                Defaults to "border".

        Returns:
            image by dst camera.
        """

        self.grid_sample = WarpingModule(
            torch.max(dst_cam.image_size), mode=mode, padding_mode=padding_mode
        )
        if uv_map is None:
            uv_map = self.generate_mapping(
                dst_cam,
            )
        dst_image = self.grid_sample(src_im, uv_map)
        return dst_image

    def calculate_fov_range(self):
        """Calculate multi camera fov by intrinsic params."""

        assert self.image_size is not None
        w, h = self.image_size
        if self.num_fov_points == 4:
            points = torch.tensor(
                [
                    [0, torch.div(h, 2)],
                    [w - 1, torch.div(h, 2)],
                    [torch.div(w, 2), 0],
                    [torch.div(w, 2), h - 1],
                ],
                dtype=self.dtype,
                device=self.device,
            )
        elif self.num_fov_points == 8:
            points = torch.tensor(
                [
                    [0, torch.div(h, 2)],
                    [w - 1, torch.div(h, 2)],
                    [torch.div(w, 2), 0],
                    [torch.div(w, 2), h - 1],
                    [0, 0],
                    [w - 1, 0],
                    [0, h - 1],
                    [w - 1, h - 1],
                ],
                dtype=self.dtype,
                device=self.device,
            )
        else:
            NotImplementedError("Only support 4/8 points to cal fov!")

        points = points.repeat((self.num_cameras, 1, 1))  # Bx8x2

        cam_points = self.project_pixel2cam(points)
        azimuth_x = torch.atan2(cam_points[..., 0], cam_points[..., 2])
        azimuth_y = torch.atan2(cam_points[..., 1], cam_points[..., 2])
        self.hfov_range = torch.stack(
            [azimuth_x.min(dim=1)[0], azimuth_x.max(dim=1)[0]], dim=-1
        )  # Bx2
        self.vfov_range = torch.stack(
            [azimuth_y.min(dim=1)[0], azimuth_y.max(dim=1)[0]], dim=-1
        )  # Bx2

    def _judge_points_in_fov_cam(self, points_3d, axis=0):
        """Verify 3d points whether in camera fov range.

        Args:
            points_3d: 3D point in camera coord. Shape:(B, N, 3).
            axis: verify axis.
                0: x axis
                1: y axis
                2: both x and y axis
        Return:
            is_valid: flag of valid. Shape:(B, N)
        """

        assert axis in [0, 1, 2], "axis type only support one of 0,1,2."
        if isinstance(points_3d, np.ndarray):
            points_3d = torch.from_numpy(points_3d).to(self.camera_matrix)
        assert isinstance(points_3d, torch.Tensor)
        assert points_3d.ndim >= 3, "Please check input dim!"

        self.calculate_fov_range()
        param_shape = pad_reshape_length(
            self.hfov_range.shape, points_3d.ndim - 1
        )

        if axis in [0, 2]:
            azimuth_x = torch.atan2(points_3d[..., 0], points_3d[..., 2])
            valid_x = torch.full(azimuth_x.shape, False)
            valid_x[
                torch.logical_and(
                    azimuth_x > self.hfov_range.reshape(param_shape)[..., :1],
                    azimuth_x < self.hfov_range.reshape(param_shape)[..., 1:],
                )
            ] = True

        if axis in [1, 2]:
            azimuth_y = torch.atan2(points_3d[..., 1], points_3d[..., 2])
            valid_y = torch.full(azimuth_y.shape, False)
            valid_y[
                torch.logical_and(
                    azimuth_y > self.vfov_range.reshape(param_shape)[..., :1],
                    azimuth_y < self.vfov_range.reshape(param_shape)[..., 1:],
                )
            ] = True

        if axis == 0:
            return valid_x
        elif axis == 1:
            return valid_y
        else:
            valid_xy = torch.full(valid_x.shape, False)
            valid_xy[torch.logical_and(valid_x, valid_y)] = True
            return valid_xy

    def judge_points_in_fov(self, points_3d, axis=0, coord_system="camera"):
        """Judge 3d points out of camera fov range.

        Args:
            points_3d: 3D point in camera coord. Shape:(N, 3).
            axis: verify axis.
                0: x axis
                1: y axis
                2: both x and y axis
            coord_system: input pts coord, camera/vcs.
        Return:
            points_3d in fov flag.
        """

        if coord_system == "camera":
            points_3d_cam = points_3d
        elif coord_system == "vcs":
            points_3d_cam = self.project_vcs2cam(points_3d)
        else:
            raise ValueError(
                f"pts_coord only support camera/vcs. Got {coord_system}."
            )

        valid_index = self._judge_points_in_fov_cam(points_3d_cam, axis)
        return valid_index


@OBJECT_REGISTRY.register
class DifferentiablePinholeCamera(DifferentiableCameraBase):
    def __init__(
        self,
        num_cameras: int = None,
        image_size: torch.Tensor = None,
        camera_matrix: torch.Tensor = None,
        distcoeffs: torch.Tensor = None,
        poseMat_vcs2cam: torch.Tensor = None,
        cameras: Optional[List[CameraBase]] = None,
        poseMat_lidar2vcs: Optional[torch.Tensor] = None,
        poseMat_lidar2cam: Optional[torch.Tensor] = None,
        affine_matrix: Optional[torch.Tensor] = None,
        parameter_type: str = "Buffer",
        num_iters: int = 5,
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,
        lr: float = 1e-1,
        early_stop_iter: int = 5,
        epsilon: float = 1e-7,
        num_fov_points: int = 8,
    ):
        super().__init__(
            num_cameras=num_cameras,
            image_size=image_size,
            camera_matrix=camera_matrix,
            distcoeffs=distcoeffs,
            poseMat_vcs2cam=poseMat_vcs2cam,
            cameras=cameras,
            poseMat_lidar2vcs=poseMat_lidar2vcs,
            poseMat_lidar2cam=poseMat_lidar2cam,
            affine_matrix=affine_matrix,
            parameter_type=parameter_type,
            num_iters=num_iters,
            optimizer=optimizer,
            lr=lr,
            early_stop_iter=early_stop_iter,
            epsilon=epsilon,
            num_fov_points=num_fov_points,
        )
        self._lens_model = PinholeProjModule()


@OBJECT_REGISTRY.register
class DifferentiableFisheyeCamera(DifferentiableCameraBase):
    def __init__(
        self,
        num_cameras: int = None,
        image_size: torch.Tensor = None,
        camera_matrix: torch.Tensor = None,
        distcoeffs: torch.Tensor = None,
        poseMat_vcs2cam: torch.Tensor = None,
        cameras: Optional[List[CameraBase]] = None,
        poseMat_lidar2vcs: Optional[torch.Tensor] = None,
        poseMat_lidar2cam: Optional[torch.Tensor] = None,
        affine_matrix: Optional[torch.Tensor] = None,
        parameter_type: str = "Buffer",
        num_iters: int = 10,
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,
        lr: float = 1e-1,
        early_stop_iter: int = 10,
        epsilon: float = 1e-8,
        num_fov_points: int = 4,
    ):
        super().__init__(
            num_cameras=num_cameras,
            image_size=image_size,
            camera_matrix=camera_matrix,
            distcoeffs=distcoeffs,
            poseMat_vcs2cam=poseMat_vcs2cam,
            cameras=cameras,
            poseMat_lidar2vcs=poseMat_lidar2vcs,
            poseMat_lidar2cam=poseMat_lidar2cam,
            affine_matrix=affine_matrix,
            parameter_type=parameter_type,
            num_iters=num_iters,
            optimizer=optimizer,
            lr=lr,
            early_stop_iter=early_stop_iter,
            epsilon=epsilon,
            num_fov_points=num_fov_points,
        )
        self._lens_model = FisheyeProjModule()


@OBJECT_REGISTRY.register
class DifferentiableCylindricalCamera(DifferentiableCameraBase):
    """Differentialble Camera with Cylindrical lens model.

    Args:
        Refer to argument docstring of DifferentiableCameraBase.

    """

    def __init__(
        self,
        num_cameras: int = None,
        image_size: torch.Tensor = None,
        camera_matrix: torch.Tensor = None,
        distcoeffs: torch.Tensor = None,
        poseMat_vcs2cam: torch.Tensor = None,
        cameras: Optional[List[CameraBase]] = None,
        poseMat_lidar2vcs: Optional[torch.Tensor] = None,
        poseMat_lidar2cam: Optional[torch.Tensor] = None,
        affine_matrix: Optional[torch.Tensor] = None,
        parameter_type: str = "Buffer",
        num_iters: int = 10,
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,
        lr: float = 1e-1,
        early_stop_iter: int = 10,
        epsilon: float = 1e-8,
        num_fov_points: int = 4,
    ):
        super().__init__(
            num_cameras=num_cameras,
            image_size=image_size,
            camera_matrix=camera_matrix,
            distcoeffs=distcoeffs,
            poseMat_vcs2cam=poseMat_vcs2cam,
            cameras=cameras,
            poseMat_lidar2vcs=poseMat_lidar2vcs,
            poseMat_lidar2cam=poseMat_lidar2cam,
            affine_matrix=affine_matrix,
            parameter_type=parameter_type,
            num_iters=num_iters,
            optimizer=optimizer,
            lr=lr,
            early_stop_iter=early_stop_iter,
            epsilon=epsilon,
            num_fov_points=num_fov_points,
        )
        self._lens_model = CylindricalProjModule()
