# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Optional

import torch
import torch.nn as nn

from .conversions import (
    convert_points_from_homogeneous,
    convert_points_to_homogeneous,
    pad_reshape_length,
)


class LensBaseModule(nn.Module):
    """Base class for lens refract method."""

    def project_3d_to_2d(self, points: torch.Tensor, *args, **kwargs):
        """Camera coordinate to lens coordinate."""
        if points.shape[-1] != 3:
            raise ValueError(f"points shape is invalid. Got {points.shape}.")

        pos_z_flag = points[..., -1:] > 0
        points = torch.where(
            pos_z_flag,
            points,
            torch.tensor(
                [1, 0, 0.1], dtype=points.dtype, device=points.device
            ).expand(points.shape),
        )

        # Convert 2D points from pixels to normalized camera coordinates
        # pnhole model max fov is 180 degree, for norm coord 100m->178.8°
        points = convert_points_from_homogeneous(points).clamp(
            min=-1e2, max=1e2
        )

        return points

    def project_2d_to_3d(self, points: torch.Tensor, *args, **kwargs):
        """Lens coordinate to camera coordinate."""
        if points.shape[-1] != 2:
            raise ValueError(f"points shape is invalid. Got {points.shape}.")

        points = convert_points_to_homogeneous(points)  # BxNx3
        return points


class PinholeProjModule(LensBaseModule):
    def project_3d_to_2d(self, points: torch.Tensor, distcoeffs: torch.Tensor):
        r"""Distortion of a set of 3D points based on the lens distortion model.

        radial :(k_1, k_2, k_3, k_4, k_4, k_6),
        tangential :(p_1, p_2), thin prism :(s_1, s_2, s_3, s_4), and tilt :(\tau_x, \tau_y)  # noqa E501
        distortion models are considered in this function.

        Args:
            points: camera coord system points with shape :(B, N, 3).
            dist: distortion coefficients
                :(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]]).  # noqa E501
                This is a vector with 4, 5, 8, 12 or 14 elements with shape :(B, n).

        Returns:
            distorted 2D points with shape :(B, N, 2).

        Example:
            >>> points = torch.rand(1, 1, 2)
            >>> dist_coeff = torch.rand(1, 4)
            >>> points_dist = project_3d_to_2d(points, dist_coeff)
        """
        if points.dim() < 2 or points.shape[-1] != 3:
            raise ValueError(f"points shape is invalid. Got {points.shape}.")

        if distcoeffs.shape[-1] not in [4, 5, 8, 12, 14]:
            raise ValueError(
                f"Invalid number of distortion coefficients. \
                    Got {distcoeffs.shape[-1]}"
            )

        # Adding zeros to obtain vector with 14 coeffs.
        if distcoeffs.shape[-1] < 14:
            distcoeffs = torch.nn.functional.pad(
                distcoeffs, [0, 14 - distcoeffs.shape[-1]]
            )

        # Compensate for tilt distortion
        if torch.any(distcoeffs[..., 12] != 0) or torch.any(
            distcoeffs[..., 13] != 0
        ):
            raise NotImplementedError("Not support tilt distortion yet!!!")

        pos_z_flag = points[..., -1:] > 0
        points = torch.where(
            pos_z_flag,
            points,
            torch.tensor(
                [1, 0, 0.1], dtype=points.dtype, device=points.device
            ).expand(points.shape),
        )

        # Convert 2D points from pixels to normalized camera coordinates
        # pnhole model max fov is 180 degree, for norm coord 100m->178.8°
        points = convert_points_from_homogeneous(points).clamp(
            min=-1e2, max=1e2
        )

        dist_points = self.distort_points(points, distcoeffs)

        return dist_points

    @staticmethod
    def distort_points(points, distcoeffs):
        x: torch.Tensor = points[..., 0]
        y: torch.Tensor = points[..., 1]

        dist_shape = pad_reshape_length(distcoeffs.shape, points.ndim - 1)
        distcoeffs = distcoeffs.reshape(dist_shape)

        # Distort points
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        rad_poly = (
            1
            + distcoeffs[..., 0:1] * r2
            + distcoeffs[..., 1:2] * r4
            + distcoeffs[..., 4:5] * r6
        ) / (
            1
            + distcoeffs[..., 5:6] * r2
            + distcoeffs[..., 6:7] * r4
            + distcoeffs[..., 7:8] * r6
        )
        xd = (
            x * rad_poly
            + 2 * distcoeffs[..., 2:3] * x * y
            + distcoeffs[..., 3:4] * (r2 + 2 * x * x)
            + distcoeffs[..., 8:9] * r2
            + distcoeffs[..., 9:10] * r4
        )
        yd = (
            y * rad_poly
            + distcoeffs[..., 2:3] * (r2 + 2 * y * y)
            + 2 * distcoeffs[..., 3:4] * x * y
            + distcoeffs[..., 10:11] * r2
            + distcoeffs[..., 11:12] * r4
        )
        return torch.stack([xd, yd], -1)

    def project_2d_to_3d(
        self,
        points: torch.Tensor,
        distcoeffs: torch.Tensor,
        norm_plane_z: torch.Tensor = None,
        num_iters: int = 5,
        *args,
        **kwargs,
    ):
        r"""Compensate for lens distortion a set of 2D image points.

        radial :(k_1, k_2, k_3, k_4, k_5, k_6),
        tangential :(p_1, p_2), thin prism :(s_1, s_2, s_3, s_4), and tilt :(\tau_x, \tau_y)  # noqa E501
        distortion models are considered in this function.

        Args:
            points: input image points on norm image plane. Shape :(*, N, 2).
            distcoeffs: distortion coefficients
                :(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]]). This is  # noqa E501
                a vector with 4, 5, 8, 12 or 14 elements with shape :(*, n).
            norm_plane_z: norm image plane from 1m to zm.
            num_iters: number of undistortion iterations. Default: 5.

        Returns:
            undistorted 3D camera points from 2D lens points with shape :(*, N, 2).

        Example:
            >>> _ = torch.manual_seed(0)
            >>> x = torch.rand(1, 4, 2)
            >>> dist = torch.rand(1, 4)
            >>> project_2d_to_3d(x, dist)
        """
        if points.dim() < 2 or points.shape[-1] != 2:
            raise ValueError(f"points shape is invalid. Got {points.shape}.")

        if norm_plane_z is not None:
            assert (
                norm_plane_z.shape[-1] == 1
            ), "norm_plane_z shape[-1] should be 1."
            assert (
                norm_plane_z.shape[:-1] == points.shape[:-1]
            ), "norm_plane_z shape should equal to points."
        else:
            norm_plane_z = 1

        if distcoeffs.shape[-1] not in [4, 5, 8, 12, 14]:
            raise ValueError(
                f"Invalid number of distortion coefficients. \
                    Got {distcoeffs.shape[-1]}"
            )

        # Adding zeros to obtain vector with 14 coeffs.
        if distcoeffs.shape[-1] < 14:
            distcoeffs = torch.nn.functional.pad(
                distcoeffs, [0, 14 - distcoeffs.shape[-1]]
            )

        # Compensate for tilt distortion
        if torch.any(distcoeffs[..., 12] != 0) or torch.any(
            distcoeffs[..., 13] != 0
        ):
            raise NotImplementedError("Not support tilt distortion yet!!!")

        undist_points = (
            self.undistort_points(
                points,
                distcoeffs,
                num_iters,
                *args,
                **kwargs,
            )
            * norm_plane_z
        )

        z = (
            torch.ones_like(
                undist_points[..., -1:],
                dtype=undist_points.dtype,
                device=undist_points.device,
            )
            * norm_plane_z
        )
        return torch.cat([undist_points, z], dim=-1)

    def _inner_undistort_points(
        self,
        points: torch.Tensor,
        distcoeffs: torch.Tensor,
        num_iters: int = 50,
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,
        lr: float = 1e-1,
        early_stop_iter: int = 5,
        epislon: float = 1e-7,
        *args,
        **kwargs,
    ):
        if torch.all(distcoeffs == 0):
            return points

        dist_shape = pad_reshape_length(distcoeffs.shape, points.ndim)
        distcoeffs = distcoeffs.reshape(dist_shape)

        undist_points = points.clone()
        undist_points.requires_grad = True
        optim = optimizer(params=[undist_points], lr=lr)
        StepLR = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.5)
        min_err = torch.inf
        smooth_err = 1e-9
        iter_cnt = 0
        loss_fn = torch.nn.MSELoss()

        for _ in range(num_iters):
            loss = loss_fn(
                self.distort_points(undist_points, distcoeffs), points
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            StepLR.step()

            if loss + smooth_err < min_err:
                min_err = loss
                iter_cnt = 0

            iter_cnt += 1
            if iter_cnt > early_stop_iter:
                break
            if loss <= epislon:
                break

        undist_points = undist_points.detach().requires_grad_(False)
        return undist_points

    # https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L481  # noqa
    def undistort_points(
        self,
        points: torch.Tensor,
        distcoeffs: torch.Tensor,
        num_iters: int = 15,
        *args,
        **kwargs,
    ):

        if torch.all(distcoeffs == 0):
            return points

        x: torch.Tensor = points[..., 0]  # (BxN - Bx1)/Bx1 -> BxN
        y: torch.Tensor = points[..., 1]  # (BxN - Bx1)/Bx1 -> BxN

        dist_shape = pad_reshape_length(distcoeffs.shape, points.ndim - 1)
        distcoeffs = distcoeffs.reshape(dist_shape)

        # Iteratively undistort points
        x0, y0 = x, y
        for _ in range(num_iters):
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r2 ** 3

            inv_rad_poly = (
                1
                + distcoeffs[..., 5:6] * r2
                + distcoeffs[..., 6:7] * r4
                + distcoeffs[..., 7:8] * r6
            ) / (
                1
                + distcoeffs[..., 0:1] * r2
                + distcoeffs[..., 1:2] * r4
                + distcoeffs[..., 4:5] * r6
            )
            deltaX = (
                2 * distcoeffs[..., 2:3] * x * y
                + distcoeffs[..., 3:4] * (r2 + 2 * x * x)
                + distcoeffs[..., 8:9] * r2
                + distcoeffs[..., 9:10] * r4
            )
            deltaY = (
                distcoeffs[..., 2:3] * (r2 + 2 * y * y)
                + 2 * distcoeffs[..., 3:4] * x * y
                + distcoeffs[..., 10:11] * r2
                + distcoeffs[..., 11:12] * r4
            )

            x = (x0 - deltaX) * inv_rad_poly
            y = (y0 - deltaY) * inv_rad_poly

        return torch.stack([x, y], dim=-1)


class FisheyeProjModule(LensBaseModule):
    """Base class for lens refract method."""

    def project_3d_to_2d(self, points: torch.Tensor, distcoeffs: torch.Tensor):
        r"""Distortion of a set of 3D points based on the lens distortion model.

        radial :(k_1, k_2, k_3, k_4),
        distortion models are considered in this function.

        Args:
            points: camera coord system points with shape :(*, N, 3).
            distcoeffs: distortion coefficients
                :(k_1,k_2,k_3,k_4).  # noqa E501
                This is a vector with 4 elements with shape :(*, 4).

        Returns:
            distorted 2D points with shape :(*, N, 2).

        Example:
            >>> points = torch.rand(1, 1, 3)
            >>> dist_coeff = torch.rand(1, 4)
            >>> points_dist = project_3d_to_2d(points, dist_coeff)
        """

        assert points.shape[-1] == 3, "points shape incorrect!"
        assert (
            distcoeffs.shape[-1] == 4
        ), "Horizon FisheyeProjModule only support 4 params!"

        # Convert 2D points from pixels to normalized camera coordinates
        chi = torch.linalg.norm(points[..., :2], dim=-1)
        theta = torch.atan2(chi, points[..., 2])  # BxN
        theta = torch.where(theta < 0, theta + torch.pi, theta)
        rho = self._theta_to_rho(theta, distcoeffs)
        lens_points = torch.divide(rho, chi)[..., None] * points[..., :2]
        # set (0, 0, 0) = -1
        lens_points[torch.logical_and(chi == 0, points[..., 2] == 0)] = -1
        return lens_points

    def _theta_to_rho(self, theta, distcoeffs):
        # reference:https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html

        if torch.all(distcoeffs == 0):
            return theta

        dist_shape = pad_reshape_length(distcoeffs.shape, theta.ndim)
        distcoeffs = distcoeffs.reshape(dist_shape)

        theta3 = theta ** 3
        theta5 = theta ** 5
        theta7 = theta ** 7
        theta9 = theta ** 9
        theta_d = (
            theta
            + distcoeffs[..., 0:1] * theta3
            + distcoeffs[..., 1:2] * theta5
            + distcoeffs[..., 2:3] * theta7
            + distcoeffs[..., 3:4] * theta9
        )
        return theta_d

    def project_2d_to_3d(
        self,
        points: torch.Tensor,
        distcoeffs: torch.Tensor,
        norm_plane_z: torch.Tensor = None,
        num_iters: int = 10,
        epsilon: float = 1e-8,
        *args,
        **kwargs,
    ):
        """Undistortion of a set of 2D points based on the lens distortion model.

        radial :(k_1, k_2, k_3, k_4),
        distortion models are considered in this function.

        Args:
            points: camera coord system points with shape :(*, N, 2).
            distcoeffs: distortion coefficients
                :(k_1,k_2,k_3,k_4).  # noqa E501
                This is a vector with 4 elements with shape :(*, 4).
            norm_plane_z: norm image plane from 1m to zm.
            num_iters: number of undistortion iterations. Default: 5.
            optimizer: torch.optim.Optimizer of solve undist points.
            lr: learning rate.
            early_stop_iter: early stop strategy.
            epsilon: tolorance max error to stop iter. Default:1e-5 m

        Returns:
            undistorted 3D camera points with shape :(*, N, 3).

        Example:
            >>> points = torch.rand(1, 1, 3)
            >>> dist_coeff = torch.rand(1, 4)
            >>> points_dist = project_2d_to_3d(points, dist_coeff)
        """

        if points.dim() < 2 or points.shape[-1] != 2:
            raise ValueError(f"points shape is invalid. Got {points.shape}.")

        if distcoeffs.shape[-1] != 4:
            raise ValueError(
                f"Invalid number of distortion coefficients. \
                    Got {distcoeffs.shape[-1]}"
            )

        if norm_plane_z is not None:
            assert (
                norm_plane_z.shape[-1] == 1
            ), "norm_plane_z shape[-1] should be 1."
            assert (
                norm_plane_z.shape[:-1] == points.shape[:-1]
            ), "norm_plane_z shape should equal to points."
        else:
            norm_plane_z = 1

        theta_d = torch.linalg.norm(points, dim=-1)

        thetas = self.solve_theta(
            theta_d,
            distcoeffs,
            num_iters,
            epsilon,
            *args,
            **kwargs,
        )
        chis = norm_plane_z * torch.sin(thetas)
        zs = norm_plane_z * torch.cos(thetas)  # BxNx1
        xy = torch.divide(chis, theta_d)[..., None]  # BxNx1
        xy = xy * points  # BxNx2
        cam_points = torch.cat((xy, zs[..., None]), dim=-1)  # BxNx3
        return cam_points

    def solve_theta(
        self,
        theta_d: torch.Tensor,
        distcoeffs: torch.Tensor,
        num_iters: int = 10,
        epsilon: float = 1e-8,
        *args,
        **kwargs,
    ):
        if torch.all(distcoeffs == 0):
            return theta_d

        dist_shape = pad_reshape_length(distcoeffs.shape, theta_d.ndim)
        distcoeffs = distcoeffs.reshape(dist_shape)

        theta = theta_d
        for _ in range(num_iters):
            theta2 = torch.pow(theta, 2)
            theta4 = torch.pow(theta2, 2)
            theta6 = theta4 * theta2
            theta8 = torch.pow(theta4, 2)
            k0_theta2 = distcoeffs[..., 0:1] * theta2
            k1_theta4 = distcoeffs[..., 1:2] * theta4
            k2_theta6 = distcoeffs[..., 2:3] * theta6
            k3_theta8 = distcoeffs[..., 3:4] * theta8
            theta_fix = torch.div(
                (
                    theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8)
                    - theta_d
                ),
                (
                    1
                    + 3 * k0_theta2
                    + 5 * k1_theta4
                    + 7 * k2_theta6
                    + 9 * k3_theta8
                ),
            )
            theta = theta - theta_fix

        # revert to within 180 degrees FOV to avoid numerical overflow
        theta[theta < 0] = 0
        theta[theta.abs() >= torch.pi] = 0.9999 * torch.pi
        return theta

    def _inner_solve_theta(
        self,
        theta_d: torch.Tensor,
        distcoeffs: torch.Tensor,
        num_iters: int = 100,
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,
        lr: float = 1e-1,
        early_stop_iter: int = 5,
        epsilon: float = 1e-5,
        *args,
        **kwargs,
    ):
        if torch.all(distcoeffs == 0):
            return theta_d

        dist_shape = pad_reshape_length(distcoeffs.shape, theta_d.ndim)
        distcoeffs = distcoeffs.reshape(dist_shape)

        theta = theta_d.clone()
        theta.requires_grad = True
        optim = optimizer(params=[theta], lr=lr)
        StepLR = torch.optim.lr_scheduler.StepLR(
            optim, step_size=early_stop_iter, gamma=0.8
        )
        min_err = torch.inf
        smooth_err = 1e-6
        iter_cnt = 0
        loss_fn = torch.nn.MSELoss()

        for _ in range(num_iters):
            loss = loss_fn(self._theta_to_rho(theta, distcoeffs), theta_d)
            optim.zero_grad()
            loss.backward()
            optim.step()
            StepLR.step()
            if loss + smooth_err < min_err:
                min_err = loss
                iter_cnt = 0

            iter_cnt += 1
            if iter_cnt > early_stop_iter:
                break
            if loss <= epsilon:
                break

        theta = theta.detach().requires_grad_(False)
        # revert to within 180 degrees FOV to avoid numerical overflow
        theta[theta < 0] = 0
        theta[theta.abs() >= torch.pi] = 0.9999 * torch.pi
        return theta


class CylindricalProjModule(LensBaseModule):
    """Cylindrical lens module."""

    def project_3d_to_2d(
        self, points: torch.Tensor, distcoeffs: Optional[torch.Tensor] = None
    ):
        r"""Distortion of a set of 3D points based on the lens distortion model.

        Args:
            points: camera coord system points with shape :(*, N, 3).
            distcoeffs: distortion coefficients
                :(k_1,k_2,k_3,k_4).  # noqa E501
                This is a vector with 4 elements with shape :(*, 4).

        Returns:
            distorted 2D points with shape :(*, N, 2).

        Example:
            >>> points = torch.rand(1, 1, 3)
            >>> points_dist = project_3d_to_2d(points)
        """

        assert points.shape[-1] == 3, "points shape incorrect!"

        # Convert 2D points from pixels to normalized camera coordinates
        theta = torch.atan2(points[..., 0], points[..., 2])  # BxN
        chi = torch.linalg.norm(points[..., [0, 2]], dim=-1)
        lens_points = torch.zeros_like(points)[..., :2]
        lens_points[..., 0] = theta
        lens_points[..., 1] = torch.divide(points[..., 1], chi)
        lens_points[chi == 0] = -1
        return lens_points

    def project_2d_to_3d(
        self,
        points: torch.Tensor,
        norm_plane_z: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        """Undistortion of a set of 2D points based on the lens distortion model.

        Two assumption.
        1. the focal length of cylinder is equal to that of fisheye;
        2. The image is backprojection to sqrt(x^2+z^2)=r_plane
        according to sqrt(x^2+z^2)=r_plane, tan(theta)=x/z,,
        z=r_plane/sqrt(1+tan^2(theta))
        input:image coordinate divide focal_length to lens_points
        output: camera coordinate point on a virtual plane

        Args:
            points: camera coord system points with shape :(*, N, 2).
            norm_plane_z: r plane.

        Returns:
            undistorted 3D camera points with shape :(*, N, 3).

        Example:
            >>> lens = CylindricalProjModule()
            >>> points = torch.rand(1, 1, 2)
            >>> points_dist = lens.project_2d_to_3d(points)
        """

        if points.dim() < 2 or points.shape[-1] != 2:
            raise ValueError(f"points shape is invalid. Got {points.shape}.")

        if norm_plane_z is not None:
            assert (
                norm_plane_z.shape[-1] == 1
            ), "norm_plane_z shape[-1] should be 1."
            assert (
                norm_plane_z.shape[:-1] == points.shape[:-1]
            ), "norm_plane_z shape should equal to points."
        else:
            norm_plane_z = torch.ones_like(points[..., 0:1])
        cam_points = torch.zeros((*points.shape[:-1], 3), device=points.device)

        theta = points[..., 0]
        tan_theta = torch.tan(theta)
        # fix projection with ρ=r_plane,Z=ρ*cos(Φ),X=ρ*sin(Φ),Y=ρ*y
        cam_points[..., 2] = torch.divide(
            norm_plane_z[..., 0], torch.sqrt(tan_theta.pow(2) + 1)
        )
        cam_points[..., 2][theta.abs() > torch.pi / 2] = torch.negative(
            cam_points[..., 2][theta.abs() > torch.pi / 2]
        )

        # v=cy+fy*y/sqrt(x^2+z^2) and  sqrt(x^2+z^2)=r_plane, \
        # normalized in camera
        cam_points[..., 1] = points[..., 1] * norm_plane_z[..., 0]
        cam_points[..., 0] = tan_theta * cam_points[..., 2]  # x = tan(theta)*z

        return cam_points
