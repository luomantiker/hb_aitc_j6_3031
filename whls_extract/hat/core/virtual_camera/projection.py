# Copyright (c) Horizon Robotics. All rights reserved.

import cv2
import numpy as np

from .utils import ensure_point_list


class Projection(object):
    """Base class for lens refract method."""

    def __init__(
        self,
    ):
        self._camera_matrix = np.eye(3)
        self._distcoeffs = np.zeros(4)

    @property
    def camera_matrix(self):
        return self._camera_matrix

    @camera_matrix.setter
    def camera_matrix(self, camera_matrix):
        self._camera_matrix = camera_matrix

    @property
    def distcoeffs(self):
        if self._distcoeffs is None:
            self._distcoeffs = np.zeros(4)
        return self._distcoeffs

    @distcoeffs.setter
    def distcoeffs(self, distcoeffs):
        self._distcoeffs = distcoeffs

    def project_3d_to_2d(self, camera_points):
        """Camera coordinate to lens coordinate."""
        raise NotImplementedError

    def project_2d_to_3d(self, lens_points, r_plane):
        """Lens coordinate to camera coordinate."""
        raise NotImplementedError

    def get_homoMat_2d_to_3d(self):
        """Calculate Homography Matrix from lens to 3d coordinate."""
        raise NotImplementedError

    def get_homoMat_3d_to_2d(self):
        """Calculate Homography Matrix from 3d to lens coordinate."""
        raise NotImplementedError


class CylindricalProjection(Projection):
    """Cylindrical plane projection method."""

    def project_3d_to_2d(self, camera_points):
        """Camera coordinate to lens coordinate."""
        camera_points = ensure_point_list(camera_points, dim=3)
        theta = np.arctan2(camera_points.T[0], camera_points.T[2])
        chi = np.sqrt(
            camera_points.T[0] * camera_points.T[0]
            + camera_points.T[2] * camera_points.T[2]
        )

        lens_points = np.zeros((camera_points.shape[0], 2))
        lens_points.T[0] = theta
        lens_points.T[1] = camera_points.T[1] * np.divide(
            1, chi, where=(chi != 0)
        )
        lens_points[chi == 0] = -1
        return lens_points

    def project_2d_to_3d(self, lens_points, r_plane):
        """Lens coordinate to camera coordinate.

        Two assumption.
        1. the focal length of cylinder is equal to that of fisheye;
        2. The image is backprojection to sqrt(x^2+z^2)=r_plane
        according to sqrt(x^2+z^2)=r_plane, tan(theta)=x/z,,
        z=r_plane/sqrt(1+tan^2(theta))
        input:image coordinate divide focal_length to lens_points
        output: camera coordinate point on a virtual plane
        """
        lens_points = ensure_point_list(lens_points, dim=2)
        r_plane = ensure_point_list(r_plane, dim=1)
        cam_points = np.zeros((lens_points.shape[0], 3))

        theta = lens_points.T[0]
        negative = np.full(theta.shape, False)
        negative[np.abs(theta) > (np.pi / 2)] = True

        t = np.tan(theta)
        # fix projection with ρ=r_plane,Z=ρ*cos(Φ),X=ρ*sin(Φ),Y=ρ*y
        cam_points.T[2] = np.divide(r_plane.flat, np.sqrt(t * t + 1))
        cam_points.T[2] = np.negative(
            cam_points.T[2], where=(negative), out=cam_points.T[2]
        )
        # v=cy+fy*y/sqrt(x^2+z^2) and  sqrt(x^2+z^2)=r_plane, \
        # normalized in camera
        cam_points.T[1] = lens_points.T[1] * r_plane.flat
        cam_points.T[0] = t * cam_points.T[2]  # x = tan(theta)*z
        return cam_points


class SphericalProjection(Projection):
    """Spherical plane projection method."""

    def project_3d_to_2d(self, camera_points):
        """Camera coordinate to lens coordinate."""
        camera_points = ensure_point_list(camera_points, dim=3)
        theta = np.arctan2(camera_points.T[0], camera_points.T[2])
        phi = np.arctan2(
            camera_points.T[1],
            np.hypot(camera_points.T[0], camera_points.T[2]),
        )

        lens_points = np.zeros((camera_points.shape[0], 2))
        lens_points.T[0] = theta
        lens_points.T[1] = phi
        return lens_points

    def project_2d_to_3d(self, lens_points: np.ndarray, r_plane: np.ndarray):
        """Lens coordinate to camera coordinate.

        Two assumption:
        1. the focal length of cylinder is equal to that of fisheye.
        2. The image is backprojection to sqrt(x^2+z^2+y^2)=r_plane
        according to sqrt(x^2+z^2+y^2)=r_plane, tan(theta)=x/z,
        y=r_plane*sin(phi)
        z=r_plane*cos(phi)*cos(theta)
        x=r_plane*cos(phi)*sin(theta)
        input:image coordinate divide focal_length to lens_points
        output: camera coordinate point on a virtual plane
        """
        lens_points = ensure_point_list(lens_points, dim=2)
        r_plane = ensure_point_list(r_plane, dim=1)
        cam_points = np.zeros((lens_points.shape[0], 3))

        theta = lens_points.T[0]
        # theta = (lens_points.T[0]) + 1e-5
        # negative = np.full(theta.shape, False)
        # negative[np.abs(theta) > (np.pi / 2 - 1e-10)] = True

        phi = lens_points.T[1]
        # negative_phi = np.full(phi.shape, False)
        # negative_phi[np.abs(phi) > (np.pi / 2 - 1e-10)] = True

        rcos_phi = r_plane.flat * np.cos(phi)
        # fix projection with r=r_plane, Z=r*cos(phi)*cos(theta)
        # , X=r*cos(phi)*sin(theta), Y=r*sin(phi)
        cam_points.T[2] = rcos_phi * np.cos(theta)
        cam_points.T[1] = r_plane.flat * np.sin(phi)
        cam_points.T[0] = rcos_phi * np.sin(theta)
        return cam_points


class FisheyeProjection(Projection):
    """Fisheye equidistance projection method."""

    @property
    def distcoeffs(self):
        return self._distcoeffs

    @distcoeffs.setter
    def distcoeffs(self, distcoeffs):
        if distcoeffs is not None:
            assert (
                len(distcoeffs) == 4
            ), "\
                Only Support KB Mode 4 params(k1->k4)"
        else:
            distcoeffs = np.zeros(4)
        self._distcoeffs = distcoeffs

    @property
    def power(self):
        return np.array([np.arange(start=3, stop=10, step=2)]).T

    def project_3d_to_2d(self, camera_points):
        """Camera coordinate to lens coordinate."""
        camera_points = ensure_point_list(camera_points, dim=3)
        chi = np.sqrt(
            camera_points.T[0] * camera_points.T[0]
            + camera_points.T[1] * camera_points.T[1]
        )
        theta = np.arctan2(chi, camera_points.T[2])  # 入射角
        theta = np.add(theta, np.pi, where=(theta < 0), out=theta)
        rho = self._theta_to_rho(theta)
        lens_points = (
            np.divide(rho, chi, where=(chi != 0))[:, np.newaxis]
            * camera_points[:, 0:2]
        )
        # set (0, 0, 0) = -1
        lens_points[(chi == 0) & (camera_points[:, 2] == 0)] = -1
        return lens_points

    def project_2d_to_3d(self, lens_points, r_plane):
        """Lens coordinate to camera coordinate."""
        lens_points = ensure_point_list(lens_points, dim=2)
        r_plane = ensure_point_list(r_plane, dim=1).reshape(r_plane.size)
        rhos = np.linalg.norm(lens_points, axis=1)
        thetas = self._rho_to_theta(rhos)
        chis = r_plane * np.sin(thetas)
        zs = r_plane * np.cos(thetas)
        xy = np.divide(chis, rhos, where=(rhos != 0))[:, np.newaxis]
        xy = xy * lens_points
        cam_points = np.hstack((xy, zs[:, np.newaxis]))
        return cam_points

    def _theta_to_rho(self, theta):
        # reference:https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
        return (
            np.dot(self.distcoeffs, np.power(np.array([theta]), self.power))
            + theta
        )

    # slow method
    def _rho_to_theta(self, rho):
        ploy_len = max(self.power)
        coeff = np.zeros((ploy_len))
        for k_i, coeff_i in zip(self.power, self.distcoeffs):
            coeff[ploy_len - k_i] = coeff_i
        coeff[-1] = 1
        results = np.zeros_like(rho)
        for i, _r in enumerate(rho):
            theta = np.roots([*coeff, -_r])
            theta = np.real(theta[theta.imag == 0])
            theta = theta[np.where(np.abs(theta) < np.pi)]
            theta = theta[np.where(theta > 0)]
            theta = (
                theta[np.argmin(np.abs(theta))]
                if theta.size > 0
                else np.array(0)
            )
            theta = np.min(theta) if theta.size > 0 else 0
            results[i] = np.array(theta)
        return results


class PinholeProjection(Projection):
    """Pinhole perspective projection method."""

    def project_3d_to_2d(
        self,
        camera_points,
    ):
        """Camera coordinate to lens coordinate."""
        # fix memery not continus
        camera_points = np.copy(camera_points[:, :3]).astype(np.float64)
        camera_points = np.expand_dims(camera_points, axis=1)

        rvec, _ = cv2.Rodrigues(np.identity(3, np.float64))
        tvec = np.zeros(shape=(3, 1), dtype=np.float64)
        # process 3d points, which is back of pinhole camera
        camera_points[camera_points[..., -1] <= 0] = [1e3, 0, 1e-3]
        pixel_points, _ = cv2.projectPoints(
            camera_points,
            np.array(rvec),
            tvec,
            self.camera_matrix,
            np.array(self.distcoeffs),
        )
        return pixel_points.reshape((-1, 2))

    def project_2d_to_3d(self, lens_points: np.ndarray, r_plane: np.ndarray):
        pixel_points = ensure_point_list(lens_points, dim=2)
        # fix memery not continus
        pixel_points = np.copy(pixel_points).astype(np.float64)
        pixel_points = np.expand_dims(pixel_points, axis=1)
        un_pixel_points = cv2.undistortPoints(
            pixel_points,
            self.camera_matrix,
            np.array(self.distcoeffs),
            None,
            self.camera_matrix,
        )  # noqa
        un_pixel_points = un_pixel_points.reshape(-1, 2)
        lens_points = un_pixel_points - self.camera_matrix[:2, 2]
        lens_points = np.divide(
            lens_points,
            [self.camera_matrix[0, 0], self.camera_matrix[1, 1]],
            out=lens_points,
        )
        r_plane = ensure_point_list(r_plane, dim=1).reshape(r_plane.size)
        zs = r_plane
        xy = lens_points * zs
        zs = np.repeat(zs, lens_points.shape[0])
        camera_points = np.hstack((xy, zs[:, np.newaxis]))
        return camera_points
