# Copyright (c) Horizon Robotics. All rights reserved.

import typing
import uuid

import cv2
import numpy as np


def format_gdc_mapping(src_cam, dst_cam, gdc_map_savepath):
    """Format the configuration file for GDC on SoC.

    GDC:Geometric Distortion Correction
    Args:
        src_cam (CameraBase): original camera instance
        dst_cam (CameraBase): target camera instance
        gdc_map_savepath (str): gdc map txt save path
    """
    u_map, v_map = src_cam.generate_mapping(dst_cam)
    width, height = dst_cam.image_size
    f = open(gdc_map_savepath, "w")
    f.write(str(1) + "\n")
    f.write(str(50) + " " + str(50) + "\n")
    f.write(str(height) + " " + str(width) + "\n")
    f.write(str(height / 2.0 - 1) + " " + str(width / 2.0 - 1) + "\n")
    np.set_printoptions(precision=2)
    for i in range(height):
        for j in range(width):
            f.write(str(*v_map[i, j]) + ":" + str(*u_map[i, j]) + " ")
        f.write("\n")
    f.close()


def ensure_point_list(points, dim, concatenate=True, crop=True):
    """Ensure input points ndim and dim is expected.

    Args:
        points (ndarray/list): shape should be (n, dim).
        dim (int): excepted dim.
        concatenate (bool): concat defaault one array on last col.
            Default: True.
        crop (bool): crop real dim to excepted dim. Default: True.
    """
    if isinstance(points, list):
        points = np.array(points)
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2

    if crop:
        for test_dim in range(4, dim, -1):
            if points.shape[1] == test_dim:
                new_shape = test_dim - 1
                assert np.array_equal(
                    points[:, new_shape], np.ones(points.shape[0])
                )
                points = points[:, 0:new_shape]

    if concatenate and points.shape[1] == (dim - 1):
        points = np.concatenate(
            (np.array(points), np.ones((points.shape[0], 1))), axis=1
        )

    if points.shape[1] != dim:
        raise AssertionError(
            "points.shape[1] == dim failed ({} != {})".format(
                points.shape[1], dim
            )
        )
    return points


def transform_euler2rotMat(rpyAngle: typing.Union[list, tuple, np.ndarray]):
    """Alibi roation, (horizon) new axis, not opencv axis."""
    roll, pitch, yaw = rpyAngle
    rotMat_z = np.zeros((3, 3))
    rotMat_z[0, 0] = np.cos(yaw)
    rotMat_z[0, 1] = -np.sin(yaw)
    rotMat_z[1, 0] = np.sin(yaw)
    rotMat_z[1, 1] = np.cos(yaw)
    rotMat_z[2, 2] = 1

    rotMat_y = np.zeros((3, 3))
    rotMat_y[0, 0] = np.cos(pitch)
    rotMat_y[0, 2] = np.sin(pitch)
    rotMat_y[1, 1] = 1
    rotMat_y[2, 0] = -np.sin(pitch)
    rotMat_y[2, 2] = np.cos(pitch)

    rotMat_x = np.zeros((3, 3))
    rotMat_x[0, 0] = 1
    rotMat_x[1, 1] = np.cos(roll)
    rotMat_x[1, 2] = -np.sin(roll)
    rotMat_x[2, 1] = np.sin(roll)
    rotMat_x[2, 2] = np.cos(roll)
    rotMat_xyz = rotMat_x @ rotMat_y @ rotMat_z
    return rotMat_xyz


def parse_extrinsicParam(config, is_virtual=False, return_extra=False):
    # fix: local to vcs coord(vcs_pt = poseMat_local2vcs * local_pt)
    poseMat_local2vcs = np.eye(4)
    rotMat_local2vcs = transform_euler2rotMat(
        np.array(config["vcs"]["rotation"])
    )
    poseMat_local2vcs[:3, :3] = rotMat_local2vcs
    poseMat_local2vcs[0:3, 3] = np.array(config["vcs"]["translation"]).T

    # fix: Horizon camera (horizon defined axis) to local coord
    # (local_pt = poseMat_Hcam2local * Hcam_pt)
    poseMat_Hcam2local = np.eye(4)
    cam_x = config["camera_x"]
    cam_y = config["camera_y"]
    cam_z = config["camera_z"]
    # default virtual camera is horizontal
    if is_virtual:
        roll = 0
        pitch = 0
        yaw = 0
    else:
        roll = config["roll"]
        pitch = config["pitch"]
        yaw = config["yaw"]
    rotMat_Hcam2local = transform_euler2rotMat([roll, pitch, yaw])
    poseMat_Hcam2local[:3, :3] = rotMat_Hcam2local
    poseMat_Hcam2local[0:3, 3] = np.array([cam_x, cam_y, cam_z]).T

    # fix: Camera coord (opencv axis) to Horizon camera coord
    # (Hcam_pt = poseMat_opencvCam2Hcam * opencv_pt)
    poseMat_opencvCam2Hcam = np.array(
        [0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1]
    ).reshape((4, 4))

    # VCS-> Local -> Camera coord (horizon defined axis)
    # -> Camera coord (opencv axis)
    poseMat_opencvcam2vcs = np.dot(
        np.dot(poseMat_local2vcs, poseMat_Hcam2local), poseMat_opencvCam2Hcam
    )
    poseMat_vcs2opencvcam = np.linalg.inv(poseMat_opencvcam2vcs)

    if return_extra:
        return (
            poseMat_vcs2opencvcam,
            poseMat_local2vcs,
            np.linalg.inv(poseMat_Hcam2local @ poseMat_opencvCam2Hcam),
        )
    else:
        return poseMat_vcs2opencvcam


def get_cam_uuid(src_cam, dst_cam=None):
    params = []
    for cam in [src_cam, dst_cam]:
        if cam is None:
            continue
        for arg in [
            cam.image_size,
            cam.fx,
            cam.fy,
            cam.cx,
            cam.cy,
            cam.distcoeffs,
            cam.rotMat_vcs2cam,
        ]:
            if isinstance(arg, np.ndarray):
                params.extend(arg.reshape(-1).tolist())
            elif isinstance(arg, (list, tuple)):
                params.extend(arg)
            else:
                params.append(arg)
    cam_uuid = uuid.uuid3(
        uuid.NAMESPACE_DNS, "_".join([f"{x:.6f}" for x in params])
    )
    return cam_uuid


class ImagePointsInterpolation(object):
    """Image points interpolation for all type of camera when annotation.

    Args:
        src_cam: camera instance for init.
        pt_interval: interp point nums between two points.
    Note docs:
        https://horizonrobotics.feishu.cn/wiki/wikcn08nxMdXnL0jYK0jh2SJxLc%EF%BC%9B

    """

    def __init__(self, src_cam, pt_interval=5):
        self.src_cam = src_cam
        self.pt_interval = pt_interval

    def make_point_pairs(self, points):
        point_pairs = []
        for point_id in range(len(points) - 1):
            point_pairs.append([points[point_id], points[point_id + 1]])
        point_pairs.append([points[-1], points[0]])

        return point_pairs

    def make_point_sequence(self, point_groups):
        point_sequence = []
        key_positions = []
        for point_group in point_groups:
            key_positions.append(len(point_sequence))
            if len(point_group) == 1:
                point_sequence.extend(point_group)
            else:
                point_sequence.extend(point_group[:-1])

        return point_sequence, key_positions

    def line_fitting(self, pts_pair):
        # fit a line between the 2 points
        # return a function: y = a * x + b
        x1, y1 = pts_pair[0]
        x2, y2 = pts_pair[1]
        a = (y1 - y2) * 1.0 / (x1 - x2 + 1e-6)
        b = (x1 * y2 - x2 * y1) * 1.0 / (x1 - x2 + 1e-6)

        def line_func(pts_x):
            pts_y = a * pts_x + b
            return pts_y

        return line_func

    def line_fitting_3d(self, pts_pair):
        # fit a line between the 3D points
        # return a function to get [y,z]
        x1, y1, z1 = pts_pair[0]
        x2, y2, z2 = pts_pair[1]

        ay = (y1 - y2) * 1.0 / (x1 - x2 + 1e-6)
        by = (x1 * y2 - x2 * y1) * 1.0 / (x1 - x2 + 1e-6)

        az = (z1 - z2) * 1.0 / (x1 - x2 + 1e-6)
        bz = (x1 * z2 - x2 * z1) * 1.0 / (x1 - x2 + 1e-6)

        def line_func(pts_x):
            pts_y = ay * pts_x + by
            pts_z = az * pts_x + bz
            return [pts_y, pts_z]

        return line_func

    def interpolate_points(self, pts_undistorted):
        point_pairs = self.make_point_pairs(pts_undistorted)
        point_groups = []
        flag_3d = False
        for point_pair in point_pairs:
            if len(point_pair[0]) == 2:
                line_func = self.line_fitting(point_pair)
            else:
                flag_3d = True
                line_func = self.line_fitting_3d(point_pair)
            start = point_pair[0][0]
            end = point_pair[1][0]
            inverse_flag = False
            if start > end:
                inverse_flag = True
                start, end = end, start
            if flag_3d:
                num_pts = 100
            else:
                num_pts = int((end - start) * 1.0 / self.pt_interval)
            if num_pts > 1:
                pts_x = np.linspace(start, end, num=num_pts)
                if inverse_flag is True:
                    pts_x = pts_x[::-1]
                pts_x = np.array(pts_x)
                pts_inter = line_func(pts_x)
                if flag_3d:
                    interpolated_pts = np.vstack([pts_x, pts_inter]).T
                else:
                    interpolated_pts = np.vstack([pts_x, pts_inter]).T
            else:
                if inverse_flag is False:
                    interpolated_pts = np.array(point_pair)
                else:
                    interpolated_pts = np.array(point_pair[::-1])
            point_groups.append(interpolated_pts.tolist())
        point_sequence, key_positions = self.make_point_sequence(point_groups)

        return point_sequence, key_positions

    def find_nn_points(self, pts_x_src, pts_x_dst):
        num_src = len(pts_x_src)
        num_dst = len(pts_x_dst)
        assert num_src >= num_dst
        assert num_dst > 2
        pts_x_src = np.reshape(pts_x_src, (num_src, 1))
        pts_x_dst = np.reshape(pts_x_dst, (num_dst, 1))
        A = np.hstack([pts_x_src] * num_dst)
        B = np.hstack([pts_x_dst] * num_src)
        dist2 = A ** 2 + B.T ** 2 - 2 * A * B.T
        ptx_idx = np.argmin(dist2, axis=0)

        return ptx_idx

    def point_sampling(self, point_sequence, key_positions, input_points):
        # split point_sequence into point_groups
        point_groups = []
        num_groups = len(key_positions)
        for idx in range(num_groups):
            key_position = key_positions[idx]
            point_sequence[key_position] = input_points[idx]
            if idx == 0:
                continue
            last_key_position = key_positions[idx - 1]
            point_group = point_sequence[
                last_key_position : (key_position + 1)
            ]
            point_groups.append(point_group)
            if idx == num_groups - 1:
                point_group = np.vstack(
                    [point_sequence[key_position:], point_sequence[0:1, :]]
                )
                point_groups.append(point_group)

        # point sampling
        sampled_point_groups = []
        for point_group in point_groups:
            assert len(point_group) >= 2
            if len(point_group) == 2:
                num_pts = 2
            else:
                point_group = np.array(point_group)
                pts_x_src = point_group[:, 0]
                diff = np.abs(pts_x_src[1:] - pts_x_src[:-1])
                max_diff = np.max(diff)
                max_diff = np.maximum(max_diff, 2)
                start = pts_x_src[0]
                end = pts_x_src[-1]
                inverse_flag = False
                if start > end:
                    inverse_flag = True
                    start, end = end, start
                num_pts = int((end - start) * 1.0 / max_diff)
                if max_diff < 20:
                    pts_y_src = point_group[:, 1]
                    num_pixels = int(np.max(pts_y_src) - np.min(pts_y_src))
                    target_num_pts = np.maximum(num_pixels / 20, 4)
                    num_pts = np.minimum(num_pts, target_num_pts)
            if num_pts > 3:
                pts_x = np.linspace(start, end, num=int(num_pts))
                if inverse_flag is True:
                    pts_x = pts_x[::-1]
                pts_x_dst = np.array(pts_x)
                sampled_pts_idx = self.find_nn_points(pts_x_src, pts_x_dst)
                sampled_point_group = point_group[sampled_pts_idx, :]
            else:
                sampled_point_group = point_group[[0, -1]]
            sampled_point_groups.append(sampled_point_group.tolist())

        point_sequence, _ = self.make_point_sequence(sampled_point_groups)

        return point_sequence

    def anno_via_point_interpolation(self, input_points, coord_system="pixel"):
        if coord_system == "pixel":
            pts_undistorted = self.src_cam.project_pixel2vcs(input_points)
        else:
            pts_undistorted = input_points
        interpolated_point_sequence, key_positions = self.interpolate_points(
            pts_undistorted
        )
        point_sequence = self.src_cam.project_vcs2pixel(
            list(interpolated_point_sequence)
        )
        point_sequence = self.point_sampling(
            point_sequence, key_positions, input_points
        )

        return point_sequence

    @staticmethod
    def draw_points(image, points, color=None, thickness=1):
        points = np.array(points).reshape(-1, 2)
        points = points.tolist()
        for idx, point in enumerate(points):
            if color is None:
                gap = 1.0 / len(points)
                cur_color = (
                    int(255.0 * (idx * gap)),
                    128 + int(128.0 * (idx * gap)),
                    255 - int(255.0 * (idx * gap)),
                )
            else:
                cur_color = color
            cv2.circle(
                image, (int(point[0]), int(point[1])), 1, cur_color, thickness
            )

        return image

    @staticmethod
    def draw_lines(image, points, color=None, thickness=1):
        points = np.array(points).astype(np.int32)
        assert points.ndim == 2, "Only support 2D points."
        points = points.tolist()
        n_pts = len(points)
        colort_gap = 1.0 / n_pts
        for idx in range(n_pts):
            if color is None:
                cur_color = (
                    int(255.0 * (idx * colort_gap)),
                    128 + int(128.0 * (idx * colort_gap)),
                    255 - int(255.0 * (idx * colort_gap)),
                )
            else:
                cur_color = color
            if idx + 1 == n_pts:
                end = 0
            else:
                end = idx + 1
            cv2.line(
                image,
                (points[idx][0], points[idx][1]),
                (points[end][0], points[end][1]),
                cur_color,
                thickness,
            )

        return image

    @staticmethod
    def visualize_results(
        image, input_points, point_sequence, color=(0, 0, 255), thickness=1
    ):
        img_out = image.copy()
        img_out = ImagePointsInterpolation.draw_points(
            img_out, point_sequence, thickness=thickness
        )
        # key points are drawn in red
        img_out = ImagePointsInterpolation.draw_points(
            img_out, input_points, (0, 0, 255), thickness=thickness
        )
        img_out = ImagePointsInterpolation.draw_lines(
            img_out, point_sequence, color, thickness
        )
        return img_out
