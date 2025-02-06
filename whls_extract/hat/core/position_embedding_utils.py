import logging
import threading
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PositionEncoder(object):
    __instance = None
    __is_init = False
    _instance_lock = threading.Lock()
    _visited_maps = {}

    def __new__(cls, *args, **kwargs):
        with PositionEncoder._instance_lock:
            if cls.__instance is None:
                cls.__instance = object.__new__(cls)
                return cls.__instance
            else:
                return cls.__instance

    def __init__(
        self,
        pe_stride: int,
        input_hw: Tuple[int, int],
        img_resize: int,
        pe_h: int,
        pe_w: int,
        default_intrinsic_mat: np.ndarray,
        default_distort: np.ndarray,
        default_pitch: float = 0,
        default_roll: float = 0,
        default_camera_z: float = 1,
        crop_roi: Tuple[int, int, int, int] = None,
        crop_stride: int = 8,
        channel: int = 3,
        ceiling_height: int = 2000,
        verbose: int = 1,
    ):
        if self.__is_init is False:
            self.pe_stride = pe_stride
            self.img_hw = [int(size * img_resize) for size in input_hw]
            self.img_resize = img_resize
            self.crop_roi = crop_roi
            self.crop_stride = crop_stride
            self.pe_h = max(input_hw[0] // pe_stride, pe_h)
            self.pe_w = max(input_hw[1] // pe_stride, pe_w)
            self.pe_c = channel
            self.default_intrinsic_mat = default_intrinsic_mat[:3, :3]
            self.default_distort = default_distort.flatten()
            self.default_pitch = default_pitch
            self.default_roll = default_roll
            self.default_camera_z = default_camera_z
            self.ceiling_height = ceiling_height
            self.verbose = verbose
            self.__is_init = True
            logger.info("Position Encoder init done.")

    def __len__(self):

        return len(self._visited_maps)

    @property
    def is_init(self):
        return self.__is_init

    @property
    def instance(self):
        return self.__instance

    def reset(self):
        self.__is_init = False
        self.__instance = None
        self._visited_maps = {}

    def is_visited(self, key: str) -> bool:

        return key in self._visited_maps

    def get_identity(
        self,
        intrinsic_mat: np.ndarray,
        distort: np.ndarray,
        pitch: float,
        roll: float,
        camz: float,
        decimal: int = 3,
    ) -> str:
        intrinsics = (
            intrinsic_mat.flatten().tolist() + distort.flatten().tolist()
        )
        intrinsic_key = "_".join(
            map(lambda x: str(round(x, decimal)), intrinsics)
        )

        extrinsics = [pitch, roll, camz]
        extrinsic_key = "_".join(
            map(lambda x: str(round(x, decimal)), extrinsics)
        )

        identity_key = "_".join((intrinsic_key, extrinsic_key))

        return identity_key

    def fetch_calib(self, calib_params: dict = None) -> Tuple[np.ndarray]:
        if calib_params is None:
            intrinsic_mat = self.default_intrinsic_mat
            distort = self.default_distort
            camz = self.default_camera_z
            roll = self.default_roll
            pitch = self.default_pitch
            if self.verbose:
                logger.info("using default calib_params for PE generation.")
        else:
            intrinsic_mat = np.zeros((3, 3), dtype=np.float32)
            intrinsic_mat[0, 0] = calib_params["focal_u"]
            intrinsic_mat[1, 1] = calib_params["focal_v"]
            intrinsic_mat[0, 2] = calib_params["center_u"]
            intrinsic_mat[1, 2] = calib_params["center_v"]
            intrinsic_mat[2, 2] = 1

            distort = calib_params["distort"]
            if "param" in distort:
                distort = distort["param"]

            distort = np.array(distort, dtype=np.float32)

            camz = calib_params["camera_z"]
            roll = calib_params["roll"]
            pitch = calib_params["pitch"]

        return intrinsic_mat, distort, pitch, roll, camz

    def __call__(self, calib_params: dict = None) -> np.ndarray:
        intrinsic_mat, distort, pitch, roll, camz = self.fetch_calib(
            calib_params
        )

        identity_key = self.get_identity(
            intrinsic_mat, distort, pitch, roll, camz
        )

        if self.is_visited(identity_key):
            return self._visited_maps[identity_key]

        extrinsic_scale = self.img_resize * self.pe_stride
        extrinsic_channel = self.compute_extrinsic_channel(
            intrinsic_mat, extrinsic_scale, pitch, roll, camz
        )

        position_encoding = self.generate_coordinate3d_map(
            extrinsic_channel,
            intrinsic_mat,
            distort,
        )

        if self.crop_roi is not None:
            position_encoding = self.crop_map(
                position_encoding,
                self.crop_roi,
                self.crop_stride,
            )

        self._visited_maps[identity_key] = position_encoding

        return self._visited_maps[identity_key]

    def crop_map(
        self,
        coordinate_map: np.ndarray,
        crop_roi: Tuple[int, int, int, int] = None,
        crop_stride: int = 8,
    ) -> np.ndarray:
        x1, y1, x2, y2 = [int(x // crop_stride) for x in crop_roi]
        coordinate_map = coordinate_map[:, y1:y2, x1:x2]

        return coordinate_map

    def calculate_rotx(self, pitch: float) -> np.ndarray:
        rotx = np.zeros((9, 1), dtype=np.float32)

        rotx[0, 0] = 1
        rotx[4, 0] = np.cos(pitch)
        rotx[5, 0] = -np.sin(pitch)
        rotx[7, 0] = np.sin(pitch)
        rotx[8, 0] = np.cos(pitch)

        return rotx.reshape(3, 3)

    def calculate_rotz(self, roll: float) -> np.ndarray:
        rotz = np.zeros((9, 1), dtype=np.float32)

        rotz[0, 0] = np.cos(roll)
        rotz[1, 0] = -np.sin(roll)
        rotz[3, 0] = np.sin(roll)
        rotz[4, 0] = np.cos(roll)
        rotz[8, 0] = 1

        return rotz.reshape(3, 3)

    def undistort_points(
        self,
        pts: np.ndarray,
        pe_w: int,
        pe_h: int,
        intrinsic_mat: np.ndarray,
        distort: np.ndarray,
    ) -> np.ndarray:
        pts[:, 0] = np.clip(pts[:, 0], 0, pe_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, pe_h - 1)
        pts = np.expand_dims(pts, axis=0)
        cam_pts = cv2.undistortPoints(pts, intrinsic_mat, distort)
        cam_pts = np.squeeze(cam_pts)

        return cam_pts

    def compute_extrinsic_channel(
        self,
        intrinsic_mat: np.ndarray,
        pe_stride: int,
        pitch: float,
        roll: float,
        camera_z: float,
    ) -> np.ndarray:
        rotation = np.dot(
            self.calculate_rotx(-pitch), self.calculate_rotz(roll)
        )

        translation_v = np.zeros((3, 1), dtype=np.float32)
        translation_v[1, 0] = -camera_z

        normal = np.array((0, -1, 0), dtype=np.float32).reshape(-1, 1)
        normal_t = normal.T

        uu = np.arange(self.pe_w, dtype=np.float32)
        vv = np.arange(self.pe_h, dtype=np.float32)
        uu = uu * pe_stride + pe_stride // 2
        vv = vv * pe_stride + pe_stride // 2
        grid_x, grid_y = np.meshgrid(uu, vv)

        uv = np.stack(
            (
                grid_x.flatten(),
                grid_y.flatten(),
                np.ones_like(grid_x.flatten()),
            ),
            axis=0,
        )

        intrinsic_inv = np.linalg.inv(intrinsic_mat)
        cam_pts = np.dot(np.dot(np.dot(normal_t, rotation), intrinsic_inv), uv)
        cam_height_t = np.dot(normal_t, translation_v)

        z_f = -cam_height_t / cam_pts
        z_f_channel = z_f.reshape(self.pe_h, self.pe_w)

        z_c = (self.ceiling_height - cam_height_t) / cam_pts
        z_c_channel = z_c.reshape(self.pe_h, self.pe_w)

        z_f_channel = z_f_channel.astype(np.float32)
        z_c_channel = z_c_channel.astype(np.float32)
        extrinsic_channel = np.maximum(z_f_channel, z_c_channel)

        extrinsic_channel = np.arctan(extrinsic_channel)

        return extrinsic_channel

    def generate_coordinate3d_map(
        self,
        extrinsic_channel: np.ndarray,
        intrinsic_mat: np.ndarray,
        distort: np.ndarray,
        plen: int = 3,
        resize_scale: int = 4,
    ) -> np.ndarray:
        img_h, img_w = self.img_hw

        uu = np.arange(img_w // resize_scale, dtype=np.float32)
        vv = np.arange(img_h // resize_scale, dtype=np.float32)
        uu = uu * resize_scale + resize_scale // 2
        vv = vv * resize_scale + resize_scale // 2
        xx, yy = np.meshgrid(uu, vv)
        pts_uv = np.stack([xx.reshape((-1)), yy.reshape((-1))], axis=1)

        pts_cam = self.undistort_points(
            pts_uv, img_w, img_h, intrinsic_mat, distort
        )
        pts_cam = pts_cam.reshape(
            img_h // resize_scale, img_w // resize_scale, 2
        )
        x_plane = pts_cam[..., 0]
        y_plane = pts_cam[..., 1]
        x_plane = cv2.resize(
            x_plane,
            dsize=(self.pe_w, self.pe_h),
            interpolation=cv2.INTER_LINEAR,
        )
        y_plane = cv2.resize(
            y_plane,
            dsize=(self.pe_w, self.pe_h),
            interpolation=cv2.INTER_LINEAR,
        )

        x_plane = x_plane.reshape(-1, 1)
        y_plane = y_plane.reshape(-1, 1)

        pts_cam = np.concatenate(
            [x_plane, y_plane, np.ones_like(x_plane, dtype=np.float32)], axis=1
        )

        extrinsic_channel = extrinsic_channel.reshape(-1, 1)

        pts_cam *= extrinsic_channel

        pts_cam = pts_cam.reshape(
            (self.pe_c // plen, self.pe_h, self.pe_w, plen)
        )
        pts_cam = pts_cam.transpose((0, 3, 1, 2))
        coordinate_map = pts_cam.reshape(self.pe_c, self.pe_h, self.pe_w)

        return coordinate_map
