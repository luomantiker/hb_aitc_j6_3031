import os
import pickle
from typing import Callable, Optional

import cv2
import msgpack
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Dataset

from hat.data.datasets.data_packer import Packer
from hat.data.utils import decode_img
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type import PackTypeMapper
from hat.utils.pack_type.utils import get_packtype_from_path

try:
    from pyquaternion import Quaternion

except ImportError:
    pyquaternion = None
    Quaternion = None

__all__ = [
    "Occ3dNuscenesPacker",
    "Occ3dNuscenesDataset",
]


@OBJECT_REGISTRY.register
class Occ3dNuscenesDataset(Dataset):
    """Occupancy Dataset object for packed NuScenes.

    Args:
        data_path: packed dataset path.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        transforms: A function transform that takes input
            sample and its target as entry and returns a transformed version.
        pack_type: The pack type.
        pack_kwargs: Kwargs for pack type.
    """

    def __init__(
        self,
        data_path: str,
        load_interval: int = 1,
        transforms: Optional[Callable] = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
    ):
        self.root = data_path
        self.transforms = transforms

        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        try:
            self.pack_type = get_packtype_from_path(data_path)
        except NotImplementedError:
            assert pack_type is not None
            self.pack_type = PackTypeMapper(pack_type.lower())

        self.pack_file = self.pack_type(
            self.root, writable=False, **self.kwargs
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()[::load_interval]

        self.sampler = Occ3DNuscenesSampler()

    def _decode(self, pack_file, sample):
        def _decode_hook(obj):
            def _decode_bytes(obj):
                if isinstance(obj, bytes):
                    obj = obj.decode("utf-8")
                return obj

            new_obj = {}
            for k, v in obj.items():
                k = _decode_bytes(k)
                if k not in [
                    "img",
                    "voxel_semantics",
                    "mask_lidar",
                    "mask_camera",
                ]:
                    v = _decode_bytes(v)
                new_obj[k] = v
            return new_obj

        sample = pack_file.read(sample)
        sample = msgpack.unpackb(sample, object_hook=_decode_hook, raw=True)
        return sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self._decode(self.pack_file, self.samples[item])
        data = self.sampler(sample)

        if self.transforms is not None:
            data = self.transforms(data)
        return data


class Occ3DNuscenesSampler(object):
    """Occupancy Dataset object for packed occ3d-nuscenes."""

    def _load_img(self, img):
        img = decode_img(img, iscolor=cv2.IMREAD_COLOR)
        return Image.fromarray(img)

    def get_sensor_transforms(self, info, cam_name):
        """Get sensor transforms.

        Args:
            info:
            cam_name: Current CAM to be read.
        Returns:
            sensor2ego: (4, 4)
            ego2global: (4, 4)
        """
        w, x, y, z = info["cams"][cam_name]["sensor2ego_rotation"]
        # sensor to ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix
        )  # (3, 3)
        sensor2ego_tran = torch.Tensor(
            info["cams"][cam_name]["sensor2ego_translation"]
        )  # (3, )
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran

        # ego to global
        w, x, y, z = info["cams"][cam_name]["ego2global_rotation"]
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix
        )  # (3, 3)
        ego2global_tran = torch.Tensor(
            info["cams"][cam_name]["ego2global_translation"]
        )  # (3, )
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_ego2imgs(self, sensor2egos, ego2globals, intrins):
        """Get ego2imgs.

        Args:
            sensor2egos: (N_views, 4, 4)
            ego2globals: (N_views, 4, 4)
            intrins:     (N_views, 3, 3)
        Returns:
            ego2imgs: (N_views,4, 4)
        """
        sensor2keyegos = self.get_sensor2keyegos(sensor2egos, ego2globals)
        N = ego2globals.shape[0]

        ego2imgs = []
        for i in range(N):
            ego2img = self.get_ego2img(sensor2keyegos[i], intrins[i])
            ego2imgs.append(ego2img.numpy())
        return ego2imgs

    def get_sensor2keyegos(self, sensor2egos, ego2globals):
        """Calculate the transformation from adj sensor to key ego.

        Args:
            sensor2egos: (N_views, 4, 4)
            ego2globals: (N_views, 4, 4)
        Returns:
            sensor2keyegos: (N_views,4, 4)
        """
        keyego2global = ego2globals[0, ...].unsqueeze(0)  # (1, 4, 4)
        global2keyego = torch.inverse(keyego2global.double())  # (1, 4, 4)
        sensor2keyegos = (
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        )  # (N_views, 4, 4)
        sensor2keyegos = sensor2keyegos.float()
        return sensor2keyegos

    def get_ego2img(self, sensor2ego, intrin):
        """Get ego2img.

        Args:
            sensor2ego: (4, 4)
            intrin: (3,3)
        Returns:
            ego2img: (4, 4)
        """
        ego2sensor = torch.inverse(sensor2ego)

        viewpad = torch.eye(4)
        viewpad[:3, :3] = intrin
        ego2img = viewpad @ ego2sensor
        return ego2img

    def __call__(self, sample):
        data = {}
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        for cam_name in sample["cams"].keys():
            cam_data = sample["cams"][cam_name]
            intrin = torch.Tensor(cam_data["cam_intrinsic"])
            sensor2ego, ego2global = self.get_sensor_transforms(
                sample, cam_name
            )

            intrins.append(intrin)  # Camera intrinsic (3, 3)
            sensor2egos.append(sensor2ego)  # camera2ego (4, 4)
            ego2globals.append(ego2global)  # ego2global (4, 4)
            imgs.append(self._load_img(cam_data["img"]))

        sensor2egos = torch.stack(sensor2egos)  # (N_views, 4, 4)
        ego2globals = torch.stack(ego2globals)  # (N_views, 4, 4)
        intrins = torch.stack(intrins)  # (N_views, 3, 3)

        data["img"] = imgs
        data["ego2img"] = self.get_ego2imgs(sensor2egos, ego2globals, intrins)

        voxel_semantics = np.frombuffer(
            sample["voxel_semantics"], dtype=np.uint8
        ).reshape(200, 200, 16)
        mask_lidar = np.frombuffer(
            sample["mask_lidar"], dtype=np.uint8
        ).reshape(200, 200, 16)
        mask_camera = np.frombuffer(
            sample["mask_camera"], dtype=np.uint8
        ).reshape(200, 200, 16)
        data["voxel_semantics"] = torch.tensor(
            voxel_semantics, dtype=torch.uint8
        )
        data["mask_lidar"] = torch.tensor(mask_lidar, dtype=torch.uint8)
        data["mask_camera"] = torch.tensor(mask_camera, dtype=torch.uint8)

        data["layout"] = "chw"
        data["color_space"] = "rgb"
        return data


class Occ3dNuscenesImageParser(data.Dataset):
    """Parser object for packed NuScenes images.

    Args:
        root: Root directory where images are downloaded to.
        annFile: Path to json annotation file,
            kitti_train.json or kitti_eval.json. (
            For ground truth, we do not use the official txt file format data,
            but use the json file marked by the Horizon Robotics.
            )
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        load_interval: int = 1,
    ):
        super(Occ3dNuscenesImageParser, self).__init__()
        self.root = root
        self.annFile = annFile
        self.load_interval = load_interval
        self.data_infos = self.load_annotations()

    def _load_img(self, img_path):
        img_path = os.path.join(self.root, img_path)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        return cv2.imencode(".jpg", img)[1].tobytes()

    def _gen_cam_info(self, info):
        for cam_name in info["cams"].keys():
            cam_data = info["cams"][cam_name]
            img_path = cam_data["data_path"]
            cam_data["img"] = self._load_img(img_path)

            # array->list
            cam_data["sensor2lidar_rotation"] = cam_data[
                "sensor2lidar_rotation"
            ].tolist()
            cam_data["sensor2lidar_translation"] = cam_data[
                "sensor2lidar_translation"
            ].tolist()
            cam_data["cam_intrinsic"] = cam_data["cam_intrinsic"].tolist()

    def load_annotations(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        with open(self.annFile, "rb") as f:
            data = pickle.load(f)
        data_infos = sorted(data["infos"], key=lambda e: e["timestamp"])
        data_infos = data_infos[:: self.load_interval]
        return data_infos

    def _load_occgt(self, info):
        occ_gt_path = info["occ_path"]
        occ_gt_path = os.path.join(self.root, occ_gt_path, "labels.npz")

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels["semantics"]
        mask_lidar = occ_labels["mask_lidar"]
        mask_camera = occ_labels["mask_camera"]

        info["voxel_semantics"] = np.array(semantics).tobytes()
        info["mask_lidar"] = np.array(mask_lidar).tobytes()
        info["mask_camera"] = np.array(mask_camera).tobytes()
        return info

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        info = self.data_infos[index]
        self._gen_cam_info(info)
        info = self._load_occgt(info)

        # array->list
        info["gt_boxes"] = info["gt_boxes"].tolist()
        info["gt_names"] = info["gt_names"].tolist()
        info["gt_velocity"] = info["gt_velocity"].tolist()
        info["num_lidar_pts"] = info["num_lidar_pts"].tolist()
        info["num_radar_pts"] = info["num_radar_pts"].tolist()
        info["valid_flag"] = info["valid_flag"].tolist()
        try:
            info["annotations"] = [
                np.stack(info["ann_infos"][0]).tolist(),
                info["ann_infos"][1],
            ]
        except Exception:
            info["annotations"] = []
        del info["ann_infos"]
        return info


class Occ3dNuscenesPacker(Packer):
    """
    Packer is used for packing occ3d-nuscenes dataset to target format.

    Args:
        src_data_dir: The dir of original kitti2D data.
        target_data_dir: Path for LMDB file.
        annFile: Path to json annotation file,
            kitti_train.json or kitti_eval.json.
        num_workers: The num workers for reading data
            using multiprocessing.
        pack_type: The file type for packing.
        num_samples: the number of samples you want to pack. You
            will pack all the samples if num_samples is None.
    """

    def __init__(
        self,
        src_data_dir: str,
        target_data_dir: str,
        annFile: str,
        num_workers: int,
        pack_type: str,
        num_samples: Optional[int] = None,
        **kwargs,
    ):
        self.dataset = Occ3dNuscenesImageParser(
            root=src_data_dir,
            annFile=annFile,
        )
        if num_samples is None:
            num_samples = len(self.dataset)
        super(Occ3dNuscenesPacker, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def pack_data(self, idx):
        info = self.dataset[idx]
        return msgpack.packb(info)
