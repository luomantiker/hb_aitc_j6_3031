# Copyright (c) Horizon Robotics. All rights reserved.
import glob
import logging
import os
from typing import List, Optional

import cv2
import msgpack
import numpy as np
import torch.utils.data as data
from PIL import Image

from hat.data.utils import decode_img
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type import PackTypeMapper
from hat.utils.pack_type.utils import get_packtype_from_path
from .data_packer import Packer

__all__ = ["Mot17Dataset", "Mot17Packer", "Mot17FromImage"]

logger = logging.getLogger(__name__)


def encode_hook(obj):
    if isinstance(obj, np.ndarray):
        if len(obj.shape) == 3:
            img_bytes = cv2.imencode(".png", obj)[1]
            obj = np.asarray(img_bytes).astype(np.uint8).tobytes()
        elif len(obj.shape) == 2:
            obj = obj.tobytes()
        elif len(obj.shape) == 1:
            obj = obj.tobytes()
    return obj


def decode_hook(obj):
    def _decode_bytes(obj):
        if isinstance(obj, bytes):
            obj = obj.decode("utf-8")
        return obj

    new_obj = {}
    for k, v in obj.items():
        k = _decode_bytes(k)

        if k == "img":
            v = decode_img(v, iscolor=cv2.IMREAD_COLOR)
        elif k == "gt_bboxes":
            v = np.array(
                np.frombuffer(v, dtype=np.float32).reshape(-1, 4)
            ).astype(np.float32)
        elif k == "gt_classes":
            v = np.array(np.frombuffer(v, dtype=np.float32)).astype(np.float32)
        elif k == "gt_ids":
            v = np.array(np.frombuffer(v, dtype=np.float32)).astype(np.float32)
        else:
            v = _decode_bytes(v)
        new_obj[k] = v
    return new_obj


@OBJECT_REGISTRY.register
class Mot17Dataset(data.Dataset):  # noqa: D205,D400
    """
    Mot17Dataset provides the method of reading Mot17 data
    from target pack type.

    Args:
        data_path: The path of packed file.
        sampler_lengths: The length of the sequence data.
        sample_mode: The sampling mode,
            only support 'fixed_interval' or 'random_interval'.
        sample_interval: The sampling interval,
            if sample_mode is 'random_interval',
            randomly select from [1, sample_interval].
        sampler_steps: Sequence length changes according to the epoch.
        transforms: Transfroms of data before using.
        pack_type: The pack type.
        pack_kwargs: Kwargs for pack type.
        to_rgb: Whether to convert to `rgb` color_space.
    """

    def __init__(
        self,
        data_path: str,
        sampler_lengths: List[int] = (1,),
        sample_mode: str = "fixed_interval",
        sample_interval: int = 10,
        sampler_steps: List[int] = None,
        transforms: Optional[List] = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
        to_rgb: bool = True,
    ):

        self.data_path = data_path
        self.transforms = transforms
        self.to_rgb = to_rgb

        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        if pack_type is not None:
            self.pack_type = PackTypeMapper[pack_type.lower()]
        else:
            self.pack_type = get_packtype_from_path(data_path)

        self.pack_file = self.pack_type(
            self.data_path, writable=False, **self.kwargs
        )
        self.lmdb_samples = self.pack_file.get_keys()
        self.sampler_lengths = sampler_lengths
        self.sample_mode = sample_mode
        self.sample_interval = sample_interval
        self.sampler_steps = sampler_steps
        self.num_frames_per_batch = max(self.sampler_lengths)

        self.num_samples = (
            len(self.lmdb_samples)
            - (self.num_frames_per_batch - 1) * self.sample_interval
        )

        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            # Enable sampling length adjustment.
            assert len(self.sampler_lengths) > 0
            assert len(self.sampler_lengths) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
            self.num_samples = (
                len(self.lmdb_samples)
                - (self.num_frames_per_batch - 1) * self.sample_interval
            )
            self.period_idx = 0
            self.num_frames_per_batch = self.sampler_lengths[0]
            self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        logger.info(
            "set epoch: epoch {} period_idx={}".format(epoch, self.period_idx)
        )
        self.num_frames_per_batch = self.sampler_lengths[self.period_idx]

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in [
            "fixed_interval",
            "random_interval",
        ], "invalid sample mode: {}".format(self.sample_mode)
        if self.sample_mode == "fixed_interval":
            sample_interval = self.sample_interval
        elif self.sample_mode == "random_interval":
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = (
            start_idx,
            start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1,
            sample_interval,
        )
        return default_range

    def pre_continuous_frames(self, start, end, interval=1):
        frame_data_list = []
        for i in range(start, end, interval):
            single_data = self._pre_single_frame(i)
            frame_data_list.append(single_data)
        data_seq = {"frame_data_list": frame_data_list}
        return data_seq

    def _pre_single_frame(self, idx: int):
        raw_data = self.pack_file.read(self.lmdb_samples[idx])
        data = msgpack.unpackb(raw_data, object_hook=decode_hook, raw=True)

        data["img_shape"] = data["img"].shape[0:2]
        data["layout"] = "hwc"
        data["color_space"] = "rgb"
        data["seq_name"] = data["img_name"].split("/")[-3]
        if not self.to_rgb:
            cv2.cvtColor(data["img"], cv2.COLOR_RGB2BGR, data["img"])
            data["color_space"] = "bgr"

        data["ori_img"] = data["img"].copy()
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        sample_start, sample_end, sample_interval = self._get_sample_range(idx)
        data_seq = self.pre_continuous_frames(
            sample_start, sample_end, sample_interval
        )

        if self.transforms is not None:
            data_seq = self.transforms(data_seq)
        data_seq["frame_length"] = len(data_seq["frame_data_list"])

        return data_seq


@OBJECT_REGISTRY.register
class Mot17Packer(Packer):  # noqa: D205,D400
    """
    Mot17Packer is used for converting MOT17 dataset
    to target DataType format.

    Args:
        src_data_dir: The dir of original mot17 data.
        target_data_dir: Path for packed file.
        split_name: Split name of data, must be train or test.
        num_workers: Num workers for reading data using multiprocessing.
        pack_type: The file type for packing.
        num_samples: the number of samples you want to pack. You
            will pack all the samples if num_samples is None.
    """

    def __init__(
        self,
        src_data_dir: str,
        target_data_dir: str,
        split_name: str,
        num_workers: int,
        pack_type: str,
        num_samples: Optional[int] = None,
        **kwargs,
    ):
        assert split_name in [
            "train",
            "test",
        ], "split_name must be one of train and test."
        if split_name == "train":
            data_path = os.path.join(src_data_dir, "train")
        if split_name == "test":
            data_path = os.path.join(src_data_dir, "test")

        self.dataset = Mot17FromImage(
            data_path=data_path,
            sampler_lengths=[1],
            sample_mode="fixed_interval",
            sample_interval=1,
        )
        if num_samples is None:
            num_samples = len(self.dataset)
        super(Mot17Packer, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )
        self.pop_keys = ["img_shape", "layout", "color_space", "seq_name"]

    def pack_data(self, idx):
        data = self.dataset[idx]
        data_pack = data["frame_data_list"][0]

        for pop_key in self.pop_keys:
            data_pack.pop(pop_key)

        return msgpack.packb(data_pack, default=encode_hook)


@OBJECT_REGISTRY.register
class Mot17FromImage(data.Dataset):
    """Mot17FromImage which gets img data and gt from the data_path.

    Args:
        data_path: The dir of mot17 data.
        sampler_lengths: The length of the sequence data.
        sample_mode: The sampling mode,
            only support 'fixed_interval' or 'random_interval'.
        sample_interval: The sampling interval,
            if sample_mode is 'random_interval',
            randomly select from [1, sample_interval].
        sampler_steps: Sequence length changes according to the epoch.
        transforms: List of transform.
        to_rgb: Whether to convert to `rgb` color_space.
    """

    def __init__(
        self,
        data_path: str,
        sampler_lengths: List[int] = (1,),
        sample_mode: str = "fixed_interval",
        sample_interval: int = 10,
        sampler_steps: List[int] = None,
        transforms: Optional[List] = None,
        to_rgb: bool = True,
    ):
        self.data_path = data_path

        self.sampler_lengths = sampler_lengths
        self.sample_mode = sample_mode
        self.sample_interval = sample_interval
        self.sampler_steps = sampler_steps
        self.transforms = transforms
        self.to_rgb = to_rgb

        self.seqs = sorted(os.listdir(self.data_path))

        self.img_files = []
        for seq in self.seqs:
            seq_path = os.path.join(self.data_path, seq, "img1")
            images = sorted(glob.glob(seq_path + "/*.jpg"))
            self.img_files.extend(images)

        self.label_files = [
            (
                x.replace("img1", "labels_with_ids")
                .replace(".png", ".txt")
                .replace(".jpg", ".txt")
            )
            for x in self.img_files
        ]
        self.video_dict = {}
        self._register_videos()

        self.num_frames_per_batch = max(self.sampler_lengths)
        self.num_samples = (
            len(self.img_files)
            - (self.num_frames_per_batch - 1) * self.sample_interval
        )

        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            # Enable sampling length adjustment.
            assert len(self.sampler_lengths) > 0
            assert len(self.sampler_lengths) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
            self.num_samples = (
                len(self.img_files)
                - (self.sampler_lengths[-1] - 1) * self.sample_interval
            )
            self.period_idx = 0
            self.num_frames_per_batch = self.sampler_lengths[0]
            self.current_epoch = 0

    def __len__(self):
        return self.num_samples

    def _register_videos(self):
        for label_name in self.label_files:
            video_name = label_name[len(self.data_path) + 1 :].split("/")[0]
            if video_name not in self.video_dict:
                logger.info(
                    "register {}-th video: {} ".format(
                        len(self.video_dict) + 1, video_name
                    )
                )
                self.video_dict[video_name] = len(self.video_dict)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        logger.info(
            "set epoch: epoch {} period_idx={}".format(epoch, self.period_idx)
        )
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in [
            "fixed_interval",
            "random_interval",
        ], "invalid sample mode: {}".format(self.sample_mode)
        if self.sample_mode == "fixed_interval":
            sample_interval = self.sample_interval
        elif self.sample_mode == "random_interval":
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = (
            start_idx,
            start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1,
            sample_interval,
        )
        return default_range

    def pre_continuous_frames(self, start, end, interval=1):
        frame_data_list = []
        for i in range(start, end, interval):
            single_data = self._pre_single_frame(i)
            frame_data_list.append(single_data)
        data_seq = {"frame_data_list": frame_data_list}
        return data_seq

    def _pre_single_frame(self, idx: int):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        img = Image.open(img_path)
        single_data = {}
        w, h = img._size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(
            img_path, w, h
        )
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(
                os.path.join(label_path), dtype=np.float32
            ).reshape(-1, 6)

            # normalized cewh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4] / 2)
            labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5] / 2)
            labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4] / 2)
            labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5] / 2)
        else:
            raise ValueError(
                "invalid label path: {}".format(os.path.join(label_path))
            )

        # video_name = '/'.join(label_path.split('/')[:-1])

        video_name = label_path[len(self.data_path) + 1 :].split("/")[0]

        img_name = img_path[len(self.data_path) + 1 :]
        obj_idx_offset = self.video_dict[video_name] * 1000000

        single_data["img"] = np.array(img)
        single_data["gt_bboxes"] = []
        single_data["gt_classes"] = []
        single_data["gt_ids"] = []
        single_data["image_id"] = idx
        single_data["img_shape"] = (h, w)
        single_data["color_space"] = "rgb"
        single_data["layout"] = "hwc"
        single_data["img_name"] = img_name
        single_data["seq_name"] = img_name.split("/")[-3]

        for label in labels:
            single_data["gt_bboxes"].append(label[2:6].tolist())
            single_data["gt_classes"].append(label[0])
            obj_id = label[1] + obj_idx_offset if label[1] >= 0 else label[1]
            single_data["gt_ids"].append(obj_id)  # relative id

        if len(single_data["gt_bboxes"]) > 0:
            single_data["gt_bboxes"] = np.array(
                single_data["gt_bboxes"], dtype=np.float32
            )
            single_data["gt_classes"] = np.array(
                single_data["gt_classes"], dtype=np.float32
            )
            single_data["gt_ids"] = np.array(
                single_data["gt_ids"], dtype=np.float32
            )
        else:
            single_data["gt_bboxes"] = np.zeros((0, 4), dtype=np.float32)
            single_data["gt_classes"] = np.zeros((0, 1), dtype=np.float32)
            single_data["gt_ids"] = np.zeros((0, 1), dtype=np.float32)

        if not self.to_rgb:
            cv2.cvtColor(
                single_data["img"], cv2.COLOR_RGB2BGR, single_data["img"]
            )
            single_data["color_space"] = "bgr"

        return single_data

    def __getitem__(self, idx: int):
        sample_start, sample_end, sample_interval = self._get_sample_range(idx)
        data_seq = self.pre_continuous_frames(
            sample_start, sample_end, sample_interval
        )
        data_seq["frame_length"] = len(data_seq["frame_data_list"])
        if self.transforms is not None:
            data_seq = self.transforms(data_seq)
        return data_seq
