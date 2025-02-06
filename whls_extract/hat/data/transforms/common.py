# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import random
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import cv2
import horizon_plugin_pytorch.nn.bgr_to_yuv444 as b2y
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, is_list_of_type, to_cuda
from hat.utils.package_helper import require_packages

try:
    from torchvision.transforms import functional as F
except ImportError:
    F = None


__all__ = [
    "ListToDict",
    "DeleteKeys",
    "RenameKeys",
    "RepeatKeys",
    "Undistortion",
    "PILToTensor",
    "TensorToNumpy",
    "AddKeys",
    "CopyKeys",
    "TaskFilterTransform",
    "Cast",
    "RandomSelectOne",
    "MultiTaskAnnoWrapper",
    "ConvertDataType",
    "BgrToYuv444",
    "BgrToYuv444V2",
    "PILToNumpy",
]


@OBJECT_REGISTRY.register
class ListToDict(object):
    """Convert list args to dict.

    Args:
        keys: keys for each object in args.
    """

    def __init__(self, keys: List[str]):
        assert is_list_of_type(
            keys, str
        ), "expect list/tuple of str, but get%s" % type(keys)
        self.keys = keys

    def __call__(self, args):
        assert len(self.keys) == len(args), "%d vs. %d" % (
            len(self.keys),
            len(args),
        )
        return {k: v for k, v in zip(self.keys, args)}


@OBJECT_REGISTRY.register
class DeleteKeys(object):
    """Delete keys in input dict.

    Args:
        keys: key list to detele

    """

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            if key in data:
                data.pop(key)
        return data


@OBJECT_REGISTRY.register
class RenameKeys(object):
    """Rename keys in input dict.

    Args:
        keys: key list to rename, in "old_name | new_name" format.

    """

    def __init__(self, keys: List[str], split: str = "|"):
        self.split = split
        self.keys = keys

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            assert self.split in key
            old_key, new_key = key.split(self.split)
            old_key = old_key.strip()
            new_key = new_key.strip()
            if old_key in data:
                data[new_key] = data.pop(old_key)
        return data


@OBJECT_REGISTRY.register
class RepeatKeys(object):
    """Repeat keys in input dict.

    Args:
        keys: key list to repeat.
        repeat_times: keys repeat times.

    """

    def __init__(self, keys: List[str], repeat_times: int):
        assert repeat_times >= 1
        self.keys = keys
        self.repeat_times = repeat_times

    def __call__(self, data):
        for repeat_key in self.keys:
            if isinstance(data[repeat_key], list):
                data[repeat_key] = data[repeat_key] * self.repeat_times
            elif isinstance(data[repeat_key], np.ndarray):
                data[repeat_key] = np.repeat(
                    data[repeat_key], self.repeat_times, axis=0
                )
            else:
                raise NotImplementedError
        return data


@OBJECT_REGISTRY.register
class Undistortion(object):  # noqa: D205,D400
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to
        undistor ``PIL Image`` or ``numpy.ndarray``.

    """

    def _to_pillow(self, data, mode="I"):
        # convert a numpy.ndarray img to pillow img
        if isinstance(data, (list, tuple)):
            return [self._to_pillow(_, mode) for _ in data]
        else:
            return Image.fromarray(data).convert(mode)

    def _undistort(self, data, intrinsic, distor_coeff):
        # scale intrinsics matrix
        intrinsic = intrinsic.copy()
        h, w = data.shape[0:2]
        intrinsic[0, :] *= w
        intrinsic[1, :] *= h

        mapx, mapy = cv2.initUndistortRectifyMap(
            intrinsic, distor_coeff, None, intrinsic, (w, h), cv2.CV_32FC1
        )
        img_ud = cv2.remap(data, mapx, mapy, interpolation=cv2.INTER_NEAREST)

        return img_ud

    def _to_undistort_array(
        self, data, intrinsic, distor_coeff, is_depth=False
    ):
        # convert a pillow img to undistor numpy.ndarray img

        if isinstance(data, (list, tuple)):
            res2 = []
            for frame in data:
                res1 = []
                for img in frame:
                    res1.append(
                        self._undistort(np.array(img), intrinsic, distor_coeff)
                    )
                res2.append(res1)
            return res2
        else:
            res = self._undistort(np.array(data), intrinsic, distor_coeff)
            return res

    def __call__(self, data):
        data["color_imgs"] = self._to_pillow(
            self._to_undistort_array(
                data["pil_imgs"], data["intrinsics"], data["distortcoef"]
            ),
            mode="RGB",
        )

        if "obj_mask" in data:
            data["obj_mask"] = self._to_pillow(
                self._to_undistort_array(
                    data["obj_mask"], data["intrinsics"], data["distortcoef"]
                ),
                mode="I",
            )

        if "front_mask" in data:
            data["front_mask"] = self._to_pillow(
                self._to_undistort_array(
                    data["front_mask"], data["intrinsics"], data["distortcoef"]
                ),
                mode="I",
            )

        return data

    def __repr__(self):
        return "Undistortion"


@OBJECT_REGISTRY.register
class PILToTensor(object):
    r"""Convert PIL Image to Tensor."""

    @require_packages("torchvision")
    def __init__(self):
        super(PILToTensor, self).__init__()

    def __call__(self, data):
        if isinstance(data, Image.Image):
            data = F.pil_to_tensor(data)
        elif isinstance(data, dict):
            for k in data:
                data[k] = self(data[k])
        elif isinstance(data, Sequence) and not isinstance(data, str):
            data = type(data)(self(d) for d in data)
        return data


@OBJECT_REGISTRY.register
class PILToNumpy(object):
    r"""Convert PIL Image to Numpy."""

    def __init__(self):
        super(PILToNumpy, self).__init__()

    def __call__(self, data):
        if isinstance(data, Image.Image):
            data = np.asarray(data, dtype=np.float32)
        elif isinstance(data, dict):
            for k in data:
                data[k] = self(data[k])
            data["layout"] = "hwc"
        elif isinstance(data, Sequence) and not isinstance(data, str):
            data = type(data)(self(d) for d in data)
        return data


@OBJECT_REGISTRY.register
class TensorToNumpy(object):
    r"""Convert tensor to numpy."""

    def __init__(self):
        super(TensorToNumpy, self).__init__()

    def __call__(self, data):
        if isinstance(data, Tensor):
            data = data.cpu().numpy().squeeze()
        if isinstance(data, dict):
            for k in data:
                data[k] = self(data[k])
        elif isinstance(data, Sequence) and not isinstance(data, str):
            data = type(data)(self(d) for d in data)
        return data


@OBJECT_REGISTRY.register
class ToCUDA(object):
    r"""
    Move Tensor to cuda device.

    Args:
        device (int, optional): The destination GPU device idx.
            Defaults to the current CUDA device.
    """

    def __init__(self, device: int = None):
        super(ToCUDA, self).__init__()
        self.device = device

    def __call__(self, data):
        return to_cuda(data, self.device)


@OBJECT_REGISTRY.register
class AddKeys(object):
    """Add new key-value in input dict.

    Frequently used when you want to add dummy keys to data dict
    but don't want to change code.

    Args:
        kv: key-value data dict.

    """

    def __init__(self, kv: Dict[str, Any]):
        assert isinstance(kv, Mapping)
        self._kv = kv

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self._kv:
            assert key not in data, f"{key} already exists in data."
            data[key] = self._kv[key]
        return data


@OBJECT_REGISTRY.register
class CopyKeys(object):
    """Copy new key in input dict.

    Frequently used when you want to cache keys to data dict
    but don't want to change code.

    Args:
        kv: key-value data dict.

    """

    def __init__(self, keys: List[str], split: str = "|"):
        self.split = split
        self.keys = keys

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            assert self.split in key
            old_key, new_key = key.split(self.split)
            old_key = old_key.strip()
            new_key = new_key.strip()
            if old_key in data:
                data[new_key] = copy.deepcopy(data[old_key])
        return data


@OBJECT_REGISTRY.register
class TaskFilterTransform(object):
    """Apply transform on assign task.

    Parameters
    ----------
    task_name: str
        Assign task name.
    """

    def __init__(self, task_name: str, transform: Callable):
        self.task_name = task_name
        self.transform = transform

    def __call__(self, data):
        if "task_name" in data:
            if (
                isinstance(data["task_name"], list)
                and data["task_name"][0] == self.task_name
            ):
                data = self.transform(data)
            if (
                isinstance(data["task_name"], str)
                and data["task_name"] == self.task_name
            ):
                data = self.transform(data)
        return data


class Cast(object):
    """Data type transformer.

    Parameters
    ----------
    dtype: str
        Transform input to dtype.
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, data):
        assert isinstance(data, np.ndarray)
        return (
            data.astype(self.dtype)
            if np.dtype(data.dtype) != np.dtype(self.dtype)
            else data
        )


def _call_ts_func(transformers, func_name, *args, **kwargs):
    for ts_i in _as_list(transformers):
        if hasattr(ts_i, func_name) and callable(getattr(ts_i, func_name)):
            getattr(ts_i, func_name)(*args, **kwargs)


@OBJECT_REGISTRY.register
class RandomSelectOne(object):
    """Select one of transforms to apply.

    Args:
        transforms: list of transformations to compose.
        p: probability of applying selected transform. Default: 0.5.
        p_trans: list of possibility of transformations.
    """

    def __init__(self, transforms: List, p: float = 0.5, p_trans: List = None):
        self.transforms = transforms
        self.num_trans = list(range(len(self.transforms)))
        self.p = p
        if p_trans is None:
            p_trans = [1 / len(transforms)] * len(transforms)
        self.p_trans = p_trans

    def __call__(self, data):
        if random.random() < self.p:
            select_num = random.choices(
                self.num_trans, weights=self.p_trans, k=1
            )[0]
            return self.transforms[select_num](data)
        return data


@OBJECT_REGISTRY.register
class MultiTaskAnnoWrapper(object):
    """Wrapper for multi-task anno generating.

    Args:
        sub_transforms: The mapping dict for task-wise transforms.
        unikeys: Keys of unique annotations in each task.
        repkeys: Keys of repeated annotations for all tasks.
    """

    def __init__(
        self,
        sub_transforms: Dict[str, Any],
        unikeys: Tuple[str] = (),
        repkeys: Tuple[str] = (),
    ):
        self.sub_transforms = sub_transforms
        self.unikeys = unikeys
        self.repkeys = repkeys

    def __call__(self, data: Mapping):
        new_data = {}
        for task, transforms in self.sub_transforms.items():
            t_data = copy.deepcopy(data)
            new_data[task] = {}
            for transform in transforms:
                t_data = transform(t_data)
            for k, v in t_data.items():
                if k in self.repkeys:
                    new_data[task][k] = v
                    new_data[k] = v
                elif k in self.unikeys:
                    new_data[task][k] = v
                elif k not in new_data:
                    new_data[k] = v
                # TODO (jiaxi.wu): data consistency from different transforms

        return new_data

    def __repr__(self):
        return "MultiTaskAnnoWrapper"


@OBJECT_REGISTRY.register
class ConvertDataType(object):
    """Convert data type.

    Args:
        convert_map: The mapping dict for to be converted data name and type.
            Only for np.ndarray and  torch.Tensor.
    """

    def __init__(
        self,
        convert_map: Optional[Dict] = None,
    ):
        self.convert_map = convert_map

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.convert_map is not None:
            for data_name, dtype in self.convert_map.items():
                if isinstance(data[data_name], np.ndarray):
                    data[data_name] = data[data_name].astype(dtype)
                elif isinstance(data[data_name], Tensor):
                    data[data_name] = data[data_name].to(dtype)
                else:
                    raise TypeError(
                        f"Unsupport convert {data_name}'s "
                        f"type {type(data[data_name])} to {dtype}"
                    )
        return data

    def __repr__(self):
        return "ConvertDataType"


@OBJECT_REGISTRY.register
class FixLengthPad(object):
    def __init__(self, keys, lengths, dims=0, **kwargs):
        self.keys = _as_list(keys)
        if not isinstance(lengths, (list, tuple)):
            lengths = [lengths] * len(keys)
        if not isinstance(dims, (list, tuple)):
            dims = [dims] * len(keys)
        self.lengths = lengths
        self.dims = dims
        self.kwargs = kwargs

    def __call__(self, data):
        for key, dim, length in zip(self.keys, self.dims, self.lengths):
            if key not in data:
                continue
            pad_length = length - data[key].shape[dim]
            ndim = len(data[key].shape)
            if isinstance(data[key], np.ndarray):
                if pad_length > 0:
                    pad_width = [[0, 0] for _ in range(ndim)]
                    pad_width[dim][1] = pad_length
                    data[key] = np.pad(data[key], pad_width, **self.kwargs)
                elif pad_length < 0:
                    data[key] = data[key].take(range(length), axis=dim)
            elif isinstance(data[key], Tensor):
                if pad_length > 0:
                    pad_width = [0 for _ in range(2 * ndim)]
                    pad_width[2 * (ndim - dim) - 1] = pad_length
                    data[key] = torch.nn.functional.pad(
                        data[key], pad_width, **self.kwargs
                    )
                elif pad_length < 0:
                    index = torch.LongTensor(list(range(length)))
                    data[key] = data[key].index_select(dim, index)
            else:
                raise TypeError(
                    f"Unsupport pad {key}'s type {type(data[key])}"
                )
        return data

    def __repr__(self):
        return "FixLengthPad"


@OBJECT_REGISTRY.register
class BgrToYuv444(object):
    """
    BgrToYuv444 is used for color format convert.

    .. note::
        Affected keys: 'img'.

    Args:
        rgb_input (bool): The input is rgb input or not.
    """

    def __init__(self, affect_key: str = "img", rgb_input: bool = False):
        self.affect_key = affect_key
        self.rgb_input = rgb_input

    def __call__(self, data):
        if isinstance(data, dict) and self.affect_key not in data:
            return data
        image = data[self.affect_key] if isinstance(data, dict) else data
        ndim = image.ndim
        if ndim == 3:
            image = torch.unsqueeze(image, 0)
        if image.dtype is not torch.uint8:
            image = image.to(dtype=torch.uint8)
        if image.shape[1] == 6:
            image1 = b2y.bgr_to_yuv444(image[:, :3], self.rgb_input).float()
            image2 = b2y.bgr_to_yuv444(image[:, 3:], self.rgb_input).float()
            image = torch.cat((image1, image2), dim=1)
        else:
            image = b2y.bgr_to_yuv444(image, self.rgb_input)
            image = image.float()
        if ndim == 3:
            image = image[0]
        if isinstance(data, dict):
            data[self.affect_key] = image
            return data
        else:
            return image


@OBJECT_REGISTRY.register
class BgrToYuv444V2(object):
    """
    BgrToYuv444V2 is used for color format convert.

    BgrToYuv444V2 implements by calling rgb2centered_yuv functions which
    has been verified to get the basically same YUV output on J5.

    .. note::
        Affected keys: 'img'.

    Args:
        rgb_input : The input is rgb input or not.
        swing: "studio" for YUV studio swing (Y: -112~107,
                U, V: -112~112).
                "full" for YUV full swing (Y, U, V: -128~127).
                default is "full"
    """

    def __init__(self, rgb_input: bool = False, swing: str = "full"):
        self.rgb_input = rgb_input
        assert swing in ["studio", "full"]
        self.swing = swing

        if self.swing == "studio":
            weight = [[66, 129, 25], [-38, -74, 112], [112, -94, -18]]
            offset = [16, 128, 128]
        else:
            weight = [[77, 150, 29], [-43, -84, 127], [127, -106, -21]]
            offset = [0, 128, 128]
        if not self.rgb_input:
            weight = [w[::-1] for w in weight]
        self.weight = (
            torch.tensor(weight).to(torch.float32).unsqueeze(2).unsqueeze(3)
        )
        self.offset = (
            torch.tensor(offset).to(torch.float32).reshape(1, 3, 1, 1)
        )
        self.bias = torch.ones(3) * 128

    def _convert_color(self, data):
        data = data.to(torch.float32)
        if self.weight.device != data.device:
            self.weight = self.weight.to(data.device)
            self.bias = self.bias.to(data.device)
            self.offset = self.offset.to(data.device)
        res = torch.nn.functional.conv2d(data, self.weight, self.bias) / 256
        res += self.offset
        return res.to(torch.int32).float()

    def __call__(self, data):
        image = data["img"] if isinstance(data, dict) else data
        ndim = image.ndim
        if ndim == 3:
            image = torch.unsqueeze(image, 0)
        if image.dtype is not torch.uint8:
            image = image.to(dtype=torch.uint8)
        if image.shape[1] == 6:
            image1 = self._convert_color(image[:, :3])
            image2 = self._convert_color(image[:, 3:])
            image = torch.cat((image1, image2), dim=1)
        else:
            image = self._convert_color(image)
        if ndim == 3:
            image = image[0]
        if isinstance(data, dict):
            data["img"] = image
            return data
        else:
            return image
