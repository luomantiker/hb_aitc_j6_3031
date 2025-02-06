# Copyright (c) Horizon Robotics. All rights reserved.
import collections
import logging
import re
from typing import Any, Callable, Dict, List, Mapping, Sequence, Union

import numpy as np
import torch
from PIL import Image

try:
    from torch._six import string_classes
except ImportError:
    string_classes = (str, bytes)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

from hat.core.virtual_camera import CameraBase
from hat.data.metas import img_metas
from hat.registry import OBJECT_REGISTRY

np_str_obj_array_pattern = re.compile(r"[SaUO]")

__all__ = [
    "collate_2d",
    "collate_2d_2pe",
    "collate_3d",
    "collate_psd",
    "CocktailCollate",
    "collate_lidar",
    "collate_real3d",
    "collate_2d_with_diff_im_hw",
    "collate_seq_with_diff_im_hw",
    "collate_nlu_with_pad",
    "collate_2d_replace_empty",
    "default_collate_v2",
    "collate_2d_pad",
    "collate_mot_seq",
    "collate_2d_cat",
    "collate_lidar3d",
    "collate_mmfusion_3d",
    "collate_argoverse",
    "collate_disp_cat",
    "collate_e2e_dynamic",
    "collate_gaze_seq",
]


def default_collate_v2(batch):
    """Entend torch.utils.data.default_collate.

    It can handle classes that cannot be converted to torch.tensor and \
    convert them to lists instead of reporting errors directly. \
    Examples: \
        batch=[dict(input_x=A), dict(input_x=B)] \
            where input_x can not be converted to torch.Tensor \
        output=dict(input_x=[A, B]).
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    "default_collate: batch must contain tensors, numpy arrays"
                    ", numbers, dicts or lists; found {}".format(elem.dtype)
                )

            return default_collate_v2([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {
            key: default_collate_v2([d[key] for d in batch]) for key in elem
        }
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(
            *(default_collate_v2(samples) for samples in zip(*batch))
        )
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                "each element in list of batch should be of equal size"
            )
        transposed = zip(*batch)
        return [default_collate_v2(samples) for samples in transposed]
    return batch


def collate_psd(batch: List[Any]):
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in parking slot detection(psd) task. \
        For collating data with inconsistent shapes.

    Args:
        batch: list of data.
    """
    elem = batch[0]
    # these key-value will skip default_collate
    list_key = ["img", "ori_img", "img_name", "label", "layout", "color_space"]
    if not isinstance(elem, dict):
        return default_collate(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return_data = {}
        for key in elem:
            if key in list_key:
                collate_data = [d[key] for d in batch]
                if key == "img":
                    collate_data = torch.stack(collate_data, dim=0)
            else:
                collate_data = default_collate([d[key] for d in batch])

            return_data.update({key: collate_data})
        return return_data


def collate_2d(
    batch: List[Any], verbose: bool = False
) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in 2d task, for collating data with inconsistent shapes.

    Args:
        batch: list of data.
    """

    elem = batch[0]
    # these key-value will skip default_collate
    list_key = [
        "gt_classes",
        "gt_bboxes",
        "ig_bboxes",
        "gt_tanalphas",
        "gt_difficult",
        "gt_labels",
        "ori_img",
        "resized_ori_img",
        "img_id",
        "layout",
        "img_shape",
        "resized_shape",
        "color_space",
        "classes",
        "bboxes",
        "crop_offset",
        "before_crop_shape",
        "crop_roi",
        "structure",
        "dgain",
        "rg_gain",
        "bg_gain",
        "bit_nums_upper",
        "bit_nums_lower",
        "channels",
        "cur_pattern",
        "raw_pattern",
        "gt_lines",
        "ori_gt_lines",
        "before_pad_shape",
        "attr_labels",
        "gt_labels_3d",
        "gt_bboxes_3d",
        "centers2d",
        "depths",
        "filename",
        "ori_shape",
        "cam2img",
        "ego2global_translation",
        "ego2global_rotation",
        "sensor2ego_translation",
        "sensor2ego_rotation",
        "token",
        "scale",
        "padded_img",
        "crop_bbox",
        "transform_meta",
        "camera_info",
        "gt_invasion_status",
        "gt_beside_valid",
        "gt_invasion_scale",
    ]
    if not isinstance(elem, dict):
        return default_collate(batch)
    elif isinstance(elem, Mapping):
        return_data = {}
        unexpected_keys = []
        for key in elem:
            if key in list_key:
                collate_data = [d[key] for d in batch]
            else:
                collate_data = default_collate([d[key] for d in batch])
            if key not in img_metas:
                unexpected_keys.append(key)

            return_data.update({key: collate_data})
        if len(unexpected_keys) > 0 and verbose:
            logging.warning(
                f"{unexpected_keys} appear in keys of dataset."
                "Please check whether it is an image task and "
                "meets expectations."
            )
        return return_data


def collate_2d_2pe(batch: List[Any]) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in 2d task, for collating data with inconsistent shapes.

    Args:
        batch: list of data.
    """
    for d in batch:
        try:
            elem = d[0]
            break
        except:  # noqa [E722]
            continue

    # these key-value will skip default_collate
    list_key = [
        "gt_classes",
        "gt_bboxes",
        "ig_bboxes",
        "gt_difficult",
        "gt_labels",
        "ori_img",
        "resized_ori_img",
        "img_id",
        "layout",
        "img_shape",
        "resized_shape",
        "color_space",
        "classes",
        "bboxes",
        "crop_offset",
        "before_crop_shape",
        "crop_roi",
        "structure",
        "bit_nums_upper",
        "bit_nums_lower",
        "channels",
        "cur_pattern",
        "raw_pattern",
        "gt_lines",
        "ori_gt_lines",
        "before_pad_shape",
        "obj_id",
    ]
    if not isinstance(elem, dict):
        return default_collate(batch)
    elif isinstance(elem, Mapping):
        return_data = {}
        unexpected_keys = []
        for key in elem:
            if key in list_key:
                collate_data = []
                for d in batch:
                    for ins in d:
                        collate_data.append(ins[key])
            else:
                collate_data = []
                for d in batch:
                    for ins in d:
                        collate_data.append(ins[key])
                collate_data = default_collate(collate_data)
            if key not in img_metas:
                unexpected_keys.append(key)

            return_data.update({key: collate_data})
        if len(unexpected_keys) > 0:
            logging.warning(
                f"{unexpected_keys} appear in keys of dataset."
                "Please check whether it is an image task and "
                "meets expectations."
            )
        return return_data


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def pad_batch_img(imgs: List[torch.Tensor]):
    # imgs[0] shape should be c * h * w
    max_size = _max_by_axis([list(img.shape) for img in imgs])
    batch_shape = [len(imgs)] + max_size
    dtype = imgs[0].dtype
    device = imgs[0].device
    pad_imgs = torch.zeros(batch_shape, dtype=dtype, device=device)
    for img, pad_img in zip(imgs, pad_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return pad_imgs


def expand_ori_data(batch: List[Dict[str, List[torch.Tensor]]], key: str):
    results = []
    for d in batch:
        for imgs in d[key]:
            results.extend(imgs)
    return results


def collate_2d_pad(batch: List[Any]) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in 2d task, for collating data with inconsistent shapes.
    Images with different shapes will pad to max shapes by axis.

    Args:
        batch: list of data.
    """

    elem = batch[0]
    # these key-value will skip default_collate
    list_key = [
        "gt_classes",
        "gt_bboxes",
        "ig_bboxes",
        "gt_difficult",
        "gt_labels",
        "ori_img",
        "resized_ori_img",
        "img_id",
        "layout",
        "img_shape",
        "resized_shape",
        "color_space",
        "classes",
        "bboxes",
        "crop_offset",
        "before_crop_shape",
        "crop_roi",
        "structure",
        "bit_nums_upper",
        "bit_nums_lower",
        "channels",
        "cur_pattern",
        "raw_pattern",
        "gt_lines",
        "ori_gt_lines",
        "before_pad_shape",
    ]
    if not isinstance(elem, dict):
        return pad_batch_img(batch)
    elif isinstance(elem, Mapping):
        return_data = {}
        for key in elem:
            if key in list_key:
                collate_data = [d[key] for d in batch if key in d]
            elif key == "img":
                collate_data = pad_batch_img([d[key] for d in batch])
                pad_img_shape = [collate_data[0].shape[-2:] for _ in batch]
                return_data.update({"batch_input_shape": pad_img_shape})
            else:
                collate_data = default_collate(
                    [d[key] for d in batch if key in d]
                )

            return_data.update({key: collate_data})
        return return_data


def collate_3d(
    batch_data: List[Any],
    ignore_keys: Sequence[str] = (),
    stack_keys: Sequence[str] = (),
    custom_handle_keys: Sequence[str] = (),
    custom_func: Callable = None,
    truncation_len: int = 4,
):
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in bev task. \
    * If output tensor from dataset shape is (n,c,h,w),concat on \
        aixs 0 directly. \
    * If output tensor from dataset shape is (c,h,w),expand_dim on \
        axis 0 and concat.

    Args:
        batch: list of data.
        ignore_keys: ignore keys in collate_3d
        stack_keys: 0-dim stack keys in collate_3d
        custom_handle_keys: apply custom_func to keys in data
        custom_func: function to handle batch data
        truncation_len: not to unsqueeze when truncation_len
    """
    if isinstance(batch_data[0], dict):
        result = {}

        all_keys = []
        for batch in batch_data:
            for key in batch.keys():
                if key not in all_keys:
                    all_keys.append(key)

        for key in all_keys:
            if key in ignore_keys:
                result[key] = [d[key] for d in batch_data if key in d]
            elif key in stack_keys:
                result[key] = torch.cat(
                    [d[key] for d in batch_data if key in d], dim=0
                )
            elif key in custom_handle_keys:
                assert custom_func is not None
                result[key] = custom_func(batch_data, key)
            else:
                result[key] = collate_3d(
                    [d[key] for d in batch_data if key in d], ignore_keys
                )
        return result
    elif isinstance(batch_data[0], (list, tuple)):
        return [collate_3d(data, ignore_keys) for data in zip(*batch_data)]
    elif isinstance(batch_data[0], torch.Tensor):
        if len(batch_data[0].shape) == truncation_len:
            return torch.cat(batch_data, dim=0)
        else:
            batch_data = [torch.unsqueeze(d, dim=0) for d in batch_data]
            return torch.cat(batch_data, dim=0)
    elif isinstance(
        batch_data[0],
        (str, bytes, int, float, CameraBase, Image.Image, np.ndarray),
    ):
        return batch_data
    else:
        raise TypeError


def collate_real3d(batch: List[Any]) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in real3d task, for collating data with inconsistent shapes.

    Args:
        batch: list of data.
    """

    elem = batch[0]
    # these key-value will skip default_collate
    list_key = [
        "annotations",
        "camera",
        "source_cam",
        "virtual_cam",
        "warp_cam",
        "view",
        "gt_bboxes",
        "gt_classes",
        "gt_bboxes_3d",
        "gt_classes_3d",
        "centers2d_prj",
        "depths",
        "org_image",
        "gt_pcl",
    ]
    if not isinstance(elem, dict):
        return default_collate(batch)
    elif isinstance(elem, Mapping):
        return_data = {}
        for key in elem:
            if key == "img":
                try:
                    collate_data = default_collate([d[key] for d in batch])
                except RuntimeError:  # image size not equal
                    collate_data = [d[key] for d in batch]
            elif key in list_key:
                collate_data = [d[key] for d in batch]
            else:
                try:
                    collate_data = default_collate([d[key] for d in batch])
                except RuntimeError:
                    raise RuntimeError

            return_data.update({key: collate_data})
        return return_data


def collate_lidar(batch_list: List[Any]) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in rad task, for collating data with inconsistent shapes.
    Rad(Realtime and Accurate 3D Object Detection).

    First converts List[Dict[str, ...] or List[Dict]] to
    Dict[str, List], then process values whoses keys are
    related to training.

    Args:
        batch: list of data.
    """
    example_merged = collections.defaultdict(list)

    # 将batch_list中每个样本中相同的键值元素放置到一个list中.
    for example in batch_list:
        if isinstance(example, list):
            for subexample in example:
                for k, v in subexample.items():
                    example_merged[k].append(v)
        else:
            for k, v in example.items():
                example_merged[k].append(v)

    # 按照Key的不同，重新编排整理.
    # 每个key的elems功能不同，处理方式存在差异.
    batch_size = len(example_merged["metadata"])
    ret = {}
    for key, elems in example_merged.items():
        # 下述key，简单拼接并丈量化.
        if key in [
            "voxels",
            "voxels_raw",
            "num_points",
            "num_points_raw",
            "num_gt",
            "voxel_labels",
            "num_voxels",
            "num_voxels_raw",
            "points_num",
            "points_num_raw",
            "voxels_pillars",
            "voxels_pillars_raw",
            "num_points_pillars",
            "num_points_pillars_raw",
            "num_voxels_pillars",
            "num_voxels_pillars_raw",
            "pose",
            "whole_equation",
        ]:
            ret[key] = torch.tensor(np.concatenate(elems, axis=0))

        # 下述key,将每个batch的所有elem拼接并张量化.
        # elems:List[List[array,...],...] -> List[tensor,tensor,...]
        elif key in [
            "sweep_voxels",
            "sweep_num_points",
            "sweep_num_voxels",
            "sweep_voxels_pillars",
            "sweep_num_points_pillars",
            "sweep_num_voxels_pillars",
        ]:
            batch_collated_list = []
            # idx为batch的索引.
            for idx in range(len(elems[0])):
                # 每个batch所有的elem放入到一个list中.
                batch_elem = [elem[idx] for elem in elems]
                batch_collated_list.append(
                    torch.tensor(np.concatenate(batch_elem, axis=0))
                )
            ret[key] = batch_collated_list

        # 下述key,将每个batch的所有elem拼接并张量化.
        # elems::List[List[array,...],...] -> List[tensor,tensor,...]
        elif key == "gt_boxes":
            task_max_gts = []
            # gt boxes为多个任务的监督列表.
            # task_id 为每个任务的gt_boxes.
            # 找到每个任务batch中最多gt_box的数量.
            for task_id in range(len(elems[0])):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, len(elems[k][task_id]))
                task_max_gts.append(max_gt)
            res = []
            # 构建每个任务的监督array.
            for idx, max_gt in enumerate(task_max_gts):
                batch_task_gt_boxes3d = np.zeros((batch_size, max_gt, 7))
                for i in range(batch_size):
                    len_elem = len(elems[i][idx])
                    # 对每个gt_box赋索引值.
                    batch_task_gt_boxes3d[i, :len_elem, :] = elems[i][idx]
                res.append(batch_task_gt_boxes3d)
            ret[key] = res

        elif key in ["metadata", "parsing"]:
            ret[key] = elems

        # 下述key,将每个elem在放回到所属batch的list中拼接并张量化.
        # elems::List[List[array,...],...] -> dict{str:tensor}
        elif key == "calib":
            ret[key] = {}
            # 将每个elem在放回到所属batch的list中.
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            # 拼接并张量化
            for k1, v1 in ret[key].items():
                ret[key][k1] = torch.tensor(np.stack(v1, axis=0))

        # 下述key,每个elem都进行pad后拼接并张量化.
        # elems::List[List[array,...],...] -> Tensor
        elif key in [
            "coordinates",
            "coordinates_raw",
            "points",
            "points_raw",
            "sample_points",
            "coordinates_pillars",
            "coordinates_pillars_raw",
        ]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = torch.tensor(np.concatenate(coors, axis=0))

        # 下述key,将每个batch的每个elem pad后拼接并张量化.
        # elems::List[List[array,...],...] -> List[tensor,tensor,...]
        elif key in [
            "sweep_coordinates",
            "sweep_points",
            "sweep_coordinates_pillars",
        ]:
            batch_collated_list = []
            for idx in range(len(elems[0])):
                batch_elem = [elem[idx] for elem in elems]
                coors = []
                for i, coor in enumerate(batch_elem):
                    coor_pad = np.pad(
                        coor,
                        ((0, 0), (1, 0)),
                        mode="constant",
                        constant_values=i,
                    )
                    coors.append(coor_pad)

                batch_collated_list.append(
                    torch.tensor(np.concatenate(coors, axis=0))
                )
            ret[key] = batch_collated_list

        # 下述key,将每个elem在放回到所属batch的list中拼接并张量化.
        # elems::List[List[array,...],...] -> List[tensor,tensor,...]
        elif (
            key
            in [
                "reg_targets",
                "reg_weights",
                "labels",
                "hm",
                "anno_box",
                "ind",
                "mask",
                "cat",
                "seg_hm",
                "kps_hm",
                "gt_boxes_tasks",
                "seg_loss_mask",
                "seg_hm",
                "map_seg_hm",
            ]
            or "hm_d" in key
        ):
            ret[key] = collections.defaultdict(list)
            res = []
            for elem in elems:
                # 将每个元素放入到所属batch的list中.
                for idx, ele in enumerate(elem):
                    ret[key][str(idx)].append(torch.tensor(ele))
            # 将每个batch的elem list进行stack后放入key的list.
            for _, vv in ret[key].items():
                res.append(torch.stack(vv))
            ret[key] = res

        # 下述key,每个elem都stack并张量化.
        # elems::List[List[array,...],...] -> List[tensor,tensor,...]
        elif key in ["gt_boxes_and_cls", "feature_trans"]:
            ret[key] = torch.tensor(np.stack(elems, axis=0))
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret


@OBJECT_REGISTRY.register
class CocktailCollate(object):
    """CocktailCollate.

    鸡尾酒（多模）算法批量数据collate的Callable类.
    默认需要处理的是 dict 类型数据的列表。

    首先，将List[Dict[str, ...]]转换成Dict[str, List]
    然后，对dict中的 'images', 'audio', 'label' 跟训练相关的数据。
    进行 pad_sequence 操作。对 'tokens' 直接跳过。
    其他的key使用default_collate


    Args:
        ignore_id: 被忽略的标签ID, 默认使用wenet中的-1即-1.
                   处理标签数据时，使用-1的值作为padding值
        batch_first: 处理批量数据时, batch 的维度是否在第1位(数组编号0).
                     如果batch_first是True, 数组为 BxTx*
                     如果batch_first是False, 数组为 TxBx*
        mode: 以什么模式进行 collates. train, calibration
    """

    def __init__(
        self,
        ignore_id: int = -1,
        batch_first: bool = True,
        mode: str = "train",
    ):
        self.ignore_id = ignore_id
        self.batch_first = batch_first
        assert mode in ["train", "calibration"]
        self.mode = mode

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        # 整理需要获取keys
        batch = [b for b in batch if b is not None]
        keys_list = [tuple(sorted(elem.keys())) for elem in batch]
        keys = set(keys_list)
        assert len(keys) == 1, f"keys in should be same, but get {keys_list}"
        keys = keys.pop()
        # 按照key重新整理
        batch = {key: [elem[key] for elem in batch] for key in keys}
        for key, batch_ in batch.items():
            logging.debug(f"{key}, {type(batch_)}")
            if key in ["images", "audio"]:
                if self.mode == "train":
                    batch_ = pad_sequence(
                        batch_,
                        batch_first=self.batch_first,
                        padding_value=0,
                    )
                else:
                    seq_len = min(elem.size(0) for elem in batch_)
                    batch_ = torch.stack(
                        [elem[:seq_len] for elem in batch_],
                        dim=(0 if self.batch_first else 1),
                    )
            elif key == "label":
                if self.mode == "train":
                    batch_ = pad_sequence(
                        batch_,
                        batch_first=self.batch_first,
                        padding_value=self.ignore_id,
                    )
                else:
                    seq_len = min(elem.size(0) for elem in batch_)
                    batch_ = torch.stack(
                        [elem[:seq_len] for elem in batch_],
                        dim=(0 if self.batch_first else 1),
                    )
            elif key in ["tokens", "text"]:
                pass
            else:
                if self.mode != "train" and key in [
                    "label_length",
                    "audio_lens",
                    "images_lens",
                ]:
                    min_lengths = min(elem for elem in batch_)
                    batch_ = [min_lengths for _ in batch_]
                batch_ = default_collate(batch_)

            batch[key] = batch_
        return batch


def collate_2d_with_diff_im_hw(batch: List[Any]) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in 2d task, for collating data with different
    image heights or widths. These inconsisten images will
    be vstacked in batch transform.

    Args:
        batch: list of data.
    """

    elem = batch[0]
    # these key-value will skip default_collate
    list_key = [
        "gt_classes",
        "gt_bboxes",
        "gt_tanalphas",
        "gt_difficult",
        "gt_labels",
        "ori_img",
        "resized_ori_img",
        "img_id",
        "layout",
        "img_shape",
        "resized_shape",
        "color_space",
        "classes",
        "bboxes",
        "crop_offset",
        "before_crop_shape",
        "crop_roi",
        "structure",
    ]
    if not isinstance(elem, dict):
        return default_collate(batch)
    elif isinstance(elem, Mapping):
        return_data = {}
        unexpected_keys = []
        for key in elem:
            # skip diff shape img collate,
            # they will be padding to batch data on batch transform
            if key in ["img", "gt_seg"]:
                collate_data = [default_collate(d[key]) for d in batch]
            elif key in list_key:
                collate_data = [d[key] for d in batch]
            else:
                collate_data = default_collate([d[key] for d in batch])
            if (key not in img_metas) and (not isinstance(elem[key], dict)):
                unexpected_keys.append(key)
            return_data.update({key: collate_data})
        if len(unexpected_keys) > 0:
            logging.warning(
                f"{unexpected_keys} appear in keys of dataset."
                f"Please check whether it is an image task and meets expectations."  # noqa
            )
        return return_data


def collate_2d_replace_empty(
    batch: List[Any],
    prob: float = 0.0,
) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    This function also replaces those detection samples that have no positive
    training targets with eligible ones.This can improve training effectiveness
    and efficiency when there are many images having no training targets in the
    dataset.

    Args:
        batch: list of data.
        prob: the probability of conducting empty-gt image replacement.
    """

    elem = batch[0]
    # do the replacement of no-gt images with a probability of prob, so that
    # there can be no-gt images in the batch to avoid false positives.
    # we only do this for detection data
    if (
        np.random.uniform(low=0.0, high=1.0) < prob
        and "gt_bboxes" in elem
        and "gt_classes" in elem
    ):
        batch_size = len(batch)

        # find which sample has gt and which has no gt
        has_gt_list = []
        no_gt_list = []
        for i in range(batch_size):
            gt_classes = batch[i]["gt_classes"]
            if (gt_classes >= 0).any():
                has_gt_list.append(i)
            else:
                no_gt_list.append(i)
        # replace those no-gt samples with one has-gt sample randomly chosen
        # from existing has-gt samples. if there is no any gt in this batch, we
        # can do nothing
        if has_gt_list:
            for no_gt_idx in no_gt_list:
                has_gt_idx = np.random.choice(has_gt_list)
                batch[no_gt_idx] = batch[has_gt_idx]

    # these key-value will skip default_collate
    list_key = [
        "gt_classes",
        "gt_bboxes",
        "gt_tanalphas",
        "gt_difficult",
        "gt_labels",
        "ig_bboxes",
        "ori_img",
        "resized_ori_img",
        "img_id",
        "layout",
        "img_shape",
        "img_height",
        "img_width",
        "pad_shape",
        "keep_ratio",
        "scale_idx",
        "resized_shape",
        "color_space",
        "classes",
        "bboxes",
        "crop_offset",
        "before_crop_shape",
        "crop_roi",
        "structure",
    ]
    if not isinstance(elem, dict):
        return default_collate(batch)
    elif isinstance(elem, Mapping):
        return_data = {}
        unexpected_keys = []
        for key in elem:
            if key in list_key:
                collate_data = [d[key] for d in batch]
            else:
                collate_data = default_collate([d[key] for d in batch])
            if key not in img_metas:
                unexpected_keys.append(key)

            return_data.update({key: collate_data})

        return return_data


def collate_seq_with_diff_im_hw(
    batch: List[Dict],
) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in sequence task, for collating data with different
    image heights or widths. These inconsisten images will
    be vstacked in batch transform.

    Args:
        batch: list of data.
    """

    elem = batch[0]["frame_data_list"][0]
    # these key-value will skip default_collate
    list_key = [
        "gt_classes",
        "gt_bboxes",
        "gt_tanalphas",
        "gt_difficult",
        "gt_labels",
        "ori_img",
        "resized_ori_img",
        "img_id",
        "layout",
        "img_shape",
        "resized_shape",
        "color_space",
        "classes",
        "bboxes",
        "crop_offset",
        "before_crop_shape",
        "crop_roi",
        "structure",
    ]

    return_data = {}
    unexpected_keys = []
    for key in elem:
        # skip diff shape img collate,
        # they will be padding to batch data on batch transform
        if key in ["img", "gt_seg"]:
            collate_data = []
            for one_seq_data in batch:
                for frame in one_seq_data["frame_data_list"]:
                    collate_data.append(default_collate(frame[key]))
        elif key in list_key:
            collate_data = []
            for one_seq_data in batch:
                for frame in one_seq_data["frame_data_list"]:
                    collate_data.append(frame[key])
        else:
            collate_data = []
            for one_seq_data in batch:
                for frame in one_seq_data["frame_data_list"]:
                    collate_data.append(frame[key])
            collate_data = default_collate(collate_data)
        if (key not in img_metas) and (not isinstance(elem[key], dict)):
            unexpected_keys.append(key)
        return_data.update({key: collate_data})
    seq_len = [d["frame_length"] for d in batch]
    if len(unexpected_keys) > 0:
        logging.warning(
            f"{unexpected_keys} appear in keys of dataset."
            f"Please check whether it is an image task and meets expectations."
        )
    return_data.update(
        num_seq=default_collate([len(seq_len)]),
        seq_len=default_collate(seq_len),
    )
    return return_data


def collate_nlu_with_pad(
    batch_dic: List[Dict], total_sequence_length: int = 30
) -> Dict:
    """Collate nlu func for dataloader."""
    batch_len = len(batch_dic)
    batch_query, batch_ids, batch_pad_masks = [], [], []
    batch_domain_labs, batch_intent_labs, batch_ner_labs = ([], [], [])
    batch_sample_weights = []
    throw_tokens = 2
    for i in range(batch_len):
        cur_dic = batch_dic[i]
        batch_query.append(cur_dic["query"])
        pad_length = (
            total_sequence_length - len(cur_dic["query_id"])
            if len(cur_dic["query_id"]) < total_sequence_length
            else 0
        )
        batch_ids.append(
            torch.tensor(cur_dic["query_id"] + [0 for _ in range(pad_length)])
        )
        batch_ner_labs.append(
            torch.tensor(cur_dic["ner"] + [0 for _ in range(pad_length)])
        )
        batch_pad_masks.append(
            torch.tensor(
                [True for _ in range(len(cur_dic["query_id"]) - throw_tokens)]
                + [False for _ in range(pad_length)]
            )
        )
        batch_domain_labs.append(cur_dic["domain"])
        batch_intent_labs.append(cur_dic["intent"])
        batch_sample_weights.append(cur_dic["sample_weight"])

    res = {
        "batch_query": batch_query,
        "batch_ids": pad_sequence(batch_ids, batch_first=True),
        "batch_pad_masks": pad_sequence(batch_pad_masks, batch_first=True),
        "batch_domain_labs": torch.tensor(batch_domain_labs),
        "batch_intent_labs": torch.tensor(batch_intent_labs),
        "batch_ner_labs": pad_sequence(batch_ner_labs, batch_first=True),
        "batch_sample_weights": torch.tensor(batch_sample_weights),
    }
    return res


def collate_mot_seq(
    batch: List[Dict],
) -> Union[torch.Tensor, Dict]:
    """Collate for mot seq data.

    Args:
        batch: list of data.
    """
    return_data = {}
    frames_data = []
    list_key = [
        "gt_classes",
        "gt_bboxes",
        "gt_difficult",
        "gt_labels",
        "ori_img",
        "resized_ori_img",
        "img_id",
        "layout",
        "img_shape",
        "resized_shape",
        "color_space",
        "classes",
        "bboxes",
        "crop_offset",
        "before_crop_shape",
        "crop_roi",
        "structure",
        "gt_ids",
        "img",
        "seq_name",
    ]
    for one_seq_data in batch:
        elem = one_seq_data["frame_data_list"][0]
        seq_collate_data = {}
        for key in elem:
            if key in list_key:
                if key == "img":
                    collate_data = []
                    for d in one_seq_data["frame_data_list"]:
                        if isinstance(d[key], np.ndarray):
                            collate_data.append(np.expand_dims(d[key], axis=0))
                        elif isinstance(d[key], torch.Tensor):
                            collate_data.append(d[key].unsqueeze(0))
                        else:
                            raise ValueError(
                                f"Unsupport image datatype: {type(d[key])}"
                            )
                else:
                    collate_data = [
                        d[key] for d in one_seq_data["frame_data_list"]
                    ]
            else:
                collate_data = default_collate(
                    [d[key] for d in one_seq_data["frame_data_list"]]
                )
            seq_collate_data.update({key: collate_data})
        frames_data.append(seq_collate_data)
    return_data.update({"frame_data_list": frames_data})

    return return_data


def collate_lidar3d(batch_list: List[Any]) -> Union[torch.Tensor, Dict]:
    example_merged = collections.defaultdict(list)
    for example in batch_list:
        if isinstance(example, list):
            for sub_example in example:
                for k, v in sub_example.items():
                    example_merged[k].append(v)
        else:
            for k, v in example.items():
                example_merged[k].append(v)

    ret = {}

    for key, elems in example_merged.items():
        if key in [
            "voxels",
            "num_points",
            "num_gt",
            "num_voxels",
        ]:
            ret[key] = torch.tensor(np.concatenate(elems, axis=0))
        elif key in ["annotations", "metadata"]:
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = torch.tensor(np.stack(v1, axis=0))
        elif key in ["coordinates"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = torch.tensor(
                np.concatenate(coors, axis=0), dtype=torch.int64
            )

        elif key in ["points", "gt_boxes", "gt_classess"]:
            points_lst = [torch.tensor(points) for points in elems]
            ret[key] = points_lst
        elif key in ["gt_seg_labels", "gt_seg_mask"]:
            ret[key] = torch.tensor(np.stack(elems, axis=0))
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret


def collate_2d_cat(batch: List[Any]) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in 2d task, for collating data with the first dimension inconsistent.
    If the data shape is (n,c,h,w), concat on aixs 0 directly.

    Args:
        batch: list of data.
    """

    elem = batch[0]
    L = [list(elem.keys()) for elem in batch]
    keys_list = list(set(L[0]).union(*L[1:]))
    goal_keys = list(filter(lambda x: "detection" in x, keys_list))
    # these key-value will skip default_collate
    list_key = [
        "gt_classes",
        "gt_bboxes",
        "ig_bboxes",
        "gt_difficult",
        "gt_labels",
        "ori_img",
        "resized_ori_img",
        "img_id",
        "layout",
        "img_shape",
        "resized_shape",
        "color_space",
        "classes",
        "bboxes",
        "crop_offset",
        "before_crop_shape",
        "crop_roi",
        "structure",
        "bit_nums_upper",
        "bit_nums_lower",
        "channels",
        "cur_pattern",
        "raw_pattern",
        "gt_lines",
        "ori_gt_lines",
        "before_pad_shape",
    ]
    if not isinstance(elem, dict):
        return pad_batch_img(batch)
    elif isinstance(elem, Mapping):
        return_data = {}
        for key in keys_list:
            if key in list_key:
                collate_data = [d[key] for d in batch if key in d]
            elif key == "img" or "detection" in key:
                collate_data = torch.cat(
                    [d[key] for d in batch if key in d], 0
                )
            else:
                if key == "num_boxes":
                    for d in batch:
                        for g_key in goal_keys:
                            if g_key not in d[key]:
                                d[key][g_key] = 0
                collate_data = default_collate_v2(
                    [d[key] for d in batch if key in d]
                )

            return_data.update({key: collate_data})
        return return_data


@OBJECT_REGISTRY.register
def collate_mmfusion_3d(batch_list):
    example_merged = collections.defaultdict(list)
    for example in batch_list:
        if isinstance(example, list):
            for subexample in example:
                for k, v in subexample.items():
                    example_merged[k].append(v)
        else:
            for k, v in example.items():
                example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if len(elems) == 0 or elems[0] is None:
            ret_value = None
            continue
        if key in [
            "timestamp",
        ]:
            ret_value = torch.tensor(np.concatenate(elems, axis=0))
        elif key in [
            "voxel_data",
            "voxel_num_points",
            "num_voxels",
            "pillar_data",
            "pillar_num_points",
            "num_pillars",
        ]:
            ret_value = []
            for idx in range(len(elems[0])):
                batch_elem = [elem[idx] for elem in elems]
                ret_value.append(
                    torch.tensor(np.concatenate(batch_elem, axis=0))
                )
        elif key in [
            "voxel_coordinates",
            "pillar_coordinates",
        ]:
            ret_value = []
            for idx in range(len(elems[0])):
                batch_elem = [elem[idx] for elem in elems]
                coords = []
                for i, coord in enumerate(batch_elem):
                    coord_pad = np.pad(
                        coord,
                        ((0, 0), (1, 0)),
                        mode="constant",
                        constant_values=i,
                    )
                    coords.append(coord_pad)

                ret_value.append(torch.tensor(np.concatenate(coords, axis=0)))
        elif key in [
            "have_lidar_input",
            "annos_dict",
        ]:  # fusion batch size = 1
            ret_value = elems[0]
        elif key in ["voxel_shape", "pillar_shape"]:
            ret_value = [torch.tensor(i) for i in elems]
        elif key in [
            "homography_temporal_lidar",
            "hm",
            "anno_box",
            "ind",
            "mask",
            "cat",
            "gt_boxes_tasks",
            "anno_box_reg",
            "ind_reg",
            "mask_reg",
        ]:
            ret_value = torch.tensor(np.stack(elems, axis=0))
        else:
            ret_value = collate_3d(elems)

        ret[key] = ret_value

    return ret


def collate_argoverse(batch: List[Any]) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in argoverse dataset, for collating data with inconsistent shapes.

    Args:
        batch: list of data.
    """

    elem = batch[0]
    # these key-value will skip default_collate
    training_key = [
        "lane_feat",
        "lane_mask",
        "traj_feat",
        "traj_mask",
        "traj_labels",
        "goals_2d",
        "goals_2d_labels",
        "goals_2d_mask",
        "instance_mask",
        "end_points",
        "feat_mask",
    ]
    if not isinstance(elem, dict):
        return default_collate(batch)
    elif isinstance(elem, Mapping):
        return_data = {}
        for key in elem:
            if key not in training_key:
                collate_data = [d[key] for d in batch]
            else:
                collate_data = default_collate([d[key] for d in batch])
            return_data.update({key: collate_data})
        return return_data


def collate_disp_cat(batch: List[Any]) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in disp pred task. For concat img data with the first
    dimension and change the img data layout to [llllrrrr].

    Args:
        batch: list of data.

    """

    elem = batch[0]
    # these key-value will skip default_collate

    list_key = [
        "gt_classes",
        "gt_bboxes",
        "ig_bboxes",
        "gt_difficult",
        "gt_labels",
        "ori_img",
        "resized_ori_img",
        "img_id",
        "layout",
        "img_shape",
        "resized_shape",
        "color_space",
        "classes",
        "bboxes",
        "crop_offset",
        "before_crop_shape",
        "crop_roi",
        "structure",
        "bit_nums_upper",
        "bit_nums_lower",
        "channels",
        "cur_pattern",
        "raw_pattern",
        "gt_lines",
        "ori_gt_lines",
        "before_pad_shape",
    ]
    if not isinstance(elem, dict):
        return pad_batch_img(batch)
    elif isinstance(elem, Mapping):
        return_data = {}
        for key in elem:
            if key in list_key:
                collate_data = [d[key] for d in batch if key in d]
            elif key == "img" or "detection" in key:
                collate_data = torch.cat([d[key] for d in batch], 0)
                left = collate_data[::2]
                right = collate_data[1::2]
                collate_data = torch.cat([left, right], 0)
            else:
                collate_data = default_collate_v2(
                    [d[key] for d in batch if key in d]
                )

            return_data.update({key: collate_data})
        return return_data


def collate_e2e_dynamic(batch_data: List[Any]):
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in e2e task.
    merge every variable in the batch_data according to the type.
    "motr_targets" and "pil_imgs" are two special keys.
    List: all lists are extend together.
    Dict: recursively merge all elements.
        including: veh_gt, vru_gt, meta_info.
    Tensor: all tensors are concatenated on axis 0 directly.
        including: homography, homo_offset, timestamp,
        homography_temporal, img, side_img, round_img, narrow_img.
    Int, Float, strs: all int and float are generate in a list.
        including: subclip_idx, pack_names, view.
    "motr_targets": merge all "motr_targets" in a list.
    "pil_imgs": extend imgs of all views and frames and in a
    single list. \
        An example:
        batch_size=3
        num_frames_per_iter=2

        `PrepareTempoDataE2EDynamic`:
            batch1 = {"img" : `torch.randn((2, 3, 512, 960))`}
            batch2 = {"img" : `torch.randn((2, 3, 512, 960))`}
            batch3 = {"img" : `torch.randn((2, 3, 512, 960))`}
        the collate will return, which stack the splited data on
        batch by idx:
            batch = [
                {"img": torch.randn((6, 3, 512, 960))},
            ]
    For more info, refer the unit-test: test_collate.py
    Args:
        batch: list of data.
    """

    if isinstance(batch_data[0], dict):
        result = {}
        for key in batch_data[0].keys():
            if key == "motr_targets":
                result[key] = [d[key] for d in batch_data]
            elif key == "origin_imgs":
                result_key = []
                for d in batch_data:
                    for imgs in d[key]:
                        result_key.extend(imgs)
                result[key] = result_key
            elif key == "odo_info":
                result[key] = torch.cat([d[key] for d in batch_data], dim=0)
            else:
                result[key] = collate_e2e_dynamic([d[key] for d in batch_data])
        return result
    elif isinstance(batch_data[0], (list, tuple)):
        return [collate_e2e_dynamic(data) for data in zip(*batch_data)]
    elif isinstance(batch_data[0], torch.Tensor):
        if len(batch_data[0].shape) == 4:
            return torch.cat(batch_data, dim=0)
        else:
            batch_data = [torch.unsqueeze(d, dim=0) for d in batch_data]
            return torch.cat(batch_data, dim=0)
    elif isinstance(batch_data[0], (str, int, float, bool)):
        return batch_data
    else:
        raise TypeError


def collate_gaze_seq(
    batch: List[Dict],
) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in gaze estimation.
    Args:
        batch: list of data.
    """
    batch_tmp = []
    for seq in batch:
        batch_tmp.extend(seq)
    return_data = {}
    return_data.update(default_collate(batch_tmp))

    return return_data
