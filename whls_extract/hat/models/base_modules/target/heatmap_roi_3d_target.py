import copy
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from hat.core.affine import point_affine_transform
from hat.core.heatmap import draw_heatmap, draw_heatmap_min
from hat.core.utils_3d import get_dense_locoffset, get_gaussian2D, get_reg_map
from hat.registry import OBJECT_REGISTRY

__all__ = ["HeatMap3DTargetGenerator"]


@OBJECT_REGISTRY.register
class HeatMap3DTargetGenerator(nn.Module):
    """
    Generate heatmap target for 3D detection.

    Note that computation is performed on cpu currently instead of gpu.

    Args:
        num_classes: Number of classes.
        normalize_depth: Whether to normalize depth.
        focal_length_default: Default focal length.
        min_box_edge: Minimum box edge.
        max_depth: Maximum depth.
        max_objs: Maximum number of objects.
        down_stride: Down stride of heatmap.
        undistort_2dcenter: Whether to undistort 2D center.
        undistort_depth_uv: Whether to undistort depth uv.
        input_padding: Padding of input image.
        depth_min_option: Whether to use depth min option.

    """

    def __init__(
        self,
        num_classes: int,
        normalize_depth: bool,
        focal_length_default: float,
        min_box_edge: int,
        max_depth: int,
        max_objs: int,
        classid_map: dict,
        down_stride: Optional[int] = 4,
        undistort_2dcenter: Optional[bool] = False,
        undistort_depth_uv: Optional[bool] = False,
        input_padding: Optional[list] = None,
        depth_min_option: Optional[bool] = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.normalize_depth = normalize_depth
        self.focal_length_default = focal_length_default
        self.classid_map = classid_map

        self.down_stride = down_stride
        self.min_box_edge = min_box_edge
        self.max_depth = max_depth
        self.max_objs = max_objs
        self.undistort_2dcenter = undistort_2dcenter
        self.undistort_depth_uv = undistort_depth_uv
        self.input_padding = input_padding
        self.depth_min_option = depth_min_option

    def forward(self, data):

        _device = data["img_wh"].device
        ignore_masks = data.pop("ignore_mask").cpu().numpy()
        img_whs = data.pop("img_wh").cpu().numpy()
        calibs = data.pop("calib").cpu().numpy()
        dist_coeffs = data.pop("distCoeffs").cpu().numpy()
        trans_matrices = data.pop("trans_matrix").cpu().numpy()
        batch_bboxes = data.pop("bboxes").cpu().numpy()
        batch_location_offsets = data.pop("location_offsets").cpu().numpy()
        batch_dims = data.pop("dims").cpu().numpy()
        batch_rotation_ys = data.pop("rotation_ys").cpu().numpy()
        batch_depths = data.pop("depths").cpu().numpy()
        batch_cls_ids = data.pop("cls_ids").cpu().numpy()
        batch_locations = data.pop("locations").cpu().numpy()
        if self.undistort_depth_uv:
            resized_eq_fus = data.pop("resized_eq_fu").cpu().numpy()
            resized_eq_fvs = data.pop("resized_eq_fv").cpu().numpy()

        ret = {
            "heatmap": [],
            "box2d_wh": [],
            "dimensions": [],
            "location_offset": [],  # noqa
            "depth": [],  # noqa
            "heatmap_weight": [],
            "point_pos_mask": [],
            "index": [],
            "index_mask": [],
            "location": [],
            "rotation_y": [],
            "dimensions_": [],
            "img_wh": [],
            "ignore_mask": [],
        }

        done_stride_affine = np.array(
            [1 / self.down_stride, 0, 0, 0, 1 / self.down_stride, 0]
        ).reshape(2, 3)

        for batch_idx in range(len(img_whs)):
            trans_matrix = trans_matrices[batch_idx]
            trans_output = trans_matrix / self.down_stride
            ignore_mask = ignore_masks[batch_idx]
            img_wh = img_whs[batch_idx]
            calib = calibs[batch_idx]
            dist_coeff = dist_coeffs[batch_idx]
            bboxes = batch_bboxes[batch_idx]
            location_offsets = batch_location_offsets[batch_idx]
            dims = batch_dims[batch_idx]
            rotation_ys = batch_rotation_ys[batch_idx]
            depths = batch_depths[batch_idx]
            cls_ids = batch_cls_ids[batch_idx]
            locations = batch_locations[batch_idx]

            output_width, output_height = [
                img_wh[0] // self.down_stride,
                img_wh[1] // self.down_stride,
            ]

            ignore_mask = cv2.warpAffine(
                ignore_mask,
                done_stride_affine,
                (output_width, output_height),
                flags=cv2.INTER_NEAREST,
            )
            ignore_mask = ignore_mask.astype(np.float32)[:, :, np.newaxis]

            hm = np.zeros(
                (output_height, output_width, self.num_classes),
                dtype=np.float32,
            )
            wh = np.zeros((output_height, output_width, 2), dtype=np.float32)
            depth = np.zeros((output_height, output_width), dtype=np.float32)
            dim = np.zeros((output_height, output_width, 3), dtype=np.float32)
            loc_offset = np.zeros(
                (output_height, output_width, 2), dtype=np.float32
            )
            weight_hm = np.zeros(
                (output_height, output_width), dtype=np.float32
            )
            weight_hm_min = 10000 * np.ones(
                (output_height, output_width), dtype=np.float32
            )
            point_pos_mask = np.zeros(
                (output_height, output_width), dtype=np.float32
            )
            # sin cos
            ind_ = np.zeros((self.max_objs), dtype=np.int64)
            ind_mask_ = np.zeros((self.max_objs), dtype=np.float32)
            rot_y_ = np.zeros((self.max_objs, 1), dtype=np.float32)
            loc_ = np.zeros((self.max_objs, 3), dtype=np.float32)
            dim_ = np.zeros((self.max_objs, 3), dtype=np.float32)

            ann_idx = -1

            valid_idx = np.where(cls_ids > 0)

            for bbox_idx in valid_idx[0]:
                location_offset = location_offsets[bbox_idx]
                bbox = copy.deepcopy(bboxes[bbox_idx])

                bbox[:2] = point_affine_transform(bbox[:2], done_stride_affine)
                bbox[2:] = point_affine_transform(bbox[2:], done_stride_affine)

                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_width - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_height - 1)
                _wh = bbox[2:] - bbox[:2]

                # filter class
                if int(self.classid_map[cls_ids[bbox_idx]]) < 0:
                    continue
                if np.any(_wh <= 0):
                    continue

                # filter bbox
                if (bboxes[bbox_idx][2] - bboxes[bbox_idx][0]) < (
                    self.min_box_edge * trans_matrix[0][0]
                ) or (bboxes[bbox_idx][3] - bboxes[bbox_idx][1]) < (
                    self.min_box_edge * trans_matrix[1][1]
                ):
                    ignore_mask[
                        int(bbox[1]) : int(bbox[3] + 1),
                        int(bbox[0]) : int(bbox[2] + 1),
                        :,
                    ] = 1.0
                    continue
                # filter by depth
                if depths[bbox_idx] > self.max_depth:
                    ignore_mask[
                        int(bbox[1]) : int(bbox[3] + 1),
                        int(bbox[0]) : int(bbox[2] + 1),
                        :,
                    ] = 1.0
                    continue

                ct = (bbox[:2] + bbox[2:]) / 2
                ct_int = tuple(ct.astype(np.int32).tolist())

                ann_idx += 1
                if ann_idx >= self.max_objs:
                    break

                ind_[ann_idx] = ct_int[1] * output_width + ct_int[0]
                ind_mask_[ann_idx] = 1
                loc_[ann_idx] = locations[bbox_idx]
                rot_y_[ann_idx] = rotation_ys[bbox_idx]
                dim_[ann_idx] = dims[bbox_idx]
                # ttfnet style
                insert_hm = get_gaussian2D(_wh)

                insert_hm_wh = insert_hm.shape[:2][::-1]

                if not self.depth_min_option:
                    insert_reg_map_list = [
                        get_reg_map(insert_hm_wh, depths[bbox_idx]),
                        get_reg_map(insert_hm_wh, _wh),
                        get_reg_map(insert_hm_wh, dims[bbox_idx]),
                        get_dense_locoffset(
                            insert_hm_wh,
                            ct_int,
                            location_offset[:2],
                            locations[bbox_idx],
                            dims[bbox_idx],
                            calib,
                            trans_output,
                            dist_coeff,
                            self.undistort_2dcenter,
                        ),
                    ]
                    reg_map_list = [
                        depth,
                        wh,
                        dim,
                        loc_offset,  # rotbin, rotres
                    ]
                    draw_heatmap(
                        hm[:, :, int(self.classid_map[cls_ids[bbox_idx]])],
                        insert_hm,
                        ct_int,
                    )
                    draw_heatmap(
                        weight_hm,
                        insert_hm,
                        ct_int,
                        reg_map_list,
                        insert_reg_map_list,
                    )

                else:
                    insert_reg_map_list = [
                        get_reg_map(insert_hm_wh, _wh),
                        get_reg_map(insert_hm_wh, dims[bbox_idx]),
                        get_dense_locoffset(
                            insert_hm_wh,
                            ct_int,
                            location_offset[:2],
                            locations[bbox_idx],
                            dims[bbox_idx],
                            calib,
                            trans_output,
                            dist_coeff,
                            self.undistort_2dcenter,
                        ),
                    ]
                    reg_map_list = [
                        wh,
                        dim,
                        loc_offset,  # rotbin, rotres
                    ]
                    draw_heatmap(
                        hm[:, :, int(self.classid_map[cls_ids[bbox_idx]])],
                        insert_hm,
                        ct_int,
                    )
                    draw_heatmap(
                        weight_hm,
                        insert_hm,
                        ct_int,
                        reg_map_list,
                        insert_reg_map_list,
                    )

                    insert_reg_map_list_min = [
                        get_reg_map(insert_hm_wh, depths[bbox_idx]),
                    ]

                    reg_map_list_min = [
                        depth,
                    ]

                    insert_hm_depth = depths[bbox_idx] * np.ones(
                        (insert_hm.shape[0], insert_hm.shape[1]),
                        dtype=np.float64,
                    )

                    draw_heatmap_min(
                        weight_hm_min,
                        insert_hm_depth,
                        ct_int,
                        reg_map_list_min,
                        insert_reg_map_list_min,
                    )

                point_pos_mask[ct_int[1], ct_int[0]] = 1

            if self.normalize_depth:
                if self.undistort_depth_uv:
                    resized_eq_fu = resized_eq_fus[batch_idx]
                    resized_eq_fv = resized_eq_fvs[batch_idx]
                    down_strid_eq_fu = cv2.warpAffine(
                        resized_eq_fu,
                        done_stride_affine,
                        (output_width, output_height),
                    )
                    down_strid_eq_fv = cv2.warpAffine(
                        resized_eq_fv,
                        done_stride_affine,
                        (output_width, output_height),
                    )
                    depth_u = (
                        depth * self.focal_length_default / down_strid_eq_fu
                    )
                    depth_v = (
                        depth * self.focal_length_default / down_strid_eq_fv
                    )
                    depth = np.stack([depth_u, depth_v], axis=-1)
                else:
                    depth *= self.focal_length_default / calib[0, 0]
                    depth = depth[:, :, np.newaxis]
                loc_offset *= self.focal_length_default / calib[0, 0]

            tmp = {
                "heatmap": hm.transpose(2, 0, 1),
                "box2d_wh": wh.transpose(2, 0, 1),
                "dimensions": dim.transpose(2, 0, 1),
                "location_offset": loc_offset.transpose(2, 0, 1),
                "depth": depth.transpose(2, 0, 1),
                "heatmap_weight": weight_hm[:, :, np.newaxis].transpose(
                    2, 0, 1
                ),
                "point_pos_mask": point_pos_mask[:, :, np.newaxis].transpose(
                    2, 0, 1
                ),
                "index": ind_,
                "index_mask": ind_mask_,
                "location": loc_,
                "rotation_y": rot_y_,
                "dimensions_": dim_,
                "img_wh": img_wh,
                "ignore_mask": ignore_mask.transpose(2, 0, 1),
            }
            if self.input_padding is not None:
                tmp = self.dense3d_pad_after_label_generator(
                    tmp, self.input_padding, self.down_stride
                )
            tmp["depth"] = tmp["depth"].astype(np.float32)

            for k, v in tmp.items():
                ret[k].append(v)
        for k, v in ret.items():
            ret[k] = torch.tensor(np.array(v), device=_device)
        return ret
