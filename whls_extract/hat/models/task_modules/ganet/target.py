# Copyright (c) Horizon Robotics. All rights reserved.

from functools import cmp_to_key

import numpy as np
import scipy.interpolate as spi
import torch
from torch import nn

from hat.core.center_utils import draw_umich_gaussian
from hat.registry import OBJECT_REGISTRY

__all__ = ["GaNetTarget"]


def assign_weight(dis, h, joints, weights=None):
    if weights is None:
        weights = [1, 0.4, 0.2]
    step = h // joints
    weight = 1
    if dis < 0:
        weight = weights[2]
    elif dis < 2 * step:
        weight = weights[0]
    else:
        weight = weights[1]
    return weight


def ploy_fitting_cube(line, h, w, sample_num=100):
    # Y->X
    line_coords = np.array(line).reshape((-1, 2))
    line_coords = np.array(sorted(line_coords, key=lambda x: x[1]))
    line_coords = line_coords[line_coords[:, 0] > 0, :]
    line_coords = line_coords[line_coords[:, 0] < w, :]

    X = line_coords[:, 1]
    Y = line_coords[:, 0]
    if len(X) < 2:
        return None
    new_x = np.linspace(max(X[0], 0), min(X[-1], h), sample_num)

    if len(X) > 3:
        ipo3 = spi.splrep(X, Y, k=3)
        iy3 = spi.splev(new_x, ipo3)
    else:
        ipo3 = spi.splrep(X, Y, k=1)
        iy3 = spi.splev(new_x, ipo3)
    return np.concatenate([iy3[:, None], new_x[:, None]], axis=1)


@OBJECT_REGISTRY.register
class GaNetTarget(nn.Module):
    """
    Target for ganet, generate info using training from label.

    Args:
        hm_down_scale: The downsample scale of heatmape for input data.
        radius: Gaussian circle radius.
    """

    def __init__(
        self,
        hm_down_scale: int,
        radius: int = 2,
    ):
        super(GaNetTarget, self).__init__()
        self.hm_down_scale = hm_down_scale
        self.radius = radius

    def get_target(self, img_shape, gt_lines):

        hm_h = int(img_shape[1] // self.hm_down_scale)
        hm_w = int(img_shape[2] // self.hm_down_scale)

        # gt init
        reg_hm = np.zeros((2, hm_h, hm_w), np.float32)
        offset_hm = np.zeros((2, hm_h, hm_w), np.float32)
        int_offset_mask = np.zeros((1, hm_h, hm_w), np.float32)
        pts_offset_mask = np.zeros((2, hm_h, hm_w), np.float32)
        gt_kpts_hm = np.zeros((1, hm_h, hm_w), np.float32)

        # gt heatmap and ins of bank

        for pts in gt_lines:  # per lane
            pts = pts / self.hm_down_scale
            pts = ploy_fitting_cube(
                pts, hm_h, hm_w, int(360 / self.hm_down_scale)
            )
            if pts is None:
                continue

            pts_tmp = sorted(pts, key=cmp_to_key(lambda a, b: b[-1] - a[-1]))

            pts = []
            for pts_info in pts_tmp:
                pts_tmp_x = min(hm_w - 1, pts_info[0])
                pts_tmp_x = max(0, pts_tmp_x)
                pts_tmp_y = min(hm_h - 1, pts_info[1])
                pts_tmp_y = max(0, pts_tmp_y)
                pts.append((pts_tmp_x, pts_tmp_y))

            if pts is not None and len(pts) > 1:
                start_point, end_point = pts[0], pts[-1]
                max_y = abs(start_point[1] - end_point[1])

                for pt in pts:
                    pt_int = (int(pt[0]), int(pt[1]))
                    gt_kpts_hm[0] = draw_umich_gaussian(
                        gt_kpts_hm[0], pt_int, radius=self.radius
                    )

                    reg_x = pt[0] - pt_int[0]
                    reg_y = pt[1] - pt_int[1]
                    reg_hm[0, pt_int[1], pt_int[0]] = reg_x
                    reg_hm[1, pt_int[1], pt_int[0]] = reg_y
                    if abs(reg_x) < 2 and abs(reg_y) < 2:
                        int_offset_mask[0, pt_int[1], pt_int[0]] = 1
                    offset_x = start_point[0] - pt[0]
                    offset_y = start_point[1] - pt[1]

                    mask_value = assign_weight(offset_y, max_y, 1)
                    pts_offset_mask[:, pt_int[1], pt_int[0]] = mask_value

                    offset_hm[0, pt_int[1], pt_int[0]] = offset_x
                    offset_hm[1, pt_int[1], pt_int[0]] = offset_y

        sample_results = {}

        sample_results["gt_kpts_hm"] = torch.from_numpy(gt_kpts_hm)
        sample_results["int_offset"] = torch.from_numpy(reg_hm)
        sample_results["pts_offset"] = torch.from_numpy(offset_hm)
        sample_results["int_offset_mask"] = torch.from_numpy(int_offset_mask)
        sample_results["pts_offset_mask"] = torch.from_numpy(pts_offset_mask)

        return sample_results

    def forward(self, data):
        device = data["img"].device
        gt_kpts_hm = []
        int_offset = []
        pts_offset = []
        int_offset_mask = []
        pts_offset_mask = []
        for img_shape, gt_lines in zip(data["img_shape"], data["gt_lines"]):
            tmp_target = self.get_target(img_shape, gt_lines)
            gt_kpts_hm.append(tmp_target["gt_kpts_hm"])
            int_offset.append(tmp_target["int_offset"])
            pts_offset.append(tmp_target["pts_offset"])
            int_offset_mask.append(tmp_target["int_offset_mask"])
            pts_offset_mask.append(tmp_target["pts_offset_mask"])

        ganet_target = {}
        ganet_target["gt_kpts_hm"] = torch.stack(gt_kpts_hm).to(device=device)
        ganet_target["int_offset"] = torch.stack(int_offset).to(device=device)
        ganet_target["pts_offset"] = torch.stack(pts_offset).to(device=device)
        ganet_target["int_offset_mask"] = torch.stack(int_offset_mask).to(
            device=device
        )
        ganet_target["pts_offset_mask"] = torch.stack(pts_offset_mask).to(
            device=device
        )
        return ganet_target
