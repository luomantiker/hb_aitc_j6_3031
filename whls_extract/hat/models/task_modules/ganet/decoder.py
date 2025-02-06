# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from hat.models.base_modules.postprocess import PostProcessorBase
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["GaNetDecoder"]


@OBJECT_REGISTRY.register
class GaNetDecoder(PostProcessorBase):  # noqa: D205,D400
    """
    Decoder for ganet, convert the output of the model to a prediction result
    in original image.

    Args:
        root_thr: Threshold of select start point.
        kpt_thr: Threshold of key points.
        cluster_thr: Distance threshold of clustering point.
        downscale: Down sampling scale for input data.
        min_points: Minimum number of key points.
    """

    def __init__(
        self,
        root_thr: float = 1.0,
        kpt_thr: float = 0.4,
        cluster_thr: float = 4.0,
        downscale: int = 8,
        min_points: int = 10,
    ):
        super(GaNetDecoder, self).__init__()
        self.root_thr = root_thr
        self.kpt_thr = kpt_thr
        self.cluster_thr = cluster_thr
        self.downscale = downscale
        self.min_points = min_points

    def forward(
        self,
        heat: torch.Tensor,
        offset: torch.Tensor,
        error: torch.Tensor,
        meta_data: Dict[str, Any],
    ):

        heat = torch.clamp(heat.sigmoid(), min=1e-4, max=1 - 1e-4)
        cpt_seeds, kpt_seeds = self._decoder(heat, offset, error)

        bs = heat.shape[0]
        results = []
        for i in range(bs):
            cpt_seeds_tmp = cpt_seeds[i].detach().cpu().numpy()
            kpt_seeds_tmp = kpt_seeds[i].detach().cpu().numpy()
            if cpt_seeds_tmp.shape[0] == 0:
                results.append([])
                continue
            scale_factor = meta_data["scale_factor"][i].cpu().numpy()
            crop_offset = meta_data["crop_offset"][i]

            kpt_groups_tmp = self._group_points_np_vector(
                kpt_seeds_tmp,
                cpt_seeds_tmp,
                self.cluster_thr,
            )
            lanes_tmp = self._lane_post_process(
                kpt_groups_tmp, self.downscale, scale_factor, crop_offset
            )
            results.append(lanes_tmp)

        return results

    def _lane_post_process(
        self, kpt_groups, downscale, scale_factor, crop_offset
    ):
        """Map the detected key points back to the original image."""
        lanes = []
        ratio_x = 1 / scale_factor[0]
        ratio_y = 1 / scale_factor[1]
        offset_x, offset_y = crop_offset[0], crop_offset[1]
        for kpt_points in kpt_groups:
            if len(kpt_points) <= self.min_points:
                continue
            kpt_points = kpt_points * downscale
            kpt_points[:, 0] = kpt_points[:, 0] * ratio_x + offset_x
            kpt_points[:, 1] = kpt_points[:, 1] * ratio_y + offset_y
            lanes.append(kpt_points)
        return lanes

    def _decoder(self, heat, offset, error):  # noqa: D205,D400
        """Analyze the starting point and other key points
        through nms and offset.
        """
        bs = heat.shape[0]

        # nms heatmap
        hmax = nn.functional.max_pool2d(
            heat, (1, 3), stride=(1, 1), padding=(0, 1)
        )
        keep = (hmax == heat).float()
        heat_nms = heat * keep

        heat_nms = heat_nms.permute(0, 3, 2, 1)  # NWHC
        offset = offset.permute(0, 3, 2, 1)  # NWHC
        error = error.permute(0, 3, 2, 1)  # NWHC

        kpt_mask = torch.gt(heat_nms, self.kpt_thr)  # all point mask
        all_points_idxs = torch.where(kpt_mask)
        all_points_idxs = torch.stack(all_points_idxs[:-1], dim=1)

        offset_bs = 0
        kpt_seeds = []
        start_points = []

        for i in range(bs):
            idx_tmp = torch.where(all_points_idxs[:, 0] == i)
            idx_tmp_len = len(idx_tmp[0])

            sample_points = all_points_idxs[
                offset_bs : offset_bs + idx_tmp_len, 1:
            ]

            sample_offset = offset[i, sample_points[:, 0], sample_points[:, 1]]
            sample_error = error[i, sample_points[:, 0], sample_points[:, 1]]

            start_point_mask = torch.lt(sample_offset[..., 1], self.root_thr)
            # start_points
            start_points_tmp = sample_points[start_point_mask].float()
            start_points.append(start_points_tmp)

            sample_offset = sample_offset + sample_points
            sample_error = sample_error + sample_points

            kpt_seeds.append(
                torch.concat([sample_error, sample_offset], dim=1)
            )
            offset_bs += idx_tmp_len

        return start_points, kpt_seeds

    def _group_points_np_vector(self, points, cluster_centers, center_thr=5):
        # begin
        groups_idx = []

        num_center_point = cluster_centers.shape[0]
        cluster_centers_tmp = cluster_centers.copy()
        cluster_centers_re = cluster_centers.repeat(num_center_point, axis=0)
        cluster_centers_tmp = np.tile(
            cluster_centers_tmp, (num_center_point, 1)
        )
        sub = cluster_centers_re - cluster_centers_tmp
        square = np.square(sub)
        sum_ = np.sum(square, axis=1)
        vector_result = np.sqrt(sum_)
        vector_result = vector_result.reshape(
            num_center_point, num_center_point
        )

        group_id_max = 0
        groups_idxs = []
        for i in range(num_center_point):
            same_group_point_idx = np.where(vector_result[i] <= center_thr)[0][
                0
            ]
            if same_group_point_idx == i:

                groups_idxs.append([i])
                groups_idx.append(group_id_max)
                group_id_max += 1
            else:
                group_id = groups_idx[same_group_point_idx]
                groups_idxs[group_id].append(i)
                groups_idx.append(group_id)

        groups_centers_mean_tmp = []
        # choose mean center
        for groups_idx_tmp in groups_idxs:
            group_center_new = np.mean(
                cluster_centers[groups_idx_tmp], axis=0, dtype=int
            )
            groups_centers_mean_tmp.append(group_center_new)

        groups_centers_mean_tmp = np.array(groups_centers_mean_tmp)
        aligns = points[:, :2]
        centers = points[:, 2:]

        num_points = centers.shape[0]
        num_centers = groups_centers_mean_tmp.shape[0]

        centers_tmp = centers.copy()

        centers_tmp_re = centers_tmp.repeat(num_centers, axis=0)
        groups_centers_mean_tmp_re = np.tile(
            groups_centers_mean_tmp, (num_points, 1)
        )

        sub = centers_tmp_re - groups_centers_mean_tmp_re
        square = np.square(sub)
        sum_ = np.sum(square, axis=1)
        vector_result_cetern = np.sqrt(sum_)
        vector_result_cetern = vector_result_cetern.reshape(
            num_points, num_centers
        )

        min_idx = np.argmin(vector_result_cetern, axis=1)
        min_value = vector_result_cetern[np.arange(num_points), min_idx]

        mask = min_value <= center_thr

        min_idx = min_idx[mask]
        aligns = aligns[mask]

        groups_tmp = []
        for i in range(num_centers):
            groups_tmp.append(aligns[min_idx == i])

        return groups_tmp
