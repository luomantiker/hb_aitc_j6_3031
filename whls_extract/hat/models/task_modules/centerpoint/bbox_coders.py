# Copyright (c) Horizon Robotics. All rights reserved.
# Source code reference to OpenMMLab.
from typing import List, Optional

import torch

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class CenterPointBBoxCoder:
    """Bbox coder for CenterPoint.

    Args:
        pc_range: Range of point cloud.
        out_size_factor: Downsample factor of the model.
        voxel_size: Size of voxel.
        post_center_range: Limit of the center.
            Default: None.
        max_num: Max number to be kept. Default: 100.
        score_threshold: Threshold to filter boxes
            based on score. Default: None.
    """

    def __init__(
        self,
        pc_range: List[float],
        out_size_factor: int,
        voxel_size: List[float],
        post_center_range: Optional[List[float]] = None,
        max_num: Optional[int] = 100,
        score_threshold: Optional[float] = None,
    ):

        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold

    def _gather_feat(
        self,
        feats: torch.Tensor,
        inds: torch.Tensor,
        feat_masks: torch.Tensor = None,
    ):
        """Given feats and indexes, returns the gathered feats.

        Args:
            feats: Features to be transposed and gathered
                with the shape of [B, W*H, C].
            inds: Indexes with the shape of [B, N].
            feat_masks: Mask of the feats.
                Default: None.

        Returns:
            feats: Gathered feats.
        """
        dim = feats.size(2)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def _topk(self, scores: torch.Tensor, K: int = 80):
        """Get indexes based on scores.

        Args:
            scores: scores with the shape of [B, N, W, H].
            K: Number to be kept. Defaults to 80.

        Returns:
            tuple[torch.Tensor]:
                - topk_score: Selected scores with the shape of [B, K].
                - topk_inds: Selected indexes with the shape of [B, K].
                - topk_clses: Selected classes with the shape of [B, K].
                - topk_ys: Selected y coord with the shape of [B, K].
                - topk_xs: Selected x coord with the shape of [B, K].
        """
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (
            (topk_inds.float() / torch.tensor(width, dtype=torch.float))
            .int()
            .float()
        )
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
        topk_inds = self._gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind
        ).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(
            batch, K
        )
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(
            batch, K
        )

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _transpose_and_gather_feat(
        self, feat: torch.Tensor, ind: torch.Tensor
    ):
        """Given feats and indexes, returns the transposed and gathered feats.

        Args:
            feat: Features to be transposed and gathered
                with the shape of [B, C, W, H].
            ind: Indexes with the shape of [B, N].

        Returns:
            torch.Tensor: Transposed and gathered feats.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def encode(self):
        pass

    def decode(
        self,
        heat: torch.Tensor,
        rot_sine: torch.Tensor,
        rot_cosine: torch.Tensor,
        hei: torch.Tensor,
        dim: torch.Tensor,
        vel: torch.Tensor,
        reg: torch.Tensor = None,
        task_id: int = -1,
    ):
        """Decode bboxes.

        Args:
            heat: Heatmap with the shape of [B, N, W, H].
            rot_sine: Sine of rotation with the shape of
                [B, 1, W, H].
            rot_cosine: Cosine of rotation with the shape of
                [B, 1, W, H].
            hei: Height of the boxes with the shape
                of [B, 1, W, H].
            dim: Dim of the boxes with the shape of
                [B, 3, W, H].
            vel: Velocity with the shape of [B, 2, W, H].
            reg: Regression value of the boxes in
                2D with the shape of [B, 2, W, H]. Default: None.
            task_id: Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
        batch, cat, _, _ = heat.size()

        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)

        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, self.max_num, 2)
            xs = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, self.max_num, 1) + 0.5
            ys = ys.view(batch, self.max_num, 1) + 0.5

        # rotation value and direction label
        rot_sine = self._transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.view(batch, self.max_num, 1)

        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.view(batch, self.max_num, 1)
        rot = torch.atan2(rot_sine, rot_cosine)

        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, self.max_num, 1)

        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, self.max_num, 3)

        # class label
        clses = clses.view(batch, self.max_num).float()
        scores = scores.view(batch, self.max_num)

        xs = (
            xs.view(batch, self.max_num, 1)
            * self.out_size_factor
            * self.voxel_size[0]
            + self.pc_range[0]
        )
        ys = (
            ys.view(batch, self.max_num, 1)
            * self.out_size_factor
            * self.voxel_size[1]
            + self.pc_range[1]
        )

        if vel is None:
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
        else:  # exist velocity, nuscene format
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, self.max_num, 2)
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)

        final_scores = scores
        final_preds = clses

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heat.device
            )
            mask = (
                final_box_preds[..., :3] >= self.post_center_range[:3]
            ).all(2)
            mask &= (
                final_box_preds[..., :3] <= self.post_center_range[3:]
            ).all(2)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    "bboxes": boxes3d,
                    "scores": scores,
                    "labels": labels,
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )

        return predictions_dicts
