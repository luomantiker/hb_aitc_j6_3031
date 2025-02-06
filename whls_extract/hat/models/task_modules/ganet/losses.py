# Copyright (c) Horizon Robotics. All rights reserved.

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["GaNetLoss"]


@OBJECT_REGISTRY.register
class GaNetLoss(nn.Module):
    """
    The loss module of YOLOv3.

    Args:
        loss_kpts_cls: Key poinits classification loss module.
        loss_pts_offset_reg: Key points regiression loss module.
        loss_int_offset_reg: Int error of points regiression loss module.
    """

    def __init__(
        self,
        loss_kpts_cls: nn.Module,
        loss_pts_offset_reg: nn.Module,
        loss_int_offset_reg: nn.Module,
    ):
        super(GaNetLoss, self).__init__()
        self.loss_kpts_cls = loss_kpts_cls
        self.loss_pts_offset_reg = loss_pts_offset_reg
        self.loss_int_offset_reg = loss_int_offset_reg

    def forward(self, kpts_hm, pts_offset, int_offset, ganet_target):

        kpts_hm = torch.clamp(torch.sigmoid(kpts_hm), min=1e-4, max=1 - 1e-4)
        kpts_cls_loss = self.loss_kpts_cls(kpts_hm, ganet_target["gt_kpts_hm"])

        pts_mask_weight = (
            ganet_target["pts_offset_mask"].bool().float().sum() + 1e-4
        )
        offset_reg_loss = self.loss_pts_offset_reg(
            pts_offset * ganet_target["pts_offset_mask"],
            ganet_target["pts_offset"] * ganet_target["pts_offset_mask"],
            avg_factor=pts_mask_weight,
        )

        int_mask_weight = (
            ganet_target["int_offset_mask"].bool().float().sum() + 1e-4
        )
        int_offset_reg_loss = self.loss_int_offset_reg(
            int_offset * ganet_target["int_offset_mask"],
            ganet_target["int_offset"] * ganet_target["int_offset_mask"],
            avg_factor=int_mask_weight,
        )

        outputs = {}
        outputs["kpts_cls_loss"] = kpts_cls_loss
        outputs["offset_reg_loss"] = offset_reg_loss
        outputs["int_offset_reg_loss"] = int_offset_reg_loss
        return outputs
