# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, Optional, Tuple

import torch

from hat.registry import OBJECT_REGISTRY

__all__ = ["FCOSLoss", "VehicleSideFCOSLoss"]


@OBJECT_REGISTRY.register
class FCOSLoss(torch.nn.Module):
    """
    FCOS loss wrapper.

    Args:
        losses (list): loss configs.

    Note:
        This class is not universe. Make sure you know this class limit before
        using it.

    """

    def __init__(
        self,
        cls_loss: torch.nn.Module,
        reg_loss: torch.nn.Module,
        centerness_loss: Optional[torch.nn.Module] = None,
    ):
        super(FCOSLoss, self).__init__()
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.centerness_loss = centerness_loss

    def forward(self, pred: Tuple, target: Tuple[Dict]) -> Dict:
        res = {}
        # `pred` is in target tuple, so we get pred from target.
        # assume fcos target is in cls/reg/centerness order.
        cls_res = self.cls_loss(**target[0])
        reg_res = self.reg_loss(**target[1])

        if not isinstance(cls_res, dict):
            cls_res = {"cls_loss": cls_res}
        if not isinstance(reg_res, dict):
            reg_res = {"reg_loss": reg_res}
        res.update(cls_res)
        res.update(reg_res)

        if self.centerness_loss is not None:
            assert len(target) == 3
            ctr_res = self.centerness_loss(**target[2])
            res.update(ctr_res)
            # Three dict shouldn't contain same key.
            assert len(res) == len(cls_res) + len(reg_res) + len(ctr_res), (
                "`cls_res`, `reg_cls` and `ctr_res` have same name keys, "
                "this may cause bugs."
            )
        return res


@OBJECT_REGISTRY.register
class VehicleSideFCOSLoss(torch.nn.Module):
    """
    VehicleSide Task FCOS Loss wrapper.

    Args:
        cls_loss: Classification Loss.
        reg_bbox_loss: Regression Loss for Vehicle Side BBox.
        reg_alpha_loss: Regression Loss for Vehicle Side Alpha.
        centerness_loss: FCOS Centerness Loss.
    """

    def __init__(
        self,
        cls_loss: torch.nn.Module,
        reg_bbox_loss: torch.nn.Module,
        reg_alpha_loss: torch.nn.Module,
        centerness_loss: torch.nn.Module,
    ):
        super(VehicleSideFCOSLoss, self).__init__()
        self.cls_loss = cls_loss
        self.reg_bbox_loss = reg_bbox_loss
        self.reg_alpha_loss = reg_alpha_loss
        self.centerness_loss = centerness_loss

    def forward(self, pred: Tuple, target: Tuple[Dict]) -> Dict:
        assert len(target) == 4
        res = {}
        # `pred` is in target tuple, so we get pred from target.
        # assume fcos target is in cls/reg/centerness order.
        cls_res = self.cls_loss(**target[0])
        reg_bbox_loss = self.reg_bbox_loss(**target[1])
        reg_alpha_loss = self.reg_alpha_loss(**target[2])
        ctr_res = self.centerness_loss(**target[3])
        res.update(cls_res)
        res.update(reg_bbox_loss)
        res.update(reg_alpha_loss)
        res.update(ctr_res)
        # Three dict shouldn't contain same key.
        assert len(res) == len(cls_res) + len(reg_bbox_loss) + len(
            reg_alpha_loss
        ) + len(ctr_res), (
            "`cls_res`, `reg_bbox_loss`, `reg_alpha_loss` and `ctr_res` "
            "have same name keys, this may cause bugs."
        )
        return res
