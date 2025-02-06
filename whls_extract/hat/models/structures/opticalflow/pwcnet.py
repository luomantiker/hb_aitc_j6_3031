# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Dict, List, Optional

import torch
from torch import nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = ["PwcNet", "PwcNetIrInfer"]


@OBJECT_REGISTRY.register
class PwcNet(nn.Module):
    """The basic structure of PWCNet.

    Args:
        backbone: backbone module or dict for building backbone module.
        neck: neck module or dict for building neck module.
        head: head module or dict for building head module.
        loss: loss module or dict for building loss module.
        loss_weights:  loss weights for each feature.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        loss_weights: Optional[List[float]] = None,
    ):
        super(PwcNet, self).__init__()

        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss
        if loss is not None:
            assert (
                loss_weights is not None
            ), "For train step, loss_weights must is not None."
        if loss_weights is not None:
            self.loss_weights = torch.Tensor(loss_weights)

    @fx_wrap()
    def _post_process(self, data, flows):
        if self.training:
            gt_labels = data["gt_flow"]
            flows = flows[::-1]
            losses = []
            for flow, gt_label, loss_weight in zip(
                flows, gt_labels, self.loss_weights
            ):
                loss = self.loss(flow, gt_label, loss_weight)
                losses.append(loss)
            return {
                "losses": losses,
                "pred_flows": flows[0],
            }
        else:
            return flows

    def forward(self, data: Dict):
        feat = self.backbone(data["img"])
        feat = self.neck(feat) if self.neck else feat
        flows = self.head(feat)
        return self._post_process(data, flows)

    def fuse_model(self):
        if self.backbone:
            self.backbone.fuse_model()
        if self.neck:
            self.neck.fuse_model()
        if self.head:
            self.head.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [
            self.backbone,
            self.neck,
            self.head,
        ]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()

    def set_calibration_qconfig(self):
        from hat.utils import qconfig_manager

        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )
        if self.loss is not None:
            self.loss.qconfig = None


@OBJECT_REGISTRY.register
class PwcNetIrInfer(nn.Module):
    """
    The basic structure of PwcNetIrInfer.

    Args:
        ir_model: The ir model.
    """

    def __init__(
        self,
        ir_model: nn.Module,
    ):
        super().__init__()
        self.ir_model = ir_model

    def forward(self, data):
        outputs = self.ir_model(data)
        return outputs
