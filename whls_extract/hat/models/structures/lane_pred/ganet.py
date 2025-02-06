# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Dict

from torch import nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = ["GaNet", "GaNetIrInfer"]


@OBJECT_REGISTRY.register
class GaNet(nn.Module):
    """The basic structure of GaNet.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        targets: Target module.
        post_process: Post process module.
        losses: Loss module.

    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        targets: nn.Module = None,
        post_process: nn.Module = None,
        losses: nn.Module = None,
    ):
        super(GaNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.losses = losses
        self.targets = targets
        self.post_process = post_process

    @fx_wrap()
    def _post_process(self, data, kpts_hm, pts_offset, int_offset):
        if self.post_process is None:
            return kpts_hm, pts_offset, int_offset

        if self.training:
            ganet_target = self.targets(data)
            outputs = self.losses(
                kpts_hm, pts_offset, int_offset, ganet_target
            )
            return outputs

        else:
            results = self.post_process(kpts_hm, pts_offset, int_offset, data)
            return results

    def forward(self, data: Dict):

        feats = self.backbone(data["img"])

        if self.neck is not None:
            feats = self.neck(feats)

        kpts_hm, pts_offset, int_offset = self.head(feats[0])

        return self._post_process(data, kpts_hm, pts_offset, int_offset)

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
        if self.losses is not None:
            self.losses.qconfig = None


@OBJECT_REGISTRY.register
class GaNetIrInfer(nn.Module):
    """
    The basic structure of GaNetIrInfer.

    Args:
        ir_model: The ir model.
        post_process: Postprocess module.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        post_process: nn.Module,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.post_process = post_process

    def forward(self, data):
        kpts_hm, pts_offset, int_offset = self.ir_model(data)
        results = self.post_process(kpts_hm, pts_offset, int_offset, data)
        return results
