# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict

from torch import nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = ["Detr", "DetrIrInfer"]


@OBJECT_REGISTRY.register
class Detr(nn.Module):
    """The basic structure of detr.

    Args:
        backbone: backbone module.
        neck: neck module.
        head: head module with transformer architecture.
        criterion: loss module.
        post_process: post process module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        criterion: nn.Module = None,
        post_process: nn.Module = None,
    ):
        super(Detr, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.criterion = criterion
        self.post_process = post_process

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck."""
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x

    @fx_wrap()
    def _post_process(self, data, outputs):
        if self.training:
            loss_dict = self.criterion(outputs, data)
            return loss_dict
        else:
            if self.post_process is None:
                return outputs
            results = self.post_process(outputs, data)
            return results

    def forward(self, data: Dict):
        imgs = data["img"]
        feats = self.extract_feat(imgs)
        outputs = self.head(feats, data)
        return self._post_process(data, outputs)

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

    def set_calibration_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_calibration_qconfig()
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "set_calibration_qconfig"):
                module.set_calibration_qconfig()


@OBJECT_REGISTRY.register
class DetrIrInfer(nn.Module):
    """
    The basic structure of DetrIrInfer.

    Args:
        ir_model: ir model
        post_process: Post process module.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        post_process: nn.Module = None,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.post_process = post_process

    def forward(self, data):
        outputs = self.ir_model(data)
        if self.post_process is not None:
            results = self.post_process(outputs, data)
            return results
        return outputs
