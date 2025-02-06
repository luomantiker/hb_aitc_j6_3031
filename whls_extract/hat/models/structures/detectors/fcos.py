# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from inspect import signature
from typing import Dict

from torch import nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = ["FCOS", "FCOSIrInfer"]


@OBJECT_REGISTRY.register
class FCOS(nn.Module):
    """The basic structure of fcos.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        targets: Target module.
        loss_cls: Classification loss module.
        loss_reg: Regiression loss module.
        loss_centerness: Centerness loss module.
        desc: Description module.
        postprocess: Postprocess module.

    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        targets: nn.Module = None,
        desc: nn.Module = None,
        post_process: nn.Module = None,
        loss_cls: nn.Module = None,
        loss_reg: nn.Module = None,
        loss_centerness: nn.Module = None,
    ):
        super(FCOS, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.loss_centerness = loss_centerness
        self.desc = desc

        self.targets = targets
        self.post_process = post_process

    def extract_feat(self, img, uv_map=None):
        """Directly extract features from the backbone + neck."""
        if "uv_map" in signature(self.backbone.forward).parameters:
            x = self.backbone(img, uv_map)
        else:
            x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x

    @fx_wrap()
    def _post_process(self, data, preds):
        if self.training and self.targets is not None:
            cls_targets, reg_targets, centerness_targets = self.targets(
                data, preds
            )

            cls_loss = self.loss_cls(**cls_targets)
            reg_loss = self.loss_reg(**reg_targets)
            centerness_loss = self.loss_centerness(**centerness_targets)
            return dict(**cls_loss, **reg_loss, **centerness_loss)
        else:
            if self.post_process is None:
                return preds
            results = self.post_process(preds, data)
            return results

    def forward(self, data: Dict):
        imgs = data["img"]
        feats = self.extract_feat(imgs, data.get("uv_map", None))
        preds = self.head(feats)
        if self.desc is not None:
            preds = self.desc(preds)
        return self._post_process(data, preds)

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


@OBJECT_REGISTRY.register
class FCOSIrInfer(nn.Module):
    """
    The basic structure of FCOSIrInfer.

    Args:
        ir_model: ir model
        num_class: The num of class.
        post_process: Postprocess module.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        num_class: int = 80,
        post_process: nn.Module = None,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.post_process = post_process
        self.num_class = num_class

    def forward(self, data):
        outputs = self.ir_model(data)
        if self.post_process is not None:
            outputs = self.post_process(outputs, data)
        return outputs
