# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from collections import OrderedDict
from inspect import signature
from typing import Optional

import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["Segmentor", "SegmentorV2", "BMSegmentor", "SegmentorIrInfer"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Segmentor(nn.Module):
    """
    The basic structure of segmentor.

    Args:
        backbone (torch.nn.Module): Backbone module.
        neck (torch.nn.Module): Neck module.
        head (torch.nn.Module): Head module.
        losses (torch.nn.Module): Losses module.
    """

    def __init__(self, backbone, neck, head, losses=None):
        super(Segmentor, self).__init__()
        self.backbone = None
        self.neck = None
        self.head = None
        self.losses = None

        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.losses = losses

    @fx_wrap()
    def _post_process(self, preds, target):
        if target is None:
            return preds

        if not self.training or self.losses is None:
            return preds, target

        losses = self.losses(preds, target)
        return preds, losses

    def forward(self, data: dict):
        image = data["img"]
        target = data.get("gt_seg", None)
        if "uv_map" in signature(self.backbone.forward).parameters:
            features = self.backbone(image, uv_map=data.get("uv_map", None))
        else:
            features = self.backbone(image)
        preds = self.neck(features)
        preds = self.head(preds)

        return self._post_process(preds, target)

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head, self.losses]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [self.backbone, self.neck, self.head]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
        if self.losses is not None:
            self.losses.qconfig = None


@OBJECT_REGISTRY.register
class SegmentorV2(nn.Module):
    """
    The basic structure of segmentor.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        loss: Loss module.
        desc: Desc module
        postprocess: Postprocess module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        target: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        desc: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.target = target
        self.loss = loss
        self.desc = desc
        self.postprocess = postprocess

    def forward(self, data: dict):
        if "uv_map" in signature(self.backbone.forward).parameters:
            feat = self.backbone(data["img"], uv_map=data.get("uv_map", None))
        else:
            feat = self.backbone(data["img"])
        if "target" in data:
            data.update({"labels": data.pop("target")})
        feat = self.neck(feat)
        pred = self.head(feat)

        if self.desc is not None:
            pred = self.desc(pred)

        output = OrderedDict(pred=pred)

        if self.loss is not None:
            target = (
                data["labels"] if self.target is None else self.target(data)
            )
            output.update(self.loss(pred, target))

        if self.postprocess is not None:
            return self.postprocess(pred)

        return output

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [self.backbone, self.neck, self.head]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()

        if self.loss is not None:
            self.loss.qconfig = None
        if self.postprocess is not None:
            self.postprocess.qconfig = None


@OBJECT_REGISTRY.register
class BMSegmentor(SegmentorV2):
    """The segmentor structure that inputs image metas into postprocess."""

    def forward(self, data: dict):
        feat = self.backbone(data["img"])
        feat = self.neck(feat)
        pred = self.head(feat)

        if self.desc is not None:
            pred = self.desc(pred)

        if self.loss is not None:
            target = (
                data["labels"]
                if self.target is None
                else self.target(data, pred)
            )
            return self.loss(pred, target)  # diff with SegmentorV2

        if self.postprocess is not None:
            return self.postprocess(pred, data)  # diff with SegmentorV2

        output = OrderedDict(pred=pred)
        return output


@OBJECT_REGISTRY.register
class SegmentorIrInfer(nn.Module):
    """
    The basic structure of SegmentorIrInfer.

    Args:
        ir_model: The ir model.
    """

    def __init__(
        self,
        ir_model,
    ):
        super().__init__()
        self.ir_model = ir_model

    def forward(self, data):
        target = data.get("gt_seg", None)
        outputs = self.ir_model(data)

        if target is not None:
            return outputs, target
        return outputs
