# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional

import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["YOLOV3", "YOLOIrInfer"]


@OBJECT_REGISTRY.register
class YOLOV3(nn.Module):
    """
    The basic structure of yolov3.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        anchor_generator: Anchor generator module.
        target_generator: Target generator module.
        loss: Loss module.
        postprocess: Postprocess module.
    """

    def __init__(
        self,
        backbone: Optional[dict] = None,
        neck: Optional[dict] = None,
        head: Optional[dict] = None,
        filter_module: Optional[dict] = None,
        anchor_generator: Optional[dict] = None,
        target_generator: Optional[dict] = None,
        loss: Optional[dict] = None,
        postprocess: Optional[dict] = None,
    ):
        super(YOLOV3, self).__init__()

        names = [
            "backbone",
            "neck",
            "head",
            "filter_module",
            "anchor_generator",
            "target_generator",
            "loss",
            "postprocess",
        ]
        modules = [
            backbone,
            neck,
            head,
            filter_module,
            anchor_generator,
            target_generator,
            loss,
            postprocess,
        ]
        for name, module in zip(names, modules):
            if module is not None:
                setattr(self, name, module)
            else:
                setattr(self, name, None)

    @fx_wrap()
    def _post_process(self, data, x):
        if self.training:
            losses = self.loss(x, (data["gt_bboxes"], data["gt_classes"]))
            return losses, x
        elif self.filter_module is not None:
            return self.filter_module(x)
        elif self.postprocess is not None:
            x = self.postprocess(x)
            data["pred_bboxes"] = x
            data.pop("gt_labels", None)
            return data
        else:
            return x

    def forward(self, data):
        image = data["img"]

        x = self.backbone(image)
        if self.neck is not None:
            x = self.neck(x)
        x = self.head(x)
        return self._post_process(data, x)

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
class YOLOIrInfer(nn.Module):
    """
    The basic structure of YOLOIrInfer.

    Args:
        ir_model: The ir model.
        postprocess: Postprocess module.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        postprocess: nn.Module = None,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.postprocess = postprocess

    def forward(self, data):
        outputs = self.ir_model(data)
        if self.postprocess is not None:
            outputs = self.postprocess(outputs)
            data["pred_bboxes"] = outputs
            data.pop("gt_labels", None)
            return data
        return outputs
