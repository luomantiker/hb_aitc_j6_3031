# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["Classifier", "ClassifierIrInfer"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Classifier(nn.Module):
    """
    The basic structure of classifier.

    Args:
        backbone: Backbone module.
        losses: Losses module.
        make_backbone_graph: whether to use cuda_graph in backbone.
        num_warmup_iters: Num of iters for warmup of cuda_graph.

    """

    def __init__(
        self,
        backbone,
        losses=None,
        make_backbone_graph=False,
        num_warmup_iters=3,
    ):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.losses = losses

        self.make_backbone_graph = make_backbone_graph
        self.num_warmup_iters = num_warmup_iters
        self.graph_backbone = None
        self.stream = None

        if self.make_backbone_graph:
            assert (
                os.environ.get("HAT_USE_CUDAGRAPH", "0") == "1"
            ), "Please set HAT_USE_CUDAGRAPH=1 while use cuda-graph."

    @fx_wrap()
    def _post_process(self, preds, target):
        if target is None:
            return preds

        if not self.training or self.losses is None:
            return preds, target

        losses = self.losses(preds, target)
        return preds, losses

    @fx_wrap()
    def _pre_process(self, data, target=None):
        if isinstance(data, torch.Tensor):
            image = data
        else:
            image = data["img"]
            target = data.get("labels", None)
        return image, target

    def forward(self, data, target=None):
        image, target = self._pre_process(data, target)

        if self.make_backbone_graph:
            if self.graph_backbone is None:
                self.graph_backbone = torch.cuda.make_graphed_callables(
                    self.backbone,
                    (image,),
                    num_warmup_iters=self.num_warmup_iters,
                )
                preds = self.graph_backbone(image)
            else:
                preds = self.graph_backbone(image)
        else:
            preds = self.backbone(image)

        return self._post_process(preds, target)

    def fuse_model(self):
        for module in [self.backbone, self.losses]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.backbone is not None:
            if hasattr(self.backbone, "set_qconfig"):
                self.backbone.set_qconfig()

        if self.losses is not None:
            self.losses.qconfig = None

    def set_calibration_qconfig(self):
        from hat.utils import qconfig_manager

        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )
        if self.losses is not None:
            self.losses.qconfig = None


@OBJECT_REGISTRY.register
class ClassifierIrInfer(nn.Module):
    """
    The basic structure of ClassifierIrInfer.

    Args:
        model_path: The path of hbir model.
    """

    def __init__(
        self,
        ir_model: nn.Module,
    ):
        super().__init__()
        self.ir_model = ir_model

    def forward(self, data):
        outputs = self.ir_model(data).squeeze(-1).squeeze(-1)
        target = data.get("labels", None)

        if target is not None:
            return outputs, target
        return outputs
