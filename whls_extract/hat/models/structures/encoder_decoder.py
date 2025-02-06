# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from horizon_plugin_pytorch.qtensor import QTensor

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

__all__ = ["EncoderDecoder", "EncoderDecoderIrInfer"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class EncoderDecoder(nn.Module):
    """
    The basic structure of encoder decoder.

    Args:
        backbone: Backbone module.
        decode_head: Decode head module.
        target: Target module for decode head. Default: None.
        loss: Loss module for decode head. Default: None.
        neck: Neck module.
            Default: None.
        auxiliary_heads: List of auxiliary head modules \
which contains of  "head", "target", "loss".
            Default: None.
        decode: decode. Defualt: None.
        with_target: Whether return target during inference.
    """

    def __init__(
        self,
        backbone: nn.Module,
        decode_head: nn.Module,
        target: Optional[object] = None,
        loss: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        auxiliary_heads: Optional[List[Dict]] = None,
        decode: Optional[object] = None,
        with_target: Optional[nn.Module] = False,
    ):
        super(EncoderDecoder, self).__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.target = target or None
        self.loss = loss or None

        if auxiliary_heads is not None:
            self.auxiliary_heads = nn.ModuleList(
                [auxiliary_head["head"] for auxiliary_head in auxiliary_heads]
            )
            self.auxiliary_target = [
                auxiliary_head["target"] for auxiliary_head in auxiliary_heads
            ]
            self.auxiliary_loss = [
                auxiliary_head["loss"] for auxiliary_head in auxiliary_heads
            ]
        else:
            self.auxiliary_heads = None
        self.neck = neck or None
        self.with_target = with_target
        self.decode = decode or None

    @fx_wrap()
    def _post_process(self, gts, preds, features):
        if self.training:
            target = self.target(gts, preds)
            decode_loss = self.loss(**target)
            aux_losses = OrderedDict()
            if self.auxiliary_heads is not None:
                features = [
                    i.as_subclass(torch.Tensor)
                    if isinstance(i, QTensor)
                    else i
                    for i in features
                ]
                for head, target_mod, loss in zip(
                    self.auxiliary_heads,
                    self.auxiliary_target,
                    self.auxiliary_loss,
                ):
                    preds = head(features)
                    target = target_mod(gts, preds)
                    aux_loss = loss(**target)
                    aux_losses.update(aux_loss)
            return {**decode_loss, **aux_losses}
        else:
            if self.decode is not None:
                preds = self.decode(preds)
            if self.with_target is False:
                return preds
            return preds, target

    def forward(self, data: dict):
        image = data["img"]
        gts = data.get("gt_seg", None)
        features = self.backbone(image)
        if self.neck is not None:
            features = self.neck(features)
        preds = self.decode_head(features)
        return self._post_process(gts, preds, features)

    def fuse_model(self):
        for mod in [self.backbone, self.neck, self.decode_head]:
            if mod is not None and hasattr(mod, "fuse_model"):
                mod.fuse_model()
        if self.auxiliary_heads is not None:
            for mod in self.auxiliary_heads:
                if mod is not None and hasattr(mod, "fuse_model"):
                    mod.fuse_model()

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [self.backbone, self.neck, self.decode_head]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()

        if self.loss is not None:
            self.loss.qconfig = None

        if self.target is not None:
            self.target.qconfig = None

        if self.decode is not None:
            self.decode.qconfig = None

        if self.auxiliary_heads is not None:
            for i in range(len(self.auxiliary_heads)):
                self.auxiliary_heads[i].qconfig = None
                self.auxiliary_target[i].qconfig = None
                self.auxiliary_loss[i].qconfig = None


@OBJECT_REGISTRY.register
class EncoderDecoderIrInfer(nn.Module):
    """
    The basic structure of EncoderDecoderIrInfer.

    Args:
        ir_model: The ir model.
        post_process: Postprocess module.
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
            results = self.post_process(outputs)
            return results
        return outputs
