# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict

import torch
from torch import nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = ["MotionForecasting", "MotionForecastingIrInfer"]


@OBJECT_REGISTRY.register
class MotionForecasting(nn.Module):
    """The basic structure of motion forecasting.

    Args:
        encoder: encoder module.
        decoder: decoder module.
        target: target generator.
        loss: loss module.
        post_process: post process module.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        target: nn.Module = None,
        loss: nn.Module = None,
        postprocess: nn.Module = None,
    ):
        super(MotionForecasting, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target = target
        self.loss = loss
        self.postprocess = postprocess

    def forward(self, data: Dict):
        traj_feat = data["traj_feat"]
        lane_feat = data["lane_feat"]
        instance_mask = data["instance_mask"]
        feats = self.encoder(traj_feat, lane_feat, instance_mask)
        feats = self.decoder(*feats, data)
        return self._post_process(feats, data)

    @fx_wrap()
    def _post_process(self, feats, data):
        if not self.training:
            if self.postprocess is not None:
                result = self.postprocess(*feats, data)
                return result
            return feats
        if self.target and self.loss:
            targets = self.target(*feats, data)
            losses = self.loss(*targets)
            return losses

        return feats

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.encoder, self.decoder]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


@OBJECT_REGISTRY.register
class MotionForecastingIrInfer(nn.Module):
    """
    The basic structure of MotionForecastingIrInfer.

    Args:
        ir_model: The ir model.
        pad_batch: The num of pad for batchdata.
        postprocess: Postprocess module.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        pad_batch: int = 30,
        postprocess: nn.Module = None,
    ):
        super().__init__()

        self.ir_model = ir_model
        self.pad_batch = pad_batch
        self.postprocess = postprocess

    def _pad(self, data):
        bs, h, w, c = data.shape
        mask = torch.ones((self.pad_batch)).to(device=data.device)
        mask[bs:] = 0
        if bs < self.pad_batch:
            left = self.pad_batch - bs
            pad = torch.zeros((left, h, w, c)).to(device=data.device)
            data = torch.cat([data, pad])
        return data, mask

    def forward(self, data):
        traj_feat, mask = self._pad(data["traj_feat"])
        lane_feat, _ = self._pad(data["lane_feat"])
        instance_mask, _ = self._pad(data["instance_mask"])
        goals, _ = self._pad(data["goals_2d"])
        goals_2d_mask, _ = self._pad(data["goals_2d_mask"])

        inputs = {
            "traj_feat": traj_feat,
            "lane_feat": lane_feat,
            "instance_mask": instance_mask,
            "goals_2d": goals,
            "goals_2d_mask": goals_2d_mask,
        }

        outputs = self.ir_model(inputs)
        for i in range(len(outputs)):
            valid = mask == 1
            outputs[i] = outputs[i][valid]
        if self.postprocess is not None:
            result = self.postprocess(*outputs, data)
            return result

        return outputs
