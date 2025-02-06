# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor, nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

__all__ = ["StereoNet", "StereoNetPlus", "StereoNetIrInfer"]


@OBJECT_REGISTRY.register
class StereoNet(nn.Module):
    """The basic structure of StereoNet.

    Args:
        backbone: backbone module.
        neck: neck module
        head: head module.
        post_process: post_process module.
        loss: loss module.
        loss_weights:  loss weights for each feature.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        post_process: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        loss_weights: Optional[List[float]] = None,
    ):
        super(StereoNet, self).__init__()

        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.post_process = post_process
        self.loss = loss
        self.maxdisp = self.head.maxdisp

        if loss is not None:
            assert (
                loss_weights is not None
            ), "For train step, loss_weights must is not None."
        if loss_weights is not None:
            self.loss_weights = torch.Tensor(loss_weights)

    @fx_wrap()
    def _post_process(
        self,
        pred_disps: List,
        data: Dict,
    ) -> Union[List[Tensor], Dict]:
        """Perform post-processing on predicted disparities.

        Args:
            pred_disps: Predicted disparities.
            data: Input data.
        """

        if self.training:
            assert self.post_process is not None
            assert self.loss is not None
            gt_disps = data["gt_disp"]
            pred_disps = self.post_process(pred_disps, gt_disps)
            mask = (gt_disps > 0) & (gt_disps < self.maxdisp)
            losses = []
            for pred_disp, loss_weight in zip(pred_disps, self.loss_weights):
                loss = self.loss(pred_disp[mask], gt_disps[mask])
                losses.append(loss_weight * loss)
            return {
                "losses": losses,
                "pred_disps": pred_disps[-1],
            }
        else:
            if self.post_process is None:
                return pred_disps[-2], pred_disps[-1]
            else:
                return self.post_process(pred_disps)

    def forward(self, data: Dict) -> Union[List, Dict]:
        """Perform the forward pass of the model.

        Args:
            data: The input data,

        """
        feat = self.backbone(data["img"])
        feat = self.neck(feat) if self.neck else feat
        pred_disps = self.head(feat)
        outputs = self._post_process(pred_disps, data)
        return outputs

    def fuse_model(self) -> None:
        """Perform model fusion on the specified modules within the class."""

        for module in [
            self.backbone,
            self.neck,
            self.head,
        ]:
            if module is not None:
                if hasattr(module, "fuse_model"):
                    module.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

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
        if self.loss is not None:
            self.loss.qconfig = None
        if self.post_process is not None:
            self.post_process.qconfig = None


@OBJECT_REGISTRY.register
class StereoNetPlus(StereoNet):
    """The basic structure of StereoNetPlus.

    Args:
        backbone: backbone module.
        neck: neck module
        head: head module.
        post_process: post_process module.
        loss: loss module.
        loss_weights:  loss weights for each feature.
        num_fpn_feat: the number of featmap use fpn.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        post_process: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        loss_weights: Optional[List[float]] = None,
        num_fpn_feat: int = 3,
    ):
        super(StereoNetPlus, self).__init__(
            backbone, neck, head, post_process, loss, loss_weights
        )
        self.num_fpn_feat = num_fpn_feat

    def forward(self, data: Dict) -> Union[List, Dict]:
        """Perform the forward pass of the model.

        Args:
            data: The input data,

        """
        feat = self.backbone(data["img"])
        if self.neck is not None:
            feat[-self.num_fpn_feat :] = self.neck(feat[-self.num_fpn_feat :])
        pred_disps = self.head(feat)
        outputs = self._post_process(pred_disps, data)
        return outputs


@OBJECT_REGISTRY.register
class StereoNetIrInfer(nn.Module):
    """
    The basic structure of StereoNetIrInfer.

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

        outputs = self.ir_model(data)
        results = self.post_process(outputs, data)
        return results
