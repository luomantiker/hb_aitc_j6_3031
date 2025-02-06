# Copyright (c) Horizon Robotics. All rights reserved.
import torch
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY
from .metric import EvalMetric

__all__ = ["EndPointError"]


@OBJECT_REGISTRY.register
class EndPointError(EvalMetric):
    """Metric for OpticalFlow task, endpoint error (EPE).

    The endpoint error measures the distance between the
    endpoints of two optical flow vectors (u0, v0) and (u1, v1)
    and is defined as sqrt((u0 - u1) ** 2 + (v0 - v1) ** 2).

    Args:
        name: metric name.
    Refs:
        https://github.com/philferriere/tfoptflow/blob/master/tfoptflow/model_pwcnet.py
    """

    def __init__(
        self,
        name="EPE",
        use_mask=False,
    ):
        super(EndPointError, self).__init__(name)
        self.name = name
        self.use_mask = use_mask

    def update(self, labels, preds, masks=None):
        if self.use_mask:
            for label, pred, mask in zip(labels, preds, masks):
                if mask.sum() == 0:
                    continue
                else:
                    D_est, D_gt = pred[mask], label[mask]
                    epe = F.l1_loss(D_est, D_gt, reduction="mean").item()
                    self.sum_metric += epe
                    self.num_inst += 1
        else:
            diff = preds - labels
            bs = preds.shape[0]
            epe = torch.norm(diff, p=2, dim=1).mean((1, 2)).sum().item()
            self.sum_metric += epe
            self.num_inst += bs
