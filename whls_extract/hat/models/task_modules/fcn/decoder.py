# Copyright (c) Horizon Robotics. All rights reserved.
import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY

__all__ = ["FCNDecoder"]


@OBJECT_REGISTRY.register
class FCNDecoder(nn.Module):
    """FCN Decoder.

    Args:
        upsample_output_scale: Output upsample scale. Default: 8.
        use_bce: Whether using binary crosse entrypy. Default: False.
        bg_cls: Background classes id. Default: 0.
        bg_threshold: Background threshold. Default: 0.25.
    """

    def __init__(
        self,
        upsample_output_scale: int = 8,
        use_bce: bool = False,
        bg_cls: int = 0,
        bg_threshold: float = 0.25,
    ):
        super(FCNDecoder, self).__init__()
        self.upsample_output_scale = upsample_output_scale
        self.resize = torch.nn.Upsample(
            scale_factor=upsample_output_scale,
            align_corners=False,
            mode="bilinear",
        )
        self.use_bce = use_bce
        self.bg_cls = bg_cls
        self.bg_threshold = bg_threshold
        self.qconfig = None

    def forward(self, pred):
        pred = self.resize(pred)

        result = pred.argmax(dim=1)
        if self.use_bce:
            score, _ = pred.sigmoid().max(1)
            bg_indices = score < self.bg_threshold
            result[bg_indices] = self.bg_cls
        return result
