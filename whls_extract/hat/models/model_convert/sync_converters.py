# Copyright (c) Horizon Robotics. All rights reserved.

import logging

import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_local_process_group
from .converters import BaseConverter

__all__ = [
    "SyncBnConvert",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class SyncBnConvert(BaseConverter):
    """Fix qscale of weight while calibration or qat stage."""

    def __init__(self):
        super(SyncBnConvert, self).__init__()

    def __call__(self, model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(
            model, get_local_process_group()
        )
        return model
