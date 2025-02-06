# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional

import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register_module
class QuantModule(nn.Module):
    """Do quant to data.

    Args:
        scale: Sacle value of quantization.
    """

    def __init__(
        self,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.quant = QuantStub(scale=scale)

    def forward(self, x):
        x = self.quant(x)
        return x

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
