# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List, Mapping

import torch
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY

__all__ = ["DequantModule"]


@OBJECT_REGISTRY.register
class DequantModule(torch.nn.Module):
    """Do dequant to data.

    Args:
        data_names: A list of data names that need dequantization.
    """

    def __init__(
        self,
        data_names: List,
    ):
        super(DequantModule, self).__init__()
        self.data_names = data_names
        self.dequant = DeQuantStub()

    def forward(self, pred_dict: Mapping, *args):
        for key, value in pred_dict.items():
            if key not in self.data_names:
                continue
            if isinstance(value, list):
                dequant_outs = []
                for each_data in value:
                    dequant_outs.append(self.dequant(each_data))
                pred_dict[key] = dequant_outs
            else:
                pred_dict[key] = self.dequant(value)

        return pred_dict

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
