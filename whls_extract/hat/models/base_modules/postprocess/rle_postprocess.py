# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Mapping

try:
    from horizon_plugin_pytorch.nn.functional import rle
except ImportError:
    rle = None
import torch

from hat.registry import OBJECT_REGISTRY

__all__ = ["RLEPostprocess"]


@OBJECT_REGISTRY.register
class RLEPostprocess(torch.nn.Module):
    """Apply run length encoding of data in pred_dict.

    Compress dense output with patches of identical value
    by run length encoding, e.g., for semantic segmentation
    result. Note that current plugin rle only support for
    value processed by argmax.

    Args:
        data_name (str): name of data to apply run length encoding.
        dtype(torch.dtype): The value field dtype in compressed result.
            !!! Note: Not compressed results dtype. Result dtype is int64 !!!
            Support torch.int8 or torch.int16. if input is torch.max
            indices out, dtype must be torch.int16
            if value dtype = torch.int8, num dtype is uint8, max num is 255
            if value dtype = torch.int16, num dtype is uint16, max num is 65535

    """

    def __init__(self, data_name: str, dtype: torch.dtype):
        super(RLEPostprocess, self).__init__()
        self.data_name = data_name
        self.dtype = dtype

    def forward(self, pred_dict: Mapping, *args):
        if isinstance(pred_dict[self.data_name], list):
            rle_datas = []
            for each_data in pred_dict[self.data_name]:
                rle_datas.append(rle(each_data, self.dtype)[0])
            pred_dict[self.data_name] = rle_datas
        elif isinstance(pred_dict[self.data_name], torch.Tensor):
            pred_dict[self.data_name] = rle(
                pred_dict[self.data_name], self.dtype
            )[0]
        else:
            raise TypeError("only support torch.tensor or list[torch.tensor]")
        return pred_dict

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
