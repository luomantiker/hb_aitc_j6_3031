# Copyright (c) Horizon Robotics. All rights reserved.

from typing import List, Mapping

import torch

from hat.registry import OBJECT_REGISTRY

__all__ = ["MaxPostProcess"]


@OBJECT_REGISTRY.register
class MaxPostProcess(torch.nn.Module):
    """Apply max of data in pred_dict.

    Args:
        data_names: names of data to apply max.
        out_names: out names of data after max, order is related to data_names.
        dim: the dimension to reduce.
        keepdim: whether the output tensor has dim retained or not.
        return_indices: whether return indices corresponding to max.
    """

    def __init__(
        self,
        data_names: list,
        out_names: List[List[str]],
        dim: int,
        keepdim: bool = False,
        return_indices: bool = True,
    ):
        super(MaxPostProcess, self).__init__()
        self.data_names = data_names
        self.out_names = out_names
        self.dim = dim
        self.keepdim = keepdim
        self.return_indices = return_indices

    def forward(self, pred_dict: Mapping, *args):
        for idx, data_name in enumerate(self.data_names):
            assert data_name in pred_dict
            values = pred_dict[data_name]
            if isinstance(values, list):
                classes_datas = []
                score_datas = []
                for value in values:
                    scores, classes = value.max(self.dim, self.keepdim)
                    score_datas.append(scores)
                    classes_datas.append(classes)
                pred_dict[self.out_names[idx][0]] = score_datas
                if self.return_indices:
                    pred_dict[self.out_names[idx][1]] = classes_datas
            elif isinstance(values, torch.Tensor):
                scores, classes = values.max(self.dim, self.keepdim)
                pred_dict[self.out_names[idx][0]] = scores
                if self.return_indices:
                    pred_dict[self.out_names[idx][1]] = classes
            else:
                raise TypeError(
                    "only support torch.tensor or list[torch.tensor]"
                )
        return pred_dict

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
