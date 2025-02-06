# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Mapping

import horizon_plugin_pytorch
import torch
from horizon_plugin_pytorch.quantization import March
from torch.quantization import DeQuantStub, QuantStub

from hat.registry import OBJECT_REGISTRY

__all__ = ["ArgmaxPostprocess", "HorizonAdasClsPostProcessor"]


# TODO(mengao.zhao, HDLT-235): refactor argmax_postprocess #
@OBJECT_REGISTRY.register
class ArgmaxPostprocess(torch.nn.Module):
    """Apply argmax of data in pred_dict.

    Args:
        data_name (str): name of data to apply argmax.
        dim (int): the dimension to reduce.
        keepdim (bool): whether the output tensor has dim retained or not.

    """

    def __init__(self, data_name: str, dim: int, keepdim: bool = False):
        super(ArgmaxPostprocess, self).__init__()
        self.data_name = data_name
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, pred_dict: Mapping, *args):
        if isinstance(pred_dict[self.data_name], list):
            argmax_datas = []
            for each_data in pred_dict[self.data_name]:
                argmax_datas.append(each_data.argmax(self.dim, self.keepdim))
            pred_dict[self.data_name] = argmax_datas
        elif isinstance(pred_dict[self.data_name], torch.Tensor):
            pred_dict[self.data_name] = pred_dict[self.data_name].argmax(
                self.dim, self.keepdim
            )
        else:
            raise TypeError("only support torch.tensor or list[torch.tensor]")
        return pred_dict

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class HorizonAdasClsPostProcessor(torch.nn.Module):
    """Apply argmax of data in pred_dict.

    Args:
        data_name (str): name of data to apply argmax.
        dim (int): the dimension to reduce.
        keepdim (bool): whether the output tensor has dim retained or not.

    """

    def __init__(
        self,
        data_name: str,
        dim: int,
        keep_dim: bool = True,
        march: str = March.BAYES,
    ):
        super(HorizonAdasClsPostProcessor, self).__init__()
        self.data_name = data_name
        self.dim = dim
        self.keep_dim = keep_dim
        self.march = march

    @torch.no_grad()
    def forward(self, pred_cls: Mapping, *args):
        assert isinstance(pred_cls, torch.Tensor), "only support torch.Tensor"
        batch_scores, batch_cls_idxs = pred_cls.max(
            dim=self.dim, keepdim=self.keep_dim
        )

        if self.march == March.BAYES:
            return torch.cat((batch_cls_idxs, batch_scores), dim=self.dim)
        else:
            return batch_cls_idxs

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class ClsArgmaxPostProcessor(torch.nn.Module):
    """Apply argmax of data in pred_dict.

    Args:
        data_name: name of data to apply argmax.
        dim: the dimension to reduce.
        keep_dim: whether the output tensor has dim retained or not.
        march: March platform.
        with_softmax: Whether output with softmax.
        argmax: Whether output with argmax.
        with_track_feat: Whether output with tracking feature.
        max_value_only: Whether only use max value.
    """

    def __init__(
        self,
        data_name: str,
        dim: int,
        keep_dim: bool = True,
        march: str = March.BAYES,
        with_softmax=False,
        argmax=True,
        with_track_feat=False,
        max_value_only=True,
    ):
        super(ClsArgmaxPostProcessor, self).__init__()
        self.data_name = data_name
        self.dim = dim
        self.keep_dim = keep_dim
        self.march = march
        self.with_softmax = with_softmax
        self.dequant = DeQuantStub()
        self.quant = QuantStub()
        if self.with_softmax:
            self.softmax = horizon_plugin_pytorch.nn.SoftmaxBernoulli2(
                dim=self.dim, max_value_only=max_value_only
            )
        self.argmax = argmax
        self.with_track_feat = with_track_feat

    @torch.no_grad()
    def forward(self, pred_cls: Mapping, *args):
        if self.with_track_feat:
            # if len(pred_cls) > 1, pred_cls[1] is extra data to output
            assert len(pred_cls) > 1
            extra_feat = pred_cls[1]
            pred_cls = pred_cls[0]
        assert isinstance(pred_cls, torch.Tensor), "only support torch.Tensor"
        output_list = []
        if self.with_track_feat:
            output_list.append(extra_feat)
        if self.argmax:
            argmax_batch_cls_scores, _ = pred_cls.max(
                dim=self.dim, keepdim=self.keep_dim
            )
            argmax_batch_cls_scores = self.dequant(argmax_batch_cls_scores)
            output_list.append(argmax_batch_cls_scores)
        if self.with_softmax:
            softmax_score = self.softmax(
                pred_cls
            )  # retrun dtype : torch.float32
            # softmax_score = self.dequant(softmax_score)
            output_list.append(softmax_score)
        if not self.with_softmax and not self.argmax:
            output_list.append(self.dequant(pred_cls))

        return [output_list]

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
