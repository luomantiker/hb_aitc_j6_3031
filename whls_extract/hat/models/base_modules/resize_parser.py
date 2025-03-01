# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Mapping, Optional, Sequence, Union

import horizon_plugin_pytorch.nn as hnn
import torch
import torch.nn.functional as F
from torch.quantization import DeQuantStub

from hat.registry import OBJECT_REGISTRY

__all__ = ["ResizeParser"]

logger = logging.getLogger(__name__)


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = input.shape[2:]
            output_h, output_w = size
            if output_h > input_h or output_w > output_h:
                if (
                    (
                        output_h > 1
                        and output_w > 1
                        and input_h > 1
                        and input_w > 1
                    )
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    logger.warning(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    output = F.interpolate(
        input.float(), size, scale_factor, mode, align_corners
    )
    if torch.is_autocast_enabled():
        output = output.to(torch.float16)
    return output


@OBJECT_REGISTRY.register
class ResizeParser(torch.nn.Module):
    """Resize multi stride preds to specific size.

        e.g. segmentation, depth, flow an so on.

    Args:
        data_name: name of original data to resize.
        resized_data_name: name of data after resize. None means update
            in data_name inplace.
        resize_kwargs: key args of resize.
        use_plugin_interpolate: whether use
            horizon_plugin_pytorch.nn.Interpolate.
        dequant_out: whether dequant output when
            use_plugin_interpolate is True.
    """

    def __init__(
        self,
        resize_kwargs: Mapping,
        data_name: str = None,
        resized_data_name: Optional[str] = None,
        use_plugin_interpolate: bool = False,
        dequant_out: bool = True,
    ):
        super(ResizeParser, self).__init__()
        self.data_name = data_name
        self.resized_data_name = resized_data_name
        self.resize_kwargs = resize_kwargs
        self.use_plugin_interpolate = use_plugin_interpolate
        self.dequant_out = dequant_out

        if use_plugin_interpolate:
            self.resize = hnn.Interpolate(**self.resize_kwargs)
        if dequant_out:
            self.dequant = DeQuantStub()

    def _resize(self, data):
        if self.use_plugin_interpolate:
            resize_data = self.resize(data)
            if self.dequant_out:
                resize_data = self.dequant(resize_data)
        else:
            resize_data = resize(data, **self.resize_kwargs)
        return resize_data

    def forward(self, preds: Union[torch.Tensor, Sequence, Mapping]):
        if isinstance(preds, Mapping) and self.data_name not in preds:
            return preds

        ori_data = (
            preds[self.data_name] if isinstance(preds, Mapping) else preds
        )
        if isinstance(ori_data, Sequence):
            resize_data = []
            for one_stride in ori_data:
                resize_data.append(self._resize(one_stride))
        else:
            resize_data = self._resize(ori_data)
        if self.resized_data_name is not None:
            preds[self.resized_data_name] = resize_data
        else:
            if self.data_name is not None:
                preds[self.data_name] = resize_data
            else:
                preds = resize_data
        return preds

    def set_qconfig(self):
        from hat.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
