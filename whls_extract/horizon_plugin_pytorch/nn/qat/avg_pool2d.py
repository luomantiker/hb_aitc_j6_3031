import logging

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.nn.modules.utils import _pair

from horizon_plugin_pytorch.march import March, get_march
from horizon_plugin_pytorch.qtensor import QTensor
from .qat_meta import QATModuleMeta

logger = logging.getLogger(__name__)


def bernoulli_avg_pool2d(
    input,
    kernel_size,
    stride,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
):
    kernel_size = _pair(kernel_size)
    hw_reciprocal = 1 / (kernel_size[0] * kernel_size[1])
    # avg = accumulator * (int(hw_reciprocal * 2 ** 9) / 2 ** 9)
    divisor_shift = 9
    return F.avg_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        1,
    ) * (
        int(hw_reciprocal * 2 ** divisor_shift) * (1.0 / (2 ** divisor_shift))
    )


class AvgPool2d(nn.AvgPool2d, metaclass=QATModuleMeta):
    def forward(self, input: QTensor) -> QTensor:
        march = get_march()
        if march == March.BERNOULLI:
            if self.divisor_override is not None:
                raise ValueError(
                    "divisor_override is not supported on bernoulli"
                )
            out = bernoulli_avg_pool2d(
                input.as_subclass(torch.Tensor),
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
            )
            if self.activation_post_process is not None:
                if input.dtype != self.activation_post_process.dtype:
                    old_dtype = self.activation_post_process.dtype
                    self.activation_post_process.reset_dtype(
                        input.dtype, False
                    )
                    logger.warning(
                        f"{self.__class__.__name__} output dtype {old_dtype} "
                        + f"will be changed to {input.dtype}."
                    )
                self.activation_post_process.disable_observer()
                self.activation_post_process.set_qparams(input.q_scale())
        else:
            out = F.avg_pool2d(
                input.as_subclass(torch.Tensor),
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        return out
