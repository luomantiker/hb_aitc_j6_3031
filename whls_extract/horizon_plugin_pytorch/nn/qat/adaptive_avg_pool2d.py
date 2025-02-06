import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.nn.modules.utils import _pair

from horizon_plugin_pytorch.march import March, get_march
from .adaptive_pool_utils import get_kernel_stride
from .avg_pool2d import bernoulli_avg_pool2d
from .qat_meta import QATModuleMeta

logger = logging.getLogger(__name__)


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, metaclass=QATModuleMeta):
    def forward(self, input):
        output_size = _pair(self.output_size)
        if (
            input.size(-1) == output_size[-1]
            and input.size(-2) == output_size[-2]
        ):
            return input

        if get_march() == March.BERNOULLI:
            kernels, strides = get_kernel_stride(
                (input.size(-2), input.size(-1)), output_size
            )
            out = bernoulli_avg_pool2d(input.dequantize(), kernels, strides)
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
                self.activation_post_process.scale.copy_(input.q_scale())
                self.activation_post_process.disable_observer()
        else:
            out = F.adaptive_avg_pool2d(
                input.as_subclass(torch.Tensor),
                self.output_size,
            )
        if self.activation_post_process is not None:
            return self.activation_post_process(out)
        return out
