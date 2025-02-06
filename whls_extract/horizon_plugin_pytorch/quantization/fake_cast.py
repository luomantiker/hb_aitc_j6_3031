from typing import Union

import torch
from torch import nn
from torch.quantization.observer import _with_args

from horizon_plugin_pytorch.qtensor import QTensor

fp16_min = float(torch.finfo(torch.float16).min)
fp16_max = float(torch.finfo(torch.float16).max)


class FakeCast(nn.Module):
    enable_clip = False
    check_overflow = False

    def __init__(self, dtype, enable_clip=None, check_overflow=None) -> None:
        assert dtype in (torch.float16, torch.float32, torch.bool)
        super().__init__()
        self.dtype = dtype
        self._enable_fake_cast = True

        if enable_clip is not None:
            self.enable_clip = enable_clip
        if check_overflow is not None:
            self.check_overflow = check_overflow

    def get_dtype(self):
        return self.dtype

    def enable_fake_quant(self, enabled):
        self._enable_fake_cast = enabled

    def forward(self, input: Union[torch.Tensor, QTensor]):
        if isinstance(input, QTensor):
            input = input.dequantize()
        if self.dtype == torch.float16 and self._enable_fake_cast:
            if not self.enable_clip and self.check_overflow:
                if (
                    torch.logical_or(input < fp16_min, input > fp16_max).sum()
                    > 0
                ):
                    raise ValueError("Input value out of fp16 range")
            return torch.ops.horizon.fake_cast(input, self.enable_clip)
        else:
            return input.float()

    def extra_repr(self):
        return f"dtype={self.dtype}"

    with_args = classmethod(_with_args)


default_fp16_fake_cast = FakeCast.with_args(dtype=torch.float16)
default_fp32_fake_cast = FakeCast.with_args(dtype=torch.float32)
