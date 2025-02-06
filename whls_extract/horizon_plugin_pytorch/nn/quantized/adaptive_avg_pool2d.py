import warnings

import torch
from torch.nn import Module
from torch.nn.modules.utils import _pair

from horizon_plugin_pytorch.nn import qat
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .functional import avg_pool2d


class AdaptiveAvgPool2d(Module):
    r"""Quantize version."""

    _QAT_MODULE = qat.AdaptiveAvgPool2d

    def __init__(
        self,
        output_size,
        out_dtype="qint8",
    ):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = _pair(output_size)
        self.out_dtype = out_dtype
        self.register_buffer("scale", torch.tensor([1], dtype=torch.float32))

    @typechecked
    def forward(self, x: QTensor) -> QTensor:
        if (
            x.size(-1) == self.output_size[-1] or self.output_size[-1] is None
        ) and (
            x.size(-2) == self.output_size[-2] or self.output_size[-2] is None
        ):
            return x

        unsqueeze_input = False
        if x.dim() == 3:
            warnings.warn(
                "Adaptive average pool 2d with 3-dimensional input is not"
                "supported, will turn input into [1, C, H, W]."
            )
            x = x.unsqueeze(0)
            unsqueeze_input = True

        kernels, strides = qat.adaptive_pool_utils.get_kernel_stride(
            (x.size(-2), x.size(-1)), self.output_size
        )
        out, out_scale = avg_pool2d(
            x.int_repr(),
            kernel_size=kernels,
            stride=strides,
            padding=(0, 0),
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None,
            input_scale=x.q_scale(),
            input_zero_point=x.q_zero_point(),
            input_dtype=x.dtype,
            scale=self.scale,
            zero_point=torch.zeros_like(self.scale).to(torch.long),
            dtype=self.out_dtype,
        )
        if unsqueeze_input:
            out = out.squeeze(0)
        return QTensor(out, out_scale, self.out_dtype)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module."""
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )

        activation_post_process = mod.activation_post_process
        out_dtype = (
            activation_post_process.dtype
            if activation_post_process is not None
            else "qint32"
        )
        qpool = cls(
            mod.output_size,
            out_dtype,
        )

        if out_dtype != "qint32":
            qpool.scale.copy_(activation_post_process.scale)
        return qpool
