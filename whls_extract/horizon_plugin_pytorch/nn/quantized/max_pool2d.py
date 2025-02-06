from torch.nn import Module
from torch.nn.modules.utils import _pair

from horizon_plugin_pytorch.nn import qat
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .functional import max_pool2d


class MaxPool2d(Module):
    r"""Quantize version."""

    _QAT_MODULE = qat.MaxPool2d

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
        out_dtype="qint8",
    ):
        super(MaxPool2d, self).__init__()
        assert dilation == 1, "MaxPool2d only supports `dilation` = 1"
        assert (
            return_indices is False
        ), "MaxPool2d does not supports `return_indices` = True"

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else None
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    @typechecked
    def forward(self, x: QTensor) -> QTensor:
        out = max_pool2d(
            x.int_repr(),
            kernel_size=self.kernel_size,
            stride=self.stride
            if self.stride is not None
            else self.kernel_size,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )
        return QTensor(out, x.q_scale(), x.dtype)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module."""
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )
        qpool = cls(
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.return_indices,
            mod.ceil_mode,
        )
        return qpool
