from typing import Union

import torch
from torch import Tensor, nn

from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.nn.qat.qat_meta import (
    init_input_preprocess,
    is_float,
    pre_process,
)
from horizon_plugin_pytorch.qtensor import QTensor
from ..ceil import Ceil as FloatCeil
from .functional_modules import FloatFunctional
from .segment_lut import SegmentLUT


class Ceil(nn.Module):
    """qat version of Ceil module."""

    _FLOAT_MODULE = FloatCeil

    def __init__(self, qconfig):
        super(Ceil, self).__init__()
        self.qconfig = qconfig
        assert qconfig is not None, "qconfig must be provided"
        assert (
            self.qconfig.activation is not None
        ), "qconfig activation must be provided"

        if self.qconfig.activation().dtype in (qint8,):
            self.ceil = SegmentLUT(
                torch.ceil,
                False,
                None,
                input_range=None,
                auto_divide_strategy="evenly",
                qconfig=qconfig,
            )
        elif self.qconfig.activation().dtype == qint16:
            self.ceil = FloatFunctional(qconfig=qconfig)
        else:
            assert self.qconfig.activation().dtype in (
                torch.float16,
                torch.float32,
            )

        self.activation_pre_process = init_input_preprocess(qconfig)
        self.is_float = is_float(qconfig.activation())

    def forward(self, input: Union[QTensor, Tensor]):
        input = pre_process(self.activation_pre_process, input)
        if self.is_float:
            return torch.ceil(input)
        if isinstance(self.ceil, SegmentLUT) or (
            hasattr(self.ceil, "_QAT_MODULE")
            and self.ceil._QAT_MODULE is SegmentLUT
        ):
            assert (
                input.dtype == qint8
            ), f"nn.Ceil only support same input/output dtype, but get input {input.dtype} while output qint8. Try nn.quantized.FloatFunctional.ceil instead."  # noqa: E501
            return self.ceil(input)
        assert (
            input.dtype == qint16
        ), f"nn.Ceil only support same input/output dtype, but get input {input.dtype} while output qint16. Try nn.quantized.FloatFunctional.ceil instead."  # noqa: E501
        return self.ceil.ceil(input)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict.

        Args: `mod` a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_ceil = cls(qconfig=qconfig)
        return qat_ceil
