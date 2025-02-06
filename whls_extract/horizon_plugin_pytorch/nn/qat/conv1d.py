"""Fused conv1d+add+relu modules."""
from torch import nn

from .. import intrinsic
from .qat_meta import QATConvMeta

__all__ = [
    "Conv1d",
    "ConvReLU1d",
    "ConvAdd1d",
    "ConvAddReLU1d",
    "ConvReLU61d",
    "ConvAddReLU61d",
]


class Conv1d(nn.Conv1d, metaclass=QATConvMeta):
    pass


class ConvReLU1d(intrinsic.ConvReLU1d, metaclass=QATConvMeta):
    pass


class ConvReLU61d(intrinsic.ConvReLU61d, metaclass=QATConvMeta):
    pass


class ConvAdd1d(intrinsic.ConvAdd1d, metaclass=QATConvMeta, input_num=2):
    pass


class ConvAddReLU1d(
    intrinsic.ConvAddReLU1d, metaclass=QATConvMeta, input_num=2
):
    pass


class ConvAddReLU61d(
    intrinsic.ConvAddReLU61d, metaclass=QATConvMeta, input_num=2
):
    pass
