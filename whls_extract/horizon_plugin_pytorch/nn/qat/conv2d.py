"""Fused conv2d+add+relu modules."""
from torch import nn

from horizon_plugin_pytorch.qat_mode import handle_relu6_trick
from .. import intrinsic
from .qat_meta import QATConvMeta

__all__ = [
    "Conv2d",
    "ConvReLU2d",
    "ConvAdd2d",
    "ConvAddReLU2d",
    "ConvReLU62d",
    "ConvAddReLU62d",
]


class Conv2d(nn.Conv2d, metaclass=QATConvMeta):
    pass


@handle_relu6_trick
class ConvReLU2d(intrinsic.ConvReLU2d, metaclass=QATConvMeta):
    _version = 2
    pass


class ConvReLU62d(intrinsic.ConvReLU62d, metaclass=QATConvMeta):
    pass


class ConvAdd2d(intrinsic.ConvAdd2d, metaclass=QATConvMeta, input_num=2):
    pass


@handle_relu6_trick
class ConvAddReLU2d(
    intrinsic.ConvAddReLU2d, metaclass=QATConvMeta, input_num=2
):
    _version = 2
    pass


class ConvAddReLU62d(
    intrinsic.ConvAddReLU62d, metaclass=QATConvMeta, input_num=2
):
    pass
