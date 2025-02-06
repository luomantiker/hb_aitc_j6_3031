from .adaptive_avg_pool1d import AdaptiveAvgPool1d
from .adaptive_avg_pool2d import AdaptiveAvgPool2d
from .avg_pool2d import AvgPool2d
from .batchnorm import BatchNorm2d, BatchNorm3d
from .channel_scale import ChannelScale2d
from .ceil import Ceil
from .compatible_ops import relu  # noqa: F401
from .conv1d import Conv1d
from .conv2d import (
    Conv2d,
    ConvAdd2d,
    ConvAddReLU2d,
    ConvAddReLU62d,
    ConvReLU2d,
    ConvReLU62d,
)
from .conv3d import (
    Conv3d,
    ConvAdd3d,
    ConvAddReLU3d,
    ConvAddReLU63d,
    ConvReLU3d,
    ConvReLU63d,
)
from .conv_bn2d import (
    ConvBN2d,
    ConvBNAdd2d,
    ConvBNAddReLU2d,
    ConvBNAddReLU62d,
    ConvBNReLU2d,
    ConvBNReLU62d,
)
from .conv_transpose2d import (
    ConvTranspose2d,
    ConvTransposeAdd2d,
    ConvTransposeAddReLU2d,
    ConvTransposeAddReLU62d,
    ConvTransposeReLU2d,
    ConvTransposeReLU62d,
)
from .correlation import Correlation
from .cumsum import CumSum
from .deform_conv2d import (
    DeformConv2d,
    DeformConvAdd2d,
    DeformConvAddReLU2d,
    DeformConvAddReLU62d,
    DeformConvReLU2d,
    DeformConvReLU62d,
)
from .detection_post_process_v1 import DetectionPostProcessV1
from .einsum import EinSum
from .div import Div
from .embedding import Embedding
from .elu import ELU
from .exp import Exp
from .floor import Floor
from .fmod import FMod
from .functional_modules import FloatFunctional
from .gelu import GELU
from .glu import GLU
from .grid_generator import BaseGridGenerator
from .hardsigmoid import HardSigmoid
from .layernorm import LayerNorm
from .leakyrelu import LeakyReLU
from .linear import (
    Linear,
    LinearAdd,
    LinearAddReLU,
    LinearAddReLU6,
    LinearReLU,
    LinearReLU6,
)
from .lstm import LSTM
from .lstm_cell import LSTMCell
from .lut import LookUpTable
from .masked_scatter import MaskedScatter
from .max_pool1d import MaxPool1d
from .max_pool2d import MaxPool2d
from .multi_scale_roi_align import MultiScaleRoIAlign
from .multiheadattention import MultiheadAttention
from .pow import Pow
from .prelu import PReLU
from .reciprocal import Reciprocal
from .remainder import Remainder
from .roi_align import RoIAlign
from .segment_lut import SegmentLUT
from .sigmoid import Sigmoid
from .silu import SiLU
from .softmax import Softmax
from .softplus import Softplus
from .stubs import DeQuantStub, QuantStub
from .tanh import Tanh
from .setitem import SetItem
from .where import Where
from .instancenorm import InstanceNorm1d, InstanceNorm2d, InstanceNorm3d
from .multi_scale_deform_attn import MultiScaleDeformableAttention
from .scatter import Scatter, ScatterAdd, ScatterReduce
from .softmax_bernoulli2 import SoftmaxBernoulli2
from .bev_pool_v2 import BevPoolV2

# import to make the monkey patch take effect
from .pad import (  # noqa: F401
    _patch_torch_modules,
)
from .interpolate import (  # noqa: F401, F811
    _patch_torch_modules,
)
from .grid_sample import *  # noqa: F401, F403
from .slice_scatter import SliceScatter


__all__ = [
    "Conv1d",
    "Conv2d",
    "ConvReLU2d",
    "ConvAdd2d",
    "ConvAddReLU2d",
    "ConvReLU62d",
    "ConvAddReLU62d",
    "ConvBN2d",
    "ConvBNReLU2d",
    "ConvBNAdd2d",
    "ConvBNAddReLU2d",
    "ConvBNReLU62d",
    "ConvBNAddReLU62d",
    "Conv3d",
    "ConvReLU3d",
    "ConvAdd3d",
    "ConvAddReLU3d",
    "ConvReLU63d",
    "ConvAddReLU63d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
    "MaxPool1d",
    "MaxPool2d",
    "QuantStub",
    "DeQuantStub",
    "RoIAlign",
    "LookUpTable",
    "Sigmoid",
    "BaseGridGenerator",
    "SiLU",
    "DetectionPostProcessV1",
    "Softmax",
    "ConvTranspose2d",
    "ConvTransposeReLU2d",
    "ConvTransposeReLU62d",
    "ConvTransposeAdd2d",
    "ConvTransposeAddReLU2d",
    "ConvTransposeAddReLU62d",
    "Tanh",
    "MultiScaleRoIAlign",
    "BatchNorm2d",
    "GELU",
    "LayerNorm",
    "Correlation",
    "SegmentLUT",
    "PReLU",
    "AdaptiveAvgPool1d",
    "GLU",
    "LeakyReLU",
    "Pow",
    "LSTMCell",
    "Linear",
    "LinearReLU",
    "LinearReLU6",
    "LinearAdd",
    "LinearAddReLU",
    "LinearAddReLU6",
    "LSTM",
    "DeformConv2d",
    "DeformConvReLU2d",
    "DeformConvReLU62d",
    "DeformConvAdd2d",
    "DeformConvAddReLU2d",
    "DeformConvAddReLU62d",
    "Exp",
    "Div",
    "MultiheadAttention",
    "Reciprocal",
    "FloatFunctional",
    "Softplus",
    "ELU",
    "Ceil",
    "Floor",
    "HardSigmoid",
    "BatchNorm3d",
    "Where",
    "Remainder",
    "SetItem",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "MultiScaleDeformableAttention",
    "EinSum",
    "Embedding",
    "Scatter",
    "ScatterAdd",
    "ScatterReduce",
    "CumSum",
    "FMod",
    "SoftmaxBernoulli2",
    "MaskedScatter",
    "ChannelScale2d",
    "BevPoolV2",
    "SliceScatter",
]
