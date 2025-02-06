from . import qat, quantized
from ._trigonometric import (
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atanh,
    Cosh,
    Erf,
    Sinh,
    Tan,
)
from .anchor_generator import AnchorGenerator
from .atan import Atan
from .atan2 import Atan2
from .bev_pool_v2 import BevPoolV2
from .bgr_to_yuv444 import BgrToYuv444
from .ceil import Ceil
from .channel_scale import ChannelScale2d
from .channel_shuffle import ChannelShuffle
from .correlation import Correlation
from .cos import Cos
from .cumsum import CumSum
from .detection_post_process import DetectionPostProcess
from .detection_post_process_v1 import DetectionPostProcessV1
from .div import Div
from .einsum import EinSum
from .exp import Exp
from .floor import Floor
from .fmod import FMod
from .grid_generator import BaseGridGenerator
from .grid_sample import GridSample
from .gru import GRU, GRUCell
from .interpolate import Interpolate
from .layer_norm import LayerNorm

# from .linalg_norm import LinalgNorm
from .linear import Identity
from .log import HardLog
from .log_softmax import LogSoftmax
from .lut import LookUpTable
from .masked_scatter import MaskedScatter
from .multi_scale_deform_attn import MultiScaleDeformableAttention
from .multi_scale_roi_align import MultiScaleRoIAlign
from .multihead_attention import MultiheadAttention
from .norm import LinalgNorm, Norm, Normalize
from .point_pillar_scatter import PointPillarsScatter
from .pow import Pow
from .rcnn_post_process import RcnnPostProcess
from .reciprocal import Reciprocal
from .remainder import Remainder
from .scatter import Scatter, ScatterAdd, ScatterReduce
from .segment_lut import SegmentLUT
from .selu import SELU
from .setitem import SetItem
from .sin import Sin
from .slice_scatter import SliceScatter
from .softmax import Softmax
from .softmax_bernoulli2 import SoftmaxBernoulli2
from .sqrt import Sqrt
from .transformer_decoder_layer import TransformerDecoderLayer
from .where import Where

__all__ = [
    "qat",
    "quantized",
    "BgrToYuv444",
    "Identity",
    "Interpolate",
    "GridSample",
    "DetectionPostProcess",
    "AnchorGenerator",
    "LookUpTable",
    "BaseGridGenerator",
    "DetectionPostProcessV1",
    "ChannelShuffle",
    "MultiScaleRoIAlign",
    "Correlation",
    "SegmentLUT",
    "LayerNorm",
    "PointPillarsScatter",
    "Pow",
    "Sin",
    "Cos",
    "Sqrt",
    "Exp",
    "Div",
    "HardLog",
    "Reciprocal",
    "RcnnPostProcess",
    "Ceil",
    "Floor",
    "MultiScaleDeformableAttention",
    "Atan",
    "Acos",
    "Cosh",
    "Asin",
    "Sinh",
    "Erf",
    "Atanh",
    "Asinh",
    "Acosh",
    "Tan",
    "Norm",
    "Normalize",
    "Where",
    "Remainder",
    "SetItem",
    "EinSum",
    "Softmax",
    "LogSoftmax",
    "TransformerDecoderLayer",
    "Scatter",
    "SoftmaxBernoulli2",
    "Atan2",
    "ScatterAdd",
    "ScatterReduce",
    "CumSum",
    "GRU",
    "GRUCell",
    "FMod",
    "MaskedScatter",
    "MultiheadAttention",
    "LinalgNorm",
    "ChannelScale2d",
    "BevPoolV2",
    "SliceScatter",
    "SELU",
]

from torch import nn

from horizon_plugin_pytorch.fx import fx_helper

fx_helper.replace_torch_op("adaptive_avg_pool1d", True)(nn.AdaptiveAvgPool1d)
fx_helper.replace_torch_op("adaptive_avg_pool2d", True)(nn.AdaptiveAvgPool2d)
# avg_pool2d func is supported by QTensor
# fx_helper.replace_torch_op("avg_pool2d", True)(nn.AvgPool2d)
# dropout is supported by QTensor
# fx_helper.replace_torch_op("dropout", True)(nn.Dropout)
# fx_helper.replace_torch_op("dropout2d", True)(nn.Dropout2d)
fx_helper.replace_torch_op("gelu", True)(nn.GELU)
fx_helper.replace_torch_op("glu", True)(nn.GLU)
fx_helper.replace_torch_op("leaky_relu", True)(nn.LeakyReLU)
# max_pool2d is supported by QTensor
# fx_helper.replace_torch_op("max_pool2d", True)(nn.MaxPool2d)
fx_helper.replace_torch_op("prelu", True)(nn.PReLU)
fx_helper.replace_torch_op("relu", True)(nn.ReLU)
fx_helper.replace_torch_op("sigmoid", False)(nn.Sigmoid)
fx_helper.replace_torch_op("hardsigmoid", True)(nn.Hardsigmoid)
fx_helper.replace_torch_op("silu", True)(nn.SiLU)
fx_helper.replace_torch_op("tanh", False)(nn.Tanh)
fx_helper.replace_torch_op("log_softmax", False)(LogSoftmax)
fx_helper.replace_torch_op("elu", True)(nn.ELU)
fx_helper.replace_torch_op("softplus", True)(nn.Softplus)
