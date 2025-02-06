import torch
import torch.nn.intrinsic as nni
import torch.nn.quantized as nnq
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.quantization import DeQuantStub, QuantStub

from horizon_plugin_pytorch import nn as nnf
from horizon_plugin_pytorch.fx.fx_helper import get_supported_method
from horizon_plugin_pytorch.nn import functional as hF  # noqa: N812
from horizon_plugin_pytorch.nn import intrinsic, qat, quantized
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from . import stubs
from .quantization_mappings import get_qat_module_mappings

HYBRID_QAT_MODULE_MAPPINGS = {
    stubs.QuantStub: qat.QuantStub,
    QuantStub: qat.QuantStub,
    DeQuantStub: qat.DeQuantStub,
    nn.Conv2d: qat.Conv2d,
    # Intrinsic modules:
    intrinsic.ConvReLU2d: qat.ConvReLU2d,
    intrinsic.ConvAdd2d: qat.ConvAdd2d,
    intrinsic.ConvAddReLU2d: qat.ConvAddReLU2d,
    nnq.FloatFunctional: qat.FloatFunctional,
    quantized.FloatFunctional: qat.FloatFunctional,
    nn.AvgPool2d: qat.AvgPool2d,
    nn.AdaptiveAvgPool2d: qat.AdaptiveAvgPool2d,
    nn.MaxPool2d: qat.MaxPool2d,
    # nn.PReLU: qat.PReLU,
    # nnv.RoIAlign: qat.RoIAlign,
    # nnf.LookUpTable: qat.LookUpTable,
    nn.Sigmoid: qat.Sigmoid,
    # nnf.BaseGridGenerator: qat.BaseGridGenerator,
    # nnf.DetectionPostProcessV1: qat.DetectionPostProcessV1,
    # nn.Softmax: qat.Softmax,
    nn.ConvTranspose2d: qat.ConvTranspose2d,
    intrinsic.ConvTransposeAdd2d: qat.ConvTransposeAdd2d,
    intrinsic.ConvTransposeReLU2d: qat.ConvTransposeReLU2d,
    intrinsic.ConvTransposeAddReLU2d: qat.ConvTransposeAddReLU2d,
    nn.SiLU: qat.SiLU,
    nn.Tanh: qat.Tanh,
    intrinsic.ConvTransposeReLU62d: qat.ConvTransposeReLU62d,
    intrinsic.ConvTransposeAddReLU62d: qat.ConvTransposeAddReLU62d,
    intrinsic.ConvReLU62d: qat.ConvReLU62d,
    intrinsic.ConvAddReLU62d: qat.ConvAddReLU62d,
    # nnf.MultiScaleRoIAlign: qat.MultiScaleRoIAlign,
    nn.BatchNorm2d: qat.BatchNorm2d,
    # nn.GELU: qat.GELU,
    # nn.LayerNorm: qat.LayerNorm,
    nn.Conv3d: qat.Conv3d,
    nni.ConvReLU3d: qat.ConvReLU3d,
    intrinsic.ConvAdd3d: qat.ConvAdd3d,
    intrinsic.ConvAddReLU3d: qat.ConvAddReLU3d,
    intrinsic.ConvReLU63d: qat.ConvReLU63d,
    intrinsic.ConvAddReLU63d: qat.ConvAddReLU63d,
    # nnf.Correlation: qat.Correlation,
    # nnf.SegmentLUT: qat.SegmentLUT,
    # nnf.LayerNorm: qat.LayerNorm,
    intrinsic.ConvBN2d: qat.ConvBN2d,
    intrinsic.ConvBNAdd2d: qat.ConvBNAdd2d,
    intrinsic.ConvBNAddReLU2d: qat.ConvBNAddReLU2d,
    intrinsic.ConvBNReLU2d: qat.ConvBNReLU2d,
    intrinsic.ConvBNReLU62d: qat.ConvBNReLU62d,
    intrinsic.ConvBNAddReLU62d: qat.ConvBNAddReLU62d,
    nn.AdaptiveAvgPool1d: qat.AdaptiveAvgPool1d,
    # nn.GLU: qat.GLU,
    nn.LeakyReLU: qat.LeakyReLU,
    nnf.Pow: qat.Pow,
    # nn.LSTMCell: qat.LSTMCell,
    nn.Linear: qat.Linear,
    intrinsic.LinearAdd: qat.LinearAdd,
    intrinsic.LinearAddReLU: qat.LinearAddReLU,
    intrinsic.LinearReLU: qat.LinearReLU,
    intrinsic.LinearReLU6: qat.LinearReLU6,
    intrinsic.LinearAddReLU6: qat.LinearAddReLU6,
    # nn.LSTM: qat.LSTM,
    # nnv.DeformConv2d: qat.DeformConv2d,
    # intrinsic.DeformConvReLU2d: qat.DeformConvReLU2d,
    # intrinsic.DeformConvReLU62d: qat.DeformConvReLU62d,
    # intrinsic.DeformConvAdd2d: qat.DeformConvAdd2d,
    # intrinsic.DeformConvAddReLU2d: qat.DeformConvAddReLU2d,
    # intrinsic.DeformConvAddReLU62d: qat.DeformConvAddReLU62d,
    nnf.Exp: qat.Exp,
}

HYBRID_SUPPORTED_METHODS = {
    "matmul",
    "sum",
    # "minimum",
    "add",
    "mean",
    "maximum",
    "cat",
    "div",
    "mul_scalar",
    "sub",
    "exp",
    "add_scalar",
    "mul",
    "size",
    "reshape",
    "permute",
    "squeeze",
    "transpose",
    "unsqueeze",
    "view",
    "clip",
    "clamp",
}

HYBRID_SUPPORTED_FUNCTIONS = {
    torch._C._TensorBase.argmax,
    # torch._C._TensorBase.argmin,
    # torch._C._TensorBase.device.__get__,
    # torch._C._TensorBase.dim,
    # torch._C._TensorBase.get_device,
    # torch._C._TensorBase.is_cuda.__get__,
    # torch._C._TensorBase.is_contiguous,
    # torch._C._TensorBase.numel,
    # torch._C._TensorBase.requires_grad.__get__,
    # torch._C._TensorBase.requires_grad.__set__,
    # torch._C._TensorBase.shape.__get__,
    # torch._C._TensorBase.size,
    # torch._C._TensorBase.ndim.__get__,
    torch.argmax,
    # torch.argmin,
    # Tensor.backward,
    # Tensor.grad.__get__,
    # Tensor.grad_fn.__get__,
    # torch._C._TensorBase.eq,
    # torch._C._TensorBase.gt,
    # torch._C._TensorBase.greater,
    # torch._C._TensorBase.greater_equal,
    # torch._C._TensorBase.ge,
    # torch._C._TensorBase.lt,
    # torch._C._TensorBase.less,
    # torch._C._TensorBase.le,
    # torch._C._TensorBase.less_equal,
    # torch.eq,
    # torch.gt,
    # torch.greater,
    # torch.greater_equal,
    # torch.ge,
    # torch.less,
    # torch.le,
    # torch.less_equal,
    # torch.lt,
    # torch._C._TensorBase.__getitem__,
    # torch._C._TensorBase.contiguous,
    # torch._C._TensorBase.detach,
    # torch._C._TensorBase.expand,
    # torch._C._TensorBase.flatten,
    torch._C._TensorBase.permute,
    # torch._C._TensorBase.repeat,
    torch._C._TensorBase.reshape,
    # torch._C._TensorBase.roll,
    torch._C._TensorBase.squeeze,
    # torch._C._TensorBase.tile,
    torch._C._TensorBase.transpose,
    torch._C._TensorBase.unsqueeze,
    torch._C._TensorBase.view,
    # torch.flatten,
    torch.permute,
    torch.reshape,
    # torch.roll,
    torch.squeeze,
    # torch.tile,
    torch.transpose,
    torch.unsqueeze,
    torch.split,
    torch.Tensor.split,
    torch.max,
    torch._C._TensorBase.max,
    # torch.min,
    # torch._C._TensorBase.min,
    torch._C._TensorBase.clamp,
    torch._C._TensorBase.clip,
    # torch._C._TensorBase.masked_fill,
    torch._C._TensorBase.mul,
    torch.clamp,
    torch.clip,
    # torch.masked_fill,
    torch.mul,
    torch.ones_like,
    torch.zeros_like,
    # functional
    F.avg_pool2d,
    # F.channel_shuffle,
    F.interpolate,
    # F.affine_grid,
    F.grid_sample,
    F.pixel_shuffle,
    # F.pixel_unshuffle, # cannot export to onnx
    F.pad,
}


HYBRID_QUANT_MODULE_MAPPINGS = {
    qat.QuantStub: quantized.Quantize,
    qat.DeQuantStub: quantized.DeQuantize,
    # add this mapping to make fx treat Identity like quantized op
    nnf.Identity: nnf.Identity,
    # Wrapper Modules:
    qat.FloatFunctional: quantized.QFunctional,
    # Intrinsic modules:
    qat.ConvReLU2d: quantized.ConvReLU2d,
    qat.ConvAdd2d: quantized.ConvAdd2d,
    qat.ConvAddReLU2d: quantized.ConvAddReLU2d,
    qat.Conv2d: quantized.Conv2d,
    qat.AvgPool2d: quantized.AvgPool2d,
    qat.AdaptiveAvgPool2d: quantized.AdaptiveAvgPool2d,
    qat.MaxPool2d: quantized.MaxPool2d,
    # qat.RoIAlign: quantized.RoIAlign,
    # qat.PReLU: quantized.PReLU,
    # qat.LookUpTable: quantized.LookUpTable,
    qat.Sigmoid: quantized.Sigmoid,
    # qat.BaseGridGenerator: quantized.BaseGridGenerator,
    # qat.DetectionPostProcessV1: quantized.DetectionPostProcessV1,
    # qat.Softmax: quantized.QuantSoftmax,
    qat.ConvTranspose2d: quantized.ConvTranspose2d,
    qat.ConvTransposeAdd2d: quantized.ConvTransposeAdd2d,
    qat.ConvTransposeReLU2d: quantized.ConvTransposeReLU2d,
    qat.ConvTransposeAddReLU2d: quantized.ConvTransposeAddReLU2d,
    qat.SiLU: quantized.SiLU,
    qat.Tanh: quantized.Tanh,
    qat.ConvReLU62d: quantized.ConvReLU2d,
    qat.ConvAddReLU62d: quantized.ConvAddReLU2d,
    qat.ConvTransposeReLU62d: quantized.ConvTransposeReLU2d,
    qat.ConvTransposeAddReLU62d: quantized.ConvTransposeAddReLU2d,
    # qat.MultiScaleRoIAlign: quantized.MultiScaleRoIAlign,
    qat.BatchNorm2d: quantized.BatchNorm2d,
    # qat.GELU: quantized.GELU,
    qat.ConvReLU3d: quantized.ConvReLU3d,
    qat.ConvAdd3d: quantized.ConvAdd3d,
    qat.ConvAddReLU3d: quantized.ConvAddReLU3d,
    qat.Conv3d: quantized.Conv3d,
    qat.ConvReLU63d: quantized.ConvReLU3d,
    qat.ConvAddReLU63d: quantized.ConvAddReLU3d,
    # qat.Correlation: quantized.Correlation,
    qat.SegmentLUT: quantized.SegmentLUT,
    qat.ConvBN2d: quantized.Conv2d,
    qat.ConvBNAdd2d: quantized.ConvAdd2d,
    qat.ConvBNAddReLU2d: quantized.ConvAddReLU2d,
    qat.ConvBNReLU2d: quantized.ConvReLU2d,
    qat.ConvBNReLU62d: quantized.ConvReLU2d,
    qat.ConvBNAddReLU62d: quantized.ConvAddReLU2d,
    qat.AdaptiveAvgPool1d: quantized.AdaptiveAvgPool1d,
    qat.Linear: quantized.Linear,
    qat.LinearAdd: quantized.LinearAdd,
    qat.LinearReLU: quantized.LinearReLU,
    qat.LinearReLU6: quantized.LinearReLU,
    qat.LinearAddReLU: quantized.LinearAddReLU,
    qat.LinearAddReLU6: quantized.LinearAddReLU,
    # qat.DeformConv2d: quantized.DeformConv2d,
    # qat.DeformConvReLU2d: quantized.DeformConvReLU2d,
    # qat.DeformConvReLU62d: quantized.DeformConvReLU2d,
    # qat.DeformConvAdd2d: quantized.DeformConvAdd2d,
    # qat.DeformConvAddReLU2d: quantized.DeformConvAddReLU2d,
    # qat.DeformConvAddReLU62d: quantized.DeformConvAddReLU2d,
    qat.Exp: quantized.Exp,
}


def get_hybrid_quantized_module_mappings():
    """Get quantized module mapping for hybrid quantization aware training."""
    return HYBRID_QUANT_MODULE_MAPPINGS


def get_hybrid_qat_module_mappings():
    """Get module mapping for hybrid quantization aware training."""
    return HYBRID_QAT_MODULE_MAPPINGS


def get_hybrid_supported_modules():
    """Get supported module for hybrid quantization aware training."""
    return set(HYBRID_QAT_MODULE_MAPPINGS.keys())


def get_hybrid_supported_functions():
    """Get supported function for hybrid quantization aware training."""
    return HYBRID_SUPPORTED_FUNCTIONS


def get_hybrid_supported_methods():
    """Get supported method for hybrid quantization aware training."""
    return HYBRID_SUPPORTED_METHODS


# hbdk4
HBDK4_QUANTIZED_SUPPORTED_MODULES_EXTRA = {
    nn.Identity,
    nnf.Identity,
    # SegmentLUT ops
    nnf.Cos,
    nnf.Sin,
    nnf.Atan,
    nnf.HardLog,
    nnf.Sqrt,
    # from _trigonometric
    nnf.Acos,
    nnf.Acosh,
    nnf.Asin,
    nnf.Asinh,
    nnf.Atanh,
    nnf.Cosh,
    nnf.Erf,
    nnf.Sinh,
    nnf.Tan,
    # leaf module supported by submodule
    nnf.MultiScaleDeformableAttention,
    # leaf module supported by function
    nn.ChannelShuffle,  # F.channel_shuffle
    nnf.ChannelShuffle,  # nnF.channel_shuffle
    nnf.AnchorGenerator,  # nnF.anchor_generator
    nnf.PointPillarsScatter,  # nnF.point_pillars_scatter
    nnf.RcnnPostProcess,  # nnQF.point_pillars_scatter
    nn.Upsample,  # F.interpolate
    nn.UpsamplingBilinear2d,  # F.interpolate
    nn.UpsamplingNearest2d,  # F.interpolate
    nn.PixelShuffle,  # F.pixel_shuffle
    nn.PixelUnshuffle,  # F.pixel_unshuffle
}


HBDK4_QUANTIZED_SUPPORTED_METHODS_EXTRA = {
    # supported by qtensor
    # "size", # not support by hbdk4
    "reshape",
    "permute",
    "squeeze",
    "transpose",
    "unsqueeze",
    "view",
    "clip",
    "clamp",
}

HBDK4_QUANTIZED_SUPPORTED_FUNCTIONS = {
    torch._C._TensorBase.argmax,
    torch._C._TensorBase.argmin,
    # torch._C._TensorBase.device.__get__,
    # torch._C._TensorBase.dim,
    # torch._C._TensorBase.get_device,
    # torch._C._TensorBase.is_cuda.__get__,
    # torch._C._TensorBase.is_contiguous,
    # torch._C._TensorBase.numel,
    # torch._C._TensorBase.shape.__get__,
    # torch._C._TensorBase.size,
    # torch._C._TensorBase.ndim.__get__,
    # torch._C._TensorBase.is_mkldnn.__get__,
    # torch._C._TensorBase.is_complex,
    torch.argmax,
    torch.argmin,
    # Tensor.backward,
    # Tensor.grad_fn.__get__,
    # torch.is_same_size,
    # torch._C._TensorBase.requires_grad.__get__,
    # torch._C._TensorBase.requires_grad.__set__,
    # torch._C._TensorBase.grad_fn.__set__,
    # torch.Tensor.retain_grad,
    # torch.Tensor.double,
    torch._C._TensorBase.eq,
    torch._C._TensorBase.gt,
    torch._C._TensorBase.greater,
    torch._C._TensorBase.ge,
    torch._C._TensorBase.greater_equal,
    torch._C._TensorBase.lt,
    torch._C._TensorBase.less,
    torch._C._TensorBase.le,
    torch._C._TensorBase.less_equal,
    torch.eq,
    # torch.equal return a bool!!!
    torch.gt,
    torch.greater,
    torch.ge,
    torch.greater_equal,
    torch.less,
    torch.le,
    torch.less_equal,
    torch.lt,
    # torch._C._TensorBase.__getitem__,
    # torch._C._TensorBase.contiguous,
    # torch._C._TensorBase.detach,
    torch._C._TensorBase.expand,
    torch._C._TensorBase.flatten,
    torch._C._TensorBase.permute,
    torch._C._TensorBase.repeat,
    torch._C._TensorBase.reshape,
    torch._C._TensorBase.roll,
    torch._C._TensorBase.squeeze,
    torch._C._TensorBase.tile,
    torch._C._TensorBase.transpose,
    torch._C._TensorBase.unsqueeze,
    torch._C._TensorBase.view,
    # torch._C._TensorBase.clone,
    torch._C._TensorBase.gather,
    torch.flatten,
    torch.permute,
    torch.reshape,
    torch.roll,
    torch.squeeze,
    torch.tile,
    torch.transpose,
    torch.unsqueeze,
    # torch.clone,
    torch.gather,
    torch.split,
    torch.Tensor.split,
    torch.max,
    torch._C._TensorBase.max,
    torch.min,
    torch._C._TensorBase.min,
    # torch.Tensor.requires_grad_,
    # torch.Tensor.grad.__get__,
    torch._C._TensorBase.clamp,
    torch._C._TensorBase.clip,
    torch._C._TensorBase.masked_fill,
    torch._C._TensorBase.mul,
    torch._C._TensorBase.topk,
    torch._C._TensorBase.abs,
    torch.clamp,
    torch.clip,
    torch.masked_fill,
    torch.mul,
    torch.ones_like,
    torch.topk,
    torch.zeros_like,
    torch.abs,
    # functional
    F.avg_pool2d,
    F.channel_shuffle,
    F.interpolate,
    F.affine_grid,
    F.grid_sample,
    F.pixel_shuffle,
    F.pixel_unshuffle,
    F.pad,
    F.relu,
    # horizon functional
    hF.filter,
    hF.point_pillars_preprocess,
    hF.point_pillars_scatter,
    hF.window_partition,
    hF.window_reverse,
}


def get_hbdk4_quantized_supported_modules():
    """Get modules that support int8/int16 in hbdk4."""
    return (
        set(get_qat_module_mappings().keys())
        | HBDK4_QUANTIZED_SUPPORTED_MODULES_EXTRA
    )


def get_hbdk4_quantized_supported_functions():
    """Get functions that support int8/int16 in hbdk4."""
    return HBDK4_QUANTIZED_SUPPORTED_FUNCTIONS


def get_hbdk4_quantized_supported_methods():
    """Get methods that support int8/int16 in hbdk4."""
    return (
        set(get_supported_method()[FloatFunctional])
        | HBDK4_QUANTIZED_SUPPORTED_METHODS_EXTRA
    )


def get_hbdk4_float_supported_modules():
    """Get modules that support float in hbdk4."""
    return get_hbdk4_quantized_supported_modules()


def get_hbdk4_float_supported_functions():
    """Get functions that support float in hbdk4."""
    return get_hbdk4_quantized_supported_functions()


def get_hbdk4_float_supported_methods():
    """Get methods that support float in hbdk4."""
    return get_hbdk4_quantized_supported_methods()
