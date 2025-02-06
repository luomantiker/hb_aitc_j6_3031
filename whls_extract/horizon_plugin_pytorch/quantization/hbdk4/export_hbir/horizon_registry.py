import logging
import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
from hbdk4.compiler import ir
from hbdk4.compiler.ops import hbir, qnt
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torch.nn.modules.utils import _pair
from torch.types import Number

import horizon_plugin_pytorch
from horizon_plugin_pytorch import nn as hnn
from horizon_plugin_pytorch.dtype import (
    QuantDType,
    qinfo,
    qint8,
    qint16,
    qint32,
)
from horizon_plugin_pytorch.nn import functional as hF  # noqa: N812
from horizon_plugin_pytorch.nn import qat
from horizon_plugin_pytorch.nn.bev_pool_v2 import bev_pool_v2
from horizon_plugin_pytorch.nn.qat.deform_conv2d import (
    _to_pow_quantized,
    deform_conv2d_torch_function,
)
from horizon_plugin_pytorch.nn.qat.functional_modules import (
    _add_scalar_stub,
    _sub_stub,
)
from horizon_plugin_pytorch.nn.quantized.functional import filter
from horizon_plugin_pytorch.qtensor import QTensor, copy_from
from horizon_plugin_pytorch.quantization import (
    FakeCast,
    FakeQuantize,
    PACTFakeQuantize,
    _LearnableFakeQuantize,
)
from horizon_plugin_pytorch.quantization.fake_cast import fp16_max, fp16_min
from .export_hbir import (
    Exporter,
    FuncConverterBase,
    JitTensor,
    ModuleConverterBase,
)
from .torch_registry import (  # DropoutOpConverter,; MaxPool2dConverter,
    AdaptiveMaxPool1dConverter,
    AdaptiveMaxPool2dConverter,
    AffineGridConverter,
    ArgmaxConverter,
    ArgminConverter,
    ArgSortConverter,
    AvgPool2dConverter,
    ChunkConverter,
    ConstOpConverter,
    DropoutOpConverter,
    ExpandConverter,
    FlipConverter,
    GatherConverter,
    GetItemConverter,
    GridSampleConverter,
    IndexSelectConverter,
    InterpolateConverter,
    LogicalAndConverter,
    LogicalNotConverter,
    LogicalOrConverter,
    MaskedFillConverter,
    MaskedSelectConverter,
    MaxConverter,
    MaxPool1dConverter,
    MaxPool2dConverter,
    MinConverter,
    NewTensorConverter,
    PadConverter,
    PermuteConverter,
    PixelShuffleConverter,
    PixelUnShuffleConverter,
    ReduceAllConverter,
    ReLUConverter,
    RepeatConverter,
    ReshapeConverter,
    RollConverter,
    SortConverter,
    TConverter,
    TensorSplitConverter,
    TileConverter,
    ToConverter,
    ToCpuConverter,
    ToCudaConverter,
    TopkConverter,
    TorchSplitConverter,
    TransparentOpConverter,
    TransposeConverter,
    UnbindConverter,
    const_ops,
    transparent_ops,
)
from .utils import (
    LayoutConverter,
    get_hbir_dtype,
    get_hbir_tensor_type,
    lut_with_march,
    to_numpy,
)

__all__ = []

logger = logging.getLogger(__name__)


def get_hbir_tensor_qtype(qtype, shape=None):
    qtype_mapping = {
        qint8: ir.IntegerType.get_signed(8),
        qint16: ir.IntegerType.get_signed(16),
        qint32: ir.IntegerType.get_signed(32),
    }
    if shape is None:
        return ir.UnrankedTensorType.get(qtype_mapping[qtype])
    else:
        return ir.RankedTensorType.get(shape, qtype_mapping[qtype])


JitTensor.register_subclass(QTensor)


def const_fake_quant_hbir(
    input_hbir, scale: Tensor, dtype: QuantDType, axis=None
):
    if axis is not None and axis < 0:
        axis = None
    narrow = (dtype.min + dtype.max) == 0
    return qnt.const_fake_quant(
        input_hbir,
        min=(scale * dtype.min).cpu().numpy().tolist(),
        max=(scale * dtype.max).cpu().numpy().tolist(),
        bits=dtype.bits,
        narrowRange=narrow,
        axis=axis,
    )


def const_fake_quant_scalar(scalar, dtype: QuantDType):
    scalar = float(scalar)
    t = torch.tensor(scalar)
    return qnt.const_fake_quant(
        hbir.constant(
            t.numpy(),
            output_type=get_hbir_tensor_type(t.dtype),
        ),
        min=[-abs(scalar)],
        max=[abs(scalar)],
        bits=dtype.bits,
        narrowRange=True,
    )


def const_fake_quant_like(input_hbir, t: QTensor):
    return const_fake_quant_hbir(
        input_hbir, t.q_scale(), t.dtype, t.q_per_channel_axis()
    )


@JitTensor.register_subclass_converter(copy_from)
class QTensorCopyFromConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, dst: JitTensor, src: JitTensor):
        dst.hbir_node = src.hbir_node
        if dst.subclass_from is not None:
            dst.subclass_from.hbir_node = src.hbir_node

        return output


qtensor_const_ops = [
    Tensor.q_scale,
    Tensor.q_zero_point,
    Tensor.q_per_channel_scales,
    Tensor.q_per_channel_zero_points,
]


@JitTensor.register_subclass_converter(*qtensor_const_ops)
class QTensorConstOpConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, *args, **kwargs):
        return None


JitTensor.register_subclass_converter(*const_ops)(ConstOpConverter)
JitTensor.register_subclass_converter(
    F.dropout, F.dropout1d, F.dropout2d, F.dropout3d
)(DropoutOpConverter)
JitTensor.register_subclass_converter(*transparent_ops)(TransparentOpConverter)
JitTensor.register_subclass_converter(Tensor.float)(TransparentOpConverter)
JitTensor.register_subclass_converter(Tensor.to)(ToConverter)
JitTensor.register_subclass_converter(Tensor.cpu)(ToCpuConverter)
JitTensor.register_subclass_converter(Tensor.cuda)(ToCudaConverter)
JitTensor.register_subclass_converter(Tensor.new_tensor)(NewTensorConverter)


def register_compare():
    compare_op_mapping = {
        (torch.eq, Tensor.eq): hbir.equal,
        (torch.less, torch.lt, Tensor.less, Tensor.lt): hbir.less,
        (
            torch.less_equal,
            torch.le,
            Tensor.less_equal,
            Tensor.le,
        ): hbir.less_equal,
        (torch.greater, torch.gt, Tensor.greater, Tensor.gt): hbir.greater,
        (
            torch.greater_equal,
            torch.ge,
            Tensor.greater_equal,
            Tensor.ge,
        ): hbir.greater_equal,
        (
            torch.not_equal,
            torch.ne,
            Tensor.not_equal,
            Tensor.ne,
        ): hbir.not_equal,
    }

    def do(torch_funcs, hbir_func):
        # def converter in a func to hold things in local scope
        class CompareConverter(FuncConverterBase):
            @classmethod
            def convert(cls, output, *args, **kwargs):
                lhs_hbir, rhs_hbir = JitTensor.gather_hbir(args)
                lhs, rhs = JitTensor.get_base(args)
                if isinstance(lhs, (int, float)):
                    lhs = float(lhs)
                elif isinstance(rhs, (int, float)):
                    rhs = float(rhs)
                else:
                    if lhs.dtype != rhs.dtype:
                        raise ValueError(
                            "Do not support compare between different "
                            "dtype {} and {}".format(lhs.dtype, rhs.dtype)
                        )
                    bigger_scale = torch.max(lhs.q_scale(), rhs.q_scale())
                    if isinstance(lhs_hbir, ir.Value):
                        lhs_hbir = const_fake_quant_hbir(
                            lhs_hbir, bigger_scale, lhs.dtype
                        )
                    if isinstance(rhs_hbir, ir.Value):
                        rhs_hbir = const_fake_quant_hbir(
                            rhs_hbir, bigger_scale, rhs.dtype
                        )

                hbir_output = hbir_func(
                    lhs_hbir,
                    rhs_hbir,
                    output_type=get_hbir_tensor_type(torch.bool, output.shape),
                )

                return JitTensor.attach_hbir_to_tensor(output, hbir_output)

        JitTensor.register_subclass_converter(*torch_funcs)(CompareConverter)

    for torch_funcs, hbir_func in compare_op_mapping.items():
        do(torch_funcs, hbir_func)


register_compare()


# @JitTensor.register_subclass_converter(Tensor.q_scale)
# class NoOpConverter(FuncConverterBase):
#     @classmethod
#     def convert_with_hbir(cls, *args, **kwargs):
#         logger.debug("Tensor.q_scale")
#         logger.debug(args)
#         return None


class QuantOutputConverterBase(FuncConverterBase):
    @classmethod
    def convert(cls, output, *args, **kwargs):
        hbir_args = JitTensor.gather_hbir(args)
        hbir_kwargs = JitTensor.gather_hbir(kwargs)
        hbir_output = cls.convert_with_hbir(*hbir_args, **hbir_kwargs)
        hbir_output = const_fake_quant_like(hbir_output, output)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


class QatModuleConverterBase(ModuleConverterBase):
    @classmethod
    def convert_fake_quantize(cls, mod: FakeQuantize, hbir_input):
        if not mod._fake_quant_enabled:
            return hbir_input

        quant_min, quant_max = mod.quant_min, mod.quant_max
        scale = mod.scale.double()
        if torch.all(mod.scale == 1):
            logger.warning(
                (
                    "Scale in FakeQuantize equals to its init value, feature "
                    "statistics may not be gathered. Please use real scale "
                    "values in model perf, or the perf result will be wrong."
                )
            )

        min_value = to_numpy(scale.double().mul(quant_min))
        max_value = to_numpy(scale.double().mul(quant_max))
        bits = math.ceil(math.log(quant_max - quant_min, 2))
        narrow = (quant_min + quant_max) == 0

        if scale.numel() > 1:
            return qnt.const_fake_quant(
                hbir_input,
                min_value.tolist(),
                max_value.tolist(),
                bits,
                narrowRange=narrow,
                axis=mod.ch_axis,
            )
        else:
            return qnt.const_fake_quant(
                hbir_input,
                [min_value.item()],
                [max_value.item()],
                bits,
                narrowRange=narrow,
            )

    @classmethod
    def convert_fake_cast(cls, mod: FakeCast, hbir_input):
        if mod.dtype == torch.float32:
            return qnt.barrier(hbir_input)
        elif mod.dtype == torch.float16:
            hbir_output = hbir.fake_cast(
                hbir_input,
                get_hbir_dtype(torch.float16),
                output_type=get_hbir_dtype(torch.float32),
            )
            if mod.enable_clip:
                hbir_output = hbir.clip(hbir_output, fp16_min, fp16_max)
                hbir_output = hbir.fake_cast(
                    hbir_output,
                    get_hbir_dtype(torch.float16),
                    output_type=get_hbir_dtype(torch.float32),
                )
            return hbir_output
        else:
            raise ValueError("Unkown FakeCast dtype {}".format(mod.dtype))

    @classmethod
    def _convert_activation_process(cls, mod, hbir_input, attr_name):
        if hasattr(mod, attr_name):
            activation_process = getattr(mod, attr_name)
            if activation_process is not None:
                if isinstance(
                    activation_process,
                    (FakeQuantize, PACTFakeQuantize, _LearnableFakeQuantize),
                ):
                    return cls.convert_fake_quantize(
                        activation_process, hbir_input
                    )
                elif isinstance(activation_process, FakeCast):
                    return cls.convert_fake_cast(
                        activation_process, hbir_input
                    )
                else:
                    return TypeError(
                        "Unknown {} type {}".format(
                            attr_name, type(activation_process)
                        )
                    )

        return hbir_input

    @classmethod
    def convert_activation_pre_process(cls, mod, hbir_input):
        return cls._convert_activation_process(
            mod, hbir_input, "activation_pre_process"
        )

    @classmethod
    def convert_activation_post_process(cls, mod, hbir_input):
        return cls._convert_activation_process(
            mod, hbir_input, "activation_post_process"
        )


@Exporter.register_converter(FakeCast)
class FakeCastConverter(QatModuleConverterBase):
    @classmethod
    def convert_with_hbir(cls, mod: FakeCast, input):
        return cls.convert_fake_cast(mod, input)

    @classmethod
    def convert_with_constant_folding(cls, mod: FakeQuantize, output, input):
        # skip constant fold for FakeCast
        return cls.convert(mod, output, input)


@Exporter.register_converter(
    FakeQuantize, _LearnableFakeQuantize, PACTFakeQuantize
)
class FakeQuantizeConverter(QatModuleConverterBase):
    @classmethod
    def convert_with_hbir(cls, mod: FakeQuantize, input):
        return cls.convert_fake_quantize(mod, input)

    @classmethod
    def convert_with_constant_folding(cls, mod: FakeQuantize, output, input):
        # skip constant fold for FakeQuantize
        return cls.convert(mod, output, input)


@JitTensor.register_subclass_converter(Tensor.dequantize)
class DequantizeConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, *args, **kwargs):
        # Do not export qnt.barrier, because dequantize in QAT model just
        # convert QTensor to Tensor, and have no effect for quantization.
        hbir_input = JitTensor.gather_hbir(args[0])
        return JitTensor.attach_hbir_to_tensor(output, hbir_input)


@JitTensor.register_converter(horizon_plugin_pytorch.abs)
@JitTensor.register_subclass_converter(
    horizon_plugin_pytorch.abs, torch.abs, Tensor.abs
)
class HorizonAbsConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls, input: ir.Value, overflow_mode: str = "saturate"
    ):
        if overflow_mode != "saturate":
            msg = "Export only support abs with overflow_mode='saturate'"
            logger.error(msg)
            raise ValueError(msg)
        return hbir.abs(input)


@JitTensor.register_subclass_converter(F.adaptive_max_pool1d)
class QTensorAdaptiveMaxPool1dConverter(QuantOutputConverterBase):
    @classmethod
    def convert(cls, output, *args, **kwargs):
        hbir_output = AdaptiveMaxPool1dConverter.convert_adaptive_max_pool(
            *args, **kwargs
        )
        hbir_output = const_fake_quant_like(hbir_output, output)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_subclass_converter(F.adaptive_max_pool2d)
class QTensorAdaptiveMaxPool2dConverter(QuantOutputConverterBase):
    @classmethod
    def convert(cls, output, *args, **kwargs):
        hbir_output = AdaptiveMaxPool2dConverter.convert_adaptive_max_pool(
            *args, **kwargs
        )
        hbir_output = const_fake_quant_like(hbir_output, output)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_converter(_add_scalar_stub)
class AddScalarConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, x: ir.Value, scalar: Number):
        return hbir.add(x, scalar)


@JitTensor.register_subclass_converter(_add_scalar_stub)
class QTensorAddScalarConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, x: JitTensor, scalar: Number):
        scalar_hbir = const_fake_quant_scalar(
            scalar,
            JitTensor.get_base(x).dtype,
        )
        hbir_output = hbir.add(JitTensor.gather_hbir(x), scalar_hbir)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_subclass_converter(F.affine_grid)
class QTensorAffineGridConverter(QuantOutputConverterBase):
    @classmethod
    def convert_with_hbir(cls, *args, **kwargs):
        return AffineGridConverter.convert_with_hbir(*args, **kwargs)


JitTensor.register_subclass_converter(torch.argmax, Tensor.argmax)(
    ArgmaxConverter
)
JitTensor.register_subclass_converter(torch.argmin, Tensor.argmin)(
    ArgminConverter
)


@JitTensor.register_subclass_converter(F.avg_pool2d)
class QTensorAvgPool2dConverter(QuantOutputConverterBase):
    @classmethod
    def convert_with_hbir(cls, *args, **kwargs):
        return AvgPool2dConverter.convert_with_hbir(*args, **kwargs)


@JitTensor.register_converter(bev_pool_v2)
class BevPoolV2Converter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls,
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
        bev_feat_shape,
    ):
        feat = hbir.transpose(feat, (0, 1, 3, 4, 2))
        output = hbir.bev_pool_v2(
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
            bev_feat_shape,
        )
        output = hbir.transpose(output, (0, 4, 1, 2, 3))
        return output


@JitTensor.register_subclass_converter(
    torch.clamp, torch.clip, Tensor.clamp, Tensor.clip
)
class QTensorClampConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output, input: JitTensor, min=None, max=None):
        hbir_input, hbir_min, hbir_max = JitTensor.gather_hbir(
            (input, min, max)
        )
        input_base = JitTensor.get_base(input)
        if isinstance(min, Tensor):
            hbir_min = const_fake_quant_like(hbir_min, input_base)
            hbir_input = hbir.max(hbir_input, hbir_min)
            min = None
        if isinstance(max, Tensor):
            hbir_max = const_fake_quant_like(hbir_max, input_base)
            hbir_input = hbir.min(hbir_input, hbir_max)
            max = None
        if min is not None or max is not None:
            hbir_input = hbir.clip(
                hbir_input,
                float(min) if min is not None else float("-inf"),
                float(max) if max is not None else float("inf"),
            )
        return JitTensor.attach_hbir_to_tensor(output, hbir_input)


@Exporter.register_converter(hnn.Correlation, qat.Correlation)
class CorrelationConverter(QatModuleConverterBase):
    @classmethod
    def convert_with_hbir(cls, mod: qat.Correlation, data1, data2):
        layout_converter = LayoutConverter()

        data1 = layout_converter.nchw_to_nhwc(data1)
        data2 = layout_converter.nchw_to_nhwc(data2)

        output = hbir.correlation(
            data1,
            data2,
            mod.kernel_size,
            mod.max_displacement,
            mod.stride1,
            mod.stride2,
            mod.pad_size,
            "multiply" if mod.is_multiply else "subtraction",
        )
        output = layout_converter.nhwc_to_nchw(output)
        return cls.convert_activation_post_process(mod, output)


@JitTensor.register_converter(deform_conv2d_torch_function)
class FuncDeformConv2dConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output: Tensor,
        input: JitTensor,
        offset: JitTensor,
        weight: JitTensor,
        bias: Optional[JitTensor] = None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        mask: Optional[JitTensor] = None,
    ):
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        input_base, weight_base, offset_base = JitTensor.get_base(
            (input, weight, offset)
        )
        groups = input_base.size(1) // weight_base.size(1)
        offset_groups = offset_base.size(1) // (
            2 * weight_base.size(2) * weight_base.size(3)
        )

        input, offset, weight, bias, mask = JitTensor.gather_hbir(
            (input, offset, weight, bias, mask)
        )
        layout_converter = LayoutConverter()

        input = layout_converter.nchw_to_nhwc(input)
        offset = LayoutConverter().nchw_to_nhwc(offset)
        if mask is not None:
            mask = LayoutConverter().nchw_to_nhwc(mask)
        weight = LayoutConverter().nchw_to_nhwc(weight)

        output_hbir = hbir.deform_conv2d(
            input,
            weight,
            offset,
            mask,
            stride,
            padding + padding,
            dilation,
            groups,
            offset_groups,
            mask is not None,
            bias=bias,
        )

        output_hbir = layout_converter.nhwc_to_nchw(output_hbir)

        return JitTensor.attach_hbir_to_tensor(output, output_hbir)


@JitTensor.register_subclass_converter(_to_pow_quantized)
class FuncToPowQuantizedConverter(FuncConverterBase):
    @classmethod
    def convert(cls, output: QTensor, input: JitTensor, *args, **kwargs):
        hbir_input = JitTensor.gather_hbir(input)
        hbir_output = const_fake_quant_hbir(
            hbir_input, output.q_scale(), output.dtype
        )
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@Exporter.register_converter(qat.DetectionPostProcessV1)
class DetectionPostProcessV1Converter(QatModuleConverterBase):
    @classmethod
    def convert(
        cls,
        mod: qat.DetectionPostProcessV1,
        output,
        data: List[Tensor],
        anchors: List[Tensor],
        image_sizes: Tuple[int, int] = None,
    ):
        if image_sizes is not None:
            raise ValueError(
                "Dynamic image_sizes is not supported when export DPP to hbir"
            )
        bs = data[0].size(0)
        num_branches = len(data)
        num_anchors = [int(a.size(1) / 4) for a in anchors]

        stride = []
        for per_branch_anchor in anchors:
            stride.extend(
                [
                    int(
                        (
                            per_branch_anchor[0, 1, 1, 0]
                            - per_branch_anchor[0, 1, 0, 0]
                        ).item()
                    ),
                    int(
                        (
                            per_branch_anchor[0, 0, 0, 1]
                            - per_branch_anchor[0, 0, 0, 0]
                        ).item()
                    ),
                ]
            )

        # list of Tensor(N, anchor_num * 4, H, W) ->
        # Tensor(num_branches * anchor_num, 4)
        anchors = torch.cat(
            [
                per_branch_anchors[0, :, 0, 0].flatten()
                for per_branch_anchors in anchors
            ]
        ).reshape(-1, 4)

        x1 = anchors[:, 0]
        y1 = anchors[:, 1]
        x2 = anchors[:, 2]
        y2 = anchors[:, 3]

        # (num_branches * anchor_num, 4): [height, width, center_y, center_x]
        anchors = torch.stack(
            [y2 - y1, x2 - x1, (y1 + y2) / 2, (x1 + x2) / 2], dim=-1
        )

        shifted_anchors = torch.ops.horizon.round(anchors * 4).to(
            dtype=torch.int32
        )

        # (num_branches * anchor_num * 4)
        flattened_anchors = shifted_anchors.flatten().tolist()

        hbir_output_list = []
        for i in range(bs):
            # list of Tensor(1, C, H, W)
            hbir_inputs = []
            for per_branch_data in data:
                input_base: QTensor = JitTensor.get_base(per_branch_data)
                _, C, H, W = input_base.shape  # noqa: N806
                hbir_input = hbir.slice(
                    JitTensor.gather_hbir(per_branch_data),
                    [i, 0, 0, 0],
                    [i + 1, C, H, W],
                    [1, 1, 1, 1],
                )
                hbir_input = LayoutConverter().nchw_to_nhwc(hbir_input)
                hbir_input = qnt.quantize(
                    hbir_input,
                    to_numpy(input_base.q_scale()).tolist(),
                    to_numpy(input_base.q_zero_point()).tolist(),
                    output_type=get_hbir_tensor_qtype(input_base.dtype),
                )
                hbir_inputs.append(hbir_input)

            # HBTensor(4095, 6)
            ret = hbir.dpp(
                hbir_inputs,
                flattened_anchors,
                num_anchors,
                math.ceil(mod.box_filter_threshold * (1 << mod.input_shift)),
                math.ceil(mod.nms_threshold * (1 << mod.kNmsThresholdShift)),
                math.ceil(mod.nms_margin * (1 << mod.input_shift)),
                1,  # seed
                [int(mod.use_clippings)] * num_branches,
                stride,
                mod.class_offsets,
                mod.image_size,
                mod.input_shift,
                maxBoxNum=mod.post_nms_top_k,
            )

            # hbir.dpp del the 'bbox_num' output in newer hbdk4 version.
            if isinstance(ret, (list, tuple, ir.OpResultList)):
                hbir_output = ret[1]
            else:
                hbir_output = ret

            hbir_output = qnt.dequantize(hbir_output, [0.25], [0])

            # Do not split hbir output, we modify torch DPP output to align
            # to hbir.
            hbir_output_list.append(hbir_output)
            # splited_hbir_output = (
            #     hbir.reshape(
            #         hbir.slice(
            #             hbir_output, [0, 0, 0], [1, 4096, 4], step=[1, 1, 1]
            #         ),
            #         (4096, 4),
            #     ),
            #     hbir.reshape(hbir.index(hbir_output, 4, dim=2), (4096,)),
            #     hbir.reshape(hbir.index(hbir_output, 5, dim=2), (4096,)),
            # )
            # if type(mod) == qat.DetectionPostProcessV1:
            #     in_scale = torch.ones(1, dtype=torch.float32) / (1 << 4)
            #     out_scale = torch.ones(1, dtype=torch.float32) / (1 << 2)
            #     splited_hbir_output = (
            #         const_fake_quant_hbir(
            #             splited_hbir_output[0],
            #             out_scale,
            #             qint16,
            #         ),
            #         const_fake_quant_hbir(
            #             splited_hbir_output[1],
            #             in_scale,
            #             qint8,
            #         ),
            #         splited_hbir_output[2],
            #     )
            # hbir_output_list.append(splited_hbir_output)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output_list)


@Exporter.register_converter(hnn.RcnnPostProcess)
class RcnnPostProcessConverter(QatModuleConverterBase):
    @classmethod
    def convert(
        cls,
        mod: hnn.RcnnPostProcess,
        output,
        boxes: List[JitTensor],
        scores: JitTensor,
        deltas: JitTensor,
        image_sizes: Optional[JitTensor] = None,
    ):
        if len(boxes) != 1:
            raise ValueError(
                "RcnnPostProcess only support batch_size=1 when export hbir."
            )
        if image_sizes is not None:
            raise ValueError(
                "RcnnPostProcess only support static imgge sizes when export "
                "hbir."
            )

        if isinstance(boxes, (list, tuple)):
            boxes = boxes[0]

        # Correct shape and layout, enable jit to export hbir for these ops
        with JitTensor.enable_jit():
            boxes = boxes.as_subclass(Tensor)
            if boxes.dim() > 3:
                boxes = boxes.flatten(0, -3)
            if boxes.dim() < 3:
                boxes = boxes.unsqueeze(0)
            if boxes.size(-1) == 4:
                boxes = torch.cat(
                    [boxes, torch.zeros_like(boxes)[..., :2]], dim=-1
                )
            scores = scores.permute(0, 2, 3, 1)
            deltas = deltas.permute(0, 2, 3, 1)

        boxes, scores, deltas, image_sizes = JitTensor.gather_hbir(
            (boxes, scores, deltas, image_sizes)
        )

        boxes = qnt.quantize(
            boxes, [0.25], [0], output_type=get_hbir_tensor_qtype(qint16)
        )

        _, hbir_float_ret = hbir.rpp_v2(
            boxes,
            scores,
            deltas,
            mod.fixed_image_h,
            mod.fixed_image_w,
            mod.nms_threshold,
            mod.box_filter_threshold,
            mod.num_classes,
            mod.post_nms_top_k,
            mod.delta_mean,
            mod.delta_std,
        )

        return JitTensor.attach_hbir_to_tensor(
            output, hbir_float_ret, check_shape=False
        )


@Exporter.register_converter(qat.div.SegmentLUTDiv)
class QatDivConverter(QatModuleConverterBase):
    @classmethod
    def convert(cls, mod: qat.div.SegmentLUTDiv, output, x, y):
        if isinstance(y, (int, float)):
            hbir_x = JitTensor.gather_hbir(x)
            hbir_output = hbir.mul(hbir_x, 1 / float(y))
        else:
            y = SegmentLUTConverter.convert(
                mod.reciprocal.reciprocal, JitTensor.get_base(y), y
            )
            hbir_x, hbir_y = JitTensor.gather_hbir((x, y))
            hbir_output = hbir.mul(hbir_x, hbir_y)
        hbir_output = cls.convert_activation_post_process(mod.mul, hbir_output)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


JitTensor.register_subclass_converter(Tensor.expand, Tensor.expand_as)(
    ExpandConverter
)


@JitTensor.register_converter(filter)
@JitTensor.register_subclass_converter(filter)
class FilterConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        inputs,
        scales,
        zero_points,
        dtypes,
        threshold,
        idx_range,
        march,
    ):
        hbir_inputs = JitTensor.gather_hbir(inputs)
        base_inputs: List[Tensor] = JitTensor.get_base(inputs)
        scales: List[Tensor] = JitTensor.get_base(scales)

        batch_size = base_inputs[0].size(0)

        per_channel_len = []
        quantized_hbir_inputs = []
        for hbir_input, input, scale in zip(hbir_inputs, base_inputs, scales):
            per_channel_len.append(input.size(1))
            quantized_hbir_inputs.append(
                qnt.quantize(
                    hbir_input,
                    scale.numpy().tolist(),
                    [0],
                    output_type=get_hbir_tensor_qtype(dtypes[0]),
                )
            )

        batched_hbir_output = []
        for i in range(batch_size):
            per_batch_hbir_inputs = []
            for x in quantized_hbir_inputs:
                per_batch_hbir_inputs.append(hbir.index(x, i, dim=0))

            if len(per_batch_hbir_inputs) == 1:
                concated_hbir_input = per_batch_hbir_inputs[0]
            else:
                # cat [c1, h, w], [c2, h, w] to [c1+c2, h, w]
                concated_hbir_input = hbir.concat(per_batch_hbir_inputs, dim=0)

            layout_converter = LayoutConverter(force_2d=True)
            concated_hbir_input = layout_converter.nchw_to_nhwc(
                concated_hbir_input
            )

            hbir_output = hbir.filter(
                concated_hbir_input,
                idx_range[0],
                idx_range[1],
                # ceil the quantized threshold, becase filter use '>='
                float(math.ceil(threshold / scales[0].item())),
                maxIndex_type=ir.UnrankedTensorType.get(
                    ir.IntegerType.get_signed(16)
                ),
                filterCoord_type=ir.UnrankedTensorType.get(
                    ir.IntegerType.get_signed(16)
                ),
            )

            max_value, max_index, coords, filtered_data = hbir_output

            hbir_output_list = [
                qnt.dequantize(max_value, scales[0].numpy().tolist(), [0]),
                max_index,
                coords,
            ]

            if len(per_channel_len) == 1:
                filtered_data = qnt.dequantize(
                    filtered_data, scales[0].numpy().tolist(), [0]
                )
                hbir_output_list.append(filtered_data)
            else:
                start = 0
                for channel_len, out_scale in zip(per_channel_len, scales):
                    end = start + channel_len
                    single_out = hbir.dynamic_slice(
                        filtered_data,
                        hbir.constant(
                            torch.tensor((start,)).numpy(),
                            output_type=get_hbir_dtype(torch.int64),
                        ),
                        hbir.constant(
                            torch.tensor((end,)).numpy(),
                            output_type=get_hbir_dtype(torch.int64),
                        ),
                        hbir.constant(
                            torch.tensor((1,)).numpy(),
                            output_type=get_hbir_dtype(torch.int64),
                        ),
                        hbir.constant(
                            torch.tensor((1,)).numpy(),
                            output_type=get_hbir_dtype(torch.int64),
                        ),
                        output_type=get_hbir_tensor_qtype(dtypes[0]),
                    )
                    single_out = qnt.dequantize(
                        single_out, out_scale.numpy().tolist(), [0]
                    )
                    hbir_output_list.append(single_out)
                    start = end

            batched_hbir_output.append(hbir_output_list)

        return JitTensor.attach_hbir_to_tensor(
            output, batched_hbir_output, check_shape=False
        )


JitTensor.register_subclass_converter(torch.t, Tensor.t)(TConverter)
JitTensor.register_subclass_converter(torch.flip, Tensor.flip)(FlipConverter)
JitTensor.register_subclass_converter(torch.gather, Tensor.gather)(
    GatherConverter
)
JitTensor.register_subclass_converter(Tensor.__getitem__)(GetItemConverter)


@JitTensor.register_converter(hF.warp)
class WarpConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(
        cls, x: ir.Value, grid: ir.Value, mode="bilinear", padding_mode="zeros"
    ):
        if padding_mode == "zeros":
            padding_mode = "constant"
            pad_value = 0
        else:
            padding_mode = padding_mode
            pad_value = None

        layout_converter = LayoutConverter()
        x = layout_converter.nchw_to_nhwc(x)

        hbir_output = hbir.warp(
            x, grid, mode, padding_mode, padValue=pad_value
        )
        return layout_converter.nhwc_to_nchw(hbir_output)


@JitTensor.register_subclass_converter(hF.warp)
class QTensorWarpConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        x: Tensor,
        grid: Tensor,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ):
        x, grid = JitTensor.gather_hbir((x, grid))
        hbir_output = WarpConverter.convert_with_hbir(
            x, grid, mode, padding_mode
        )
        hbir_output = const_fake_quant_like(hbir_output, output)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_subclass_converter(F.grid_sample)
class QTensorGridSampleConverter(GridSampleConverter):
    @classmethod
    def convert(
        cls,
        output,
        input: ir.Value,
        grid: ir.Value,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: Optional[bool] = None,
    ):
        grid_type = qint16
        input_h, input_w = JitTensor.get_base(input).shape[-2:]
        if align_corners:
            grid_scale_x = (
                1 if input_w == 1 else (2 / (input_w - 1) + 1)
            ) / grid_type.max
            grid_scale_y = (
                1 if input_h == 1 else (2 / (input_h - 1) + 1)
            ) / grid_type.max
        else:
            grid_scale_x = (2 / input_w + 1) / grid_type.max
            grid_scale_y = (2 / input_h + 1) / grid_type.max
        grid_scale = torch.tensor([grid_scale_x, grid_scale_y])

        grid_hbir = JitTensor.gather_hbir(grid)
        grid_hbir = const_fake_quant_hbir(
            grid_hbir, grid_scale, grid_type, axis=3
        )
        hbir_output = cls.convert_with_hbir(
            JitTensor.gather_hbir(input),
            grid_hbir,
            mode,
            padding_mode,
            align_corners,
        )
        hbir_output = const_fake_quant_like(hbir_output, output)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


JitTensor.register_subclass_converter(torch.index_select, Tensor.index_select)(
    IndexSelectConverter
)


@JitTensor.register_subclass_converter(F.interpolate)
class QTensorInterpolateConverter(InterpolateConverter):
    @classmethod
    def step_fq(cls, step: Tuple[float, float]):
        from horizon_plugin_pytorch.nn.qat import interpolate

        if interpolate._fake_quant_step:
            step_frac_bitnum = 16
            scale = 1 << step_frac_bitnum
            logger.debug("Interpolate step {}".format(step))
            step = tuple(float(math.floor(x * scale)) / scale for x in step)
            logger.debug("Interpolate fq step {}".format(step))
        return step

    @classmethod
    def convert(
        cls,
        output,
        input: JitTensor,
        size: Optional[Union[int, List[int]]] = None,
        scale_factor=None,
        mode="nearest",
        align_corners=False,
        recompute_scale_factor=None,
        antialias=False,
    ):
        hbir_output = cls.convert_interpolate(
            output,
            input,
            size,
            scale_factor,
            mode,
            align_corners,
            recompute_scale_factor,
            antialias,
        )
        hbir_output = const_fake_quant_like(hbir_output, output)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@Exporter.register_converter(hnn.LayerNorm)
class HorizonLayerNormConverter(ModuleConverterBase):
    @classmethod
    def convert(cls, mod: hnn.LayerNorm, output, input: ir.Value):
        hbir_input, hbir_weight, hbir_bias = JitTensor.gather_hbir(
            (input, mod.weight, mod.bias)
        )
        input_base, weight_base = JitTensor.get_base((input, mod.weight))

        if weight_base is not None and input_base.ndim > weight_base.ndim:
            weight_shape = [1] * (input_base.ndim - weight_base.ndim) + list(
                weight_base.shape
            )

            hbir_weight = hbir.reshape(hbir_weight, weight_shape)
            hbir_bias = hbir.reshape(hbir_bias, weight_shape)

        if mod.dim is None:
            dims = list(range(-len(mod.normalized_shape), 0, 1))
            output_hbir = hbir.layernorm(
                hbir_input,
                dims,
                mod.eps,
                weight=hbir_weight,
                bias=hbir_bias,
            )
        else:
            mean = hbir.reduce_mean(hbir_input, [mod.dim], True)
            diff = hbir.sub(hbir_input, mean)
            diff_square = hbir.mul(diff, diff)
            var = hbir.reduce_mean(diff_square, [mod.dim], True)
            dev_rec = hbir.rsqrt(var)
            output_hbir = hbir.mul(diff, dev_rec)
            if mod.elementwise_affine:
                output_hbir = hbir.mul(output_hbir, hbir_weight)
                output_hbir = hbir.add(output_hbir, hbir_bias)

        return JitTensor.attach_hbir_to_tensor(output, output_hbir)


JitTensor.register_subclass_converter(torch.logical_and, Tensor.logical_and)(
    LogicalAndConverter
)
JitTensor.register_subclass_converter(torch.logical_not, Tensor.logical_not)(
    LogicalNotConverter
)
JitTensor.register_subclass_converter(torch.logical_or, Tensor.logical_or)(
    LogicalOrConverter
)
JitTensor.register_subclass_converter(torch.max, Tensor.max)(MaxConverter)
JitTensor.register_subclass_converter(torch.min, Tensor.min)(MinConverter)
JitTensor.register_subclass_converter(F.max_pool1d, F.max_pool1d_with_indices)(
    MaxPool1dConverter
)
JitTensor.register_subclass_converter(F.max_pool2d, F.max_pool2d_with_indices)(
    MaxPool2dConverter
)


@JitTensor.register_subclass_converter(torch.mul, Tensor.mul)
class QTensorMulConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, input: ir.Value, other: Number):
        if not isinstance(other, (int, float)):
            msg = "QTensor.mul only support mul with int or float"
            logger.error(msg)
            raise ValueError(msg)
        if isinstance(other, int):
            other = float(other)
        return hbir.mul(input, other)


@JitTensor.register_subclass_converter(torch.masked_fill, Tensor.masked_fill)
class QTensorMaskedFillConverter(QuantOutputConverterBase):
    @classmethod
    def convert(cls, *args, **kwargs):
        return MaskedFillConverter.convert(*args, **kwargs)


@Exporter.register_converter(qat.MultiScaleRoIAlign)
class MultiScaleRoIAlignConverter(QatModuleConverterBase):
    _quantized_hbir_impl = True

    @classmethod
    def convert(
        cls,
        mod: qat.MultiScaleRoIAlign,
        output: QTensor,
        x: JitTensor,
        box_lists: Union[JitTensor, List[JitTensor]],
    ):
        if mod.sampling_ratio != 1:
            raise ValueError("Only support sampling_ratio=1")
        if isinstance(box_lists, (list, tuple)):
            if len(box_lists) != 1:
                raise ValueError("Do not support multi boxes")
            box_lists = box_lists[0]

        hbir_inputs, hbir_boxes = JitTensor.gather_hbir((x, box_lists))
        input_bases, boxes_base = JitTensor.get_base((x, box_lists))
        input_bases: List[QTensor]
        boxes_base: QTensor

        channel_len = input_bases[0].size(1)
        batch_size = boxes_base.size(0)
        box_num = boxes_base.size(1)
        output_size = _pair(mod.output_size)

        if boxes_base.ndim != 3 or boxes_base.size(-1) != 6:
            raise ValueError("Boxes must be of shape [b, n, 6] (RPP output).")

        layout_converter = LayoutConverter(force_2d=True)
        hbir_inputs = [layout_converter.nchw_to_nhwc(x) for x in hbir_inputs]

        if cls._quantized_hbir_impl:
            hbir_inputs = [
                qnt.quantize(
                    h,
                    to_numpy(x.q_scale()).tolist(),
                    to_numpy(x.q_zero_point()).tolist(),
                    output_type=get_hbir_tensor_qtype(x.dtype),
                )
                for h, x in zip(hbir_inputs, input_bases)
            ]
            hbir_boxes = qnt.quantize(
                hbir_boxes,
                to_numpy(boxes_base.q_scale()).tolist(),
                to_numpy(boxes_base.q_zero_point()).tolist(),
                output_type=get_hbir_tensor_qtype(boxes_base.dtype),
            )

        # inputs: [
        #   roi(from DPP)([n, 6]/[batch, n, 6] dtype=int16),
        #   features(HWC, NHWC) ...
        # ]
        # output: [nHWC, BnHWC]
        hbir_output = hbir.roi_align(
            [hbir_boxes] + hbir_inputs,
            output_size,
            mod.feature_strides,
            mod.sampling_ratio,
            mod.mode,
            mod.canonical_box_size,
            mod.canonical_level,
            output_type=(
                get_hbir_tensor_qtype(output.dtype)
                if cls._quantized_hbir_impl
                else None
            ),
        )
        if cls._quantized_hbir_impl:
            hbir_output = qnt.dequantize(
                hbir_output,
                to_numpy(output.q_scale()).tolist(),
                to_numpy(output.q_zero_point()).tolist(),
            )
        hbir_output = layout_converter.nhwc_to_nchw(hbir_output)
        hbir_output = hbir.reshape(
            hbir_output,
            (
                batch_size * box_num,
                channel_len,
                output_size[0],
                output_size[1],
            ),
        )
        hbir_output = cls.convert_activation_post_process(mod, hbir_output)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_subclass_converter(
    torch.neg, torch.negative, Tensor.negative, Tensor.neg
)
class QTensorNegConverter(FuncConverterBase):
    @classmethod
    def convert_with_hbir(cls, input: ir.Value):
        return hbir.neg(input)


@JitTensor.register_subclass_converter(F.pad)
class QTensorPadConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output: JitTensor,
        input: JitTensor,
        pad: Sequence[int],
        mode: str = "constant",
        value: Optional[float] = None,
    ):
        from horizon_plugin_pytorch.nn.quantized.functional import quantize

        if mode == "constant":
            input_base = JitTensor.get_base(input)
            value = float(
                quantize(
                    torch.tensor([0.0 if value is None else float(value)]),
                    input_base.q_scale().cpu(),
                    input_base.q_zero_point().cpu(),
                    -1,
                    input_base.dtype,
                )[0]
                * input_base.q_scale().cpu(),
            )

        return PadConverter.convert(
            output,
            input,
            pad,
            mode,
            value,
        )


JitTensor.register_subclass_converter(torch.permute, Tensor.permute)(
    PermuteConverter
)

JitTensor.register_subclass_converter(F.pixel_shuffle)(PixelShuffleConverter)
JitTensor.register_subclass_converter(F.pixel_unshuffle)(
    PixelUnShuffleConverter
)


@JitTensor.register_converter(hF.point_pillars_preprocess)
class PointPillarsPreprocessConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        points_list: List[Tensor],
        pc_range: Tensor,
        voxel_size: Tensor,
        max_voxels: int,
        max_points_per_voxel: int,
        use_max: bool,
        norm_range: Tensor,
        norm_dims: Tensor,
    ):
        if not use_max:
            msg = "use_max must be true for deploy"
            logger.error(msg)
            raise ValueError(msg)

        args = (
            to_numpy(pc_range).tolist(),
            to_numpy(voxel_size).tolist(),
            max_voxels,
            max_points_per_voxel,
            to_numpy(norm_dims).tolist(),
        )
        kwargs = {
            "normRanges": to_numpy(norm_range).tolist(),
            "coords_type": ir.UnrankedTensorType.get(
                ir.IntegerType.get_signed(32)
            ),
        }

        hbir_points_list = JitTensor.gather_hbir(points_list)

        hbir_features = []
        hbir_coords = []

        coord_shape = list(output[1].shape)
        coord_shape.insert(0, 1)

        for points in hbir_points_list:
            rets = hbir.point_pillar_preprocess(
                points,
                *args,
                **kwargs,
            )
            hbir_features.append(rets[0])
            hbir_coords.append(rets[1])

        hbir_features = hbir.concat(hbir_features, 1)
        hbir_coords = hbir.concat(
            hbir_coords,
            0,
            output_type=ir.UnrankedTensorType.get(
                ir.IntegerType.get_signed(32)
            ),
        )

        layout_converter = LayoutConverter()
        layout_converter.ori_rank = 4
        hbir_features = layout_converter.nhwc_to_nchw(hbir_features)

        return JitTensor.attach_hbir_to_tensor(
            output, (hbir_features, hbir_coords)
        )


@JitTensor.register_converter(hF.point_pillars_scatter)
@JitTensor.register_subclass_converter(hF.point_pillars_scatter)
class PointPillarsScatterConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        voxel_features: Tensor,
        coords: Tensor,
        output_shape: Union[Tensor, List[int]],
    ):
        if isinstance(output_shape, Tensor):
            output_shape = to_numpy(output_shape).tolist()
        output_shape = list(output_shape)
        output_shape[1], output_shape[2], output_shape[3] = (
            output_shape[2],
            output_shape[3],
            output_shape[1],
        )
        for i in range(4):
            if isinstance(output_shape[i], Tensor):
                output_shape[i] = output_shape[i].item()

        hbir_features, hbir_coords = JitTensor.gather_hbir(
            (voxel_features, coords)
        )

        hbir_output = hbir.point_pillar_scatter(
            hbir_features, hbir_coords, output_shape
        )
        hbir_output = hbir.transpose(hbir_output, (0, 3, 1, 2))

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


# Convert the whole module because we clip the input
@Exporter.register_converter(qat.Reciprocal)
class QATReciprocalConverter(QatModuleConverterBase):
    @classmethod
    def convert(cls, mod: qat.Reciprocal, output, input):
        return SegmentLUTConverter.convert(mod.reciprocal, output, input)


# Convert the whole module because we clip the input
@Exporter.register_converter(
    hnn.Asin,
    hnn.Asinh,
    hnn.Acos,
    hnn.Acosh,
    hnn.Atanh,
    hnn.Cosh,
    hnn.Erf,
    hnn.Sinh,
    hnn.Tan,
)
class QATSegmentLUTConverter(QatModuleConverterBase):
    @classmethod
    def convert(cls, mod: hnn.Asin, output, input):
        base_input = JitTensor.get_base(input)
        if isinstance(base_input, QTensor):
            return SegmentLUTConverter.convert(mod.func, output, input)
        else:
            converter = JitTensor._converters.get(mod.func.simulated_func)
            return converter.convert(output, input)


JitTensor.register_subclass_converter(torch.relu, Tensor.relu, F.relu)(
    ReLUConverter
)


JitTensor.register_subclass_converter(Tensor.repeat)(RepeatConverter)


JitTensor.register_subclass_converter(
    torch.reshape,
    Tensor.reshape,
    Tensor.view,
    torch.flatten,
    Tensor.flatten,
    torch.squeeze,
    Tensor.squeeze,
    torch.unsqueeze,
    Tensor.unsqueeze,
)(ReshapeConverter)


JitTensor.register_subclass_converter(torch.roll, Tensor.roll)(RollConverter)


@Exporter.register_converter(
    qat.SegmentLUT, qat.segment_lut.QuantizedQATSegmentLUT
)
class SegmentLUTConverter(QatModuleConverterBase):
    @classmethod
    def convert(
        cls,
        mod: qat.SegmentLUT,
        output: QTensor,
        input: JitTensor,
    ):
        from horizon_plugin_pytorch.nn import quantized

        if isinstance(mod, qat.segment_lut.QuantizedQATSegmentLUT):
            quantized_mod = mod.quantized_mod
            mod = mod.qat_mod
        else:
            quantized_mod = quantized.SegmentLUT.from_float(mod)
        hbir_input = JitTensor.gather_hbir(input)
        input_base = JitTensor.get_base(input)

        input_type = input_base.dtype

        if (
            hasattr(mod, "activation_pre_process")
            and mod.activation_pre_process is not None
        ):
            input_type = mod.activation_pre_process.dtype
            input_base = mod.activation_pre_process(input_base)
            hbir_input = cls.convert_activation_pre_process(mod, hbir_input)

        output_type = mod.activation_post_process.dtype
        input_hbir_type = get_hbir_tensor_qtype(
            input_type, list(input_base.shape)
        )
        output_hbir_type = get_hbir_tensor_qtype(
            output_type, list(input.shape)
        )

        hbir_input = qnt.quantize(
            hbir_input,
            to_numpy(input_base.q_scale()).tolist(),
            to_numpy(input_base.q_zero_point()).tolist(),
            output_type=input_hbir_type,
        )

        if input_type == qint8 and output_type == qint8:
            # roll the table because input is casted to uint8 before look up
            table = (
                quantized_mod._init_single_table_params(input_base.q_scale())
                .to(torch.int8)
                .roll(128)
            )
            # repeat the table for HW requirement
            table = torch.stack([table, table], dim=-1).flatten()
            hbir_output = lut_with_march(
                hbir_input,
                to_numpy(table),
                output_type=output_hbir_type,
            )
        else:
            (
                table,
                alpha,
                beta,
                left_shift,
                right_shift,
                segment_max,
            ) = quantized_mod._init_multi_table_params(
                input_base.q_scale(), input_base.dtype
            )
            config = torch.stack(
                [
                    beta.to(torch.int16),
                    alpha.to(torch.int16),
                    torch.bitwise_or(
                        left_shift.to(torch.int16) << 8,
                        right_shift.to(torch.int16),
                    ),
                    segment_max.to(torch.int16),
                ]
            )
            config = torch.cat(
                [
                    table.flatten().to(torch.int16),
                    torch.zeros(
                        64,
                        dtype=torch.int16,
                        device=table.device,
                    ),
                    config.transpose(0, 1).flatten(),
                    torch.zeros(
                        32,
                        dtype=torch.int16,
                        device=table.device,
                    ),
                ]
            )

            # symmetricMode in ("ASYM", "YSYM", "CSYM")
            hbir_output = lut_with_march(
                hbir_input,
                to_numpy(config),
                symmetricMode="CSYM" if mod.is_centrosymmetric else "ASYM",
                output_type=output_hbir_type,
            )
        hbir_output = qnt.dequantize(
            hbir_output,
            to_numpy(mod.activation_post_process.scale).tolist(),
            to_numpy(mod.activation_post_process.zero_point).tolist(),
        )
        hbir_output = cls.convert_activation_post_process(mod, hbir_output)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@Exporter.register_converter(hnn.Cos, hnn.Sin)
class CosSinConverter(QatModuleConverterBase):
    _limit_input_in_one_period = True

    @classmethod
    def limit_periodic_function_input_range(cls, hbir_input, input_base, mod):
        """
        Limit the input range of periodic function to 1 period.

        1. p_num = input / p, p_num_scale = input_scale / p
        2. p_num_floor = floor(p_num),
            p_num_floor_scale = ceil(input_scale * qtype_max / p) / qtype_max
        3. p_num = p_num - p_num_floor, p_num_scale = 1 / qtype_max
        4. new_input = p_num * p, new_input_scale = p / qtype_max
        """
        period = 2 * math.pi

        if (
            not cls._limit_input_in_one_period
            or period >= input_base.q_scale() * qinfo(input_base.dtype).max
        ):
            return hbir_input, input_base.q_scale()

        logger.info(
            f"input range of {mod.simulated_func} is too large to keep segment"
            f" lut precision, will limit input range to 1 period ({period})"
        )

        hbir_input = hbir.mul(hbir_input, 1 / period)
        hbir_input = const_fake_quant_hbir(
            hbir_input,
            input_base.q_scale() / period,
            input_base.dtype,
        )
        hbir_input_floor = hbir.floor(hbir_input)
        # hbdk will change the scale of floor output to one automatically and
        # rescale to the scale we set to keep the accuracy of quantized sub.
        floor_out_scale = (
            torch.ceil(
                input_base.q_scale() * qinfo(input_base.dtype).max / period
            )
            / qinfo(input_base.dtype).max
        )
        hbir_input_floor = const_fake_quant_hbir(
            hbir_input_floor,
            floor_out_scale,
            input_base.dtype,
        )
        hbir_input = hbir.sub(hbir_input, hbir_input_floor)
        hbir_input = const_fake_quant_hbir(
            hbir_input,
            torch.tensor([1 / qinfo(input_base.dtype).max]),
            input_base.dtype,
        )
        hbir_input = hbir.mul(hbir_input, period)

        input_scale = torch.tensor([period / qinfo(input_base.dtype).max])
        hbir_input = const_fake_quant_hbir(
            hbir_input,
            input_scale,
            input_base.dtype,
        )

        return hbir_input, input_scale

    @classmethod
    def convert(cls, mod, output, input):
        from horizon_plugin_pytorch.nn import quantized

        lut_mod = mod.cos if isinstance(mod, hnn.Cos) else mod.sin
        if isinstance(lut_mod, hnn.SegmentLUT):
            # For float, directly export the function calls.
            with JitTensor.enable_jit():
                return mod.forward(input)
        elif isinstance(lut_mod, qat.segment_lut.QuantizedQATSegmentLUT):
            quantized_mod = lut_mod.quantized_mod
            mod = lut_mod.qat_mod
        else:
            quantized_mod = quantized.SegmentLUT.from_float(lut_mod)
            mod = lut_mod
        hbir_input = JitTensor.gather_hbir(input)
        input_base = JitTensor.get_base(input)

        input_type = input_base.dtype

        if (
            hasattr(mod, "activation_pre_process")
            and mod.activation_pre_process is not None
        ):
            input_type = mod.activation_pre_process.dtype
            input_base = mod.activation_pre_process(input_base)
            hbir_input = cls.convert_activation_pre_process(mod, hbir_input)

        output_type = mod.activation_post_process.dtype
        input_hbir_type = get_hbir_tensor_qtype(
            input_type, list(input_base.shape)
        )
        output_hbir_type = get_hbir_tensor_qtype(
            output_type, list(input.shape)
        )

        hbir_input, input_scale = cls.limit_periodic_function_input_range(
            hbir_input, input_base, mod
        )

        hbir_input = qnt.quantize(
            hbir_input,
            to_numpy(input_scale).tolist(),
            to_numpy(input_base.q_zero_point()).tolist(),
            output_type=input_hbir_type,
        )

        if input_type == qint8 and output_type == qint8:
            # roll the table because input is casted to uint8 before look up
            table = (
                quantized_mod._init_single_table_params(input_scale)
                .to(torch.int8)
                .roll(128)
            )
            # repeat the table for HW requirement
            table = torch.stack([table, table], dim=-1).flatten()
            hbir_output = lut_with_march(
                hbir_input,
                to_numpy(table),
                output_type=output_hbir_type,
            )
        else:
            (
                table,
                alpha,
                beta,
                left_shift,
                right_shift,
                segment_max,
            ) = quantized_mod._init_multi_table_params(
                input_scale, input_base.dtype
            )
            config = torch.stack(
                [
                    beta.to(torch.int16),
                    alpha.to(torch.int16),
                    torch.bitwise_or(
                        left_shift.to(torch.int16) << 8,
                        right_shift.to(torch.int16),
                    ),
                    segment_max.to(torch.int16),
                ]
            )
            config = torch.cat(
                [
                    table.flatten().to(torch.int16),
                    torch.zeros(
                        64,
                        dtype=torch.int16,
                        device=table.device,
                    ),
                    config.transpose(0, 1).flatten(),
                    torch.zeros(
                        32,
                        dtype=torch.int16,
                        device=table.device,
                    ),
                ]
            )

            # symmetricMode in ("ASYM", "YSYM", "CSYM")
            hbir_output = lut_with_march(
                hbir_input,
                to_numpy(config),
                symmetricMode="CSYM" if mod.is_centrosymmetric else "ASYM",
                output_type=output_hbir_type,
            )
        hbir_output = qnt.dequantize(
            hbir_output,
            to_numpy(mod.activation_post_process.scale).tolist(),
            to_numpy(mod.activation_post_process.zero_point).tolist(),
        )
        hbir_output = cls.convert_activation_post_process(mod, hbir_output)
        return JitTensor.attach_hbir_to_tensor(output, hbir_output)

    @classmethod
    def limit_input_in_one_period(cls, enable):
        cls._limit_input_in_one_period = enable


JitTensor.register_subclass_converter(torch.sort, Tensor.sort)(SortConverter)
JitTensor.register_subclass_converter(torch.split)(TorchSplitConverter)
JitTensor.register_subclass_converter(Tensor.split)(TensorSplitConverter)
JitTensor.register_subclass_converter(torch.chunk, Tensor.chunk)(
    ChunkConverter
)


@JitTensor.register_subclass_converter(_sub_stub)
class SubConverter(FuncConverterBase):
    @classmethod
    def convert(
        cls,
        output,
        x: Union[JitTensor, float, int],
        y: Union[JitTensor, float, int],
    ):
        if isinstance(x, (float, int)):
            x_hbir = const_fake_quant_scalar(
                x,
                JitTensor.get_base(y).dtype,
            )
        else:
            x_hbir = JitTensor.gather_hbir(x)
        if isinstance(y, (float, int)):
            y_hbir = const_fake_quant_scalar(
                y,
                JitTensor.get_base(x).dtype,
            )
        else:
            y_hbir = JitTensor.gather_hbir(y)
        hbir_output = hbir.sub(x_hbir, y_hbir)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


JitTensor.register_subclass_converter(torch.tile, Tensor.tile)(TileConverter)
JitTensor.register_subclass_converter(torch.topk, Tensor.topk)(TopkConverter)
JitTensor.register_subclass_converter(torch.transpose, Tensor.transpose)(
    TransposeConverter
)
JitTensor.register_subclass_converter(torch.all, Tensor.all)(
    ReduceAllConverter
)
JitTensor.register_subclass_converter(torch.argsort, Tensor.argsort)(
    ArgSortConverter
)
JitTensor.register_subclass_converter(
    torch.masked_select, Tensor.masked_select
)(MaskedSelectConverter)


class TriluConverter(FuncConverterBase):
    torch_func = None

    @classmethod
    def convert(cls, output: Tensor, input: JitTensor, diagonal=0):
        input_base = JitTensor.get_base(input)
        mask = torch.ones_like(input_base, dtype=torch.bool)
        mask = cls.torch_func(mask, diagonal)

        hbir_input = JitTensor.gather_hbir(input)
        hbir_mask = JitTensor.gather_hbir(mask)

        hbir_output = hbir.where(hbir_mask, hbir_input, 0.0)
        hbir_output = const_fake_quant_like(hbir_output, input_base)

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)


@JitTensor.register_subclass_converter(torch.tril, Tensor.tril)
class TrilConverter(TriluConverter):
    torch_func = torch.tril


@JitTensor.register_subclass_converter(torch.triu, Tensor.triu)
class TriuConverter(TriluConverter):
    torch_func = torch.triu


JitTensor.register_subclass_converter(torch.unbind, Tensor.unbind)(
    UnbindConverter
)


@JitTensor.register_subclass_converter(torch.sign, Tensor.sign)
class SignConvert(FuncConverterBase):
    @classmethod
    def convert(cls, output: Tensor, input: JitTensor):
        hbir_input = JitTensor.gather_hbir(input)

        hbir_output = hbir.sign(hbir_input)
        hbir_output = const_fake_quant_hbir(
            hbir_output, output.q_scale(), output.dtype
        )

        return JitTensor.attach_hbir_to_tensor(output, hbir_output)
