from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.onnx import register_custom_op_symbolic
import torch.onnx.symbolic_helper as sym_helper
from torch.onnx.symbolic_helper import parse_args
import torch.onnx.symbolic_opset9 as sym_opset9

if TYPE_CHECKING:
    import io


def export_onnx(
    model: torch.nn.Module,
    dummy_inputs: Union[Tuple, torch.Tensor],
    onnx_file: Union[str, "io.BytesIO"],
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    opset_version: int = 11,
    dynamic_axes: Optional[
        Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]
    ] = None,
    **kwargs,
) -> None:
    """将torch模型(torch.nn.Module)导出成onnx模型(ModelProto).

    Args:
        model: 用于导出的torch模型
        dummy_inputs: torch模型输入, 用于完成模型tracing
        onnx_file: 用于保存导出onnx模型的路径名或者字节流对象
        input_names: 导出onnx模型的输入名称序列
        output_names: 导出onnx模型的输出名称序列
        opset_version: 导出onnx模型的opset版本
        dynamic_axes: 导出onnx模型输入和输出的动态维度
        **kwargs: 其他公版torch导出onnx模型所需参数
    """
    assert isinstance(
        model,
        torch.nn.Module,
    ), "The input model is not of type torch.nn.Module."
    model.eval()
    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_file,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        **kwargs,
    )


@parse_args("v", "v")
def _maximum(g, self, other):
    return g.op("Max", self, other)


@parse_args("v", "v")
def _minimum(g, self, other):
    return g.op("Min", self, other)


@parse_args("v", "v", "v", "i", "i", "i", "b", "b", "b", "s", "s")
def _scale_quanti(
    g,
    self,
    scale,
    zero_point,
    vector_dim,
    quant_min,  # noqa: ARG001
    quant_max,
    saturate,  # noqa: ARG001
    in_place,  # noqa: ARG001
    compat_mask=True,  # noqa: ARG001
    approximate_mode="bpu_round",  # noqa: ARG001
    march="bayes",  # noqa: ARG001
):
    num_bits = 8 if quant_max == 127 else 16
    # set zero_point to 0 and make onnx graph concise.
    # "constant" will be folded and its onnx graph is more concise than "cast"
    if vector_dim == -1:
        zero_point = g.op(
            "Constant",
            value_t=torch.zeros(1, dtype=torch.int8),
        )
    else:
        zero_point = g.op(
            "Constant",
            value_t=torch.zeros(zero_point.type().sizes(), dtype=torch.int8),
        )

    return g.op(
        "horizon::HzDequantize",
        g.op("horizon::HzQuantize", self, scale, zero_point, bits_i=num_bits),
        scale,
        zero_point,
    )


@parse_args("v", "v", "v", "i", "i", "i")
def _fake_quantize_per_channel_affine(
    g,
    self,
    scale,
    zero_point,
    axis,
    quant_min=-128,
    quant_max=127,
):
    if (quant_min, quant_max) != (-128, 127) and (quant_min, quant_max) != (
        -32768,
        32767,
    ):
        raise RuntimeError(
            f"Horizon defines [-128, 127] for qint8 and [-32768, 32767] "
            f"for qint16, but got [{quant_min}, {quant_max}]",
        )

    # Horizon defines zero_point to be int8 or int16
    if quant_min == -128:
        num_bits = 8
        zero_point = g.op(
            "Cast",
            zero_point,
            to_i=sym_helper.cast_pytorch_to_onnx["Char"],
        )
    else:
        num_bits = 16
        zero_point = g.op(
            "Cast",
            zero_point,
            to_i=sym_helper.cast_pytorch_to_onnx["Short"],
        )

    return g.op(
        "horizon::HzDequantize",
        g.op(
            "horizon::HzQuantize",
            self,
            scale,
            zero_point,
            bits_i=num_bits,
            axis_i=axis,
        ),
        scale,
        zero_point,
        axis_i=axis,
    )


@parse_args("v", "t", "i", "i", "i")
def _fake_quantize_per_tensor_affine(
    g,
    self,
    scale,
    zero_point,
    quant_min=-128,
    quant_max=127,
):
    if (quant_min, quant_max) != (-128, 127) and (quant_min, quant_max) != (
        -32768,
        32767,
    ):
        raise RuntimeError(
            f"Horizon defines [-128, 127] for qint8 and [-32768, 32767] "
            f"for qint16, but got [{quant_min}, {quant_max}]",
        )
    scale = scale.float().data  # Avoid exporter generating double type
    zero_point_dtype = torch.int8 if quant_min == -128 else torch.int16
    num_bits = 8 if quant_min == -128 else 16
    zero_point = torch.tensor(
        zero_point,
        dtype=zero_point_dtype,
    )  # ONNX requires zero_point to be tensor
    return g.op(
        "horizon::HzDequantize",
        g.op("horizon::HzQuantize", self, scale, zero_point, bits_i=num_bits),
        scale,
        zero_point,
    )


@parse_args("v", "v", "i", "i", "i")
def _grid_sampler(g, self, grid, interpolation_mode, padding_mode, align_corners=False):
    interpolation_mode_list = ["bilinear", "nearest", "bicubic"]
    padding_mode_list = ["zeros", "border", "reflection"]
    return g.op(
        "horizon::GridSample",
        self,
        grid,
        mode_s=interpolation_mode_list[interpolation_mode],
        padding_mode_s=padding_mode_list[padding_mode],
        align_corners_i=align_corners,
    )


@parse_args("v", "v")
def _bitwise_and(g, self, other):
    return g.op("horizon::BitwiseAnd", self, other)


@parse_args("v", "v")
def _bitwise_or(g, self, other):
    return g.op("horizon::BitwiseOr", self, other)


@parse_args("v", "v")
def _bitwise_xor(g, self, other):
    return g.op("horizon::BitwiseXor", self, other)


@parse_args("v", "v")
def _tile(g, self, repeats):
    if not sym_helper._is_value(repeats):  # noqa: SLF001
        repeats = g.op("Constant", value_t=torch.LongTensor(repeats))
    if sym_helper._is_packed_list(repeats):  # noqa: SLF001
        repeat_size_len = len(sym_helper._unpack_list(repeats))  # noqa: SLF001
    else:
        const_repeats = sym_helper._maybe_get_const(repeats, "is")  # noqa: SLF001
        repeat_size_len = len(const_repeats)
    if self.isCompleteTensor():
        sizes = self.type().sizes()
        diff_dims = repeat_size_len - len(sizes)
        if diff_dims > 0:
            self = sym_opset9.view(
                g,
                self,
                g.op("Constant", value_t=torch.tensor([1] * diff_dims + sizes)),
            )
        if diff_dims < 0:
            const_repeats = sym_helper._maybe_get_const(repeats, "is")  # noqa: SLF001
            repeats = g.op(
                "Constant",
                value_t=torch.tensor(
                    [1] * (-diff_dims) + const_repeats,
                    dtype=sym_helper.scalar_type_to_pytorch_type[4],
                ),
            )
    return g.op("Tile", self, repeats)


@parse_args("v")
def _relu6_opset10(g, self):
    return g.op("Clip", self, min_f=0, max_f=6)


@parse_args("v")
def _acosh(g, self):
    return g.op("Acosh", self)


@parse_args("v")
def _asinh(g, self):
    return g.op("Asinh", self)


@parse_args("v")
def _atanh(g, self):
    return g.op("Atanh", self)


@parse_args("v")
def _cosh(g, self):
    return g.op("Cosh", self)


@parse_args("v")
def _sinh(g, self):
    return g.op("Sinh", self)


@parse_args("v")
def _relu6(g, self):
    dtype = self.type().scalarType()
    if dtype is None:
        dtype = 6  # float
    else:
        dtype = sym_helper.scalar_type_to_onnx.index(
            sym_helper.cast_pytorch_to_onnx[dtype]
        )
    dtype = sym_helper.scalar_type_to_pytorch_type[dtype]
    return g.op(
        "Clip",
        self,
        torch.tensor(0, dtype=dtype),
        torch.tensor(6, dtype=dtype),
    )


@parse_args("v", "i")
def _channel_shuffle(g, self, other):
    return g.op("horizon::HzChannelShuffle", self, group_i=other, data_format_s="NCHW")


@parse_args("v", "i")
def _pixel_unshuffle(g, self, other):
    in_channel = sym_helper._get_tensor_dim_size(self, 1)  # noqa: SLF001
    self = g.op("SpaceToDepth", self, blocksize_i=other)
    indices = [
        i + j * in_channel for i in range(in_channel) for j in range(other * other)
    ]
    return g.op(
        "Gather",
        self,
        g.op("Constant", value_t=torch.LongTensor(indices)),
        axis_i=1,
    )


# Export pytorch operator to onnx opset9 operator.
register_custom_op_symbolic("::tile", _tile, 9)
# Export pytorch operator to onnx opset10 operator.
register_custom_op_symbolic("::maximum", _maximum, 10)
register_custom_op_symbolic("::minimum", _minimum, 10)
register_custom_op_symbolic("horizon::scale_quanti", _scale_quanti, 10)
register_custom_op_symbolic("::grid_sampler", _grid_sampler, 10)
register_custom_op_symbolic(
    "::fake_quantize_per_channel_affine",
    _fake_quantize_per_channel_affine,
    10,
)
register_custom_op_symbolic(
    "::fake_quantize_per_tensor_affine",
    _fake_quantize_per_tensor_affine,
    10,
)
register_custom_op_symbolic("::bitwise_and", _bitwise_and, 10)
register_custom_op_symbolic("::bitwise_or", _bitwise_or, 10)
register_custom_op_symbolic("::bitwise_xor", _bitwise_xor, 10)
register_custom_op_symbolic("::relu6", _relu6_opset10, 10)
register_custom_op_symbolic("::channel_shuffle", _channel_shuffle, 10)
register_custom_op_symbolic("::pixel_unshuffle", _pixel_unshuffle, 10)
register_custom_op_symbolic("::acosh", _acosh, 10)
register_custom_op_symbolic("::asinh", _asinh, 10)
register_custom_op_symbolic("::atanh", _atanh, 10)
register_custom_op_symbolic("::cosh", _cosh, 10)
register_custom_op_symbolic("::sinh", _sinh, 10)
# Export pytorch operator to onnx opset11 operator.
register_custom_op_symbolic("::relu6", _relu6, 11)
# Export pytorch operator to onnx opset13 operator.
register_custom_op_symbolic("::tile", _tile, 13)
