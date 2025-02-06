from typing import Optional

from hbdk4.compiler.frontend.adaptor import NodeAdaptor
from hbdk4.compiler.frontend.convertor import OpConvertor
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.ops import hbir, qnt
from hbdk4.compiler.ops.common import get_value_or_create_const


class Opset19(OpConvertor):
    def __init__(self, name: str):
        super().__init__(name, "onnx", 19, True)


class QuantizeLinear(Opset19):
    def __init__(self):
        super().__init__("QuantizeLinear")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        y_scale: mlir.Value,
        y_zero_point=None,
        *,
        saturate=1,
        axis=1,
    ):
        # only support int8/uint8 quantize linear, saturate is of no use
        y_scale = adaptor.operands[1].value
        if len(y_scale.shape) == 0:  # handle rank0 tensor
            y_scale = y_scale.reshape([1])
        y_zero_point = adaptor.operands[2].value
        if len(y_zero_point.shape) == 0:  # handle rank0 tensor
            y_zero_point = y_zero_point.reshape([1])

        if isinstance(y_scale, mlir.Value):
            raise ValueError("Operator QuantizeLinear scale must be constant")
        if isinstance(y_zero_point, mlir.Value):
            raise ValueError("Operator QuantizeLinear zero must be constant")

        if y.element_type.is_unsigned:
            raise ValueError("Operator QuantizeLinear does not support unsigned")

        return qnt.quantize(
            x,
            scales=y_scale,
            zeros=y_zero_point,
            output_type=y,
            axis=axis,
            narrowRange=False,
        )


QuantizeLinear()

# torch not currently supported opset19

# class DeformConv(Opset19):
#     def __init__(self):
#         super().__init__("DeformConv")

#     def emit_mlir_op(
#         self,
#         adaptor: NodeAdaptor,
#         y: mlir.Type,
#         x: mlir.Value,
#         w: mlir.Value,
#         offset: mlir.Value,
#         b: Optional[mlir.Value] = None,
#         mask: Optional[mlir.Value] = None,
#         *args,
#         dilations=(1, 1),
#         group=1,
#         kernel_shape=None,
#         offset_group=1,
#         pads=(0, 0, 0, 0),
#         strides=(1, 1),
#     ):
#         kernel_dim = len(kernel_shape)
#         if kernel_dim != 2:
#             raise ValueError(
#                 f"Only support DeformConv 2D Operator, got kernel_dim={kernel_dim}"
#             )
#         # input trans: nchw->nhwc
#         # output trans: nhwc->nchw
#         mask_shape = offset.type.shape
#         mask_shape[1] = mask_shape[1] // 2
#         use_mask = True
#         if mask is None:
#             use_mask = False
#             mask = get_value_or_create_const(np.ones(mask_shape))

#         x = hbir.transpose(x, [0, 2, 3, 1])
#         w = hbir.transpose(w, [0, 2, 3, 1])
#         offset = hbir.transpose(offset, [0, 2, 3, 1])
#         mask = hbir.transpose(mask, [0, 2, 3, 1])
#         x = hbir.deform_conv2d(
#             x,
#             w,
#             offset,
#             mask,
#             strides,
#             pads,
#             dilations,
#             group,
#             offset_group,
#             use_mask,
#             bias=b,
#         )
#         x = hbir.transpose(x, [0, 3, 1, 2], output_type=y)
#         return x


# DeformConv()
