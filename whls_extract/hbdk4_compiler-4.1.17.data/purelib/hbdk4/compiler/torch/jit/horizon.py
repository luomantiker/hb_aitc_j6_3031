from hbdk4.compiler.frontend.adaptor import NodeAdaptor
from hbdk4.compiler.frontend.convertor import OpConvertor
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.ops import qnt
from hbdk4.compiler.ops import hbir
from hbdk4.compiler.frontend.common import nchw_to_nhwc, nhwc_to_nchw

import torch

import math


class scale_quanti(OpConvertor):
    def __init__(self):
        super().__init__("scale_quanti", "horizon", 0, False)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        scale: mlir.Value,
        zeros: mlir.Value,
        axis: int,
        quant_min: int,
        quant_max: int,
        sat: bool,
        inplace: bool,
        compact_mask: mlir.Value,
        round: str,
        march: str,
    ):

        if march != "bayes":
            raise ValueError("invalid march {}".format(march))

        if round != "bpu_round":
            raise ValueError("invalid round {}".format(round))

        bits = math.ceil(math.log(quant_max - quant_min, 2))
        narrow = (quant_min + quant_max) == 0

        if isinstance(adaptor.operands[1].value, mlir.Value):
            raise ValueError("scale should be constant")

        if isinstance(adaptor.operands[2].value, mlir.Value):
            raise ValueError("zero should be constant")

        scale = adaptor.operands[1].value

        if scale.size(0) == 1:
            scale = float(scale[0])
            min_value = scale * quant_min
            max_value = scale * quant_max
            return qnt.const_fake_quant(
                x, [min_value], [max_value], bits, narrow, output_type=y
            )
        else:
            scale = scale.to(torch.double).detach().numpy()

            min_value = scale * quant_min
            max_value = scale * quant_max
            return qnt.const_fake_quant(
                x, min_value, max_value, bits, narrow, axis=axis, output_type=y
            )


scale_quanti()


class HorizonCvt(OpConvertor):
    def __init__(self, name: str):
        super().__init__(name, "horizon", 0, True)


class quanti_resize(HorizonCvt):
    def __init__(self):
        super().__init__("quanti_resize")
        # horizon::quanti_resize(Tensor _0, Tensor _1, Tensor _2, str _3, bool _4, int _5, int _6, float _7, float _8, bool _9, str _10) -> Tensor _0

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        scale: mlir.Value,
        zero_point: mlir.Value,
        mode: str,
        align_corners: bool,
        out_height: int,
        out_width: int,
        ratio_height: float,
        ratio_width: float,
        quantized_forward: bool,
        march: str,
    ):
        isize = adaptor.operands[0].type.shape[2:4]
        osize = adaptor.results[0].type.shape[2:4]

        if ratio_height > 0 and ratio_width > 0:
            ratio = [ratio_height, ratio_width]
        else:
            ratio = [osize[0] / isize[0], osize[1] / isize[1]]

        if align_corners:
            step = [(isize[0] - 1) / (osize[0] - 1), (isize[1] - 1) / (osize[1] - 1)]
            offset = [0.5 / ratio[0] - 0.5, 0.5 / ratio[1] - 0.5]
        else:
            step = [1 / ratio[0], 1 / ratio[1]]
            offset = [0.0, 0.0]

        x = nchw_to_nhwc(x)
        x = hbir.resize2d(x, step, mode="nearest", ratio=ratio, initialOffset=offset)
        return nhwc_to_nchw(x, otype)


quanti_resize()
