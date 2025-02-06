from typing import List, Optional, Union

from hbdk4.compiler.frontend.adaptor import NodeAdaptor
from hbdk4.compiler.frontend.convertor import OpConvertor
from hbdk4.compiler.frontend.common import nchw_to_nhwc
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.frontend.common import nhwc_to_nchw
from hbdk4.compiler.ops import hbir, qnt, b25
import os
import numpy as np
from hbdk4.compiler.ops.common import get_value_or_create_const


class OpsetConvertor(OpConvertor):
    def __init__(self, name: str, foldable: bool):
        super().__init__(name, "onnx", 1, foldable)


class HzCalibrationConvertor(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzCalibration", False)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        scales: Union[float, List[float]] = None,
        bits: int = None,  # deprecated in recent horizon nn
        axis: int = None,
        zero_point: int = 0,
        qtype: str = None,  # new in horizon nn
        **ignored_attrs,
    ):
        if qtype is not None:
            qtype = qtype.decode()
            if qtype == "float16":
                return hbir.fake_cast(x, mlir.F16Type.get(), output_type=y)
            if qtype == "float32":
                return x
            bit_map = {
                "int8": 8,
                "uint8": 8,
                "int16": 16,
                "uint16": 16,
                "int32": 32,
            }
            zp_map = {"uint8": -128, "uint16": -32768}

            if qtype not in bit_map:
                raise ValueError(f"Operator HzCalibration has unknown qtype {qtype}")

            bits = bit_map[qtype]

            if qtype in zp_map:
                zero_point = zp_map[qtype]

        if isinstance(zero_point, list):
            zero_point = zero_point[0]

        quant_min = -1 * 2 ** (bits - 1) - zero_point
        quant_max = 2 ** (bits - 1) - 1 - zero_point

        if scales is None:
            raise ValueError(
                f"Operator HzCalibration scale is not valid but qtype is {qtype}"
            )

        scales = np.array(scales)

        min_value = scales * quant_min
        max_value = scales * quant_max
        narrow = (quant_min + quant_max) == 0

        if axis is None:
            return qnt.const_fake_quant(
                x, [min_value[0]], [max_value[0]], bits, narrow, output_type=y
            )
        else:
            return qnt.const_fake_quant(
                x,
                min_value.tolist(),
                max_value.tolist(),
                bits,
                narrow,
                axis=axis,
                output_type=y,
            )


class HzPreprocessConvertor(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzPreprocess", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        w: mlir.Value,
        b: Optional[mlir.Value] = None,
        *,
        channel_group,
        data_format="NCHW",
        input_type,
    ):
        strides = (1, 1)
        pads = (0, 0, 0, 0)
        dilations = (1, 1)
        data_format = str(data_format, "utf-8")
        x = hbir.transpose(x, [0, 2, 3, 1])
        w = hbir.transpose(w, [0, 2, 3, 1])
        x = hbir.conv2d(x, w, strides, pads, dilations, channel_group, bias=b)
        x = hbir.transpose(x, [0, 3, 1, 2], output_type=y)
        return x


calib = HzCalibrationConvertor()
HzPreprocess = HzPreprocessConvertor()


class HzLutConvertor(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzLut", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        param: mlir.Value,
        *,
        lut_type,
        data_format="NHWC",
    ):
        lut_param = adaptor.operands[1].value
        lut_param = np.roll(lut_param, 128)
        config = np.array([item for item in lut_param for _ in range(2)])
        if data_format == "NCHW":
            x = hbir.transpose(x, [0, 2, 3, 1])
        x = b25.lut(x, config, output_type=y)
        if data_format == "NCHW":
            x = hbir.transpose(x, [0, 3, 1, 2], output_type=y)
        return x


HzLut = HzLutConvertor()


class HzLut2LayerConvertor(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzLut2Layer", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        bias,
        bound,
        data_format="NHWC",
        left_shift,
        lut_type,
        output_bit,
        right_shift,
        rounding_method="ROUND",
        scale,
        symmetry_mode,
        table_param,
    ):
        config = np.zeros(512, dtype=np.int16)
        config[:384] = np.array(table_param).astype(np.int16)
        bias = adaptor.attributes["bias"]  # 1
        scale = adaptor.attributes["scale"]  # 2
        left_shift = adaptor.attributes["left_shift"]  # 3
        right_shift = adaptor.attributes["right_shift"]  # 4
        bound = adaptor.attributes["bound"]  # 5
        last_list = []
        for index in range(len(bias.value)):
            shift = (np.int16(np.int8(left_shift.value[index])) << 8) | np.int16(
                np.int8(right_shift.value[index])
            )
            tmp_list = [
                np.int16(bias.value[index]),
                np.int16(scale.value[index]),
                shift,
                np.int16(bound.value[index]),
            ]
            last_list.extend(tmp_list)
        config[448:480] = np.array(last_list).astype(np.int16)
        if data_format == "NCHW":
            x = hbir.transpose(x, [0, 2, 3, 1])
        symmetry_mode = (
            symmetry_mode if type(symmetry_mode) == str else symmetry_mode.decode()
        )
        if symmetry_mode == "NONE":
            sym_mode = "ASYM"
        elif symmetry_mode == "ORIGIN_SYMMETRIC":
            sym_mode = "CSYM"
        elif symmetry_mode == "Y_SYMMETRIC":
            sym_mode = "YSYM"
        else:
            raise ValueError(
                f"Operator HzLut2Layer does not support symmetry_mode {symmetry_mode}"
            )
        rounding_method = (
            rounding_method
            if type(rounding_method) == str
            else rounding_method.decode()
        )
        if rounding_method != "ROUND":
            raise ValueError(
                f"Operator HzLut2Layer does not support rounding_method {rounding_method}"
            )
        x = b25.lut(x, config, symmetricMode=sym_mode, output_type=y)
        if data_format == "NCHW":
            x = hbir.transpose(x, [0, 3, 1, 2], output_type=y)
        return x


HzLut2Layer = HzLut2LayerConvertor()

# class HzDepthToSpaceConvertor(OpsetConvertor):
#     def __init__(self):
#         OpsetConvertor.__init__(self, "HzDepthToSpace", True)

#     def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value, *,
#                      blocksize, data_format, mode):
#         input_dim = len(adaptor.operands[0].type.shape)
#         if input_dim != 4:
#             raise ValueError(f"Operator HzDepthToSpace does not support input_dim not euqal to 4, got {input_dim}")

#         if data_format == "NCHW":
#             x = hbir.transpose(x, [0, 2, 3, 1], output_type=y)
#             n, c, h, w = adaptor.operands[0].type.shape
#         else:
#             n, h, w, c, = adaptor.operands[0].type.shape

#         x = hbir.reshape(x, (n, h, w, blocksize, blocksize, c //
#                              (blocksize * blocksize)))
#         x = hbir.transpose(x, [0, 3, 4, 1, 5, 2])
#         return hbir.reshape(x, (n, h * blocksize, w * blocksize, c //
#                                 (blocksize * blocksize)),
#                             output_type=y)

# HzDepthToSpace = HzDepthToSpaceConvertor()


class HzSpaceToDepthConvertor(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzSpaceToDepth", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        block_height,
        block_width,
    ):
        if block_height != block_width:
            raise ValueError(
                f"Operator HzSpaceToDepth does not support block height and weight not equal: {block_height} vs {block_width}"
            )
        input_dim = len(adaptor.operands[0].type.shape)
        if input_dim != 4:
            raise ValueError(
                f"Operator HzSpaceToDepth does not support input_dim not euqal to 4, got {input_dim}"
            )
        n, c, h, w = adaptor.operands[0].type.shape
        x = hbir.reshape(
            x, (n, c, h // block_height, block_height, w // block_width, block_width)
        )
        x = hbir.transpose(x, [0, 3, 5, 1, 2, 4])
        return hbir.reshape(
            x,
            (n, c * (block_height * block_width), h // block_height, w // block_width),
            output_type=y,
        )


HzSpaceToDepth = HzSpaceToDepthConvertor()


class HzQuantizeConvertor(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzQuantize", False)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Type,
        scale: mlir.Value,
        zero_point=None,
        *,
        bits=8,
        axis=None,
    ):
        scale = adaptor.operands[1].value
        zero_point = [0] * len(scale)
        if len(adaptor.operands) > 2:
            zero_point = tuple(int(v) for v in adaptor.operands[2].value)
        return qnt.quantize(
            data, scale, zero_point, axis=axis, narrowRange=False, output_type=y
        )


HzQuantize = HzQuantizeConvertor()


class HzDequantizeConvertor(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzDequantize", False)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Type,
        scale: mlir.Value,
        zero_point=None,
        axis=None,
    ):
        scale = adaptor.operands[1].value
        zero_point = [0] * len(scale)
        if len(adaptor.operands) > 2:
            zero_point = tuple(int(v) for v in adaptor.operands[2].value)
        return qnt.dequantize(data, scale, zero_point, axis=axis, output_type=y)


HzDequantize = HzDequantizeConvertor()


class HzRsqrt(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzRsqrt", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        lhs: mlir.Value,
        *,
        epsilon=0,
        is_sqrt_add_reciprocal=0,
    ):
        epsilon = np.array([epsilon]).astype(np.float32)
        if is_sqrt_add_reciprocal == 0:
            lhs = hbir.add(lhs, epsilon, output_type=y)
            lhs = hbir.sqrt(lhs, output_type=y)
            lhs = hbir.reciprocal(lhs, output_type=y)
        else:
            lhs = hbir.sqrt(lhs, output_type=y)
            lhs = hbir.add(lhs, epsilon, output_type=y)
            lhs = hbir.reciprocal(lhs, output_type=y)
        return lhs


HzRsqrt()


class HzGelu(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzGelu", True)

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, approximate="none"
    ):
        return hbir.gelu(lhs, approximate=approximate, output_type=y)


HzGelu()


class GridSample(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "GridSample", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        Grid: mlir.Value,
        *,
        align_corners=0,
        mode="bilinear",
        padding_mode="zeros",
    ):
        padding_mode = (
            padding_mode if type(padding_mode) == str else padding_mode.decode()
        )

        x = hbir.transpose(x, [0, 2, 3, 1])
        if padding_mode == "zeros":
            x = hbir.grid_sample(
                x, Grid, mode=mode, alignCorner=bool(align_corners), padValue=0
            )
        elif padding_mode == "border":
            x = hbir.grid_sample(
                x,
                Grid,
                mode=mode,
                alignCorner=bool(align_corners),
                expansionMode=padding_mode,
            )
        else:
            raise ValueError(
                f"Operator GridSample does not support padding_mode {padding_mode}"
            )
        return hbir.transpose(x, [0, 3, 1, 2], output_type=y)


GridSample()


class GridSamplePlugin(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "GridSamplePlugin", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        Grid: mlir.Value,
        *,
        align_corners,
        mode="bilinear",
        padding_mode="zeros",
    ):
        if not align_corners:
            raise ValueError(
                f"Operator GridSamplePlugin does not support align_corners be False"
            )
        padding_mode = (
            padding_mode if type(padding_mode) == str else padding_mode.decode()
        )
        if padding_mode != "zeros":
            raise ValueError(
                f"Operator GridSamplePlugin does not support padding_mode {padding_mode}"
            )

        x = hbir.transpose(x, [0, 2, 3, 1])
        x = hbir.warp(x, Grid, mode=mode, padValue=0)
        return hbir.transpose(x, [0, 3, 1, 2], output_type=y)


GridSamplePlugin()


class PyOp(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "PyOp", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        *args,
        class_name: mlir.Value,
        compute: mlir.Value,
        ext_compute: mlir.Value,
        input_types: mlir.Value,
        module: mlir.Value,
        output_shape: mlir.Value,
        output_types: mlir.Value,
    ):
        ext_compute = ext_compute if type(ext_compute) == str else ext_compute.decode()

        module = module if type(module) == str else module.decode()

        srcPath = os.getcwd() + "/" + "/".join(module.split(".")) + ".py"
        entryFuncName = ext_compute
        return hbir.custom(
            args, srcPath, entryFuncName, outputs_type=y if isinstance(y, list) else [y]
        )


PyOp()


class HzReluX(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzReluX", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        clip_value,
        data_format="NCHW",
    ):
        if data_format == "NCHW":
            x = hbir.transpose(x, [0, 2, 3, 1])
        max_res = hbir.max(x, np.array([0]).astype(np.float32), output_type=y)
        ret = hbir.min(
            max_res, np.array([clip_value]).astype(np.float32), output_type=y
        )
        if data_format == "NCHW":
            ret = hbir.transpose(ret, [0, 3, 1, 2], output_type=y)
        return ret


HzReluX()


class HzSoftmax(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzSoftmax", True)

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value, *, axis=-1
    ):
        return hbir.softmax(x, axis, output_type=y)


HzSoftmax()


class HzCorrelation(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzCorrelation", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data1: mlir.Value,
        data2: mlir.Value,
        *,
        is_multiply=1,
        kernel=1,
        max_displacement=1,
        pad=0,
        strides,
    ):
        if is_multiply != 1:
            raise ValueError(
                f"Operator HzCorrelation does not support attributes is_multiply with value: {is_multiply}"
            )
        data1 = hbir.transpose(data1, [0, 2, 3, 1])
        data2 = hbir.transpose(data2, [0, 2, 3, 1])
        ret = hbir.correlation(
            data1,
            data2,
            kernel=kernel,
            max_d=max_displacement,
            stride1=strides[0],
            stride2=strides[1],
            pad=pad,
        )
        ret = hbir.transpose(ret, [0, 3, 1, 2], output_type=y)
        return ret


HzCorrelation()


class HzPad(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzPad", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        pad_value: mlir.Value,
        *,
        pads,
        mode="constant",
        data_format="NCHW",
    ):
        if data_format == "NHWC":
            x = hbir.transpose(x, [0, 2, 3, 1])
        pad_value = float(adaptor.operands[1].value[0])
        mode = mode if type(mode) == "str" else mode.decode()
        if mode not in ["constant", "edge"]:
            raise ValueError(f"Operator HzPad does not support mode with value: {mode}")
        if mode == "edge":
            mode = "border"

        pad_length = len(pads)
        if data_format == "NHWC":
            ret = hbir.pad(
                x,
                begin=pads[: pad_length // 2],
                end=pads[pad_length // 2 :],
                padValue=pad_value,
                expansionMode=mode,
            )
            ret = hbir.transpose(ret, [0, 3, 1, 2], output_type=y)
        else:
            ret = hbir.pad(
                x,
                begin=pads[: pad_length // 2],
                end=pads[pad_length // 2 :],
                padValue=pad_value,
                expansionMode=mode,
                output_type=y,
            )
        return ret


HzPad()


class HzFilter(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzFilter", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: List[mlir.Type],
        x: mlir.Value,
        *args,
        idx_range,
        threshold,
    ):
        if len(y) > 4:  # maxValue, maxIndex, coord, data
            raise ValueError(
                f"No support for HzFilter {adaptor.name} that has multiple inputs"
            )
        x = nchw_to_nhwc(x)
        ret = hbir.filter(
            x,
            channelBegin=idx_range[0],
            channelEnd=idx_range[1],
            threshold=threshold,
            maxValue_type=mlir.UnrankedTensorType.get(y[0].element_type),
            maxIndex_type=mlir.UnrankedTensorType.get(y[1].element_type),
            filterCoord_type=mlir.UnrankedTensorType.get(y[2].element_type),
            filterData_type=mlir.UnrankedTensorType.get(y[3].element_type),
        )
        return ret


HzFilter()


class HzLayerNorm(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzLayerNorm", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        scale: mlir.Value,
        bias: Optional[mlir.Value] = None,
        *,
        axis=-1,
        epsilon=1e-05,
    ):
        itype = adaptor.operands[0].type
        rank = len(itype.shape)
        assert (-rank <= axis) and (
            axis < rank
        ), 'invalid "aixs" attr of HzLayerNorm op'
        norm_dims = (
            list(range(rank + axis, rank)) if (axis < 0) else list(range(axis, rank))
        )

        # align scale rank to targeted rank
        stype = adaptor.operands[1].type
        expansion = [1] * (itype.rank - stype.rank)
        scale = hbir.reshape(scale, [*expansion, *stype.shape])

        if bias is not None:  # align bias rank to targeted rank
            btype = adaptor.operands[2].type
            expansion = [1] * (itype.rank - btype.rank)
            bias = hbir.reshape(bias, [*expansion, *btype.shape])

        x = hbir.layernorm(
            x, dims=norm_dims, eps=epsilon, weight=scale, bias=bias, output_type=otype
        )
        return x


HzLayerNorm()


# The actual version is aligned with the public version, and ptq requires that it does not start with hz
class BevPoolV2(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "BevPoolV2", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        feat: mlir.Value,
        ranks_depth: mlir.Value,
        ranks_feat: mlir.Value,
        ranks_bev: mlir.Value,
        bev_feat_shape: mlir.Value,
        interval_starts: mlir.Value,
        interval_lengths: mlir.Value,
        *args,
    ):
        bev_feat_shape_list = []
        for i in range(otype.rank):
            bev_feat_shape_list.append(otype.shape[i])

        # only support rank == 4 and 5
        assert otype.rank == 4 or otype.rank == 5, "now only support 4d or 5d"
        if otype.rank == 5:
            # bev_feat_shape.permute(0 ,4, 1, 2, 3) = fout.shape
            total_dims = len(bev_feat_shape_list)
            batch_dims_count = total_dims - 5
            batch_dims = bev_feat_shape_list[:batch_dims_count]
            last_five_dims = bev_feat_shape_list[batch_dims_count:]
            permute_order = [0, 2, 3, 4, 1]
            new_shape = batch_dims + [last_five_dims[i] for i in permute_order]
            bev_feat_shape_list = new_shape
            feat = hbir.transpose(feat, (0, 1, 3, 4, 2))
        elif otype.rank == 4:
            # bev_feat_shape.permute(3, 0, 1, 2) = fout.shape
            original_array = np.array(bev_feat_shape_list)
            bev_feat_shape_list = np.transpose(original_array, (1, 2, 3, 0)).tolist()
            feat = hbir.transpose(feat, (0, 3, 2, 1))

        x = hbir.bev_pool_v2(
            depth=x,
            feat=feat,
            ranks_depth=ranks_depth,
            ranks_feat=ranks_feat,
            ranks_bev=ranks_bev,
            interval_starts=interval_starts,
            interval_lengths=interval_lengths,
            bev_feat_shape=bev_feat_shape_list,
            # output_type=otype, // In order to separate permute from hbtl, the output type cannot be used here, otherwise infertype will not pass.
        )
        return nhwc_to_nchw(x)


BevPoolV2()


class DeformConv(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "DeformConv", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        w: mlir.Value,
        offset: mlir.Value,
        mask: mlir.Value,
        bias: mlir.Value,
        stride_h: mlir.Value,
        stride_w: mlir.Value,
        pad_h: mlir.Value,
        pad_w: mlir.Value,
        dil_h: mlir.Value,
        dil_w: mlir.Value,
        n_weight_grps: mlir.Value,
        n_offset_grps: mlir.Value,
        use_mask: mlir.Value,
        *args,
    ):
        if len(adaptor.operands) != 14:
            raise ValueError(
                f"Invalid number of operands of DeformConv2d op: should be 14, given {len(adaptor.operands)}"
            )
        strides = [1, 1]
        pads = [0, 0, 0, 0]
        dilations = [1, 1]

        strides_height = int(adaptor.operands[5].value.astype(np.int64))
        strides_width = int(adaptor.operands[6].value.astype(np.int64))
        pad_height = int(adaptor.operands[7].value.astype(np.int64))
        pad_width = int(adaptor.operands[8].value.astype(np.int64))
        dialtion_height = int(adaptor.operands[9].value.astype(np.int64))
        dialtion_width = int(adaptor.operands[10].value.astype(np.int64))
        weight_grps = int(adaptor.operands[11].value.astype(np.int64))
        offset_grps = int(adaptor.operands[12].value.astype(np.int64))
        mask_flag = bool(adaptor.operands[13].value.astype(np.bool8))

        strides = [strides_height, strides_width]
        pads = (
            pad_height,
            pad_width,
            pad_height,
            pad_width,
        )
        dilations = (dialtion_height, dialtion_width)

        # input trans: nchw->nhwc
        # output trans: nhwc->nchw
        mask_shape = offset.type.shape
        mask_shape[1] = mask_shape[1] // 2
        if mask_flag is False:
            mask = get_value_or_create_const(np.ones(mask_shape))

        x = hbir.transpose(x, [0, 2, 3, 1])
        w = hbir.transpose(w, [0, 2, 3, 1])
        offset = hbir.transpose(offset, [0, 2, 3, 1])
        mask = hbir.transpose(mask, [0, 2, 3, 1])
        x = hbir.deform_conv2d(
            x,
            w,
            offset,
            mask,
            strides,
            pads,
            dilations,
            weight_grps,
            offset_grps,
            mask_flag,
            bias=bias,
        )
        x = hbir.transpose(x, [0, 3, 1, 2], output_type=y)
        return x


DeformConv()


class HzPointPillarsPreprocess(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzPointPillarsPreprocess", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        points: mlir.Value,
        pc_ranges: mlir.Value,
        voxel_sizes: mlir.Value,
        norm_ranges: mlir.Value,
        norm_dims: mlir.Value,
        *,
        max_points_per_voxel: int,
        max_voxels: int,
        use_max: int,
    ):
        pcRanges = tuple(float(v) for v in adaptor.operands[1].value)
        voxelSizes = tuple(float(v) for v in adaptor.operands[2].value)
        normRanges = tuple(float(v) for v in adaptor.operands[3].value)
        normDims = tuple(int(v) for v in adaptor.operands[4].value)
        voxels, coords = hbir.point_pillar_preprocess(
            points,
            pcRanges,
            voxelSizes,
            max_voxels,
            max_points_per_voxel,
            normDims,
            normRanges=normRanges,
        )
        # ptq op output is NCHW
        voxels = hbir.transpose(voxels, [0, 3, 1, 2])
        return voxels, coords


HzPointPillarsPreprocess()


class HzScatter(OpsetConvertor):
    def __init__(self):
        OpsetConvertor.__init__(self, "HzScatter", True)

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        coords: mlir.Value,
        sizes: List[int],
    ):
        # ptq 'sizes' is nhwc already, so do not transpose the value
        if len(sizes) != 4:
            raise ValueError(f"Only support 4D output shape, given {len(sizes)}")
        res = hbir.point_pillar_scatter(x, coords, sizes)
        return nhwc_to_nchw(res)


HzScatter()
