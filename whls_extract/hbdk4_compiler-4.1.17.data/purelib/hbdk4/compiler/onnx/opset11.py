from typing import Optional
from hbdk4.compiler.frontend.convertor import OpConvertor
from hbdk4.compiler.frontend.adaptor import NodeAdaptor
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.ops import hbir
import numpy as np
import logging


class Opset11(OpConvertor):
    def __init__(self, name: str):
        super().__init__(name, "onnx", 11, True)


class Resize(Opset11):
    def __init__(self):
        super().__init__("Resize")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        roi: mlir.Value = None,
        scales: mlir.Value = None,
        sizes: mlir.Value = None,
        *,
        coordinate_transformation_mode="half_pixel",
        cubic_coeff_a="-0.75",
        exclude_outside=0,
        extrapolation_value=0,
        mode="nearest",
        nearest_mode="round_prefer_floor",
    ):
        input_shape = np.array(adaptor.operands[0].type.shape)
        output_shape = np.array(adaptor.results[0].type.shape)
        rank = len(input_shape)
        roi = (
            None
            if (roi is None or (len(adaptor.operands[1].value) == 0))
            else adaptor.operands[1].value
        )
        # 仅支持feature为4维的resize操作
        assert len(input_shape) == 4
        scale_tensor = (
            None
            if (scales is None or (len(adaptor.operands[2].value) == 0))
            else adaptor.operands[2].value
        )
        if scale_tensor is not None:
            resize_shape = input_shape * scale_tensor
        else:
            resize_shape = adaptor.operands[3].value
            scale_tensor = np.array(
                [
                    1,
                    1,
                    resize_shape[2] / input_shape[2],
                    resize_shape[3] / input_shape[3],
                ]
            )

        length_original = input_shape[2:].astype(np.int32)
        length_resized = resize_shape[2:].astype(np.int32)
        coordinate_transformation_mode = (
            coordinate_transformation_mode.decode()
            if isinstance(coordinate_transformation_mode, bytes)
            else coordinate_transformation_mode
        )
        if coordinate_transformation_mode != "tf_crop_and_resize" and np.array_equal(
            length_original, length_resized
        ):
            x = hbir.transpose(x, [0, 2, 3, 1])
            x = hbir.resize2d(
                x, step=[1.0, 1.0], initialOffset=[0.0, 0.0], ratio=(1.0, 1.0)
            )
            x = hbir.transpose(x, [0, 3, 1, 2], output_type=y)
            return x

        # resize 计算公式:
        # hw_in = initial_offset + hw_out * step
        resize_scale = scale_tensor[2:]
        if coordinate_transformation_mode in ["pytorch_half_pixel"]:
            step = 1 / resize_scale
            initial_offset = 0.5 / resize_scale - 0.5
            for index, resized_num in enumerate(length_resized):
                if resized_num <= 1:
                    step[index] = 0.0
                    initial_offset[index] = 0.0

        elif coordinate_transformation_mode in ["half_pixel"]:
            step = 1 / resize_scale
            initial_offset = 0.5 / resize_scale - 0.5
        elif coordinate_transformation_mode in ["align_corners"]:
            step = (
                (length_original[0] - 1) / (length_resized[0] - 1),
                (length_original[1] - 1) / (length_resized[1] - 1),
            )
            initial_offset = [0.0, 0.0]
        elif coordinate_transformation_mode in ["asymmetric"]:
            step = 1 / resize_scale
            initial_offset = [0.0, 0.0]

        elif coordinate_transformation_mode in ["tf_half_pixel_for_nn"]:
            step = 1 / resize_scale
            initial_offset = 0.5 / resize_scale
        elif coordinate_transformation_mode in ["tf_crop_and_resize"]:
            start_x = roi[rank - 2 : rank]
            end_x = roi[2 * rank - 2 : 2 * rank]
            step = [1, 1]
            initial_offset = [0, 0]
            for index in range(len(length_resized)):
                if length_resized[index] > 1:
                    step[index] = (
                        (end_x[index] - start_x[index])
                        * (length_original[index] - 1)
                        / (length_resized[index] - 1)
                    )
                    initial_offset[index] = start_x[index] * (
                        length_original[index] - 1
                    )
                else:
                    step[index] = 0
                    initial_offset[index] = (
                        0.5
                        * (start_x[index] + end_x[index])
                        * (length_original[index] - 1)
                    )
        else:
            raise ValueError(
                f"Operator Resize does not support coordinate_transformation_mode: {coordinate_transformation_mode}"
            )

        if isinstance(mode, bytes):
            mode = mode.decode()
        if isinstance(nearest_mode, bytes):
            nearest_mode = nearest_mode.decode()

        if mode == "linear":
            mode = "bilinear"
        if mode == "cubic":
            mode = "bicubic"
        if mode not in ["nearest", "bilinear", "bicubic"]:
            raise ValueError(f"Operator Resize does not support resize mode: {mode}")
        if mode == "nearest":
            if nearest_mode == "round_prefer_ceil":
                initial_offset += np.array(
                    [np.finfo(np.float32).eps, np.finfo(np.float32).eps]
                )
            if nearest_mode == "round_prefer_floor":
                initial_offset -= np.array(
                    [np.finfo(np.float32).eps, np.finfo(np.float32).eps]
                )
            elif nearest_mode == "floor":
                initial_offset -= np.array([0.5, 0.5])
            elif nearest_mode == "ceil":
                initial_offset += np.array([0.5, 0.5])
            else:
                raise ValueError(
                    f"Operator Resize does not support nearest_mode mode: {nearest_mode}"
                )

        # 1. when there is a ratio parameter, if the value of ratio is negative, you need to correct the value of initialOffset;
        # 2. when there is a size parameter, if the value of step is negative, you need to correct the value of initialOffset;
        rank = len(input_shape)
        numOfResizeAxis = 2
        for i in range(numOfResizeAxis):
            axis = rank - numOfResizeAxis - 1 + i
            if step[i] < 0:
                initial_offset[i] = float(input_shape[axis])

        x = hbir.transpose(x, [0, 2, 3, 1])
        x = hbir.resize2d(
            x, step, initial_offset, mode, size=output_shape[2:], expansionMode="border"
        )
        x = hbir.transpose(x, [0, 3, 1, 2], output_type=y)
        return x


Resize()


class Clip(Opset11):
    def __init__(self):
        super().__init__("Clip")

    def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, *args):
        if len(args) == 1:
            logging.info("Input max and min missing, using default max and min.")
            x = adaptor.operands[0].value
            max = 3.4028234663852886e38
            min = -3.4028234663852886e38
        elif len(args) == 3:
            x = adaptor.operands[0].value
            min = float(adaptor.operands[1].value)
            max = float(adaptor.operands[2].value)
        else:
            raise ValueError(
                f"Operator Clip does not support input number given: {len(args)}"
            )

        return hbir.clip(x, min, max, output_type=y)


Clip()


# input_c is optional in opset 11
class Gemm(Opset11):
    def __init__(self):
        super().__init__("Gemm")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        input_a: mlir.Value,
        input_b: mlir.Value,
        *args,
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=0,
    ):
        assert len(adaptor.operands[0].type.shape) == 2
        assert len(adaptor.operands[1].type.shape) == 2
        input_a = adaptor.operands[0].value
        input_b = adaptor.operands[1].value

        if transA:
            input_a = hbir.transpose(input_a, (1, 0))
        if transB:
            input_b = hbir.transpose(input_b, (1, 0))

        # res = alpha * (a * b) + beta * c
        res = hbir.mul(
            alpha, hbir.matmul(input_a, input_b, output_type=y), output_type=y
        )
        if args:
            res = hbir.add(res, hbir.mul(beta, args[0]), output_type=y)
        return res


Gemm()


class GatherElements(Opset11):
    def __init__(self):
        super().__init__("GatherElements")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        indices: mlir.Value,
        *,
        axis=0,
    ):
        return hbir.gather_elements(data, indices, dim=axis)


GatherElements()


class Gather(Opset11):
    def __init__(self):
        super().__init__("Gather")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        indices: mlir.Value,
        axis=0,
    ):
        return hbir.index(x, index=indices, dim=axis, output_type=y)


Gather()


class GatherND(Opset11):
    def __init__(self):
        super().__init__("GatherND")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        indices: mlir.Value,
        *,
        batchDims=0,
    ):
        return hbir.gather_nd(data, indices, batchDim=batchDims, output_type=y)


GatherND()


class TopK(Opset11):
    def __init__(self):
        super().__init__("TopK")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        k: mlir.Value,
        *,
        axis: mlir.Value,
        largest: mlir.Value = 1,
        sorted: mlir.Value = 1,
    ):
        k = adaptor.operands[1].value.tolist()[0]
        largest = bool(largest)
        sorted = bool(sorted)
        return hbir.topk(
            data,
            k=k,
            dim=axis,
            largest=largest,
            sorted=sorted,
            values_type=y[0],
            indices_type=y[1],
        )


TopK()


class ScatterND(Opset11):
    def __init__(self):
        super().__init__("ScatterND")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        indices: mlir.Value,
        updates: mlir.Value,
    ):
        return hbir.scatter_nd(data, indices, updates, output_type=y)


ScatterND()


class ScatterElements(Opset11):
    def __init__(self):
        super().__init__("ScatterElements")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        indices: mlir.Value,
        updates: mlir.Value,
        axis=0,
    ):
        return hbir.scatter_elements(data, indices, updates, axis, output_type=y)


ScatterElements()


class Neg(Opset11):
    def __init__(self):
        super().__init__("Neg")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
    ):
        return hbir.neg(x, output_type=y)


Neg()


class Pad(Opset11):
    def __init__(self):
        super().__init__("Pad")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        pads: mlir.Value,
        value: mlir.Value = None,
        *,
        mode="constant",
    ):
        pads = adaptor.operands[1].value
        value = 0
        if len(adaptor.operands) == 3:
            value = adaptor.operands[2].value
        pad_length = len(pads)
        if not mlir.IntegerType.isinstance(y.element_type):
            value = float(value)
        else:
            value = int(value)
        return hbir.pad(
            x,
            begin=pads[: pad_length // 2],
            end=pads[pad_length // 2 :],
            padValue=value,
            output_type=y,
        )


Pad()


class DepthToSpaceConvertor(Opset11):
    def __init__(self):
        super().__init__("DepthToSpace")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *,
        blocksize,
        mode=b"DCR",
    ):
        input_dim = len(adaptor.operands[0].type.shape)
        if input_dim != 4:
            raise ValueError(
                f"Operator DepthToSpace does not support input_dim not euqal to 4, got {input_dim}"
            )
        n, c, h, w = adaptor.operands[0].type.shape
        if c % (blocksize**2) != 0:
            raise ValueError(
                "The channel of the input shape must be divisible by the square of blocksize."
            )

        if mode == b"DCR":
            x = hbir.reshape(x, (n, blocksize, blocksize, c // (blocksize**2), h, w))
            x = hbir.transpose(x, [0, 3, 4, 1, 5, 2])
        elif mode == b"CRD":
            x = hbir.reshape(x, (n, c // (blocksize**2), blocksize, blocksize, h, w))
            x = hbir.transpose(x, [0, 1, 4, 2, 5, 3])
        else:
            raise ValueError(
                f"Operator DepthToSpace only support DCR mode and CRD mode, got {mode}"
            )
        return hbir.reshape(
            x,
            (n, c // (blocksize**2), h * blocksize, w * blocksize),
            output_type=y,
        )


DepthToSpace = DepthToSpaceConvertor()


# opSet14 has the same definition with 11
class CumSum(Opset11):
    def __init__(self):
        super().__init__("CumSum")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        axis: mlir.Value,
        *,
        exclusive=0,
        reverse=0,
    ):
        axis = adaptor.operands[1].value.tolist()
        return hbir.cumsum(x, axis, exclusive=exclusive, reverse=reverse)


CumSum()


class Round(Opset11):
    def __init__(self):
        super().__init__("Round")

    def emit_mlir_op(self, adaptor: NodeAdaptor, y: mlir.Type, x: mlir.Value):
        return hbir.round(x, 0, output_type=y)


Round()


class NonMaxSuppression(Opset11):
    def __init__(self):
        super().__init__("NonMaxSuppression")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        boxes: mlir.Value,
        scores: mlir.Value,
        max_output_boxes_per_class: mlir.Value = None,
        iou_threshold: mlir.Value = None,
        score_threshold: mlir.Value = None,
        *,
        center_point_box=0,
    ):

        boxes_shape = np.array(adaptor.operands[0].type.shape)
        scores_shape = np.array(adaptor.operands[1].type.shape)
        assert len(boxes_shape) == 3 and boxes_shape[2] == 4
        assert len(scores_shape) == 3 and boxes_shape[1] == scores_shape[2]

        maxOutputBoxesPerClass = 0
        if len(adaptor.operands) > 2:
            maxOutputBoxesPerClass = int(adaptor.operands[2].value)

        iouThreshold = 0.0
        if len(adaptor.operands) > 3:
            iouThreshold = float(adaptor.operands[3].value)

        scoreThreshold = 0.0
        if len(adaptor.operands) > 4:
            scoreThreshold = float(adaptor.operands[4].value)

        if center_point_box == 0:
            mode = "yxyx"
        elif center_point_box == 1:
            mode = "xywh"
        else:
            raise ValueError(
                f"Operator NonMaxSuppression does not support center_point_box: {center_point_box}"
            )
        if isinstance(mode, bytes):
            mode = mode.decode()

        if maxOutputBoxesPerClass == 0:
            raise ValueError(
                f"Operator NonMaxSuppression does not support max_output_boxes_per_class: {maxOutputBoxesPerClass}"
            )

        return hbir.nms(
            boxes, scores, mode, iouThreshold, scoreThreshold, maxOutputBoxesPerClass
        )


NonMaxSuppression()
