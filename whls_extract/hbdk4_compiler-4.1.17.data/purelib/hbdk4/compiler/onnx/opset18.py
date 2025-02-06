from typing import Callable
from hbdk4.compiler.frontend.adaptor import NodeAdaptor
from hbdk4.compiler.frontend.convertor import OpConvertor
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.ops import hbir
import numpy as np
import math


class Opset18(OpConvertor):
    def __init__(self, name: str):
        super().__init__(name, "onnx", 18, True)


class Reduce(Opset18):
    def __new__(cls, *args, **kwargs):
        if cls is Reduce:
            raise TypeError(f"Only children of '{cls.__name__}' may be instantiated")
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, name: str, mlir_op_func: Callable):
        super().__init__(name)
        self.mlir_op_func = mlir_op_func

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *args,
        keepdims=1,
        noop_with_empty_axes=0,
    ):
        axes = None
        if len(args) != 0:
            axes = adaptor.operands[1].value.tolist()

        if axes is None:
            axes = []
            if noop_with_empty_axes == 0:
                # Reduce all axes
                axes = list(range(mlir.ShapedType(x.type).rank))
            else:
                # act like identity operands, here convert to reshape
                return hbir.reshape(x, adaptor.operands[0].type.shape)

        return self.mlir_op_func(x, dims=axes, keepDim=bool(keepdims), output_type=y)


class ReduceMin(Reduce):
    def __init__(self):
        super().__init__("ReduceMin", hbir.reduce_min)


ReduceMin()


class ReduceMean(Reduce):
    def __init__(self):
        super().__init__("ReduceMean", hbir.reduce_mean)


ReduceMean()


class ReduceMax(Reduce):
    def __init__(self):
        super().__init__("ReduceMax", hbir.reduce_max)


ReduceMax()


class Split(Opset18):
    def __init__(self):
        super().__init__("Split")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        *args,
        axis=0,
        num_outputs=0,
    ):
        split = None
        if len(args) != 0:
            split = adaptor.operands[1].value.tolist()
        # format: ele = ceil(dim / num), the last one is dim - ele * (num -1)
        # for example, num_outputs is 3, dim is 128, get split [43, 43, 42]
        if split is None:
            split = []
            tmp_shape = adaptor.operands[0].type.shape
            ele = int(math.ceil(tmp_shape[axis] / num_outputs))
            if tmp_shape[axis] % num_outputs == 0:
                split = [int(tmp_shape[axis] / num_outputs)] * num_outputs
            else:
                split = [ele] * (num_outputs - 1)
                split.append(tmp_shape[axis] - ele * (num_outputs - 1))

        ret_list = []
        shape = adaptor.operands[0].type.shape
        dim = len(shape)
        asum = 0
        for i in range(len(split)):
            begin = np.array([0 if i != axis else asum for i in range(dim)])
            asum += split[i]
            end = np.array([shape[i] if i != axis else asum for i in range(dim)])
            step = np.ones(dim, dtype=np.int64)
            output_type = y[i] if isinstance(y, list) else y
            ret_list.append(
                hbir.slice(x, begin=begin, end=end, step=step, output_type=output_type)
            )
        return ret_list


Split()


class Reshape(Opset18):
    def __init__(self):
        super().__init__("Reshape")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        x: mlir.Value,
        shape: mlir.Value,
        *,
        allowzero=0,
    ):
        input_shape = adaptor.operands[0].type.shape
        out_shape = adaptor.operands[1].value
        new_shape = np.copy(out_shape)
        if allowzero == 0:
            zeros_index = np.where(out_shape == 0)
            new_shape[zeros_index] = np.array(object=input_shape)[zeros_index]
        return hbir.reshape(x, shape=new_shape)


Reshape()


class Flatten(Opset18):
    def __init__(self):
        super().__init__("Flatten")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        y: mlir.Type,
        data: mlir.Value,
        *,
        axis=1,
    ):
        original_shape = adaptor.operands[0].type.shape
        total_elem_num = np.prod(original_shape).astype(int)
        new_shape = [1, total_elem_num]
        if axis != 0:
            new_shape[0] = np.prod(original_shape[:axis]).astype(int)
            new_shape[1] = int(total_elem_num / new_shape[0])

        return hbir.reshape(data, shape=new_shape, output_type=y)


Flatten()


class Resize(Opset18):
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
        antialias=0,
        axes=None,
        coordinate_transformation_mode="half_pixel",
        cubic_coeff_a="-0.75",
        exclude_outside=0,
        extrapolation_value=0,
        keep_aspect_ratio_policy="stretch",
        mode="nearest",
        nearest_mode="round_prefer_floor",
    ):
        input_shape = np.array(adaptor.operands[0].type.shape)
        length_original = input_shape[2:].astype(np.int32)
        output_shape = np.array(adaptor.results[0].type.shape)
        rank = len(input_shape)

        # 仅支持feature为4维的resize操作
        assert len(input_shape) == 4
        if roi is not None and len(adaptor.operands[1].value) > 0:
            roi = adaptor.operands[1].value
        scale_tensor = (
            None
            if (scales is None or (len(adaptor.operands[2].value) == 0))
            else adaptor.operands[2].value
        )
        output_size = (
            None
            if (sizes is None or (len(adaptor.operands[3].value) == 0))
            else adaptor.operands[3].value
        )
        coordinate_transformation_mode = (
            coordinate_transformation_mode.decode()
            if isinstance(coordinate_transformation_mode, bytes)
            else coordinate_transformation_mode
        )
        if isinstance(mode, bytes):
            mode = mode.decode()
        if isinstance(nearest_mode, bytes):
            nearest_mode = nearest_mode.decode()
        if isinstance(keep_aspect_ratio_policy, bytes):
            keep_aspect_ratio_policy = keep_aspect_ratio_policy.decode()

        if antialias != 0:
            raise ValueError("Operator Resize does not support antialias mode.")
        if output_size is None and scale_tensor is None:
            raise ValueError(
                "Operator Resize: one of scales and sizes must be specified."
            )
        if output_size is not None and scale_tensor is not None:
            raise ValueError(
                "Operator Resize: only one of scales and sizes can be specified."
            )

        if axes is not None:
            if len(axes) > 2:
                raise ValueError("Operator Resize supports specifying up to 2 axes.")
            axes_num = len(axes)
            if output_size is not None:
                new_output_size = np.copy(input_shape)
                new_output_size[axes] = output_size
                output_size = new_output_size

            if scale_tensor is not None:
                new_scales = np.ones(rank, dtype=np.float32)
                new_scales[axes] = scale_tensor
                scale_tensor = new_scales

            if roi is not None:
                new_roi_begin = np.zeros(rank, dtype=np.float32)
                new_roi_end = np.ones(rank, dtype=np.float32)
                new_roi_begin[axes] = roi[:axes_num]
                new_roi_end[axes] = roi[axes_num:]
                roi = np.concatenate((new_roi_begin, new_roi_end))

        if scale_tensor is not None:
            resize_shape = input_shape * scale_tensor
            length_resized = resize_shape[2:].astype(np.int32)
        else:
            resize_shape = output_size
            length_resized = resize_shape[2:].astype(np.int32)
            scale_tensor = np.array(
                [
                    1,
                    1,
                    resize_shape[2] / input_shape[2],
                    resize_shape[3] / input_shape[3],
                ]
            )

            if keep_aspect_ratio_policy != "stretch":
                if keep_aspect_ratio_policy == "not_larger":
                    scale = np.min(resize_shape[2:] / input_shape[2:])
                elif keep_aspect_ratio_policy == "not_smaller":
                    scale = np.max(resize_shape[2:] / input_shape[2:])
                else:
                    raise ValueError(
                        f"Invalid keep_aspect_ratio_policy={keep_aspect_ratio_policy!r}"
                    )

                scale_tensor[2] = scale_tensor[3] = scale

                resize_shape = input_shape * scale_tensor
                length_resized = [int(elem + 0.5) for elem in resize_shape[2:]]

        if coordinate_transformation_mode != "tf_crop_and_resize" and np.array_equal(
            length_original, length_resized
        ):
            x = hbir.transpose(x, [0, 2, 3, 1])
            x = hbir.resize2d(
                x, step=[1.0, 1.0], initialOffset=[0.0, 0.0], ratio=(1.0, 1.0)
            )
            x = hbir.transpose(x, [0, 3, 1, 2], output_type=y)
            return x

        # resize calculation formula:
        # hw_in = initial_offset + hw_out * step
        resize_scale = scale_tensor[2:]
        if coordinate_transformation_mode in ["pytorch_half_pixel"]:
            step = 1 / resize_scale
            initial_offset = 0.5 / resize_scale - 0.5
            for index, resized_num in enumerate(length_resized):
                if resized_num <= 1:
                    step[index] = 0.0
                    initial_offset[index] = 0.0
        elif coordinate_transformation_mode in ["half_pixel_symmetric"]:
            adjustment = length_resized / resize_shape[2:]
            center = length_original / 2
            offset = center * (1 - adjustment)
            step = 1 / resize_scale
            initial_offset = 0.5 / resize_scale - 0.5 + offset
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

        # Currently only nearest and bilinear are supported.
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


class BitwiseAndOp(Opset18):
    def __init__(self):
        super().__init__("BitwiseAnd")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, rhs: mlir.Value
    ):
        return hbir.bitwise_and(lhs, rhs, output_type=y)


BitwiseAndOp()


class BitwiseOrOp(Opset18):
    def __init__(self):
        super().__init__("BitwiseOr")

    def emit_mlir_op(
        self, adaptor: NodeAdaptor, y: mlir.Type, lhs: mlir.Value, rhs: mlir.Value
    ):
        return hbir.bitwise_or(lhs, rhs, output_type=y)


BitwiseOrOp()
