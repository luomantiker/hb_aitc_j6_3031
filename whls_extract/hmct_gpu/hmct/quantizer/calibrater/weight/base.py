from typing import TYPE_CHECKING, List

import numpy as np

from ..base import CalibrationPass

if TYPE_CHECKING:
    from hmct.ir import OnnxModel, OnnxNode


class Calibrater(CalibrationPass):
    def run_impl(self, calibrated_model: "OnnxModel", **kwargs) -> "OnnxModel":  # noqa: ARG002
        """计算模型权重的校准阈值.

        不依赖校准数据, 不同类型节点的权重阈值计算逻辑如下:
        1. Conv/HzPreProcess, 采用perchannel计算阈值, channel在第0维度;
        2. ConvTranspose, 需对权重值进行reshape和transpose变换, 采用perchannel计算阈值,
            channel在第1维度;
        TODO(zsq): 3. Add/Sub/Mul节点;
        TODO(zsq): 4. Other节点;

        Args:
            calibrated_model: 模型权重待校准的模型(已插入校准节点)
            **kwargs: 其他未被使用到的传入参数

        Returns:
            模型权重完成校准的模型
        """
        calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
        for node in calibration_nodes:
            if node.tensor_type == "weight":
                value = node.inputs[0].value
                param_shape = node.inputs[0].shape
                next_op_types = {next_op.op_type for next_op in node.next_ops}
                if next_op_types.issubset({"Conv", "HzPreprocess"}):
                    group = self.check_and_obtain_group_attribute(node)
                    node.group = group
                    if node.thresholds is None:
                        if self.block_size == 0:
                            node.axis = 0
                            node.thresholds = self.calibrate_channel_thresholds(
                                value, 0
                            )
                        else:
                            node.axis = 1
                            node.thresholds = self.calibrate_block_thresholds(value, 1)
                            node.block_sizes = [self.block_size]
                            node.quantize_type = self.quantize_type
                        node.constant = 1
                elif next_op_types.issubset({"ConvTranspose"}):
                    node.tensor_type = "convt_weight"
                    group = self.check_and_obtain_group_attribute(node)
                    node.group = group
                    if node.thresholds is None:
                        if self.block_size == 0:
                            node.axis = 1
                            value = value.reshape(
                                group,
                                param_shape[0] // group,
                                *param_shape[1:],
                            )
                            value = np.swapaxes(value, axis1=1, axis2=2)
                            value = value.reshape(
                                param_shape[1] * group,
                                param_shape[0] // group,
                                *param_shape[2:],
                            )
                            node.thresholds = self.calibrate_channel_thresholds(
                                value, 0
                            )
                        else:
                            node.axis = 0
                            value = value.reshape(
                                group,
                                param_shape[0] // group,
                                *param_shape[1:],
                            )
                            node.thresholds = self.calibrate_block_thresholds(value, 1)
                            node.block_sizes = [self.block_size]
                            node.quantize_type = self.quantize_type
                        node.constant = 1
        return calibrated_model

    def calibrate_channel_thresholds(self, value: np.ndarray, axis: int) -> np.ndarray:
        """计算per-channel量化粒度下的校准阈值."""
        assert isinstance(
            value, np.ndarray
        ), f"type(value) should be np.ndarray, but got {type(value)}."
        assert (
            value.ndim > axis
        ), f"channel axis should be less then {value.ndim}, but got {axis}."
        return self.max_percentile * abs(value).max(
            tuple(_ for _ in range(value.ndim) if _ != axis),
        ).clip(np.finfo(np.float32).tiny, np.finfo(np.float32).max)

    def calibrate_tensor_thresholds(self, value: np.ndarray) -> np.ndarray:
        """计算per-tensor量化粒度下的校准阈值."""
        assert isinstance(
            value, np.ndarray
        ), f"type(value) should be np.ndarray, but got {type(value)}."
        return self.max_percentile * np.array(
            [
                abs(value)
                .max()
                .clip(np.finfo(np.float32).tiny, np.finfo(np.float32).max),
            ],
        )

    def calibrate_block_thresholds(self, value: np.ndarray, axis: int) -> np.ndarray:
        assert isinstance(value, np.ndarray), (
            "type(value) should be np.ndarray, " + f"but got {type(value)}."
        )
        assert value.ndim > axis, (
            "channel axis should be less then " + f"{value.ndim}, but got {axis}."
        )

        axis = axis % value.ndim
        pre_size = int(np.prod(value.shape[0:axis]))
        axis_size = int(np.prod(value.shape[axis : axis + 1]))
        post_size = int(np.prod(value.shape[axis + 1 :]))
        new_value = value.reshape(pre_size, axis_size, post_size)

        pad_width = (
            (0, 0),
            (
                0,
                (self.block_size - new_value.shape[1] % self.block_size)
                % self.block_size,
            ),
            (0, 0),
        )
        new_value = np.pad(new_value, pad_width)
        new_value = new_value.reshape(pre_size, -1, self.block_size, post_size)
        return (
            self.max_percentile
            * abs(new_value)
            .max(2)
            .clip(np.finfo(np.float32).tiny, np.finfo(np.float32).max)
            .flatten()
        )

    def check_and_obtain_group_attribute(self, onnx_node: "OnnxNode") -> int:
        groups: List[int] = [
            next_op.attributes.get("group", 1) for next_op in onnx_node.next_ops
        ]
        assert (
            len(set(groups)) == 1
        ), "It's invalid that different next_op has different group attribute value."

        return groups[0]
