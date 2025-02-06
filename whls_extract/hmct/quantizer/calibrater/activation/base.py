from collections import defaultdict
from copy import deepcopy
import logging
from typing import TYPE_CHECKING, Dict, Union

from hmct.common import (
    Dataset,
    modify_flexible_batch,
)
from hmct.executor import ORTExecutor
from hmct.ir import DataType, OnnxModel

from ..base import CalibrationPass
from ..calibration_method import CalibrationMethod
from ..quantization_type import QuantizationType

if TYPE_CHECKING:
    import numpy as np


class Calibrater(CalibrationPass):
    """Calibrater is designed to calibrate model with calibration method and dataset."""

    def __init__(self, **kwargs):  # noqa: ARG002
        self.qtype = QuantizationType()

    def _calibrate(
        self,
        calibrated_model: OnnxModel,
        calibration_method: CalibrationMethod,
        calibration_dataset: Dataset,
        **kwargs,  # noqa: ARG002
    ) -> OnnxModel:
        # 校准方法写入模型cali_attr属性,并为模型增加输出用于获取阈值
        calibration_model = add_calibration_methods(
            deepcopy(calibrated_model),
            calibration_method,
            calibration_dataset,
        )
        # 推理calibration_model获取阈值结果
        calibration_thresholds = get_calibration_thresholds(
            calibration_model,
            calibration_method,
            calibration_dataset,
        )
        # 阈值结果写入到CalibrationNode.calibration_thresholds属性
        calibrated_model = set_calibration_thresholds(
            calibrated_model,
            calibration_thresholds,
        )
        return calibrated_model  # noqa: RET504


def add_calibration_methods(
    calibrated_model: OnnxModel,
    calibration_method: CalibrationMethod,
    calibration_dataset: Dataset,
) -> OnnxModel:
    """Add calibration method for HzCalibration nodes.

    Only add calibration method for HzCalibration nodes with
    zero constant and switch attribute is "ON", append output
    for node and model output.

    Args:
        calibrated_model: The input calibrated model.
        calibration_method: The calibration method to be added.
        calibration_dataset: The calibration dataset for calibration.

    Returns:
        The updated model with the calibration methods added.
    """
    calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
    for node in calibration_nodes:
        if node.constant == 0 and node.switch == "ON":
            for method in calibration_method:
                # update percentile with number of samples
                percentile = (
                    1.0
                    - (1.0 - method.percentile) * calibration_dataset.number_of_samples
                )
                percentile = max(percentile, 0.5)
                node.add_calibration_method(
                    method=method.type,
                    max_num_bin=method.max_num_bin,
                    num_bin=method.num_bin,
                    percentile=percentile,
                    per_channel=method.per_channel,
                    axis=method.axis.get(node.name, 1),
                    block_size=method.block_size,
                )
                node.append_output(
                    name=node.name + "_" + node.input_names[0] + "_" + str(method),
                    dtype=DataType.FLOAT32,
                )
                calibrated_model.graph.append_output(node.outputs[-1])
    return calibrated_model


def get_calibration_thresholds(
    calibrated_model: OnnxModel,
    calibration_method: CalibrationMethod,
    calibration_dataset: Dataset,
) -> Dict[str, Dict[str, "np.ndarray"]]:
    """Infer calibrated model with calibration dataset to get thresholds value.

    Args:
        calibrated_model: Calibrated model with calibration methods added.
        calibration_method: The calibration method.
        calibration_dataset: The calibration dataset for calibration.

    Returns:
        Dictionary with key is calibration method,
        value is ordered dictionary (key is calibration node name,
        value is calibration thresholds).

        Example as follows:
            {
                'max': {
                    'xxx_HzCalibration': np.ndarray, ...
                },
                'kl:num_bins=1024': {
                    'xxx_HzCalibration': np.ndarray, ...
                }
            }
    """

    def get_batch_size(model: OnnxModel) -> int:
        # check all the non-initializer model inputs
        input_support = all(input_var.shape[0] == 1 for input_var in model.graph.inputs)
        # check if pyop exists
        pyop_exist = any(node.op_type == "PyOp" for node in model.graph.nodes)
        # all inputs dim0 is 1, and pyop not exist, set batch 8
        return 8 if input_support and not pyop_exist else 1

    # 推理batch模型得到输出batch_outputs
    batch_size, batch_model = (
        get_batch_size(calibrated_model),
        modify_flexible_batch(calibrated_model),
    )
    logging.info(f"Calibration using batch {batch_size}")
    executor = ORTExecutor(batch_model).create_session()
    batch_outputs = executor.forward_with_batch(
        calibration_dataset,
        batch_size=batch_size,
        progressbar="calibration in progress",
    )
    # 汇总batch_outputs到calibration_thresholds中
    calibration_thresholds = defaultdict(dict)
    calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
    calibration_method = list(map(str, calibration_method))
    for node in calibration_nodes:
        for output, method in zip(node.outputs[1:], calibration_method):
            calibration_thresholds[method][node.name] = batch_outputs[output.name][-1]
    return calibration_thresholds


def set_calibration_thresholds(
    calibrated_model: OnnxModel,
    calibration_thresholds: Dict[str, Dict[str, "np.ndarray"]],
) -> OnnxModel:
    """Sets the calibration_thresholds for the given calibrated_model.

    Args:
        calibrated_model: Calibrated model with calibration methods added.
        calibration_thresholds: Dictionary with key is calibration method,
            value is ordered dictionary (key is calibration node name,
            value is calibration thresholds).

    Returns:
        Onnx model with CalibrationNode.calibration_thresholds set.
    """
    calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
    for method, node_thresholds in calibration_thresholds.items():
        for node in calibration_nodes:
            if node.name in node_thresholds:
                node.calibration_thresholds[method] = node_thresholds[node.name]
    return calibrated_model


def set_calibration_sensitivities(
    calibrated_model: OnnxModel,
    calibration_method: Union[CalibrationMethod, str],
    calibration_sensitivities: Dict[str, Dict[str, Union[float, str]]],
) -> OnnxModel:
    """Sets the sensitivities for the given calibrated_model.

    Args:
        calibrated_model: Calibration with sensitivities to be set.
        calibration_method: The calibration method of calibration
            sensitivities.
        calibration_sensitivities: A dictionary with key is calibration
            node name, value is another dictionary with loss metric key
            and sensitivity value.

    Returns:
        Onnx model with CalibrationNode.sensitivities set.
    """
    calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
    for node in calibration_nodes:
        if node.name in calibration_sensitivities:
            node.sensitivities[str(calibration_method)][node.qtype] = (
                calibration_sensitivities[node.name]
            )
    return calibrated_model


def set_calibration_method(
    calibrated_model: OnnxModel,
    calibration_method: Union[str, CalibrationMethod],
) -> OnnxModel:
    """Sets calibration method to the given calibrated_model.

    Args:
        calibrated_model: Calibrated model with calibration thresholds added.
        calibration_method: The calibration method.

    Returns:
        Calibrated model with quantization parameters(thresholds,qtype) setted
            according to the specific calibration method.
    """
    calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
    if isinstance(calibration_method, CalibrationMethod):
        calibration_method = str(calibration_method)
    for node in calibration_nodes:
        node.set_calibration_method(calibration_method)
    return calibrated_model
