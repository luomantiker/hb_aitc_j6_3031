import logging
from typing import TYPE_CHECKING, Any, Dict

from .base import (
    Calibrater,
    CalibrationMethod,
    set_calibration_method,
)

if TYPE_CHECKING:
    from hmct.common import Dataset
    from hmct.ir import OnnxModel


class MaxPerblockCalibrater(Calibrater):
    def __init__(self, activation_config: Dict[str, Any], **kwargs):
        """Initialization function for the MaxPerblock Calibrater class.

        Args:
            activation_config: activation calibration config for max-block calibrater:
                1. block_size: The size of the block that share scale.
                2. quantize_type: The type of scale, float or shift.
            **kwargs: Other unused parameters.
        """
        super().__init__(**kwargs)
        self.block_size = activation_config.get("block_size", 32)
        self.quantize_type = activation_config.get("quantize_type", "scale")

    @property
    def name(self) -> str:
        return "max_block_calibrater"

    def run_impl(
        self,
        calibrated_model: "OnnxModel",
        calibration_dataset: "Dataset",
        **kwargs,  # noqa: ARG002
    ) -> "OnnxModel":
        """Run the calibration process using the max-block method.

        Args:
            calibrated_model: The calibrated model to be used for calibration.
            calibration_dataset: The calibration dataset to be used for
                calibration.
            **kwargs: Other unused parameters.

        Returns:
            OnnxModel: The calibrated model with thresholds set.
        """
        # 根据算子类型指定划分block的维度
        self.axis = {}
        for node_kind in ["Conv", "ConvTranspose", "HzPreprocess"]:
            for node in calibrated_model.graph.type2nodes[node_kind]:
                self.axis[node.inputs[0].src_op.name] = 1
                node.inputs[0].src_op.group = node.attributes.get("group", 1)
        for node in calibrated_model.graph.type2nodes["MatMul"]:
            self.axis[node.inputs[0].src_op.name] = -1
            self.axis[node.inputs[1].src_op.name] = -2

        self.calibration_methods = CalibrationMethod().set(
            "max", block_size=self.block_size, axis=self.axis
        )
        self.qtype.set_method(str(self.calibration_methods))
        logging.info(f"Run calibration model with {self.calibration_methods} method.")

        # 静态校准统计阈值
        calibrated_model = self._calibrate(
            calibrated_model, self.calibration_methods, calibration_dataset
        )
        calibrated_model = set_calibration_method(
            calibrated_model, self.calibration_methods
        )

        # 写入必要的属性
        calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
        for node in calibration_nodes:
            if node.constant == 0 and node.switch == "ON":
                node.quantize_type = self.quantize_type
                node.block_sizes = [self.block_size]
                node.axis = self.axis.get(node.name, 1)

        return calibrated_model
