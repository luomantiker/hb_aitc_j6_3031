from copy import deepcopy
from typing import TYPE_CHECKING

from hmct.common import Dataset, modify_model_by_cpp_func
from hmct.ir.horizon_onnx import quantizer

from .base import (
    Calibrater,
    add_calibration_methods,
    get_calibration_thresholds,
)
from ..calibration_method import CalibrationMethod

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


class ActivationPerchannel(Calibrater):
    @property
    def name(self) -> str:
        return "activation_perchannel"

    def run_impl(
        self,
        calibrated_model: "OnnxModel",
        calibration_dataset: Dataset,
        **kwargs,  # noqa: ARG002
    ) -> "OnnxModel":
        """Set channel scales for calibration in the provided calibrated model.

        Args:
            calibrated_model: The input calibrated model.
            calibration_dataset: The calibration dataset to be used for calibration.
            **kwargs: Other unused parameters.

        Returns:
            The calibrated model with channel scale attributes.
        """
        calibration_method = CalibrationMethod().set(
            "max", per_channel=True, per_channel_mask=True
        )
        # calibration_method写入模型cali_attr属性,并为模型增加输出用于获取阈值
        calibration_model = add_calibration_methods(
            deepcopy(calibrated_model), calibration_method, calibration_dataset
        )
        # 推理calibration_model获取阈值结果
        calibration_thresholds = get_calibration_thresholds(
            calibration_model, calibration_method, calibration_dataset
        )
        return modify_model_by_cpp_func(
            calibrated_model,
            quantizer.set_calibration_channel_scale,
            calibration_thresholds[str(calibration_method)],
        )
