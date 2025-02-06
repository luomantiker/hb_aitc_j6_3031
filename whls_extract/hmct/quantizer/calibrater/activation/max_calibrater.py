from typing import TYPE_CHECKING, Any, Dict

from .base import Calibrater
from ..calibration_method import CalibrationMethod

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


class MaxCalibrater(Calibrater):
    def __init__(self, activation_config: Dict[str, Any], **kwargs):
        """Initialization function for Max Calibrater.

        Args:
            activation_config: activation calibration config for max calibrater:
                1. max_percentile: Maximum percentile value to be used for
                calibration. If max_percentile < 1.0, the calibration method
                is set to "max-percentile", otherwise, it is set to "max".
                2. per_channel: Whether to enable per-channel activation quantization.
                3. asymmetric: Whether to enable asymmetric activation quantization.
            **kwargs: Other unused parameters.
        """
        super().__init__(**kwargs)
        self.calibration_methods = CalibrationMethod()
        max_percentiles = activation_config.get("max_percentile", 1.0)
        if isinstance(max_percentiles, float):
            max_percentiles = [max_percentiles]
        per_channels = activation_config.get("per_channel", False)
        if isinstance(per_channels, bool):
            per_channels = [per_channels]
        asymmetries = activation_config.get("asymmetric", False)
        if isinstance(asymmetries, bool):
            asymmetries = [asymmetries]
        for percentile in sorted(max_percentiles):
            method = "max-percentile" if percentile < 1.0 else "max"
            for asymmetric in sorted(asymmetries):
                for per_channel in sorted(per_channels):
                    self.calibration_methods.set(
                        method=method,
                        percentile=percentile,
                        per_channel=per_channel,
                        asymmetric=asymmetric,
                    )

    @property
    def name(self) -> str:
        return "max_calibrater"

    def run_impl(self, calibrated_model: "OnnxModel", **kwargs) -> "OnnxModel":  # noqa: ARG002
        return calibrated_model
