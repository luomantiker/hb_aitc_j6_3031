from typing import TYPE_CHECKING, Any, Dict

from .base import Calibrater
from ..calibration_method import CalibrationMethod

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


class KLCalibrater(Calibrater):
    def __init__(self, activation_config: Dict[str, Any], **kwargs):
        """Initialization function for Kullback-Leibler Calibrater.

        Args:
            activation_config: activation calibration config for kl calibrater:
                1. num_bin: Number of bins to use in the calibration method.
                2. max_num_bin: Maximum number of bins that can be used.
                3. per_channel: Whether to enable per-channel activation quantization.
                4. asymmetric: Whether to enable asymmetric activation quantization.
            **kwargs: Other unused parameters.
        """
        super().__init__(**kwargs)
        self.calibration_methods = CalibrationMethod()
        num_bins = activation_config.get("num_bin", 1024)
        if isinstance(num_bins, int):
            num_bins = [num_bins]
        max_num_bin = activation_config.get("max_num_bin", 16384)
        per_channels = activation_config.get("per_channel", False)
        if isinstance(per_channels, bool):
            per_channels = [per_channels]
        asymmetries = activation_config.get("asymmetric", False)
        if isinstance(asymmetries, bool):
            asymmetries = [asymmetries]
        for num_bin in sorted(num_bins):
            for asymmetric in sorted(asymmetries):
                for per_channel in sorted(per_channels):
                    self.calibration_methods.set(
                        "kl",
                        num_bin=num_bin,
                        max_num_bin=max_num_bin,
                        per_channel=per_channel,
                        asymmetric=asymmetric,
                    )

    @property
    def name(self) -> str:
        return "kl_calibrater"

    def run_impl(self, calibrated_model: "OnnxModel", **kwargs) -> "OnnxModel":  # noqa: ARG002
        return calibrated_model
