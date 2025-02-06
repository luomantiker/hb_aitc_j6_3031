import logging
from typing import TYPE_CHECKING

from .base import Calibrater, set_calibration_method
from ..calibration_method import CalibrationMethod

if TYPE_CHECKING:
    from hmct.common import Dataset
    from hmct.ir import OnnxModel


class MinMaxCalibrater(Calibrater):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calibration_method = CalibrationMethod().set("min-max")
        self.qtype.set_method(str(self.calibration_method))

    @property
    def name(self) -> str:
        return "min_max_calibrater"

    def run_impl(
        self,
        calibrated_model: "OnnxModel",
        calibration_dataset: "Dataset",
        **kwargs,  # noqa: ARG002
    ) -> "OnnxModel":
        """Run the calibration process using the min-max method.

        Args:
            calibrated_model: The calibrated model to be used for calibration.
            calibration_dataset: The calibration dataset to be used for calibration.
            **kwargs: Other unused parameters.

        Returns:
            The calibrated model with thresholds set.
        """
        logging.info(f"Run calibration model with {self.calibration_method} method.")
        calibration_model = self._calibrate(
            calibrated_model,
            self.calibration_method,
            calibration_dataset,
        )
        return set_calibration_method(calibration_model, self.calibration_method)
