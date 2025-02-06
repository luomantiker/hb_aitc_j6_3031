import logging
from typing import TYPE_CHECKING

import numpy as np

from ..base import CalibrationPass

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


class RefineThreshold(CalibrationPass):
    @property
    def name(self) -> str:
        return "refine_threshold"

    def run_impl(self, calibrated_model: "OnnxModel", **kwargs) -> "OnnxModel":  # noqa: ARG002
        """Refine illegal thresholds after calibration.

        This function iterates over the calibration nodes in the model and
        identifies any infinite, zero or nan thresholds. If infinite thresholds
        are found, they are set to the maximum value of float32. If zero or
        negative thresholds are found, they are set to a default value of 1.0.
        If nan thresholds are found, raise a ValueError.
        """
        calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
        for node in calibration_nodes:
            is_inf = np.isinf(node.thresholds)
            if np.sum(is_inf):
                logging.warning(
                    f"find inf threshold in {node.name}, it will "
                    "be set to max of fp32.",
                )
                node.thresholds = np.where(
                    is_inf,
                    np.finfo(np.float32).max,
                    node.thresholds,
                )
            is_zeros = np.array(node.thresholds) <= 0.0
            if np.sum(is_zeros):
                logging.warning(
                    f"find illegal threshold in {node.name}, it "
                    "will be set default.",
                )
                node.thresholds = np.where(is_zeros, 1.0, node.thresholds)
            is_nan = np.isnan(node.thresholds)
            if np.sum(is_nan):
                raise ValueError(
                    f"find NaN threshold in {node.name}, Please "
                    "check if your model has been fully trained and if the "
                    "calibration data is reasonable.",
                )
        return calibrated_model
