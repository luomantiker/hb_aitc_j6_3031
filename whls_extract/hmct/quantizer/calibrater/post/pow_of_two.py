from typing import TYPE_CHECKING

from ..base import CalibrationPass

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


class PowOfTwo(CalibrationPass):
    def name(self) -> str:
        return "pow_of_two"

    def run_impl(self, calibrated_model: "OnnxModel", **kwargs) -> "OnnxModel":  # noqa: ARG002
        """Updates the quantize_type of HzCalibration nodes to "shift".

        For bernoulli march, quantize_type must only be "shift".
        """
        calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
        for node in calibration_nodes:
            node.quantize_type = "shift"
        return calibrated_model
