from copy import deepcopy
import logging
from typing import TYPE_CHECKING, Any, Dict

from hmct.common import Dataset, set_model_switch

from .base import (
    compare_calibration_methods,
    set_calibration_method,
    shared_calibration_methods,
)
from ..base import CalibrationPass
from ..quantization_type import QuantizationType

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


class ModelWiseSearch(CalibrationPass):
    def __init__(
        self,
        modelwise_search: Dict[str, Any],
        **kwargs,
    ):
        """Initialization function for ModelWise Search class.

        Args:
            modelwise_search: Modelwise search config.
                1. metric: Metric for selecting calibration methods.
            **kwargs: Other unused parameters.
        """
        super().__init__(**kwargs)
        self.metric = modelwise_search.get("metric", "cosine-similarity")
        self.qtype = QuantizationType()

    @property
    def name(self) -> str:
        return "modelwise_search"

    def run_impl(
        self,
        calibrated_model: "OnnxModel",
        calibration_dataset: Dataset,
        **kwargs,  # noqa: ARG002
    ) -> "OnnxModel":
        """Run the calibration process using the modelwise method.

        Args:
            calibrated_model: The calibrated model to be used for calibration.
            calibration_dataset: The calibration dataset to be used for
                calibration.
            **kwargs: Other unused parameters.

        Returns:
            The calibrated model with thresholds set.
        """
        # fetch shared calibration methods.
        calibration_methods = shared_calibration_methods(calibrated_model)
        if len(calibration_methods) == 0:
            return calibrated_model
        if len(calibration_methods) == 1:
            self.qtype.set_method(str(calibration_methods))
            if self.qtype.perchannel:
                logging.info("Perchannel quantization is enabled.")
            if self.qtype.asymmetric:
                logging.info("Asymmetric quantization is enabled.")
            return set_calibration_method(calibrated_model, calibration_methods[0])

        # compare multiple calibration methods, select the one with best similarity.
        sorted_methods = compare_calibration_methods(
            input_data=calibration_dataset.get_data(),
            float_model=set_model_switch(deepcopy(calibrated_model), "OFF"),
            calibrated_model=calibrated_model,
            calibration_methods=calibration_methods,
            metric=self.metric,
        )
        logging.info(f"Select {sorted_methods[0][0]} method.")
        self.qtype.default = True
        self.qtype.set_method(sorted_methods[0][0])
        if self.qtype.perchannel:
            logging.info("Perchannel quantization is enabled.")
        if self.qtype.asymmetric:
            logging.info("Asymmetric quantization is enabled.")
        return set_calibration_method(calibrated_model, sorted_methods[0][0])
