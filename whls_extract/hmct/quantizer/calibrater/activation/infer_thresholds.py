from copy import deepcopy
import logging
from typing import TYPE_CHECKING, Any, Dict

from hmct.common import Dataset, set_model_switch

from .activation_asymmetric import ActivationAsymmetric
from .activation_perchannel import ActivationPerchannel
from .base import Calibrater
from ..search_methods.base import compare_calibration_methods

if TYPE_CHECKING:
    from hmct.ir import OnnxModel

    from ..calibration_method import CalibrationMethod


def set_asymmetric_thresholds(
    calibrated_model: "OnnxModel", calibration_method: "CalibrationMethod"
) -> "OnnxModel":
    """依据对称阈值设置非对称阈值.

    Args:
        calibrated_model: Calibrated Model with non-asymmetric thresholds.
        calibration_method: Asymmetric calibration method.
    """
    calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
    for node in calibration_nodes:
        calibration_thresholds = node.calibration_thresholds
        for method in calibration_method:
            if calibration_thresholds.get(str(method)) is not None:
                continue
            method.asymmetric = False
            thresholds = calibration_thresholds.get(str(method))
            method.asymmetric = True
            if thresholds is not None:
                calibration_thresholds[str(method)] = thresholds
    return calibrated_model


class InferThresholds(Calibrater):
    def __init__(
        self,
        calibration_methods: "CalibrationMethod",
        modelwise_search: Dict[str, Any],
        **kwargs,
    ):
        """Initialization function for Infer Thresholds class.

        Args:
            calibration_methods: calibration methods used for inferring thresholds.
            modelwise_search: Prune the calibration methods for inferring thresholds.
                1. similarity: Set similarity less than 1.0, there is no need
                to infer perchannel calibrated model when the similarity of
                non-perchannel model no less than the similarity set by user.
            **kwargs: Other unused parameters.
        """
        super().__init__(**kwargs)
        self.calibration_methods = calibration_methods
        self.similarity = modelwise_search.get("similarity", 1.0)

    @property
    def name(self) -> str:
        return "infer_thresholds"

    def run_impl(
        self,
        calibrated_model: "OnnxModel",
        calibration_dataset: Dataset,
        **kwargs,  # noqa: ARG002
    ) -> "OnnxModel":
        """Run the calibration process to infer model thresholds.

        Args:
            calibrated_model: The calibrated model to be used for calibration.
            calibration_dataset: The calibration dataset to be used for
                calibration.
            **kwargs: Other unused parameters.

        Returns:
            The calibrated model with calibration thresholds set.
        """
        if len(self.calibration_methods) > 1:
            method = "modelwise search"
        else:
            method = str(self.calibration_methods)
        logging.info(f"Run calibration model with {method} method.")
        calibration_methods = self.calibration_methods
        if len(self.calibration_methods.subset(asymmetric=True)):
            calibrated_model = ActivationAsymmetric()(calibrated_model)
            if len(self.calibration_methods.subset(asymmetric=False)):
                # asymmetries为[True, False],asymmetric:True和asymmetric:False校准方法
                # 会得到相同的阈值,但推理阈值耗时会翻倍增加,因此只需推理其中一组即可.
                calibration_methods = self.calibration_methods.subset(asymmetric=False)

        # infer non-perchannel thresholds.
        if len(calibration_methods.subset(per_channel=False)):
            calibration_method = calibration_methods.subset(per_channel=False)
            self._calibrate(
                calibrated_model=calibrated_model,
                calibration_method=calibration_method,
                calibration_dataset=calibration_dataset,
            )
            if self.similarity < 1.0:
                sorted_methods = compare_calibration_methods(
                    input_data=calibration_dataset.get_data(),
                    float_model=set_model_switch(deepcopy(calibrated_model), "OFF"),
                    calibrated_model=calibrated_model,
                    calibration_methods=calibration_method,
                )
                if sorted_methods[0][1] >= self.similarity:
                    return calibrated_model

        # infer perchannel thresholds.
        if len(calibration_methods.subset(per_channel=True)):
            calibrated_model = ActivationPerchannel()(
                calibrated_model,
                calibration_dataset=calibration_dataset,
            )
            calibration_method = calibration_methods.subset(per_channel=True)
            self._calibrate(
                calibrated_model=calibrated_model,
                calibration_method=calibration_method,
                calibration_dataset=calibration_dataset,
            )

        # get asymmetric thresholds.
        if len(self.calibration_methods.subset(asymmetric=True)):
            calibrated_model = set_asymmetric_thresholds(
                calibrated_model,
                self.calibration_methods.subset(asymmetric=True),
            )
        return calibrated_model
