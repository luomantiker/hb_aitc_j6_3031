from typing import TYPE_CHECKING

import numpy as np

from hmct.common import modify_model_by_cpp_func
from hmct.ir.horizon_onnx import quantizer

from ..base import CalibrationPass
from ..utils import convert_to_ptq_model

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


class PostCalibration(CalibrationPass):
    @property
    def name(self) -> str:
        return "post_calibration"

    def run_impl(self, calibrated_model: "OnnxModel", **kwargs) -> "OnnxModel":  # noqa: ARG002
        """Create post calibrated model from calibrated model.

        The post calibration includes various modifications such as
        complement calibration node, adjust threshold.
        """
        post_calibrated_model = modify_model_by_cpp_func(
            calibrated_model,
            quantizer.create_post_calibrated_model,
        )

        # 将qtype为float16的算子的阈值修改为per-tensor粒度的,
        # 以减少模型体积
        calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
        for node in calibration_nodes:
            if node.qtype == "float16" and node.thresholds is not None:
                node.thresholds = [np.max(np.array(node.thresholds))]

        post_calibrated_model = convert_to_ptq_model(post_calibrated_model)
        post_calibrated_model.check_validity()

        return post_calibrated_model
