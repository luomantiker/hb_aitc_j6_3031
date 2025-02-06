from typing import TYPE_CHECKING

import numpy as np

from hmct.common import find_input_calibration

from ..base import CalibrationPass

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


class AdjustEvenStrideDWS(CalibrationPass):
    def __init__(self, march: str):
        self.march = march

    @property
    def name(self) -> str:
        return "adjust_even_stride_dws"

    def run_impl(self, calibrated_model: "OnnxModel", **kwargs) -> "OnnxModel":  # noqa: ARG002
        """将偶数stride的Depthwise卷积转为普通卷积来满足硬件约束."""
        conv_nodes = calibrated_model.graph.type2nodes["Conv"]
        for node in conv_nodes:
            input_calib = find_input_calibration(node, 0)
            weight_calib = find_input_calibration(node, 1)
            if not input_calib or not weight_calib:
                continue
            if input_calib.qtype != "int16":
                continue
            weight_value = weight_calib.inputs[0].value
            weight_shape = weight_calib.inputs[0].shape
            out_channel = weight_shape[0]
            strides = node.attributes.get("strides")
            group = node.attributes.get("group", 1)
            group_num = out_channel // group
            if (
                out_channel % group == 0
                and weight_shape[1] == 1
                and strides
                and (
                    (
                        self.march in ["nash"]
                        and any(s % 2 == 0 and s != 2 for s in strides)
                    )
                    or (
                        self.march in ["bayes", "bayes-e"]
                        and any(s % 2 == 0 for s in strides)
                    )
                )
            ):
                plain_weight = np.zeros(weight_shape, np.float32).repeat(group, 1)
                for c in range(out_channel):
                    plain_weight[c][c // group_num] = weight_value[c][0]
                weight_calib.group = 1
                weight_calib.inputs[0].value = plain_weight
                node.set_attribute(attr_name="group", attr_val=1)
        calibrated_model.infer_shapes()
        return calibrated_model
