from typing import TYPE_CHECKING

from hmct.common import find_input_calibration, find_output_calibration

from ..base import CalibrationPass

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


class AdjustConvQuantParams(CalibrationPass):
    @property
    def name(self) -> str:
        return "adjust_conv_quant_params"

    def run_impl(self, calibrated_model: "OnnxModel", **kwargs) -> "OnnxModel":  # noqa: ARG002
        conv_nodes = calibrated_model.graph.type2nodes["Conv"]
        for node in conv_nodes:
            input_calib = find_input_calibration(node, 0)
            weight_calib = find_input_calibration(node, 1)
            output_calib = find_output_calibration(node)
            if not input_calib or not weight_calib or not output_calib:
                continue
            if input_calib.qtype != "int16":
                continue
            weight_shape = weight_calib.inputs[0].shape
            if len(weight_shape) != 4:
                continue
            if weight_shape[1] * weight_shape[2] * weight_shape[3] <= 256:
                continue
            input_thres = max(input_calib.thresholds)
            weight_thres = list(weight_calib.thresholds)
            output_thres = list(output_calib.thresholds)
            if len(output_thres) == 1:
                output_thres[0] = max(
                    max(weight_thres) * input_thres,
                    output_thres[0],
                )
            else:
                output_thres = [
                    max(
                        weight_thres[c] * input_thres,
                        output_thres[c],
                    )
                    for c in range(len(weight_thres))
                ]
            if output_thres != list(output_calib.thresholds):
                calibration_node = calibrated_model.graph.create_node(
                    op_type="HzCalibration",
                    domain="horizon",
                    attributes=output_calib.attributes,
                    inputs=[output_calib.inputs[0]],
                    num_outputs=1,
                ).insert_before(output_calib)
                output_calib.replace_input(0, calibration_node.outputs[0])
                calibration_node.qtype = "int16"
                calibration_node.thresholds = output_thres
        calibrated_model.infer_shapes()
        return calibrated_model
