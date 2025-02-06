from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..base import CalibrationPass

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


def scaling_weight(w_value: np.ndarray, f_scales: Sequence[float]) -> np.ndarray:
    """Apply channel scaling on weight value base on perchannel feature scales.

    To specific: weight value was recorded as [cout, cin/groups, kh, kw],
    we firstly normalize feature scales by maximum scale value, then
    rescaling weight value on cin axis.
    """
    w_shape, group = w_value.shape, len(f_scales) // w_value.shape[1]
    w_value = w_value.reshape(group, w_shape[0] // group, *w_shape[1:])
    w_value = np.swapaxes(w_value, axis1=1, axis2=2)
    w_value = w_value.reshape(w_shape[1] * group, w_shape[0] // group, *w_shape[2:])
    channel_scales = np.expand_dims(
        np.array(f_scales, dtype=np.float32) / max(f_scales),
        axis=tuple(range(1, w_value.ndim)),
    )
    w_value = w_value * channel_scales
    w_value = w_value.reshape(group, w_shape[1], w_shape[0] // group, *w_shape[2:])
    w_value = np.swapaxes(w_value, axis1=1, axis2=2)

    return w_value.reshape(w_shape)


class ActivationEqualization(CalibrationPass):
    @property
    def name(self) -> str:
        return "activation_equalization"

    def run_impl(self, calibrated_model: "OnnxModel", **kwargs) -> "OnnxModel":  # noqa: ARG002
        """Perform activation equalization transform on per-channel calibration.

        Args:
            calibrated_model: Onnx model to perform equalization.
            **kwargs: Collected unused parameters.

        Returns:
            Onnx model with per-channel calibration scaling.
        """
        calibration_nodes = calibrated_model.graph.type2nodes["HzCalibration"]
        for node in calibration_nodes:
            if (
                node.tensor_type == "feature"
                and len(node.scales) > 1
                and (node.block_sizes is None or node.block_sizes[0] == 0)
            ):
                for next_op in node.next_ops:
                    if next_op.op_type != "Conv":
                        continue
                    # scaling conv weight value by conv feature scales.
                    next_op.inputs[1].src_op.inputs[0].value = scaling_weight(
                        next_op.inputs[1].src_op.inputs[0].value,
                        node.scales,
                    )
                    # create calibration node before conv node.
                    calibration_node = calibrated_model.graph.create_node(
                        op_type="HzCalibration",
                        domain="horizon",
                        attributes=node.attributes,
                        inputs=[node.inputs[0]],
                        num_outputs=1,
                    ).insert_before(node)
                    calibration_node.set_attribute(attr_name="axis", attr_val=1)
                    calibration_node.set_attribute(
                        attr_name="dequantize_scale", attr_val=max(node.scales)
                    )
                    # replace next op input with calibration node output.
                    next_op.replace_input(0, calibration_node.outputs[0])
                # destroy useless calibration node.
                if not node.is_used:
                    node.destroy()
        calibrated_model.infer_shapes()
        return calibrated_model
