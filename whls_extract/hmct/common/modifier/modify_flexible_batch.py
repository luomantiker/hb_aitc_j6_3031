from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


def modify_flexible_batch(calibrated_model: "OnnxModel") -> "OnnxModel":
    """Modify onnx model with dynamic batch size.

    Args:
        calibrated_model: The calibrated model with fixed batch size.

    Returns:
        Modified onnx model with dynamic batch size.
    """
    input_vars = calibrated_model.graph.inputs
    for input_var in input_vars:
        input_var.shape[0] = "?"
    calibrated_model.infer_shapes()
    return calibrated_model
