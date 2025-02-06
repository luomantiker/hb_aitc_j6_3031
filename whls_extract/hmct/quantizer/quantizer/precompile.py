from typing import TYPE_CHECKING, Dict

from hmct.common import (
    convert_reshape_target_shape_to_positive,
    infer_shapes,
    modify_model_by_cpp_func,
)
from hmct.ir.horizon_onnx import quantizer

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


def precompile(
    calibrated_model: "OnnxModel",
    original_model: "OnnxModel",
    batched_input_shapes: Dict,
) -> "OnnxModel":
    """Convert calibrated model for hbdk4.

    Returns:
        ptq onnx model after precompile.
    """
    ptq_model = infer_shapes(
        calibrated_model,
        original_model,
        input_shape=batched_input_shapes,
    )

    ptq_model = convert_reshape_target_shape_to_positive(ptq_model)
    ptq_model = modify_model_by_cpp_func(ptq_model, quantizer.convert_for_hbdk4)

    return ptq_model  # noqa: RET504
