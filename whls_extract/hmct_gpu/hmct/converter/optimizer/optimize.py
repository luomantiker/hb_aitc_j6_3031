from typing import TYPE_CHECKING

from hmct.common import constant_folding, modify_model_by_cpp_func
from hmct.ir.horizon_onnx import quantizer

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


def optimize(original_model: "OnnxModel") -> "OnnxModel":
    optimized_model = constant_folding(original_model)
    optimized_model = modify_model_by_cpp_func(
        onnx_model=optimized_model, func=quantizer.optimize
    )

    return optimized_model  # noqa: RET504
