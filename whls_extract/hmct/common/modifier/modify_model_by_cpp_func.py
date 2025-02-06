from typing import Callable

from hmct.ir import OnnxModel, serialize_model


def modify_model_by_cpp_func(
    onnx_model: OnnxModel,
    func: Callable,
    *args,
    **kwargs,
) -> OnnxModel:
    """Modify onnx model by cpp function.

    Args:
        onnx_model: The onnx model to be modified.
        func: The cpp function to modify the onnx model.
        *args: The positional args of the cpp function.
        **kwargs: The keyword args of the cpp function.

    Returns:
        The modified onnx model.
    """
    onnx_model.reset_proto(func(serialize_model(onnx_model), *args, **kwargs))

    return onnx_model
