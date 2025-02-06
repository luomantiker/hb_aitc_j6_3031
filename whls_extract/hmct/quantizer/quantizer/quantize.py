from typing import TYPE_CHECKING, Any, Dict

from hmct.common import (
    InputDictParser,
    constant_folding,
    modify_model_by_cpp_func,
)
from hmct.ir.horizon_onnx import quantizer

from .add_transpose_node_at_input import add_transpose_node_at_input
from .remove_transpose_node_at_input import remove_transpose_node_at_input

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


def quantize(
    calibrated_model: "OnnxModel",
    input_dict_parser: InputDictParser,
) -> "OnnxModel":
    """Quantize onnx model for model convert.

    Returns:
        quantized onnx model after fixed-point quantization.
    """
    quantized_model = modify_model_by_cpp_func(calibrated_model, quantizer.quantize)
    # TODO(df.li): in C++ quantize pass phase, there is a constant-folding
    # func call after all quantize pass done.
    # This C++ constant-folding pass will fold:
    #   1. op with 4dim-const input will has a NCHW2NHWC layout-convert
    #      Transpose
    #   2. gridsample with const-input gird will has 2 SQAdd,1 concat,
    #      1 SQSub input nodes.
    quantized_model = constant_folding(quantized_model)
    quantized_model = change_quantized_model_input_layout(
        quantized_model,
        input_dict_parser,
    )
    return quantized_model  # noqa: RET504


def change_quantized_model_input_layout(
    onnx_model: "OnnxModel",
    input_dict_parser: Dict[str, Dict[str, Any]],
) -> "OnnxModel":
    input_names = onnx_model.graph.input_names
    for input_name in input_names:
        if input_name in input_dict_parser.get_input_layouts():
            expected_layout = input_dict_parser.get_input_layouts()[input_name].get(
                "expected_input_layout",
            )
            original_layout = input_dict_parser.get_input_layouts()[input_name].get(
                "original_input_layout",
            )
            # TODO 这里的逻辑可以做简化, 后续需要梳理这里的layout情况
            if expected_layout != "NOTSET" and original_layout != expected_layout:
                dest_nodes = onnx_model.graph.input_mappings[input_name].dest_ops
                for dest_node in dest_nodes:
                    if dest_node.op_type == "Transpose":
                        onnx_model = remove_transpose_node_at_input(
                            onnx_model,
                            input_name,
                        )
                    else:
                        if original_layout == "NHWC" and expected_layout == "NCHW":
                            perm = [0, 2, 3, 1]
                        else:
                            perm = [0, 3, 1, 2]
                        onnx_model = add_transpose_node_at_input(
                            onnx_model,
                            input_name,
                            perm,
                        )
    return onnx_model
