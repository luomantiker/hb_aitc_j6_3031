from collections import defaultdict
import logging
from typing import TYPE_CHECKING, Dict, Optional, Union

from hmct.custom.op_registration import check_custom_op_impl

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


def check_for_convert(
    onnx_model: "OnnxModel",
    input_batch_dict: Optional[Dict[str, int]] = None,
) -> None:
    """Performs a validity check for onnx model.

    Here we include various checks for model domain, opset and IR version,
    shape inference, unsupported ops, and input batch validity.

    Args:
        onnx_model: ONNX model to check the validity.
        input_batch_dict: Dictionary specifying the batch size for each input.
    """
    check_model_input_output(onnx_model)
    check_model_domain(onnx_model)
    check_unsupported_op(onnx_model)
    check_custom_op_impl(onnx_model)
    if input_batch_dict is not None:
        check_input_batch_validity(onnx_model, input_batch_dict)


def check_model_input_output(onnx_model: "OnnxModel") -> None:
    """Check for the validity of model input and output info.

    Raises:
        ValueError: If number of model input or output is 0.
        ValueError: If any model input shape is invalid.
    """
    # obtain all the model's non-initializer inputs
    inputs = onnx_model.graph.inputs
    if len(inputs) == 0:
        raise ValueError("can't find input in model.")

    for input_var in inputs:
        input_name = input_var.name
        input_shape = input_var.shape
        if any(not isinstance(item, int) or item == -1 for item in input_shape):
            raise ValueError(
                f"For input {input_name}, the input shape {input_shape} "
                f"is invalid as there exist missing or dynamic dimension, "
                f"which can not be converted."
            )

    if len(onnx_model.graph.outputs) == 0:
        raise ValueError("can't find output in model.")


def check_model_domain(onnx_model: "OnnxModel") -> None:
    """Checks the domain of nodes in the ONNX model.

    Raises:
        ValueError: If there exist any node with unsupported domains.
    """
    domain_warning_map = defaultdict(list)
    supported_domain = [
        "",
        "ai.onnx",
        "ai.onnx.ml",
        "horizon.custom",
        "horizon",
        "ai.onnx.contrib",
    ]
    for node in onnx_model.graph.nodes:
        # 不检查PyOp的domain问题, 因为不同的自定义算子实现需要放进不同的domain中,
        # 以此来达到实现多种自定义算子的功能. 因此这里不应该对PyOp的domain进行检查.
        # None means missing domain field, equivalent to "" && "ai.onnx" domain
        if (
            node.op_type != "PyOp"
            and node.domain is not None
            and node.domain not in supported_domain
        ):
            domain_warning_map[node.domain].append(node.name)
    for k, v in domain_warning_map.items():
        warning_str = (
            f"nodes: {v} are specified as domain: {k}, which are "
            f"not supported by official onnx. Please check whether "
            f"these ops are official onnx ops or defined by yourself."
        )
        logging.error(warning_str)
    if len(domain_warning_map) != 0:
        logging.error(
            f"Unsupported domain detected: {domain_warning_map}, "
            f"supported domain: {supported_domain}"
        )
        raise ValueError("unsupported domain detected in model")


def check_unsupported_op(onnx_model: "OnnxModel") -> None:
    """Check for unsupported operations in the ONNX model.

    Raises:
        ValueError: If the model contains unsupported operations
        (e.g. op with subgraph or plugin custom op).
    """
    op_with_subgraph_set = {"Loop", "Scan", "SequenceMap"}
    op_with_plugin_custom_set = {
        "MultiScaleRoIAlign",
        "DetectionPostProcessV1",
        "point_pillars_preprocess",
        "point_pillars_scatter",
        "rcnn_post_process",
        "roi_align_list",
        "roi_align_tensor",
        "base_grid_generator",
        "rle",
        "AnchorGenerator",
        "Correlation",
    }
    op_with_subgraph_name_list = []
    op_with_plugin_custom_name_list = []

    # get name of all unsupported op
    for node in onnx_model.graph.nodes:
        if node.op_type in op_with_subgraph_set:
            op_with_subgraph_name_list.append(node.name)
        if node.op_type in op_with_plugin_custom_set:
            op_with_plugin_custom_name_list.append(node.name)
    if op_with_subgraph_name_list:
        raise ValueError(
            f"ERROR: The model contains ops with subgraph: "
            f"{op_with_subgraph_name_list}, which are not supported "
            f"by horizon_nn. Please remove these unsupported op(s)."
        )
    if op_with_plugin_custom_name_list:
        raise ValueError(
            f"ERROR: The model contains plugin custom ops: "
            f"{op_with_plugin_custom_name_list}, which are not supported "
            f"by horizon_nn. Please use horizon_plugin_pytorch to deploy your model."
        )


def check_input_batch_validity(
    onnx_model: "OnnxModel",
    input_batch_dict: Union[Dict[str, int], None],
) -> None:
    """Checks the validity of input batch parameters for the model.

    Args:
        onnx_model: ONNX model to check the validity.
        input_batch_dict: Input batch dictionary.

    Raises:
        ValueError: If the input_batch parameter is set for multi-batch model.
    """
    if input_batch_dict is None:
        input_batch_dict = {}
    for input_var in onnx_model.graph.inputs:
        input_name = input_var.name
        input_shape = input_var.shape
        # if input batch param is not given, set it to 1
        if input_batch_dict.get(input_name, 1) > 1 and input_shape[0] > 1:
            raise ValueError(
                "param input_batch can not be set when "
                "input model is multi-batch model"
            )
