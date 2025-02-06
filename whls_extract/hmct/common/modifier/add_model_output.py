import logging
from typing import Optional, Sequence, Set

from hmct.ir import OnnxModel, OnnxVariable, save_model


def add_model_output(
    onnx_model: OnnxModel,
    output_nodes: Optional[Sequence[str]] = None,
    output_tensors: Optional[Sequence[str]] = None,
    output_op_types: Optional[Sequence[str]] = None,
    keep_original_output: bool = True,
) -> OnnxModel:
    """根据给定output_nodes/output_tensors/output_op_types为模型添加输出.

    Args:
        onnx_model: 待添加输出的onnx模型
        output_nodes: 目标节点, 将每个节点的所有输出都添加为模型输出
        output_tensors: 目标tensor, 将每个tensor添加为模型输出
        output_op_types: 目前节点类型, 将每个类型的所有节点的输出添加为模型输出
        keep_original_output: 是否保留onnx_model的原始输出

    Returns:
        添加输出后的onnx模型
    """
    output_nodes = [] if output_nodes is None else output_nodes
    output_tensors = [] if output_tensors is None else output_tensors
    output_op_types = [] if output_op_types is None else output_op_types

    if not keep_original_output:
        onnx_model.graph.clear_outputs()

    added_output_vars: Set[OnnxVariable] = set()
    # add model output based on output_nodes
    node_mappings = onnx_model.graph.node_mappings
    for node_name in output_nodes:
        added_output_vars.update(node_mappings[node_name].outputs)
    # add model output based on output_tensors
    variable_mappings = onnx_model.graph.variable_mappings
    added_output_vars.update(
        variable_mappings[output_name] for output_name in output_tensors
    )
    # add model output based on output_op_types
    for op_type in output_op_types:
        for onnx_node in onnx_model.graph.type2nodes[op_type]:
            added_output_vars.update(onnx_node.outputs)

    # add collected output variables
    onnx_model.graph.extend_outputs(
        added_output_vars.difference(onnx_model.graph.outputs)
    )

    try:
        onnx_model.infer_shapes().check_validity()
    except Exception as e:
        save_model(onnx_model, "add_model_output_fail.onnx")
        logging.error(
            "onnx model validation failed, invalid "
            "model saved as add_model_output_fail.onnx",
        )
        raise e

    return onnx_model
