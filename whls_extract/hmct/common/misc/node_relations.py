from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Set, Tuple

from .find_calibration_node import find_input_calibration, find_output_calibration

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


def node_inputs_calibration_relations(
    model: "OnnxModel",
) -> Tuple[Dict[str, Dict[int, str]], Dict[str, Set[str]]]:
    """获取普通节点和校准节点的映射关系.

    Returns:
        返回tuple包括普通节点的输入校准节点, 以及校准节点影响的哪些普通节点输入.
    """
    node_to_input_calibration = defaultdict(dict)
    calibration_to_output_node = defaultdict(set)
    for node in model.graph.nodes:
        if node.op_type == "HzCalibration":
            continue
        for i in range(len(node.inputs)):
            calibration_node = find_input_calibration(node, i)
            if calibration_node is None:
                continue
            node_to_input_calibration[node.name][i] = calibration_node.name
            calibration_to_output_node[calibration_node.name].add(node.name)

    return node_to_input_calibration, calibration_to_output_node


def node_outputs_calibration_relations(
    model: "OnnxModel",
) -> Tuple[Dict[str, Dict[int, str]], Dict[str, Set[str]]]:
    """获取普通节点和校准节点的映射关系.

    Returns:
        返回tuple包括普通节点的输出校准节点, 以及校准节点影响哪些普通节点输出.
    """
    node_to_output_calibration = defaultdict(dict)
    calibration_to_input_node = defaultdict(set)
    for node in model.graph.nodes:
        if node.op_type == "HzCalibration":
            continue
        for i in range(len(node.outputs)):
            calibration_node = find_output_calibration(node, i)
            if calibration_node is None:
                continue
            node_to_output_calibration[node.name][i] = calibration_node.name
            calibration_to_input_node[calibration_node.name].add(node.name)

    return node_to_output_calibration, calibration_to_input_node
