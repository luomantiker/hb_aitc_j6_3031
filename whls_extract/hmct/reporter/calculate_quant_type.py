from typing import TYPE_CHECKING, Any, Container, Dict, Optional

from hmct.common import find_input_calibration, find_output_calibration

if TYPE_CHECKING:
    from hmct.ir import OnnxModel, OnnxNode


def get_node_quant_info(node: "OnnxNode"):
    """获取指定节点的量化信息."""
    input_names = []
    input_thresholds = []
    input_qtypes = []
    output_names = []
    output_thresholds = []
    output_qtypes = []

    # 获取节点输入的量化相关信息
    for idx, input_var in enumerate(node.inputs):
        input_names.append(input_var.name)

        cal_node = find_input_calibration(node, idx)
        if cal_node is not None:
            input_thresholds.append(
                [round(thres_, 6) for thres_ in cal_node.thresholds]
            )
            input_qtypes.append(cal_node.qtype)
        else:
            input_thresholds.append([])
            input_qtypes.append("--")

    # 获取节点输出的量化信息
    for idx, output in enumerate(node.outputs):
        output_names.append(output.name)
        cal_node = find_output_calibration(node, idx)
        if cal_node is not None:
            output_thresholds.append(
                [round(thres_, 6) for thres_ in cal_node.thresholds]
            )
            output_qtypes.append(cal_node.qtype)
        else:
            output_thresholds.append([])
            output_qtypes.append("--")

    return (
        input_names,
        input_thresholds,
        input_qtypes,
        output_names,
        output_thresholds,
        output_qtypes,
    )


def calculate_quant_type(
    onnx_model: "OnnxModel",
    node_names: Optional[Container] = None,
) -> Dict[str, Dict[str, Any]]:
    """获取节点量化信息, 用于后续的模型可视化相关操作."""
    node_quant_info = {}
    for node in onnx_model.graph.nodes:
        node_name = node.name
        # 获取指定节点的量化信息
        if node_names is not None and node_name not in node_names:
            continue

        if node.op_type in ["HzQuantize", "HzDequantize", "HzCalibration"]:
            continue

        (
            input_names,
            input_thresholds,
            input_qtypes,
            output_names,
            output_thresholds,
            output_qtypes,
        ) = get_node_quant_info(node)

        node_quant_info[node_name] = {
            "type": node.op_type,
            "input_name": input_names,
            "input_threshold": input_thresholds,
            "input_qtype": input_qtypes,
            "output_name": output_names,
            "output_threshold": output_thresholds,
            "output_qtype": output_qtypes,
        }

    return node_quant_info
