from collections import defaultdict
import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np

from hmct.common import Loss, add_model_output, set_model_switch
from hmct.executor import ORTExecutor

if TYPE_CHECKING:
    from hmct.ir import OnnxModel

SIMILARITY_BLACK_OP_LIST = ["HzQuantize", "HzDequantize", "HzCalibration"]


def calculate_similarity(
    calibrated_model: "OnnxModel",
    calibration_data_dict: Dict[str, np.ndarray],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Check consistency between ptq model and optimized float model.

    Args:
        calibrated_model: calibrated ptq model.
        calibration_data_dict: inference data used to compare.

    Returns:
        A dictionary containing information about each node.
    """

    def _find_uncomparable_nodes(model):
        return [
            node.name
            for node in model.graph.nodes
            for output_shape in node.output_shapes
            if (
                not output_shape
                or len(output_shape) == 0
                or any(isinstance(s, str) for s in output_shape)
                or sum(output_shape) == 0
            )
        ]

    def _search_calibration_node_output(node_item):
        next_ops = node_item.next_ops
        for next_op in next_ops:
            # 如果指定节点后面能找到HzCalibration
            if next_op.op_type == "HzCalibration":
                # 则返回该HzCalibration的经过伪量化后的输出
                return next_op.outputs[0].name
        # 否则就直接使用这个节点的输出
        return node_item.outputs[0].name

    # 这里不再设置return_outputs参数.
    # 如果需要返回相应的结果, 则需要在调用时传入一个空的dict
    # 该函数会对传入的dict修改并返回.
    uncomparable_nodes = _find_uncomparable_nodes(calibrated_model)

    node_similarity_info = {}
    node_output_tensor_dict = {}
    # 1. 初始化node_output_tensor_dict和node_similarity_info
    for node_item in calibrated_model.graph.nodes:
        if (
            node_item.op_type not in SIMILARITY_BLACK_OP_LIST
            and node_item.name not in uncomparable_nodes
        ):
            # if node_item.op_type not in black_list:
            out = _search_calibration_node_output(node_item)
            if out:
                node_output_tensor_dict[node_item.name] = out
            node_similarity_info[node_item.name] = {
                "Cosine Similarity": "--",
                "L1 Distance": "--",
                "L2 Distance": "--",
                "Chebyshev Distance": "--",
            }

    # 3. 分别推理校准模型和浮点模型推理结果, 获取每一层的量化误差
    # 3.1 推理calibrated_model, 获取伪量化的推理结果
    modified_calibrated_model = copy.deepcopy(calibrated_model)
    modified_calibrated_model = add_model_output(
        modified_calibrated_model,
        output_tensors=list(node_output_tensor_dict.values()),
    )
    calibration_executor = ORTExecutor(modified_calibrated_model).create_session()
    try:
        calibration_output = calibration_executor.forward(calibration_data_dict)
    except Exception:
        try:
            calibration_executor.to("cpu")
            calibration_output = calibration_executor.forward(calibration_data_dict)
        except Exception:
            logging.warning("Failed to get output of calibrated_model!")
            calibration_output = defaultdict(list)
    del calibration_executor

    # 3.2 删除calibrated模型中的fake quanti, 使其输出float模型推理结果
    float_model = set_model_switch(copy.deepcopy(calibrated_model), "OFF")
    modified_float_model = add_model_output(
        float_model,
        output_tensors=list(node_output_tensor_dict.values()),
    )
    float_executor = ORTExecutor(modified_float_model).create_session()
    try:
        float_output = float_executor.forward(calibration_data_dict)
    except Exception:
        try:
            float_executor.to("cpu")
            float_output = float_executor.forward(calibration_data_dict)
        except Exception:
            logging.warning("Failed to get output of float_model!")
            float_output = defaultdict(list)
    del float_executor
    # 3.3 计算浮点模型推理结果和伪量化模型推理结果的余弦相似度等信息
    loss_functions = {
        "Cosine Similarity": Loss.create("cosine-similarity"),
        "L1 Distance": Loss.create("mre"),
        "L2 Distance": Loss.create("mse"),
        "Chebyshev Distance": Loss.create("chebyshev"),
    }
    for node_name, tensor_name in node_output_tensor_dict.items():
        lvec = np.array(calibration_output[tensor_name]).astype(np.float32).flatten()
        rvec = np.array(float_output[tensor_name]).astype(np.float32).flatten()
        if lvec.size == rvec.size and lvec.size > 0:
            for loss_name, loss_func in loss_functions.items():
                node_similarity_info[node_name].update(
                    {loss_name: f"{loss_func.run(lvec, rvec):.6f}"}
                )

    # 输出模型输出层的量化误差信息
    output_similarity_info = defaultdict(dict)
    for output in calibrated_model.graph.outputs:
        lvec = np.array(calibration_output[output.name]).astype(np.float32).flatten()
        rvec = np.array(float_output[output.name]).astype(np.float32).flatten()
        if lvec.size != rvec.size or lvec.size == 0:
            output_similarity_info[output.name] = {
                "Cosine Similarity": "--",
                "L1 Distance": "--",
                "L2 Distance": "--",
                "Chebyshev Distance": "--",
            }
        else:
            for loss_name, loss_func in loss_functions.items():
                output_similarity_info[output.name].update(
                    {loss_name: f"{loss_func.run(lvec, rvec):.6f}"}
                )

    return node_similarity_info, output_similarity_info
