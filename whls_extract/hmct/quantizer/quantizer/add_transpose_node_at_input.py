import logging
from typing import List

import numpy as np

from hmct.ir import OnnxModel, save_model


def add_transpose_node_at_input(
    onnx_model: OnnxModel,
    input_name: str,
    perm: List[int],
) -> OnnxModel:
    """在模型的给定输入前插入Transpose节点.

    Args:
        onnx_model: 待插入Transpose节点的onnx模型
        input_name: 要在其后插入Transpose节点的输入name
        perm: 插入Transpose节点的perm属性

    Returns:
        插入Transpose节点后的onnx模型
    """
    target_input = onnx_model.graph.input_mappings[input_name]

    dest_ops = target_input.dest_ops
    transpose_node = onnx_model.graph.create_node(
        op_type="Transpose",
        name=input_name + "_Transposed",
        attributes={"perm": perm},
        inputs=[target_input],
        num_outputs=1,
    ).prepend_on()
    # replace all the uses exist in node inputs
    for dest_op in dest_ops:
        dest_op.replace_input(target_input, transpose_node.outputs[0])
    # replace all the uses exist in graph outputs
    for idx, output in enumerate(onnx_model.graph.outputs):
        if target_input is output:
            onnx_model.graph.outputs[idx] = transpose_node.outputs[0]

    # 通过给定的perm调整输入shape
    inv_perm = np.argsort(perm)
    new_shape = [target_input.shape[i] for i in inv_perm]
    target_input.shape = new_shape

    try:
        onnx_model.infer_shapes().check_validity()
    except Exception as e:
        save_model(onnx_model, "add_transpose_fail.onnx")
        logging.error(
            "onnx model validation failed, invalid model "
            "saved as add_transpose_fail.onnx.",
        )
        raise e

    return onnx_model
