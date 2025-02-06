import logging

from hmct.ir import OnnxModel, save_model


def remove_transpose_node_at_input(onnx_model: OnnxModel, input_name: str) -> OnnxModel:
    """删除模型给定输入后的Transpose节点.

    Args:
        onnx_model: 待删除Transpose节点的onnx模型
        input_name: 要删除后接Transpose节点的模型输入name

    Returns:
        删除Transpose节点后的onnx模型
    """
    if input_name in onnx_model.graph.input_mappings:
        input_var = onnx_model.graph.input_mappings[input_name]
    else:
        raise ValueError(
            f"Unexpected input name: {input_name} not found "
            "in model inputs when remove transpose node.",
        )

    user_nodes = input_var.dest_ops
    perm_list = []
    for user_node in user_nodes:
        if user_node.op_type != "Transpose":
            raise ValueError(
                f"The all user nodes for input {input_name} should "
                f"be the Transpose.",
            )
        perm_list.append(user_node.attributes["perm"])

    if not all(perm == perm_list[0] for perm in perm_list):
        raise ValueError(
            f"The all user nodes for input {input_name} should "
            f"have the same Transpose perm attribute.",
        )

    # 更新模型输入shape
    input_var.shape = [input_var.shape[i] for i in perm_list[0]]

    for transpose_node in user_nodes:
        # 获取Transpose节点输出
        transpose_output = transpose_node.outputs[0]
        # 使用新的输入替换原始的输入
        transpose_output.replace_all_uses_with(input_var)
        # 删除Transpose节点
        transpose_node.destroy()

    try:
        onnx_model.infer_shapes().check_validity()
    except Exception as e:
        save_model(onnx_model, "remove_transpose_fail.onnx")
        logging.error(
            "onnx model validation failed, invalid model "
            "saved as remove_transpose_fail.onnx",
        )
        raise e

    return onnx_model
