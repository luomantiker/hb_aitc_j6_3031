from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from hmct.ir import CalibrationNode, OnnxNode


def _can_be_skipped(node: "OnnxNode") -> bool:
    """判断向前或者向后查找校准节点的时候, 遇到的节点是否可以跳过."""
    return node.op_type in [
        "MaxPool",
        "GlobalMaxPool",
        "Relu",
        "Clip",
        "Reshape",
        "Transpose",
        "ReduceMax",
        "Split",
        "Slice",
        "Gather",
        "ScatterND",
    ]


def find_input_calibration(
    node: "OnnxNode",
    index: Optional[int] = None,
) -> Optional["CalibrationNode"]:
    """找到一个普通节点输入的校准节点.

    Args:
        node: 普通非校准节点, 用于寻找其输入的校准节点.
        index: 指定需要寻找校准节点的输入索引.
            如果是None, 依次遍历节点各个输入并返回首个找到的校准节点.

    Returns:
        若找到返回相应校准节点, 否则返回None.
    """
    # ScatterND节点的输入0是data, 若未明确指定index, 默认查找data输入上的校准节点.
    if index is None and node.op_type in ["ScatterND"]:
        index = 0

    # 获取当前节点的前序节点, 用以遍历查找校准节点.
    prev_ops: Iterable["OnnxNode"]
    if index is None:
        prev_ops = node.prev_ops
    else:
        src_op = node.inputs[index].src_op
        prev_ops = [src_op] if src_op is not None else []

    # 依次遍历前序节点, 查找校准节点.
    for prev_op in prev_ops:
        if prev_op.op_type == "HzCalibration":
            return prev_op
    # 若未找到校准节点, 且前序节点可以跳过, 则递归查找校准节点.
    for prev_op in prev_ops:
        if _can_be_skipped(prev_op):
            return find_input_calibration(prev_op)

    # 若未找到校准节点, 则返回None.
    return None


def find_output_calibration(
    node: "OnnxNode",
    index: Optional[int] = None,
) -> Optional["CalibrationNode"]:
    """找到一个普通节点输出的校准节点.

    Args:
        node: 普通非校准节点, 用于寻找其输出的校准节点.
        index: 指定需要寻找校准节点的输出索引.
            如果是None, 依次遍历节点各个输出并返回首个找到的校准节点.

    Returns:
        若找到返回相应校准节点, 否则返回None.
    """
    # Conv+ResNetAdd结构需要找到ResNetAdd后面的校准节点.
    if (
        node.op_type == "Conv"
        and len(node.outputs[0].dest_ops) == 1
        and node.outputs[0] not in node.owning_graph.outputs
        and node.outputs[0].dest_op.op_type == "Add"
    ):
        return find_output_calibration(node.outputs[0].dest_op)

    # 获取当前节点的后继节点, 用以遍历查找校准节点.
    next_ops = (
        list(node.next_ops) if index is None else list(node.outputs[index].dest_ops)
    )
    # 层序遍历查找校准节点.
    while next_ops:
        next_op = next_ops.pop(0)
        if next_op.op_type == "HzCalibration":
            return next_op
        # 若后继节点可以跳过, 则将其后继节点加入待查找列表.
        if _can_be_skipped(next_op):
            next_ops.extend(next_op.next_ops)

    # 若未找到校准节点, 则返回None.
    return None
