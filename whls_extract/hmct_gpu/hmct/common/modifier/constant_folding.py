import logging
from typing import TYPE_CHECKING, Container, Optional, Union

import numpy as np

from hmct.ir import OnnxModel, load_model

from .passes.const_node_to_initializer import ConstNodeToInitializer
from .passes.constant_to_initializer import ConstantToInitializer
from .passes.identity_to_initializer import IdentityToInitializer
from .passes.input_shape_to_initializer import InputShapeToInitializer
from .passes.pass_base import run_optimize_passes
from .passes.unfold_if_subgraph import IfSubgraphUnfold

if TYPE_CHECKING:
    from hmct.ir.onnx_utils import ModelProto


def constant_folding(
    onnx_model_or_proto: Union[OnnxModel, "ModelProto", bytes, str],
    max_inc: Optional[int] = None,
    ignore_nodes: Optional[Container[str]] = None,
) -> OnnxModel:
    """对输入onnx模型执行常量折叠优化.

    Args:
        onnx_model_or_proto: 输入待常量折叠的onnx模型
        max_inc: 若是某个节点折叠前后, 模型参数增加量超过max_inc, 则忽略对本节点的折叠
        ignore_nodes: 若是折叠时节点的名字包含于ignore_nodes, 则忽略对本节点的折叠

    Returns:
        完成常量折叠的onnx模型
    """
    if max_inc is not None:
        assert max_inc >= 0, "max_inc should be an integer greater than or equal to 0."
    onnx_model = load_model(onnx_model_or_proto)

    # 统计常量折叠之前的节点数和参数量
    before_node_num = len(onnx_model.nodes)
    before_param_num = 0
    for onnx_var in onnx_model.variables:
        if onnx_var.is_param:
            before_param_num += int(np.prod(onnx_var.value.shape))

    # 创建并执行常量折叠相关的优化pass
    passes = [
        InputShapeToInitializer(),
        ConstantToInitializer(),
        IdentityToInitializer(max_inc=max_inc, ignore_nodes=ignore_nodes),
        ConstNodeToInitializer(max_inc=max_inc, ignore_nodes=ignore_nodes),
        IfSubgraphUnfold(),
    ]
    onnx_model = run_optimize_passes(
        onnx_model=onnx_model,
        passes=passes,
        iterate=True,
    )

    # 统计常量折叠之后的节点数和参数量
    after_node_num = len(onnx_model.nodes)
    after_param_num = 0
    for onnx_var in onnx_model.variables:
        if onnx_var.is_param:
            after_param_num += int(np.prod(onnx_var.value.shape))

    # 整理并输出常量折叠优化pass的日志记录
    # 1) 首先, 整理整体性的日志输出
    logging.debug("Summary info for constant_folding:")
    logging.debug(
        f"  After constant_folding, the number of nodes has "
        f"changed from {before_node_num} to {after_node_num}."
    )
    logging.debug(
        f"  After constant_folding, the number of parameters has "
        f"changed from {before_param_num} to {after_param_num}."
    )
    # 2) 其次, 整理具体性的日志输出
    logging.debug("Detailed info for constant_folding:")
    sum_log = sorted(
        [
            item
            for p in passes
            if isinstance(p, ConstNodeToInitializer)
            for item in p.sum_log
        ],
        # 根据每个节点折叠后导致的参数增加量递减排序
        key=lambda item: int(item[:-1].split()[-1]),
        reverse=True,
    )
    logging.debug("\n".join([2 * " " + line for line in sum_log]))

    return onnx_model
