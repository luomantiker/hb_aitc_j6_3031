from typing import TYPE_CHECKING, Container, Optional

import numpy as np

from .pass_base import PredicateBasedPass

if TYPE_CHECKING:
    from hmct.ir import OnnxModel, OnnxNode


class IdentityToInitializer(PredicateBasedPass):
    def __init__(
        self,
        max_inc: Optional[int] = None,
        ignore_nodes: Optional[Container[str]] = None,
    ) -> None:
        super().__init__()
        self._max_inc = max_inc
        self._ignore_nodes = ignore_nodes
        self.sum_log = []

    @property
    def name(self) -> str:
        return "replace_identity_with_initializer"

    def match_pattern(self, onnx_node: "OnnxNode") -> bool:
        # ignore_nodes为用户配置参数, 用于忽略对指定name的节点的折叠
        if self._ignore_nodes is not None and onnx_node.name in self._ignore_nodes:
            return False
        return onnx_node.op_type == "Identity" and onnx_node.inputs[0].is_param

    def apply_transform(self, onnx_node: "OnnxNode", onnx_model: "OnnxModel") -> bool:
        # calculate the number of parameters to delete
        del_num = 0
        # 考虑节点输入中可能存在的常量OnnxVariable对象
        if len(onnx_node.inputs[0].dest_ops) == 1:
            del_num += int(np.prod(onnx_node.inputs[0].value.shape))
        # calculate the number of parameters to add
        add_num = int(np.prod(onnx_node.inputs[0].value.shape))
        # 若参数增加量超出给定阈值, 则忽略对该节点的折叠
        if self._max_inc is not None and add_num - del_num > self._max_inc:
            return False
        # 添加折叠该节点的详细信息到输出日志
        self.sum_log.append(
            f"After folding node (op_name: {onnx_node.name}, "
            f"op_type: {onnx_node.op_type}), the number of "
            f"increased parameters is {add_num - del_num}."
        )

        # 对当前算子进行常量折叠
        output_var = onnx_model.graph.create_variable(
            name=onnx_node.outputs[0].name,
            is_param=True,
            value=onnx_node.inputs[0].value,
        )
        onnx_node.outputs[0].replace_all_uses_with(output_var)
        # destroy folded node
        onnx_node.destroy()

        return True
