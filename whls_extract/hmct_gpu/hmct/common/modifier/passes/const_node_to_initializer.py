from typing import Container, Optional

import numpy as np

from hmct.executor import ORTExecutor
from hmct.ir import OnnxModel, OnnxNode, OnnxVariable, extract_submodel

from .pass_base import PredicateBasedPass


class ConstNodeToInitializer(PredicateBasedPass):
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
        return "replace_const_node_with_initializer"

    def match_pattern(self, onnx_node: OnnxNode) -> bool:
        # ignore_nodes为用户配置参数, 用于忽略对指定name的节点的折叠
        if self._ignore_nodes is not None and onnx_node.name in self._ignore_nodes:
            return False
        if onnx_node.op_type in (
            "Shape",
            "Size",
            "Constant",
            "Identity",
            "If",
            "HzQuantize",
            "QuantizeLinear",
            "HzCalibration",
        ):
            return False
        for input_var in onnx_node.inputs:
            if not input_var.is_param and not input_var.is_dummy:
                return False
        return True

    def apply_transform(self, onnx_node: OnnxNode, onnx_model: OnnxModel) -> bool:
        # execute model for node constant outputs
        single_node_model = extract_submodel(
            onnx_model=onnx_model,
            input_vars=onnx_node.inputs,
            output_vars=onnx_node.outputs,
            infer_shapes=False,
            check_model=False,
        )
        output_dict = ORTExecutor(single_node_model).inference(None)

        # calculate the number of parameters to delete
        del_num = 0
        # 考虑节点输入中可能存在的常量OnnxVariable对象
        for input_var in onnx_node.inputs:
            if input_var.is_param and len(input_var.dest_ops) == 1:
                del_num += int(np.prod(input_var.value.shape))
        # 考虑节点属性中可能存在的常量OnnxVariable对象
        for attr_val in onnx_node.attributes.values():
            attr_vec = attr_val if isinstance(attr_val, list) else [attr_val]
            for item_val in attr_vec:
                if isinstance(item_val, OnnxVariable):
                    del_num += int(np.prod(item_val.value.shape))
        # calculate the number of parameters to add
        add_num = 0
        for output_value in output_dict.values():
            add_num += int(np.prod(output_value.shape))
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
        for output_idx, output_var in enumerate(onnx_node.outputs):
            const_value = output_dict[output_var.name]
            const_var = onnx_model.graph.create_variable(
                name=output_var.name,
                is_param=True,
                value=const_value,
            )
            onnx_node.outputs[output_idx].replace_all_uses_with(const_var)
        # destroy folded node
        onnx_node.destroy()

        return True
