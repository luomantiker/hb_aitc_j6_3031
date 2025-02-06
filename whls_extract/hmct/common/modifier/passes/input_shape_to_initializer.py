from typing import TYPE_CHECKING

import numpy as np

from .pass_base import PredicateBasedPass

if TYPE_CHECKING:
    from hmct.ir import OnnxModel, OnnxNode


class InputShapeToInitializer(PredicateBasedPass):
    @property
    def name(self) -> str:
        return "replace_input_shape_with_initializer"

    def initialize_pass(self, onnx_model: "OnnxModel", **kwargs) -> None:
        super().initialize_pass(onnx_model, **kwargs)
        # to reduce time cost, we will call infer_shape func
        # only if there exist Shape or Size nodes
        if (
            len(onnx_model.graph.type2nodes["Shape"])
            + len(onnx_model.graph.type2nodes["Size"])
            > 0
        ):
            onnx_model.infer_shapes()

    def match_pattern(self, onnx_node: "OnnxNode") -> bool:
        return (
            onnx_node.op_type in ("Shape", "Size")
            and not onnx_node.inputs[0].is_shape_dynamic
        )

    def apply_transform(self, onnx_node: "OnnxNode", onnx_model: "OnnxModel") -> bool:
        # 为Shape算子计算常量输出
        if onnx_node.op_type == "Shape":
            opset_ver = (
                onnx_model.opset_import[""]
                if "" in onnx_model.opset_import
                else onnx_model.opset_import["ai.onnx"]
            )
            if opset_ver < 15:
                output_value = np.array(onnx_node.inputs[0].shape, dtype=np.int64)
            else:
                # 从opset15开始, Shape算子拥有start和end属性指定输出的shape维度范围
                start_dim = onnx_node.attributes.get("start", None)
                end_dim = onnx_node.attributes.get("end", None)
                output_value = np.array(
                    onnx_node.inputs[0].shape[start_dim:end_dim], dtype=np.int64
                )
        # 为Size算子计算常量输出
        elif onnx_node.op_type == "Size":
            output_value = np.array(
                int(np.prod(onnx_node.inputs[0].shape)), dtype=np.int64
            )
        else:
            raise ValueError("op_type should be either Shape or Size.")

        # 对当前算子进行常量折叠
        output_var = onnx_model.graph.create_variable(
            name=onnx_node.outputs[0].name,
            is_param=True,
            value=output_value,
        )
        onnx_node.outputs[0].replace_all_uses_with(output_var)
        # destroy folded node
        onnx_node.destroy()

        return True
