from typing import TYPE_CHECKING

from .pass_base import PredicateBasedPass

if TYPE_CHECKING:
    from hmct.ir import OnnxGraph, OnnxModel, OnnxNode


class IfSubgraphUnfold(PredicateBasedPass):
    @property
    def name(self) -> str:
        return "unfold_if_subgraph"

    def match_pattern(self, onnx_node: "OnnxNode") -> bool:
        return onnx_node.op_type == "If" and onnx_node.inputs[0].is_param

    def apply_transform(self, onnx_node: "OnnxNode", onnx_model: "OnnxModel") -> bool:
        # 根据If节点的常量输入取到需要被展开的子图
        subgraph: OnnxGraph
        if onnx_node.inputs[0].value:
            subgraph = onnx_node.attributes["then_branch"]
        else:
            subgraph = onnx_node.attributes["else_branch"]
        # delete graph attribute in If node to avoid duplicated
        # node && variable names during the following sub-graph unfold
        onnx_node.clear_attributes()

        # copy sub-graph elements to main-graph
        onnx_model.graph.copy_from(
            variables=subgraph.variables,
            graphs=subgraph.graphs,
            nodes=subgraph.nodes,
        )
        # 将子图各个输出name修改为If节点相应的输出name,
        # 保证展开后的子图能和主图其余部分对接上
        for output_idx, subgraph_output_name in enumerate(subgraph.output_names):
            subgraph_output_var = onnx_model.graph.variable_mappings[
                subgraph_output_name
            ]
            subgraph_output_var.name = onnx_node.outputs[output_idx].name
            onnx_node.outputs[output_idx].replace_all_uses_with(subgraph_output_var)
        # After modifying sub-graph outputs names to If node output names,
        # it's necessary to make topological sort.
        onnx_model.sort_topologically()

        # destroy unfolded If node
        onnx_node.destroy()

        return True
