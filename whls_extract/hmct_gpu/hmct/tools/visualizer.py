import argparse
import sys

import pydot

from hmct.ir import OnnxModel, OnnxNode, load_model
from hmct.ir.utils import onnx_dtype_to_numpy_dtype
from hmct.reporter import calculate_quant_type


class InOutNode:
    def __init__(self, name, dtype, pynode_name=None):
        self.name = name
        self.dtype = dtype
        self.pynode_name = pynode_name

    def dot(self):
        return pydot.Node(
            name=self.name if self.pynode_name is None else self.pynode_name,
            label=self.label,
            color=self.color,
            style="rounded",
            shape="ellipse",
            margin=0.05,
        )

    @property
    def label(self) -> str:
        if self.dtype is None:
            label = f"<{self.name}>"
        else:
            label = f"<{self.name} <br/> {self.dtype}>"
        return label

    @property
    def color(self) -> str:
        if self.dtype in ["float32", "float16"]:
            return "red"
        return "#000000"


class GraphNode:
    def __init__(self, op_type, name):
        self.op_type = op_type
        self.name = name
        self.tae = False

    def dot(self):
        return pydot.Node(
            name=self.name.replace(":", "_"),
            label=self.label,
            color=self.color,
            style="rounded, filled",
            shape="rect",
            margin=0.2,
        )

    @property
    def label(self) -> str:
        if self.op_type in ["Quantize", "Dequantize"]:
            label = f"<<b>{self.op_type}</b>>"
        else:
            label = f"<<b>{self.op_type}</b><br/>{self.name}>"
        return label

    @property
    def color(self) -> str:
        if self.op_type in ["Conv", "ConvTranspose", "MatMul"] or self.tae is True:
            return "#4884b4"
        if self.op_type in [
            "Reshape",
            "Transpose",
            "Concat",
            "Slice",
            "Split",
            "Expand",
            "Tile",
            "Gather",
            "GatherElements",
        ]:
            return "#8ebc8d"
        if self.op_type in ["Cast", "Quantize", "Dequantize"]:
            return "#808080"
        return "#45fffe"


class GraphEdge:
    def __init__(self, src, dst, shape, qtype, thresholds):
        self.src = src
        self.dst = dst
        self.shape = shape
        self.qtype = qtype
        self.thresholds = thresholds

    def dot(self):
        return pydot.Edge(
            src=self.src,
            dst=self.dst,
            label=self.label,
            color=self.color,
            fontsize="12pt",
        )

    @property
    def label(self) -> str:
        if self.qtype is None or self.qtype == "--":
            return f"<{self.shape}>"
        if len(self.thresholds) == 1:
            return f"<{self.shape} <br/> QT={self.qtype} <br/> TH={self.thresholds[0]}>"
        if len(self.thresholds) > 1:
            return (
                f"<{self.shape} <br/> QT={self.qtype} <br/>"
                "TH=[{self.thresholds[0]}...]>"
            )
        return f"<{self.shape} <br/> QT={self.qtype}>"

    @property
    def color(self) -> str:
        if self.qtype in ["int8", "uint8"]:
            return "green"
        if self.qtype in ["int16"]:
            return "orange"
        if self.qtype in ["float16", "float32"]:
            return "red"
        return "black"


def can_add_fused_into_conv(node: OnnxNode) -> bool:
    """判断Add节点是否可以融合到前面的Conv节点中."""
    return (
        node.op_type == "Add"
        and len(node.prev_ops) == 2
        and (
            (node.prev_ops[0].op_type == "Conv" and len(node.prev_ops[0].next_ops) == 1)
            or (
                node.prev_ops[1].op_type == "Conv"
                and len(node.prev_ops[1].next_ops) == 1
            )
        )
    )


def can_relu_fused_into_conv(node: OnnxNode) -> bool:
    """判断Relu和Clip节点是否可以融合到前面的Conv/ConvTranspose节点中."""
    if node.op_type in ["Relu", "Clip"]:
        prev_op = node.inputs[0].src_op
        if prev_op.op_type in ["Conv", "ConvTranspose"] and len(prev_op.next_ops) == 1:
            return True
        if (
            prev_op.op_type == "Add"
            and len(prev_op.next_ops) == 1
            and can_add_fused_into_conv(prev_op) is True
        ):
            return True
    return False


def can_node_fused_into_conv(node: OnnxNode) -> bool:
    """判断指定节点是否可以融合到前面的Conv节点中."""
    return can_add_fused_into_conv(node) or can_relu_fused_into_conv(node)


def create_svg_graph(model: OnnxModel, output_file: str):
    qtype_dict = calculate_quant_type(model)
    graph_inputs = []
    graph_outputs = []
    graph_nodes = []
    graph_edges = []

    graph_inputs.extend(
        InOutNode(
            name=input_var.name, dtype=onnx_dtype_to_numpy_dtype(input_var.dtype).name
        )
        for input_var in model.graph.inputs
    )

    for node in model.graph.nodes:
        if node.op_type in ["HzQuantize", "HzDequantize", "HzCalibration"]:
            continue

        graph_nodes.append(GraphNode(op_type=node.op_type, name=node.name))

        # 判断节点是否可以融合到Conv中.
        if node.op_type in ["Relu", "Clip", "Add"]:
            graph_nodes[-1].tae = can_node_fused_into_conv(node)

        for idx, input_var in enumerate(node.inputs):
            if input_var.is_param:
                continue

            input_qtype = None
            input_thresholds = []
            if idx < len(qtype_dict[node.name]["input_qtype"]):
                input_qtype = qtype_dict[node.name]["input_qtype"][idx]
            if idx < len(qtype_dict[node.name]["input_threshold"]):
                input_thresholds = qtype_dict[node.name]["input_threshold"][idx]

            input_node = input_var.src_op
            if input_node.op_type in ["HzCalibration", "HzQuantize", "HzDequantize"]:
                if input_node.inputs[0] in model.graph.inputs:
                    input_node = input_node.inputs[0]
                elif input_node.inputs[0].src_op is None:
                    continue
                else:
                    input_node = input_node.inputs[0].src_op
                    if input_node.op_type in [
                        "HzCalibration",
                        "HzQuantize",
                        "HzDequantize",
                    ]:
                        input_node = input_node.inputs[0].src_op
            graph_edges.append(
                GraphEdge(
                    src=input_node.name.replace(":", "_"),
                    dst=node.name.replace(":", "_"),
                    shape=input_var.shape,
                    qtype=input_qtype,
                    thresholds=input_thresholds,
                )
            )

    for output_var in model.graph.outputs:
        # 避免模型输出的名字跟中间节点的名字重复.
        pynode_name = output_var.name + "_inout"
        graph_outputs.append(
            InOutNode(
                name=output_var.name,
                dtype=onnx_dtype_to_numpy_dtype(output_var.dtype).name,
                pynode_name=pynode_name,
            )
        )
        input_node = output_var.src_op
        if input_node.op_type == "HzCalibration":
            input_node = input_node.inputs[0].src_op
        graph_edges.append(
            GraphEdge(
                src=input_node.name,
                dst=pynode_name,
                shape=output_var.shape,
                qtype=None,
                thresholds=[],
            )
        )

    # Create svg graph
    svg_graph = pydot.Dot("onnx_model", rankdir="TB")
    for input_var in graph_inputs:
        svg_graph.add_node(input_var.dot())

    for output_var in graph_outputs:
        svg_graph.add_node(output_var.dot())

    for node in graph_nodes:
        svg_graph.add_node(node.dot())

    for edge in graph_edges:
        svg_graph.add_edge(edge.dot())

    svg_graph.write_svg(output_file)


def get_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_model",
        "-om",
        type=str,
        required=True,
        help="Input onnx model(.onnx) file.",
    )
    parser.add_argument(
        "--output", "-o", default="test.svg", type=str, help="Output svg file."
    )
    return parser.parse_args(args)


def main(args):
    args = get_args(args)
    onnx_model = load_model(args.onnx_model)
    create_svg_graph(onnx_model, args.output)


if __name__ == "__main__":
    main(sys.argv[1:])
