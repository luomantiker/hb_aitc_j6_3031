from functools import wraps
from typing import Dict, Optional

import onnx
import torch
from onnx import TensorProto, helper
from torch.fx import Node

from horizon_plugin_pytorch.utils.location_info import TorchLocationInfo
from .jit_scheme import GraphModule, Tracer


def _torch_to_onnx_dtype(torch_dtype):
    mapping = {
        None: TensorProto.UNDEFINED,
        torch.float16: TensorProto.FLOAT16,
        torch.float32: TensorProto.FLOAT,
        torch.float64: TensorProto.DOUBLE,
        torch.bfloat16: TensorProto.BFLOAT16,
        torch.int8: TensorProto.INT8,
        torch.int16: TensorProto.INT16,
        torch.int32: TensorProto.INT32,
        torch.int64: TensorProto.INT64,
        torch.uint8: TensorProto.UINT8,
        torch.bool: TensorProto.BOOL,
    }
    return mapping[torch_dtype]


def _legalize_node_inputs(node: Node):
    inputs = []
    attrs = {}

    for i, x in enumerate(node.args):
        if isinstance(x, Node):
            inputs.append(x.name)
        elif isinstance(x, (list, tuple)):
            for j, y in enumerate(x):
                if isinstance(y, Node):
                    inputs.append(y.name)
                else:
                    attrs["arg{}_{}".format(i, j)] = str(y)
        else:
            attrs["arg{}".format(i)] = str(x)
    for n, x in node.kwargs.items():
        if isinstance(x, Node):
            inputs.append(x.name)
        else:
            attrs[n] = str(x)

    return inputs, attrs


def _make_node_for_module(
    model: torch.nn.Module, qualified_name: str, inputs, outputs, **attrs
):
    mod = model.get_submodule(qualified_name)

    return helper.make_node(
        TorchLocationInfo.format_op_name(mod),
        inputs,
        outputs,
        qualified_name,
        str(mod),
        qualified_name,
        **attrs,
    )


def visualize(
    model: GraphModule,
    output_path: str,
    node_name_to_dtype: Optional[Dict] = None,
    node_name_to_shape: Optional[Dict] = None,
    node_name_to_location: Optional[Dict] = None,
):
    """Visualize a FX Graph by save it as an onnx model.

    Args:
        model (GraphModule): The GraphModule is basically a torch.nn.Module,
            and contains a attr called 'graph' of type `torch.fx.Graph`
            associating with the forward computation of the model.
        output_path (str): Output onnx file path.
        node_name_to_dtype (dict, optional): A mapping from node name to its
            output dtype. If provided as argument of model attr, this info will
            be added to output onnx.
            Defaults to None.
        node_name_to_shape (dict, optional): A mapping from node name to its
            output shape. If provided as argument of model attr, this info will
            be added to output onnx.
            Defaults to None.
        node_name_to_location (dict, optional): A mapping from node name to its
            location. If provided as argument of model attr, this info will be
            added to output onnx.
            Defaults to None.
    """
    if not output_path.endswith(".onnx"):
        output_path = output_path + ".onnx"
    if node_name_to_dtype is None:
        node_name_to_dtype = getattr(model, "node_name_to_dtype", {})
    if node_name_to_shape is None:
        node_name_to_shape = getattr(model, "node_name_to_shape", {})
    if node_name_to_location is None:
        node_name_to_location = getattr(model, "node_name_to_location", {})

    input_values = []
    output_values = []
    node_defs = []

    @wraps(helper.make_node)
    def add_node(*args, **kwargs):
        node_defs.append(helper.make_node(*args, **kwargs))

    for node in model.graph.nodes:
        node: Node

        dtype = _torch_to_onnx_dtype(node_name_to_dtype.get(node.name, None))
        shape = node_name_to_shape.get(node.name, None)
        loc: TorchLocationInfo = node_name_to_location.get(node.name, None)

        common_node_attrs = {
            "output_dtype": dtype,
            "output_shape": shape,
        }
        if loc is not None:
            common_node_attrs.update(loc.to_dict())

        if node.op == "placeholder":
            input_values.append(
                helper.make_tensor_value_info(node.name, dtype, shape)
            )
        elif node.op == "get_attr":
            attrs = {"attr_name": node.target}
            attrs.update(common_node_attrs)
            node_defs.append(
                _make_node_for_module(
                    model, node.target, [], [node.name + "_target"], **attrs
                )
            )
            add_node(
                node.op,
                [node.name + "_target"],
                [node.name],
                domain=loc.mod_name if loc is not None else None,
                **attrs,
            )
        elif node.op == "call_function":
            if node.target is Tracer.scope_end:
                continue
            inputs, attrs = _legalize_node_inputs(node)
            attrs.update(common_node_attrs)
            add_node(
                TorchLocationInfo.format_op_name(node.target),
                inputs,
                [node.name],
                domain=loc.mod_name if loc is not None else None,
                **attrs,
            )
        elif node.op == "call_module":
            inputs, attrs = _legalize_node_inputs(node)
            attrs.update(common_node_attrs)
            node_defs.append(
                _make_node_for_module(
                    model, node.target, inputs, [node.name], **attrs
                )
            )
        elif node.op == "call_method":
            inputs, attrs = _legalize_node_inputs(node)
            attrs.update(common_node_attrs)
            add_node(
                "method_{}".format(node.target),
                inputs,
                [node.name],
                domain=loc.mod_name if loc is not None else None,
                **attrs,
            )
        elif node.op == "output":
            output_nodes = node.args[0]
            if isinstance(output_nodes, Node):
                output_nodes = (output_nodes,)
            for n in output_nodes:
                dtype = _torch_to_onnx_dtype(
                    node_name_to_dtype.get(n.name, None)
                )
                shape = node_name_to_shape.get(n.name, None)
                output_values.append(
                    helper.make_tensor_value_info(n.name, dtype, shape)
                )
        else:
            raise ValueError("Unknown node type {}".format(node.op))

    graph_def = helper.make_graph(
        node_defs,
        "fx_graph_model",
        input_values,
        output_values,
    )
    model_def = helper.make_model(graph_def, producer_name="onnx-example")

    onnx.save_model(model_def, output_path)
