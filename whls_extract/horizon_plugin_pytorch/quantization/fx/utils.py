from numbers import Real
from typing import Dict, Optional, Sequence

from tabulate import tabulate
from torch.fx import GraphModule
from torch.fx.node import Node

from horizon_plugin_pytorch.fx.fx_helper import (
    get_supported_method,
    match_node_operation,
)
from horizon_plugin_pytorch.nn.quantized import FloatFunctional

__all__ = ["replace_function_with_module", "graph_repr"]


def _parent_name(target):
    """Get the parent name and current module name from a qualified name."""
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]


def _get_generatd_mod_name(model, current_scope_name, op_name):
    module_idx = 0
    module_name = "{}_generated_{}_{}".format(
        current_scope_name, op_name, module_idx
    )

    def _hasattr(obj, attr):
        try:
            left, right = attr.split(".", 1)
        except Exception:
            return hasattr(obj, attr)
        return _hasattr(getattr(obj, left), right)

    while _hasattr(model, module_name):
        module_idx += 1
        module_name = "{}_generated_{}_{}".format(
            current_scope_name, op_name, module_idx
        )

    return module_name


def _check_scalar_node(node):
    # x.shape
    if (
        node.op == "call_function"
        and node.target == getattr
        and node.args[-1] == "shape"
    ):
        return True
    # x.size(d)
    if node.op == "call_method" and node.target == "size":
        return True
    return False


def _mark_scalar_node(model: GraphModule):
    """Mark scalar node to avoid auto replace of ops on scalar.

    Currently only mark tensor.shape operation.
    """

    def _is_scalar_args(args):
        if isinstance(args, Node):
            return args._is_scalar

        if isinstance(args, (Real, str)):
            return True

        if isinstance(args, Sequence) and not isinstance(args, str):
            return all([_is_scalar_args(x) for x in args])

        return False

    for node in list(model.graph.nodes):
        args = list(node.args) + list(node.kwargs.values())

        if _check_scalar_node(node):
            node._is_scalar = True
        elif args and _is_scalar_args(args):
            node._is_scalar = True
        else:
            node._is_scalar = False


def replace_function_with_module(
    model: GraphModule, node_name_to_scope: Dict[str, str]
):
    """Refine this docstring in the future.

    Replace function type operations in a model
    with corresponding horizon.nn module or FloatFunctional

    Args:
        model (GraphModule): The input model
        node_name_to_scope (Dict[str, str]): Mapping from node
            name to the owner module name and type.
    """
    from horizon_plugin_pytorch.fx.fx_helper import (
        _torch_horizon_nn_op_mapping,
        _torch_horizon_op_mapping,
    )

    supported_functional = get_supported_method()[FloatFunctional]
    _mark_scalar_node(model)

    for node in list(model.graph.nodes):
        node: Node
        # replace torch func with horizon.nn module
        op_name = match_node_operation(node, _torch_horizon_op_mapping)
        if op_name is not None and not node._is_scalar:
            # add Module to model
            current_scope_name = node_name_to_scope[node.name]
            module_name = _get_generatd_mod_name(
                model, current_scope_name, op_name
            )
            model.add_submodule(
                module_name, _torch_horizon_op_mapping[op_name]()
            )

            # modify graph
            model.graph.inserting_before(node)
            call_module_node = model.graph.call_module(
                module_name,
                args=node.args,
                kwargs=node.kwargs,
            )
            node_name_to_scope[call_module_node.name] = current_scope_name
            node.replace_all_uses_with(call_module_node)
            node_name_to_scope.pop(node.name)

            continue

        # replace torch.nn.functional func with torch.nn module
        op_name = match_node_operation(node, _torch_horizon_nn_op_mapping)
        if op_name is not None and not node._is_scalar:
            # add Module to model
            current_scope_name = node_name_to_scope[node.name]
            module_name = _get_generatd_mod_name(
                model, current_scope_name, op_name
            )

            tensor_args = []
            non_tensor_args = []
            tensor_kwargs = {}
            non_tensor_kwargs = {}

            for arg in node.args:
                if isinstance(arg, Node):
                    tensor_args.append(arg)
                else:
                    non_tensor_args.append(arg)
            for k, v in node.kwargs.items():
                if isinstance(v, Node):
                    tensor_kwargs[k] = v
                else:
                    non_tensor_kwargs[k] = v

            tensor_args = tuple(tensor_args)
            non_tensor_args = tuple(non_tensor_args)

            model.add_submodule(
                module_name,
                _torch_horizon_nn_op_mapping[op_name](
                    *non_tensor_args, **non_tensor_kwargs
                ),
            )

            # modify graph
            model.graph.inserting_before(node)
            call_module_node = model.graph.call_module(
                module_name,
                args=tensor_args,
                kwargs=tensor_kwargs,
            )
            node_name_to_scope[call_module_node.name] = current_scope_name
            node.replace_all_uses_with(call_module_node)
            node_name_to_scope.pop(node.name)

            continue

        # replace torch func with FloatFunctional
        op_name = match_node_operation(node, supported_functional)
        if op_name is not None and not node._is_scalar:
            # add FloatFunctional to model
            current_scope_name = node_name_to_scope[node.name]
            module_name = _get_generatd_mod_name(
                model, current_scope_name, op_name
            )
            ff_mod = FloatFunctional()
            ff_mod._last_called_method_name = op_name
            model.add_submodule(module_name, ff_mod)

            # modify graph
            model.graph.inserting_before(node)
            get_attr_node = model.graph.get_attr(module_name)
            node_name_to_scope[get_attr_node.name] = current_scope_name
            call_method_node = model.graph.call_method(
                op_name,
                args=(get_attr_node,) + node.args,
                kwargs=node.kwargs,
            )
            node_name_to_scope[call_method_node.name] = current_scope_name
            node.replace_all_uses_with(call_method_node)
            node_name_to_scope.pop(node.name)

    # delete the nodes of function
    # only delete replaced nodes make dead code still in graph, which leads to
    # compatibility problems. So directly delete dead node
    model.graph.eliminate_dead_code()
    for node in model.graph.nodes:
        if hasattr(node, "_is_scalar"):
            del node._is_scalar


def graph_repr(graph, additional_attrs: Optional[Dict] = None):
    headers = ["opcode", "name", "target", "args", "kwargs"]
    node_specs = [
        [n.op, n.name, n.target, n.args, n.kwargs] for n in graph.nodes
    ]

    if additional_attrs is not None:
        for name, attr in additional_attrs.items():
            attr: Dict
            headers.append(name)
            for item in node_specs:
                item.append(attr[item[1]])

    return tabulate(node_specs, headers=headers)
