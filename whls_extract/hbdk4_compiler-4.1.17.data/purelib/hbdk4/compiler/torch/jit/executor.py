from typing import Union

import importlib

import torch

from hbdk4.compiler.torch.jit.utils import get_graph, Schema


def _run_prim_node(node: torch.Node, *args, **kwargs):
    op = node.kind()
    if op == "prim::Constant":
        return node.output().toIValue()
    if op == "prim::TupleConstruct":
        return args
    if op == "prim::ListConstruct":
        return list(args)
    if op == "prim::TupleUnpack":
        if len(args[0]) == 1:
            return args[0][0]  # in python, single return decards list/tuple
        return args[0]
    if op == "prim::ListUnpack":
        if len(args[0]) == 1:
            return args[0][0]  # in python, single return decards list/tuple
        return args[0]
    if op == "prim::DictConstruct":
        return {args[idx * 2]: args[idx * 2 + 1] for idx in range(len(args) // 2)}
    if op == "prim::NumToTensor":
        return torch.as_tensor(args[0])
    if op == "prim::GetAttr":
        return getattr(args[0], getattr(node, node.kindOf("name"))("name"))
    if op == "prim::IgnoredPythonOp":
        attrs = node.inputsAt(0).type().annotation_str.split(".")
        if attrs[0] == "__torch__":
            mod_name = attrs[1:-1]
            func_name = attrs[-1]
            mod = importlib.import_module(".".join(mod_name))
            func = getattr(mod, func_name)
            return func(*args[1:])

    raise ValueError("unexpected node", node)


def _run_jit_node(node: torch.Node, *args):
    op = node.kind()
    if op == "aten::__getitem__":
        return args[0][args[1]]

    schema = Schema(node.schema())
    args, kwargs = schema.format(*args)
    func = torch._C._jit_get_operation(node.kind())[0]
    return func(*args, **kwargs)


def run_node(node: str, *args):
    op = node.kind()
    if op.startswith("prim::"):
        return _run_prim_node(node, *args)
    return _run_jit_node(node, *args)


def run_graph(graph: torch.Graph, *graph_args):

    stack = dict()

    for i, v in zip(list(graph.inputs()), graph_args):
        stack[i] = v

    for node in graph.nodes():
        node_args = [stack[i] for i in node.inputs()]
        node_rets = run_node(node, *node_args)

        if len(list(node.outputs())) == 1:
            stack[list(node.outputs())[0]] = node_rets
        else:
            for o, v in zip(list(node.outputs()), node_rets):
                stack[o] = v

    graph_rets = [stack[o] for o in graph.outputs()]
    if len(graph_rets) == 1:
        return graph_rets[0], stack
    return graph_rets, stack


def run(jit: Union[torch.jit.ScriptModule, torch.jit.ScriptFunction], *args):
    graph = get_graph(jit)
    if isinstance(jit, torch.jit.ScriptModule):
        return run_graph(graph, jit, *args)
    return run_graph(graph, *args)
