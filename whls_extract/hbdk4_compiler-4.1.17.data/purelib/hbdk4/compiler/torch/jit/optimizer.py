from typing import Union
import torch

from hbdk4.compiler.torch.jit.utils import get_graph


def replace_prim_CallFunction(graph, module, *args):
    if isinstance(module, torch.jit.TracedModule):
        module = module._actual_script_module

    mini_stack = {}
    for i, v in zip(list(graph.inputs()), args):
        mini_stack[i] = v

    for node in list(graph.nodes()):
        if node.kind() == "prim::Constant":
            mini_stack[node.outputsAt(0)] = node.output().toIValue()

        if node.kind() == "prim::GetAttr":
            obj = mini_stack[node.inputsAt(0)]
            attr = node.s("name")
            mini_stack[node.outputsAt(0)] = getattr(obj, attr)

        if node.kind() == "prim::CallMethod":
            sub_module = mini_stack[node.inputsAt(0)]
            if isinstance(sub_module, torch.jit.TracedModule):
                sub_module = sub_module._actual_script_module
            attr = node.s("name")
            method = getattr(sub_module, attr)

            args = []
            for i in node.inputs():
                args.append(mini_stack[i] if i in mini_stack else None)

            replace_prim_CallFunction(method.graph, sub_module, *args)

        if node.kind() == "prim::CallFunction":
            # func_name = node.inputsAt(0).type().annotation_str
            new_node = graph.create("prim::IgnoredPythonOp")
            new_node.insertAfter(node)

            for i in list(node.inputs()):
                new_node.addInput(i)

            new_node.output().setType(node.output().type())
            new_node.copyMetadata(node)
            node.output().replaceAllUsesWith(new_node.output())
            node.destroy()


def optimize(jit: torch.ScriptModule):
    # replace prim::CallFunction to prim::IgnoredPythonOp so that jit.freeze will not inline it
    graph = get_graph(jit)
    args = [None] * len(list(graph.inputs()))
    args[0] = jit
    replace_prim_CallFunction(graph, jit, *args)

    jit = torch.jit.freeze(jit)

    graph = get_graph(jit)

    torch._C._jit_pass_peephole(graph)
    torch._C._jit_pass_fuse_addmm(graph)
    torch._C._jit_pass_remove_mutation(graph)
    torch._C._jit_pass_remove_inplace_ops(graph)
    torch._C._jit_pass_peephole(graph)

    torch._C._jit_pass_constant_pooling(graph)

    return jit
