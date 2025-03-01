import torch

from horizon_plugin_pytorch._compat import node_get

__all__ = ["get_script_subgraph", "arange"]


def _get_script_attr(model, node):
    mod_inputs = list(model.graph.inputs())
    inputs = list(node.inputs())
    if inputs[0].type().kind() == "ClassType":
        if mod_inputs[0] == inputs[0]:
            node_name = node_get(node, "name")
            return getattr(model, node_name)
        else:
            top_node = _get_script_attr(model, inputs[0].node())
            node_name = node_get(node, "name")
            return getattr(top_node, node_name)


def get_script_subgraph(model, node):
    # get submodule's graph from scripted module
    if node.kind() != "prim::CallMethod":
        return None
    inputs = list(node.inputs())
    return _get_script_attr(model, inputs[0].node())


@torch.jit.script
def arange(start: int, end: int, step: int, device_like: torch.Tensor):
    return torch.arange(start, end, step, device=device_like.device)
