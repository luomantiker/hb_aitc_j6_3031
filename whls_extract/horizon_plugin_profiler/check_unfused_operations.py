import warnings
from copy import deepcopy
from typing import Any

from horizon_plugin_profiler.utils.model_helper import (
    _as_tuple,
    attach_qualified_name,
    register_hook_on_leaf,
    swap_ff_with_horizonff,
)
from horizon_plugin_profiler.utils.typeguard import typechecked

import torch
from tabulate import tabulate
from torch import Tensor, nn

from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.qtensor import QTensor


@typechecked
def check_unfused_operations(
    model: torch.nn.Module, example_inputs: Any, print_tabulate: bool = True
):
    """Check unfused modules in a model.

    NOTE: This function is only capable to check unfused modules. For the
          correctness of fusion, please use `featuremap_similarity`
          to compare the feature between fused and unfused model.

    Args:
        model (torch.nn.Module):  The input model.
        example_inputs (Any[Tensor]): The input data feed to model.
        print_tabulate (bool, optional): Whether print the result as tabulate.
            Defaults to True.

    Returns:
        List[List[str]]:
            The qualified name of modules that can be fused.
    """
    model = deepcopy(model)

    swap_ff_with_horizonff(model)
    attach_qualified_name(model)

    output_to_module_mapping = {}
    virtual_input_node = nn.Identity()
    virtual_input_node._qualified_name = "placeholder"
    virtual_input_node._output_to = []

    def _add_virtual_input_node(inputs):
        if isinstance(inputs, (Tensor, QTensor)):
            output_to_module_mapping[inputs] = virtual_input_node
        # in HAT config, dict is ConfigDict, a subclass of dict
        elif issubclass(type(inputs), dict):
            for value in list(inputs.values()):
                _add_virtual_input_node(value)
        elif type(inputs) in (list, tuple):
            for i in inputs:
                _add_virtual_input_node(i)

    _add_virtual_input_node(example_inputs)

    def _record_output_hook(module, input, output):
        if isinstance(output, (list, tuple)):
            for x in output:
                _record_output_hook(module, input, x)
        elif isinstance(output, (Tensor, QTensor)):
            output_to_module_mapping[output] = module

    def _make_graph_hook(module, input):
        if not hasattr(module, "_input_from"):
            module._input_from = []
        if not hasattr(module, "_output_to"):
            module._output_to = []

        if isinstance(input, (list, tuple)):
            for x in input:
                _make_graph_hook(module, x)
        elif isinstance(input, (Tensor, QTensor)):
            # if input from functions like reshape
            # use virtual_input_node as input
            input_from = (
                virtual_input_node
                if input not in output_to_module_mapping
                else output_to_module_mapping[input]
            )
            if input_from not in module._input_from:
                module._input_from.append(input_from)
            if module not in input_from._output_to:
                input_from._output_to.append(module)

    register_hook_on_leaf(model, _record_output_hook, _make_graph_hook)

    example_inputs = _as_tuple(example_inputs)
    model(*example_inputs)

    def match_node(module, expected_type):
        if hasattr(module, "_matched"):
            return False
        if not type(module) == expected_type:
            return False
        if expected_type is FloatFunctional:
            return module._last_called_method_name == "add"
        return True

    def match_pattern(root: torch.nn.Module, pattern):
        if match_node(root, pattern[0]):
            if len(pattern) == 1:
                root._matched = True
                return [[root]]
            else:
                tile = []
                for next_node in root._output_to:
                    tile += match_pattern(next_node, pattern[1:])
                for matched_seq in tile:
                    root._matched = True
                    matched_seq.insert(0, root)
                return tile
        else:
            return []

    def get_unmatched_next(root: torch.nn.Module, visited_nodes):
        if root in visited_nodes:
            return set()
        next_nodes = set()
        visited_nodes.add(root)
        for node in root._output_to:
            if hasattr(node, "_matched"):
                next_nodes |= get_unmatched_next(node, visited_nodes)
            else:
                if node not in visited_nodes:
                    next_nodes.add(node)

        return next_nodes

    def search_pattern(root: torch.nn.Module, patterns):
        current_roots = [root]
        ret = []

        visited_nodes = set()
        while len(current_roots) > 0:
            next_roots = set()
            for node in current_roots:
                if not hasattr(node, "_matched"):
                    matched_seqs = []
                    for pattern in patterns:
                        matched_seqs = match_pattern(node, pattern)
                        if len(matched_seqs) > 0:
                            ret += matched_seqs
                            break

                next_roots |= get_unmatched_next(node, visited_nodes)
            if current_roots == next_roots:
                warnings.warn(
                    "There are circles in the graph, "
                    "related nodes are {}".format(
                        [m._qualified_name for m in current_roots]
                    ),
                    UserWarning,
                )
                break
            current_roots = next_roots

        return ret

    from horizon_plugin_pytorch.quantization.fuse_modules import (
        get_op_list_to_fuser_mapping,
    )

    fuse_patterns = list(get_op_list_to_fuser_mapping().keys())
    fuse_patterns.sort(key=len, reverse=True)

    matched_seqs = search_pattern(virtual_input_node, fuse_patterns)
    module_to_fuse = []
    for matched_seq in matched_seqs:
        valid = True
        shared_conv = False
        if len(matched_seq[0]._output_to) > 1:
            shared_conv = True
        for m in matched_seq[1:]:
            if isinstance(m, FloatFunctional):
                if len(m._input_from) > 2 or len(m._output_to) > 1:
                    valid = False
                    break
            else:
                if len(m._input_from) > 1 or len(m._output_to) > 1:
                    valid = False
                    break

        if valid:
            module_to_fuse.append(
                [
                    (
                        module._qualified_name
                        + ("(shared)" if i == 0 and shared_conv else ""),
                        type(module),
                    )
                    for i, module in enumerate(matched_seq)
                ]
            )

    if print_tabulate:
        if len(module_to_fuse) == 0:
            print("Do not find any fusable modules")
        else:
            print("Fusable modules are listed below\n")
            for item in module_to_fuse:
                print(
                    tabulate(
                        item + [["", ""]],
                        headers=["name", "type"],
                    )
                )

    del model

    return module_to_fuse
