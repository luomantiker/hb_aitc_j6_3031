# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict, List, Optional

from hat.models.structures.multitask_graph_model import MultitaskGraphModel
from hat.registry import OBJECT_REGISTRY
from .converters import BaseConverter

try:
    from hatbc.workflow.symbol import Node
except ImportError:
    Node = None


__all__ = ["GraphModelSplit", "GraphModelInputKeyMapping"]


@OBJECT_REGISTRY.register
class GraphModelSplit(BaseConverter):
    """Split graph model in deploy mode."""

    def __init__(
        self,
        split_nodes: List[str],
        next_bases: List[str],
        save_models: Optional[List[str]] = None,
        pick_models_index: Optional[int] = None,
    ):
        super(GraphModelSplit, self).__init__()
        assert len(split_nodes) == len(next_bases)
        assert all([n in ["top", "bottom"] for n in next_bases])
        if save_models is not None:
            assert len(split_nodes) == len(save_models)
            assert all(
                [
                    k in [None, "", "top", "bottom", "top,bottom"]
                    for k in save_models
                ]
            )
        else:
            save_models = ["top,bottom"] * len(split_nodes)
        self.split_nodes = split_nodes
        self.next_bases = next_bases
        self.save_models = save_models
        self.pick_models_index = pick_models_index

    def __call__(self, graph_model: MultitaskGraphModel):
        top_models = []
        bottom_models = []
        for s_n, n_b, k_m in zip(
            self.split_nodes, self.next_bases, self.save_models
        ):
            top_model, bottom_model, _, _ = graph_model.split_module(
                out_names=None, split_node_name=s_n, common_module_flatten=True
            )
            if k_m in ["top", "top,bottom"]:
                top_models.append(top_model)
            if k_m in ["bottom", "top,bottom"]:
                bottom_models.append(bottom_model)
            if n_b == "top":
                graph_model = top_model
            else:
                graph_model = bottom_model
        split_models = top_models + bottom_models
        if self.pick_models_index is not None:
            return split_models[self.pick_models_index]
        else:
            return split_models


@OBJECT_REGISTRY.register
class GraphModelInputKeyMapping(BaseConverter):
    """Mapping input key in graph model for deploy mode."""

    def __init__(
        self,
        input_key_mapping: Dict[str, str],
    ):
        super(GraphModelInputKeyMapping, self).__init__()
        self.input_key_mapping = input_key_mapping

    def __call__(self, graph_model: MultitaskGraphModel):
        graph = graph_model.get_sub_graph(graph_model._output_names)
        graph_replace_key = graph.copy(deepcopy=False)
        _record_node = {}
        created_place_holders = {}

        def _replace_input_key(node):
            for input_i in node.inputs:
                if input_i.op != "PLACEHOLDER":
                    continue
                if input_i.name in self.input_key_mapping:
                    if (
                        self.input_key_mapping[input_i.name]
                        not in created_place_holders
                    ):
                        created_place_holders[
                            self.input_key_mapping[input_i.name]
                        ] = Node.create_placeholder(
                            name=self.input_key_mapping[input_i.name],
                            attr=input_i.attr,
                        )
                    create_node = created_place_holders[
                        self.input_key_mapping[input_i.name]
                    ]
                    _record_node[node._inputs.pop(input_i).name] = create_node
                    node._inputs[create_node] = create_node

            def _replace_inner(inp):
                if isinstance(inp, (list, tuple)):
                    new_list = []
                    for item in inp:
                        new_item = _replace_inner(item)
                        new_list.append(new_item)
                    return type(inp)(new_list)
                elif isinstance(inp, dict):
                    new_dict = {}
                    for key, value in inp.items():
                        new_value = _replace_inner(value)
                        if key in _record_node:
                            _key = _record_node[key].name
                        else:
                            _key = key
                        new_dict[_key] = new_value
                    return new_dict
                elif isinstance(inp, Node) and inp.name in _record_node:
                    return _record_node[inp.name]
                else:
                    return inp

            node._args = _replace_inner(node._args)
            node._kwargs = _replace_inner(node._kwargs)

        graph_replace_key.post_order_dfs_visit(fvisit=_replace_input_key)
        graph_replace_key = graph_replace_key.copy(deepcopy=False)

        graph_model_relace_key = graph_model.from_existed_graph_and_modules(
            graph_replace_key,
            node2name=graph_model.node2name,
            flatten_outputs=graph_model.flatten_outputs,
            output_names=graph_model._output_names,
            name2inds_fmts=graph_model._name2inds_fmts,
        )
        return graph_model_relace_key
