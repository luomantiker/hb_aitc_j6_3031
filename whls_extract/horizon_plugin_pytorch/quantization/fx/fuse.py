import inspect
from typing import Any, Callable, Dict, Tuple

from torch.fx import GraphModule, Node, map_arg
from torch.fx.graph import Graph

from horizon_plugin_pytorch.utils.model_helper import is_ff_node
from .fusion_patterns import FuseHandler, MatchAllNode
from .graph_module import FusedGraphModule
from .pattern_utils import get_default_fusion_patterns, is_match
from .quantization_types import Pattern


class DuplicateMatchError(Exception):
    pass


class Fuser:
    def fuse(
        self,
        model: GraphModule,
        fuse_custom_config_dict: Dict[str, Any] = None,
    ) -> GraphModule:
        """Fuse model with graph modification and recompile.

        Args:
            model (GraphModule): Input model, must has `graph` attr.
            fuse_custom_config_dict (Dict[str, Any], optional): Unused.
        """
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}

        input_root = model
        input_graph = model.graph
        self.modules = dict(input_root.named_modules())

        fusion_patterns = get_default_fusion_patterns()
        # find fusion
        fusion_pairs = self._find_matches(
            input_root, input_graph, fusion_patterns
        )

        self.fused_graph = Graph()
        env: Dict[Any, Any] = {}

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])

        module_called_times = self._get_module_called_times(input_graph, model)
        fused_map = {}

        for node in input_graph.nodes:
            root_node, obj = fusion_pairs.get(node.name, (None, None))
            if root_node is node:
                assert obj is not None
                obj.fuse(
                    self,
                    env,
                    load_arg,
                    module_called_times,
                    fused_map=fused_map,
                )
            elif root_node is None:
                env[node.name] = self.fused_graph.node_copy(node, load_arg)
            # node matched in patterns and is not root is removed here

        preserved_attributes = fuse_custom_config_dict.get(
            "preserved_attributes", []
        )
        model = FusedGraphModule(
            input_root, self.fused_graph, preserved_attributes + ["qconfig"]
        )

        return model

    def fuse_with_ori_graph(
        self,
        model: GraphModule,
        fuse_custom_config_dict: Dict[str, Any] = None,
    ) -> GraphModule:
        """Fuse model without graph modification and recompile.

        Model forward code is not modified, so we must replace <the mod being
        fused into other mod> with nn.Identity.

        Args:
            model (GraphModule): Input model, must has `graph` attr.
            fuse_custom_config_dict (Dict[str, Any], optional): Unused.
        """
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}

        input_root = model
        input_graph = model.graph
        self.modules = dict(input_root.named_modules())

        fusion_patterns = get_default_fusion_patterns()
        # find fusion
        fusion_pairs = self._find_matches(
            input_root, input_graph, fusion_patterns
        )

        module_called_times = self._get_module_called_times(input_graph, model)
        fused_map = {}
        fused_modules = {}

        for node in input_graph.nodes:
            root_node, obj = fusion_pairs.get(node.name, (None, None))
            if root_node is node:
                assert obj is not None
                obj.fuse_on_module(
                    self,
                    module_called_times,
                    fused_map=fused_map,
                    fused_modules=fused_modules,
                )

        return model

    def _get_module_called_times(self, graph, model):
        module_called_times = {}
        for node in graph.nodes:
            if node.op == "call_module":
                if node.target in module_called_times:
                    module_called_times[node.target] += 1
                else:
                    module_called_times[node.target] = 1
            elif is_ff_node(node, model):
                mod_name = node.args[0].target
                if mod_name in module_called_times:
                    module_called_times[mod_name] += 1
                else:
                    module_called_times[mod_name] = 1

        return module_called_times

    def _find_matches(
        self,
        root: GraphModule,
        graph: Graph,
        patterns: Dict[Pattern, Callable],
    ) -> Dict[str, Tuple[Node, FuseHandler]]:
        modules = dict(root.named_modules())
        match_map: Dict[
            str, Tuple[Node, FuseHandler]
        ] = {}  # node name -> (root_node, match_value)

        def apply_match(pattern, node, match):
            """Add matched nodes and Handler to match_map."""
            if isinstance(pattern, tuple):
                s, *args = pattern
                owner = None
                if inspect.ismethod(s):
                    for cls in inspect.getmro(s.__self__.__class__):
                        if s.__name__ in cls.__dict__:
                            owner = cls
                else:
                    owner = getattr(
                        inspect.getmodule(s),
                        s.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[
                            0
                        ],
                        None,
                    )
                if inspect.isclass(owner) and owner is not s:
                    # call_method(obj, arg0, arg1, ...)
                    apply_match(s, node, match)
                    getattr_node = node.args[0]
                    if len(getattr_node.users) == 1:
                        apply_match(getattr, getattr_node, match)
                    for subpattern, arg in zip(args, node.args[1:]):
                        apply_match(subpattern, arg, match)
                else:
                    # call_function or call_module
                    apply_match(s, node, match)
                    for subpattern, arg in zip(args, node.args):
                        apply_match(subpattern, arg, match)
            elif pattern is MatchAllNode:
                return
            else:
                assert is_match(modules, node, pattern)
                if node.name not in match_map:
                    match_map[node.name] = match
                else:
                    # raise DuplicateMatchError to ignore the pattern
                    # including node already used by other pattern
                    raise DuplicateMatchError(
                        "node {} already matched".format(node.name)
                    )

        for node in reversed(graph.nodes):
            if node.name not in match_map:
                for pattern, value in patterns.items():
                    if is_match(modules, node, pattern):
                        try:
                            apply_match(
                                pattern, node, (node, value(self, node))
                            )
                        except DuplicateMatchError:
                            pass

        return match_map
