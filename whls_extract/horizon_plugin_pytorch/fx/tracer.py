import copy
from typing import Any

import numpy as np
from torch import Tensor, fx
from torch.fx.graph import Graph
from torch.fx.node import Argument, Node
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_unflatten

from horizon_plugin_pytorch.utils._swap_horizon_float_nn import (
    swap_nn_with_horizonnn,
)
from horizon_plugin_pytorch.utils.misc import pytree_convert
from horizon_plugin_pytorch.utils.model_helper import (
    HookAndTorchFunctionHelper,
)
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .fx_helper import FxWrapManager

__all__ = [
    "CustomTracer",
    "HookAndTorchFunctionTracer",
]


class HookAndTorchFunctionTracer(HookAndTorchFunctionHelper):
    """Getting graph with module hook and __torch_function__ of tensor."""

    # Only for accessing HookAndTorchFunctionTracer obj in TracerTensor
    _current = None

    def __init__(self) -> None:
        self._graph = None
        self._submodule_paths = None

    class TracerTensor(HookAndTorchFunctionHelper.TracerTensor):
        """Patch Tensor for tracing."""

        src: Node

        def __new__(cls, data: Tensor, src: Node):
            instance = super().__new__(cls, data)
            return instance

        def __init__(self, data: Tensor, src: Node):
            super().__init__(data)
            self.src = src

        @classmethod
        def _get_node_args(cls, args):
            """Convert function or module args to fx node args."""
            flatten_args_src = []
            flatten_args, spec = tree_flatten(args)
            for flatten_arg in flatten_args:
                if isinstance(flatten_arg, cls):
                    flatten_args_src.append(flatten_arg.src)
                elif isinstance(flatten_arg, Tensor):
                    flatten_args_src.append("constant tensor")
                else:
                    flatten_args_src.append(flatten_arg)
            return tree_unflatten(flatten_args_src, spec)

        @classmethod
        def _torch_function_postprocess(
            cls, func, types, args, kwargs, func_ret
        ):
            """Postprocess of __torch_function__."""
            flatten_func_ret, _ = tree_flatten(func_ret)
            if not any([isinstance(i, Tensor) for i in flatten_func_ret]):
                # don't record constant operation
                return func_ret

            node = HookAndTorchFunctionTracer._current._graph.create_node(
                "call_function",
                func,
                cls._get_node_args(args),
                cls._get_node_args(kwargs),
            )

            def func(x):
                return cls(x, node)

            func_ret = pytree_convert(func_ret, Tensor, func)

            return func_ret

        def __deepcopy__(self, memo):
            return self.__class__(copy.deepcopy(self._base), self.src)

    def _forward_pre_hook(self, mod, args, kwargs):
        """Implement module forward pre hook."""
        # Do not unwrap TracerTensor in this hook,
        # or we cannot get node in _foward_hook.
        pass

    def _forward_hook(self, mod, args, kwargs, output):
        """Implement module forward hook."""
        flatten_output, _ = tree_flatten(output)
        if not any([isinstance(i, Tensor) for i in flatten_output]):
            # don't record constant operation
            return output

        node = self._graph.create_node(
            "call_module",
            mod._qualified_name,
            self.TracerTensor._get_node_args(args),
            self.TracerTensor._get_node_args(kwargs),
        )

        def func(x):
            return self.TracerTensor(x, node)

        output = pytree_convert(output, Tensor, func)

        return output

    def _record_graph_output(self, output):
        """Implement hook to record output and add output node in graph."""
        output_args = self.TracerTensor._get_node_args(output)
        self._graph.create_node(
            "output",
            "output",
            output_args if isinstance(output_args, tuple) else (output_args,),
        )

    def _example_inputs_preprocess(
        self, example_inputs, example_kw_inputs: dict
    ):
        """Preprocess example inputs before running forward."""
        flatten_inputs, spec = tree_flatten(
            (example_inputs, example_kw_inputs)
        )
        new_flatten_inputs = []
        for i, flatten_input in enumerate(flatten_inputs):
            if isinstance(flatten_input, Tensor):
                new_flatten_input = self.TracerTensor(
                    flatten_input,
                    self._graph.create_node(
                        "placeholder",
                        f"model_input_{i}",
                        None,
                        None,
                    ),
                )
                new_flatten_inputs.append(new_flatten_input)
            else:
                new_flatten_inputs.append(flatten_input)

        return tree_unflatten(new_flatten_inputs, spec)

    @typechecked
    def trace(
        self,
        model: Module,
        example_inputs: Any = None,
        example_kw_inputs: Any = None,
    ) -> Graph:
        """
        Trace model to get graph.

        Args:
            model (Module): The model being traced.
            example_inputs (Any): Inputs used to run model forward.

        Returns:
            Graph: FX graph.
        """
        # Freeze bn when trace.
        def _freeze_bn(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                m.eval()

        model.apply(_freeze_bn)

        old_obj = HookAndTorchFunctionTracer._current
        HookAndTorchFunctionTracer._current = self

        self._graph = Graph(owning_module=model)
        self._submodule_paths = {v: k for k, v in model.named_modules()}

        model_ret = self._register_hook_and_forward(
            model, example_inputs, example_kw_inputs
        )
        self._record_graph_output(model_ret)
        self._graph.eliminate_dead_code()

        HookAndTorchFunctionTracer._current = old_obj

        return self._graph


class CustomTracer(fx.Tracer):
    def __init__(self):
        super().__init__()

    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        return FxWrapManager.is_wrapped_module(m) or super().is_leaf_module(
            m, module_qualified_name
        )

    def create_arg(self, a: Any) -> Argument:
        if a is self.root:
            return self.create_node("get_attr", "_self", (), {})
        if isinstance(a, np.number):
            return a
        return super().create_arg(a)

    def trace(self, root, *args, **kwargs):
        # Freeze bn when trace.
        def _freeze_bn(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                m.eval()

        if isinstance(root, Module):
            root.apply(_freeze_bn)

        FxWrapManager.apply_wrap()

        if isinstance(root, Module):
            swap_nn_with_horizonnn(root)

        if self.is_leaf_module(root, ""):
            self.root = root
            fn = type(root).forward
            if "tracer_cls" in Graph.__init__.__code__.co_varnames:
                self.graph = Graph(tracer_cls=getattr(self, "__class__", None))
            else:
                self.graph = Graph()
            fn, args = self.create_args_for_root(fn, True)
            output = self.create_node(
                "call_method",
                "__call__",
                (self.create_arg(root),) + tuple(arg.node for arg in args[1:]),
                {},
            )
            self.create_node(
                "output",
                "output",
                (output,),
                {},
            )

            return self.graph

        return super(CustomTracer, self).trace(root, *args, **kwargs)
