# Copyright (c) Horizon Robotics. All rights reserved.

import functools
import inspect
import logging
import types
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import torch.nn as nn

try:
    from hatbc.patch import Patcher
    from hatbc.workflow.symbol import Node, Symbol
    from hatbc.workflow.trace import (
        execute_and_trace,
        get_current_graph_tracer,
    )
except ImportError:
    Patcher, Node, Symbol = None, None, None
    execute_and_trace = None
    get_current_graph_tracer = None
from torch.nn import Module

from hat.utils.apply_func import flatten, regroup
from hat.utils.package_helper import require_packages

logger = logging.getLogger(__name__)

__all__ = [
    "TorchModulePatch",
    "merge_symbol_nodes",
    "ModuleShareError",
]


@require_packages("hatbc")
def _make_class_traceable(
    obj, *, return_patch_func_only=False, allow_input_skip=False
):
    """Wrap __call__ function of class `obj` to make it traceable."""
    assert hasattr(  # noqa
        obj, "__call__"
    ), "object should have the __call__ function"  # noqa
    assert isinstance(
        obj.__call__, types.FunctionType
    ), "the __call__ function of object should not be static"

    fn = obj.__call__

    @functools.wraps(fn)
    def _wrap(self, *args, disable_trace=False, **kwargs):
        graph_tracer = get_current_graph_tracer()
        traceable = getattr(self, "_traceable", False)
        if graph_tracer is None or disable_trace or not traceable:
            return fn(self, *args, **kwargs)
        else:
            logger.debug("call execute_and_trace")
            return execute_and_trace(
                self,
                args,
                kwargs,
                __workflow_allow_input_skip__=allow_input_skip,
            )  # noqa

    _wrap.__patched__ = True
    _wrap.__workflow_allow_input_skip__ = allow_input_skip

    if return_patch_func_only:
        return _wrap
    else:
        obj.__call__ = _wrap
        return obj


@dataclass
class _OpID:
    cls: Any
    args: List[Any]
    kwargs: Dict[str, Any]
    obj: Any

    def __eq__(self, __o: object) -> bool:
        if self.cls != __o.cls:
            return False
        if self.args != __o.args:
            return False
        if self.kwargs != __o.kwargs:
            return False
        return True

    def not_equal_reason(self, __o: object) -> Optional[str]:
        if self.cls != __o.cls:
            return "module class not equal! {} != {}".format(self.cls, __o.cls)
        if self.args != __o.args:
            return "module args not equal! {} != {}".format(
                self.args, __o.args
            )
        if self.kwargs != __o.kwargs:
            return "module kwargs not equal! {} != {}".format(
                self.kwargs, __o.kwargs
            )
        return None


class ModuleShareError(RuntimeError):
    def __init__(self, err_msg: str) -> None:
        self.err_msg = err_msg

    def __str__(self):
        return self.err_msg

    def __repr__(self) -> str:
        return self.__str__()


class TorchModulePatch(object):
    """Patch class that change all torch.nn.Module to have more features.

    Module sharing:
        share torch.nn.Module by set `node_name` in module initializer.

    Traceable by hatbc.workflow:
        Set `traceable` in module initializer to make it traceable. default
        is True.
        For traceable module, you can also specify `disable_trace` in call
        function to disable tracing.

    Args:
        key (str, optional): The key of workspace to use. Workspaces with
            the same key share all resources. Defaults to "global".

    """

    _current = None

    _workspaces = {}

    @require_packages("hatbc")
    def __init__(self, key="global") -> None:
        self._old_scope = None
        if key in TorchModulePatch._workspaces:
            resources = TorchModulePatch._workspaces[key]
        else:
            resources = {}
            TorchModulePatch._workspaces[key] = resources
            resources["named_op_instance"] = {}

        self._resources = resources
        self._env = None
        self._patcher = Patcher()
        self.patched_module = set()

        if not getattr(nn.Module, "__patched__", False):
            self._patcher.patch(
                nn.Module,
                "__call__",
                _make_class_traceable(nn.Module, return_patch_func_only=True),
            )

    def clear_named_ops(self) -> None:
        self._resources["named_op_instance"] = {}

    @property
    def named_op_instance(self) -> Dict[str, _OpID]:
        return self._resources["named_op_instance"]

    def patch_module(self, cls):
        # only patch new module
        if not (
            inspect.isclass(cls)
            and issubclass(cls, Module)
            and not hasattr(cls, f"__module_patched_{cls.__name__}")
        ):
            return

        cls_new = cls.__new__

        @functools.wraps(cls.__new__)
        def __new__(_cls, *args, node_name=None, traceable=True, **kwargs):
            ret = None

            _cls_obj = _cls if inspect.isclass(_cls) else _cls.__class__
            if node_name is None:
                if cls_new == Module.__new__:
                    ret = cls_new(_cls_obj)
                else:
                    ret = cls_new(_cls_obj, *args, **kwargs)

            else:
                op_id = _OpID(cls=cls, args=args, kwargs=kwargs, obj=None)
                global_inst = TorchModulePatch.current().named_op_instance
                if node_name in global_inst:
                    if global_inst[node_name] != op_id:
                        raise ModuleShareError(
                            "Cannot share module with node_name={}. Reason: {}".format(  # noqa
                                node_name,
                                global_inst[node_name].not_equal_reason(op_id),
                            )
                        )
                    ret = global_inst[node_name].obj
                else:
                    if cls_new == Module.__new__:
                        ret = cls_new(_cls_obj)
                    else:
                        ret = cls_new(_cls_obj, *args, **kwargs)
            return ret

        cls_init = cls.__init__

        @functools.wraps(cls.__init__)
        def __init__(obj, *args, node_name=None, traceable=True, **kwargs):
            if node_name is None:
                cls_init(obj, *args, **kwargs)
                obj._traceable = False
            else:
                assert isinstance(node_name, str)
                global_inst = TorchModulePatch.current().named_op_instance
                # remember to skip initialization if skipped.
                if node_name not in global_inst:
                    cls_init(obj, *args, **kwargs)
                    global_inst[node_name] = _OpID(
                        cls=obj.__class__, args=args, kwargs=kwargs, obj=obj
                    )
                else:
                    # remember to skip initialization if skipped,
                    # and checck _traceable
                    assert (
                        obj._traceable == traceable
                    ), "traceable param dispatch for node {}".format(node_name)

                obj._traceable = traceable
            obj.__node_name = node_name

        @functools.wraps(cls.__new__)
        def default_new(_cls, *args, **kwargs):
            return super(cls, _cls).__new__(_cls)

        # set default __new__ to prevent error.
        cls.__new__ = default_new

        self._patcher.patch(cls, "__new__", __new__)
        self._patcher.patch(cls, "__init__", __init__)

        self._patcher.patch(cls, f"__module_patched_{cls.__name__}", True)

        self.patched_module.add(cls)

    def patch_all_module(self):
        for cls in self._env.values():
            self.patch_module(cls)

    def __enter__(self):
        self._old_scope = TorchModulePatch._current
        # just set current to new one and does not update any resource.
        TorchModulePatch._current = self
        self._env = inspect.currentframe().f_back.f_globals
        self._patcher.__enter__()
        self.patch_all_module()
        return self

    def __exit__(self, ptype, value, trace):
        self._patcher.__exit__(ptype, value, trace)
        TorchModulePatch._current = self._old_scope

    @staticmethod
    def current():
        return TorchModulePatch._current

    @staticmethod
    def is_active():
        return TorchModulePatch._current is not None


def merge_symbol_nodes(sym1: Symbol) -> Mapping[Node, Node]:
    """Merge nodes whose op and args are identical.

    Can be moved to hatbc.workflow in the future.

    Args:
        sym1 (Symbol): The symbol graph to merge.

    Returns:
        None: _description_
    """
    name2node = OrderedDict()
    name2child = OrderedDict()  # the op input
    name2parent = OrderedDict()  # the op that use this name as input.

    def _fvisit(node: Node):
        """Construct adjacency maps."""
        assert node.name not in name2node
        name2node[node.name] = node
        name2child[node.name] = set(node.inputs)
        if node.name not in name2parent:
            name2parent[node.name] = set()
        for input in node.inputs:
            if input.name not in name2parent:
                name2parent[input.name] = set()
            name2parent[input.name].add(node)

    def node_equal(node1: Node, node2: Node) -> bool:
        # special ops are not comparable.
        if isinstance(node1.op, str) or isinstance(node2.op, str):
            return False
        if node1.op != node2.op:
            return False

        if node1.num_inputs != node2.num_inputs:
            return False

        args1 = flatten(node1.args)[0]
        args2 = flatten(node2.args)[0]
        if len(args1) != len(args2):
            return False

        for in1, in2 in zip(args1, args2):
            if in1 != in2:
                return False

        kwargs1 = node1._kwargs
        kwargs2 = node2._kwargs
        if len(kwargs1) != len(kwargs2):
            return False
        for k in kwargs1.keys():
            if k not in kwargs2:
                return False
            sub_args1 = flatten(kwargs1[k])[0]
            sub_args2 = flatten(kwargs2[k])[0]
            if len(sub_args1) != len(sub_args2):
                return False
            for in1, in2 in zip(sub_args1, sub_args2):
                if in1 != in2:
                    return False

        return True

    def update_node_args_kwargs(node: Node, src: Node, target: Node):
        args, fmts = flatten(node._args)
        args = list(args)
        for i in range(len(args)):
            if args[i] == src:
                args[i] = target
        node._args = regroup(args, fmts)[0]

        kwargs, fmts = flatten(node._kwargs)
        kwargs = list(kwargs)
        for i in range(len(kwargs)):
            if kwargs[i] == src:
                kwargs[i] = target
        node._kwargs = regroup(kwargs, fmts)[0]

        node._inputs.pop(src)
        node._inputs[target] = target

    def node_replace(src: Node, target: Node):
        for parent in name2parent[src.name]:
            # change all parent that use src node, and change it to dst node
            update_node_args_kwargs(parent, src, target)

    sym1.post_order_dfs_visit(_fvisit)
    # post_order_names = list(name2node.keys())
    post_order_nodes = list(name2node.values())
    merged_nodes = {}
    keeped = []
    for node in post_order_nodes:
        need_keep = True
        for v in keeped:
            if node_equal(v, node):
                # just change parant node dependency.
                # no need to change graph info.
                logger.debug(f"replace {node} with {v}")
                node_replace(node, v)
                need_keep = False
                merged_nodes[node] = v
                break
        if need_keep:
            keeped.append(node)

    # check whether last output can be merged
    for i in range(len(sym1._outputs)):
        node = sym1._outputs[i]
        if node in merged_nodes:
            sym1._outputs[i] = merged_nodes[node]

    return merged_nodes
