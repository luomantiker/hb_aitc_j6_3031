import logging
import sys
from distutils.version import LooseVersion
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.fx import Graph, Node
from torch.utils._pytree import tree_flatten

from horizon_plugin_pytorch.utils.location_info import get_user_stack_info
from horizon_plugin_pytorch.utils.misc import pytree_convert
from horizon_plugin_pytorch.utils.model_helper import is_leaf_module
from .qtensor import QTensor

logger = logging.getLogger(__name__)


class Dispatcher:
    """A class to dispatch func call on DispatchedTensorWrapper."""

    _current = None
    _strict_mode = False

    def __init__(self) -> None:
        super().__init__()
        # A mapping from torch func to its registered impls,
        # the dict value is a sequantial of callable, which will be
        # used in order.
        self.call_seq: Dict[Any, Union[List[Any], Tuple[Any, ...]]] = {}
        # A mapping from torch func to its called times during self
        # is activated.
        self.called_count: Dict[Any, int] = {}
        self._current_mod = None

    def enter(self, owner_mod):
        """Activate self to handle func call on DispatchedTensorWrapper."""
        self.old_obj = Dispatcher._current
        Dispatcher._current = self
        self._current_mod = owner_mod

    def exit(self):
        """Exit current instance and recover the old one (if exist)."""
        if not self._strict_mode:
            for k in self.called_count:
                if self.called_count[k] != len(self.call_seq[k]):
                    logger.warning(
                        "Forward graph varied after tracing! Func {} is called"
                        " {} times in tracing time, but seen {} in "
                        "last forward, set {}._strict_mode=True to get more "
                        "accurate exception info".format(
                            k,
                            len(self.call_seq[k]),
                            self.called_count[k],
                            "{}.{}".format(
                                self.__module__, self.__class__.__name__
                            ),
                        )
                    )
        else:
            msg = (
                "Following func in traced graph is not called"
                " in current forward:\n"
            )
            should_raise = False
            for k in self.called_count:
                if self.called_count[k] != len(self.call_seq[k]):
                    should_raise = True
                    for _, func_info in self.call_seq[k][
                        self.called_count[k] :
                    ]:
                        msg += "Function {}\n{}\n".format(k, func_info)
            if should_raise:
                raise RuntimeError(msg)

        for k in self.called_count:
            self.called_count[k] = 0

        Dispatcher._current = self.old_obj

    def add_call(self, func, dispatch_to, func_info=""):
        """Add one func call to dispatch.

        Args:
            func (callable): Torch func to be handled.
            dispatch_to (callable): Custom impl to dispatch to.
                Must accept `current_mod` as first argument to get generated
                module from it.
            func_info (str, optional): Info about the func, such as related
                python code.
                Defaults to "".
        """
        if func in self.call_seq:
            self.call_seq[func].append((dispatch_to, func_info))
        else:
            self.call_seq[func] = [(dispatch_to, func_info)]
            self.called_count[func] = 0

    def _call(self, func, args, kwargs):
        index = self.called_count[func]
        if index > len(self.call_seq[func]) - 1:
            raise RuntimeError(
                "Forward graph varied after tracing! Func {} is called"
                " {} times in tracing time, but trying to call the {}th"
                " times".format(func, len(self.call_seq[func]), index + 1),
                self.call_seq[func],
            )
        dispatch_to, recorded_func_info = self.call_seq[func][index]

        if self._strict_mode:
            if get_user_stack_info() != recorded_func_info:
                raise RuntimeError(
                    "Current call stack is different with recorded call stack "
                    "in traced graph, which is:\n{}".format(recorded_func_info)
                )

        self.called_count[func] = index + 1

        return dispatch_to(self._current_mod, *args, **kwargs)

    @classmethod
    def current(cls):
        return cls._current

    @classmethod
    def is_handled_func(cls, func):
        return cls._current is not None and func in cls._current.call_seq

    @classmethod
    def call(cls, func, args, kwargs):
        """Handle one call to origin torch func.

        Input argumants is same as Tensor.__torch_function__
        """
        return cls._current._call(func, args, kwargs)


def _is_call_ordinal_mod(n: Node):
    if n.op == "call_module" and not n.target.split(".")[-1].startswith(
        "_generated_"
    ):
        return n.target
    if (
        n.op == "call_method"
        and n.args[0].op == "get_attr"
        and not n.args[0].target.split(".")[-1].startswith("_generated_")
    ):
        return n.args[0].target
    return ""


def _is_call_generated_mod(n: Node):
    if n.op == "call_module" and n.target.split(".")[-1].startswith(
        "_generated_"
    ):
        return n.target
    if (
        n.op == "call_method"
        and n.args[0].op == "get_attr"
        and n.args[0].target.split(".")[-1].startswith("_generated_")
    ):
        return n.args[0].target
    return ""


def _search_for_prev_ordinal_mods(n: Node):
    mod_name_list = []
    for prev_node in tree_flatten((n.args, n.kwargs))[0]:
        if not isinstance(prev_node, Node):
            continue
        mod_name = _is_call_generated_mod(prev_node)
        if mod_name:
            continue
        mod_name = _is_call_ordinal_mod(prev_node)
        if mod_name:
            mod_name_list.append(mod_name)
        else:
            mod_name_list += _search_for_prev_ordinal_mods(prev_node)
    return mod_name_list


def _search_for_next_ordinal_mods(n: Node):
    mod_name_list = []
    for next_node in n.users:
        mod_name = _is_call_generated_mod(next_node)
        if mod_name:
            continue
        mod_name = _is_call_ordinal_mod(next_node)
        if mod_name:
            mod_name_list.append(mod_name)
        else:
            mod_name_list += _search_for_next_ordinal_mods(next_node)
    return mod_name_list


def _get_mod_before_after_generated_mod(graph: Graph):
    mod_before_generated_mod = []
    mod_after_generated_mod = []
    mod_after_mod_before_generated_mod = []

    for n in graph.nodes:
        if _is_call_generated_mod(n):
            mod_before_generated_mod += _search_for_prev_ordinal_mods(n)
            mod_after_generated_mod += _search_for_next_ordinal_mods(n)

    for n in graph.nodes:
        if _is_call_ordinal_mod(n) in mod_before_generated_mod:
            mod_after_mod_before_generated_mod += (
                _search_for_next_ordinal_mods(n)
            )

    return set(mod_before_generated_mod), set(
        mod_after_generated_mod + mod_after_mod_before_generated_mod
    )


def _get_called_mods(graph: Graph):
    mod_name_list = []
    for n in graph.nodes:
        if n.op == "call_module":
            mod_name_list.append(n.target)
        if n.op == "call_method" and n.args[0].op == "get_attr":
            mod_name_list.append(n.args[0].target)

    return mod_name_list


# Must subclass Tensor to support python operators such as `+`
class DispatchedTensorWrapper(Tensor):
    def __new__(cls, t: Tensor):
        return t.as_subclass(cls)

    def __init__(self, t: Tensor):
        self._t = t

    @classmethod
    def _wrap(cls, x):
        if isinstance(x, cls):
            return x
        elif isinstance(x, QTensor):
            return cls(x)
        elif x.dtype.is_floating_point:
            return cls(x)
        else:
            return x

    @classmethod
    def wrap(cls, inputs):
        return pytree_convert(
            inputs,
            Tensor,
            cls._wrap,
        )

    @classmethod
    def unwrap(cls, inputs):
        return pytree_convert(
            inputs, DispatchedTensorWrapper, lambda x: x._t, strict_type=True
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        try:
            args = DispatchedTensorWrapper.unwrap(args)
            kwargs = DispatchedTensorWrapper.unwrap(kwargs)

            if Dispatcher.is_handled_func(func):
                rets = Dispatcher.call(func, args, kwargs)
            else:
                rets = func(*args, **kwargs)

            return DispatchedTensorWrapper.wrap(rets)
        except Exception as e:
            # Convert ValueError to RuntimeError, or torch will convert
            # ValueError to 'TypeError: Unsupported operation ...'
            raise RuntimeError(
                str(e.args[0]) if len(e.args) > 0 else str(e)
            ).with_traceback(sys.exc_info()[-1])

    @classmethod
    def tensor_wrapper_forward_pre_hook_with_kw(cls, mod, args, kwargs):
        return cls.unwrap(args), cls.unwrap(kwargs)

    @classmethod
    def tensor_wrapper_forward_pre_hook(cls, mod, args):
        return cls.unwrap(args)

    @classmethod
    def tensor_wrapper_forward_hook(cls, mod, args, output):
        return cls.wrap(output)

    @classmethod
    def _wrap_model_forward(
        cls, mod: torch.nn.Module, pre_process=True, post_process=True
    ):
        hook_handles = []
        if pre_process:
            if LooseVersion(torch.__version__) >= LooseVersion("2.0"):
                hook_handles.append(
                    mod.register_forward_pre_hook(
                        cls.tensor_wrapper_forward_pre_hook_with_kw,
                        with_kwargs=True,
                    )
                )
            else:
                logger.warning(
                    "Torch >= 2.0 is need to handle DispatchedTensorWrapper "
                    "in kwargs, for torch < 2.0, do not pass tensor input"
                    " as kwargs"
                )
                hook_handles.append(
                    mod.register_forward_pre_hook(
                        cls.tensor_wrapper_forward_pre_hook
                    )
                )

        if post_process:
            hook_handles.append(
                mod.register_forward_hook(cls.tensor_wrapper_forward_hook)
            )
        return hook_handles

    @classmethod
    def decorate_with_tensor_wrapper(
        cls, model: torch.nn.Module, graph: Optional[Graph] = None
    ):
        hook_handles = []
        if graph is None:
            # Decorate all leaf modules by default.
            for _, mod in model.named_children():
                if is_leaf_module(mod, False):
                    hook_handles += cls._wrap_model_forward(
                        mod,
                        post_process=mod.__class__.__name__ != "DeQuantStub",
                    )
                else:
                    cls.decorate_with_tensor_wrapper(mod)
        else:
            # If graph is given, only decorate necessary mods.
            (
                mod_before_generated_mod,
                mod_after_generated_mod,
            ) = _get_mod_before_after_generated_mod(graph)
            for mod_name in mod_before_generated_mod:
                hook_handles += cls._wrap_model_forward(
                    model.get_submodule(mod_name), False, True
                )
            for mod_name in mod_after_generated_mod:
                hook_handles += cls._wrap_model_forward(
                    model.get_submodule(mod_name), True, False
                )
            # Decorate qat mods not in graph, in case they are
            # only for training.
            called_mods = _get_called_mods(graph)
            for name, mod in model.named_children():
                if (
                    is_leaf_module(mod, False)
                    and hasattr(mod, "qconfig")
                    and name not in called_mods
                ):
                    hook_handles += cls._wrap_model_forward(
                        mod,
                        post_process=mod.__class__.__name__ != "DeQuantStub",
                    )

        return hook_handles
