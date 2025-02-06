r"""Extended tracer and wrap of torch.fx.

This file defines a inherit tracer of torch.fx.Tracer and a extended wrap to
allow wrapping of user-defined Module or method, which help users do some
optimization of their own module by torch.fx
"""
import builtins
import inspect
import logging
import string
from functools import partial
from inspect import isclass, isfunction, ismethod
from types import FrameType, FunctionType
from typing import Union

import torch
from torch.fx.node import Node
from torch.nn import Module
from torch.utils._pytree import tree_flatten

__all__ = [
    "wrap",
    "get_supported_method",
    "replace_torch_op",
    "convert_fx_node_name",
    "is_fx_node_name_match_module_name",
    "get_fx_node_input_output",
]


from torch.fx._symbolic_trace import (
    _wrapped_fns_to_patch,
    _wrapped_methods_to_patch,
)

logger = logging.getLogger(__name__)


class FxWrapManager:
    """A class to manage the wraped objs for fx."""

    # A global list of custom-defined modules to be traced as leaf module
    _wrapped_modules_to_patch = []
    # all things to be patched before trace, contain (obj, frame)
    _wrapped_objs_to_patch = set()

    @classmethod
    def _apply_a_wrap(
        cls,
        obj: Union[FunctionType, str, type],
        def_frame: FrameType,
        skip_compile: bool,
    ):
        """
        Patch an object for symbolic trace.

        Args:
            obj (Union[FunctionType, str, type]): Object to be patched.
            def_frame (FrameType): The frame when object is defined.
            skip_compile (bool, optional):
                Whether wrapped obj is skipped in compile, used by
                `horizon_plugin_pytorch.quantization.fx.split_compilable_model
                .split_compilable_model`. Defaults to False.
        """
        from horizon_plugin_pytorch.quantization.fx.split_compilable_model import (  # noqa E501
            _compilable_functions,
            _compilable_methods,
            _compilable_modules,
        )

        if isinstance(obj, str):
            # wrap("sum")
            fn_name = obj

            if not hasattr(builtins, fn_name):
                raise ValueError(
                    "Invalid builtin func name {}".format(fn_name)
                )

            f = def_frame.f_back
            if isinstance(_wrapped_fns_to_patch, dict):
                _wrapped_fns_to_patch[(id(f.f_globals), fn_name)] = f.f_globals
            else:
                _wrapped_fns_to_patch.append((f.f_globals, fn_name))

            if not skip_compile:
                _compilable_functions.add(getattr(builtins, fn_name))

        elif isfunction(obj):
            owner = None
            # ismethod only recognize the method of instance, for example
            # class A:
            #     def func(self):
            #         pass
            # a = A()
            # ismethod(a.func) == True
            # ismethod(A.func) == False
            if ismethod(obj):
                if inspect.isclass(obj.__self__):
                    # for classmethod, obj.__self__ is a class
                    class_entrance = obj.__self__
                else:
                    # for method, obj.__self__ is a instance
                    class_entrance = obj.__self__.__class__

                for cls in inspect.getmro(class_entrance):
                    if obj.__name__ in cls.__dict__:
                        owner = cls
            else:
                owner = getattr(
                    inspect.getmodule(obj),
                    obj.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[
                        0
                    ],
                    None,
                )

            if isclass(owner):
                # class CLASS():
                #     def method(self):
                #         pass
                #
                # wrap(CLASS.method)
                wrapped_class_method = (owner, obj.__name__)
                if wrapped_class_method not in _wrapped_methods_to_patch:
                    _wrapped_methods_to_patch.append(wrapped_class_method)

                if not skip_compile:
                    _compilable_methods[obj.__name__].add(owner)

            else:
                # def func():
                #     pass
                #
                # wrap(func)
                fn_name = obj.__code__.co_name
                f = def_frame.f_back
                if isinstance(_wrapped_fns_to_patch, dict):
                    _wrapped_fns_to_patch[
                        (id(f.f_globals), fn_name)
                    ] = f.f_globals
                else:
                    _wrapped_fns_to_patch.append((f.f_globals, fn_name))

                if not skip_compile:
                    _compilable_functions.add(obj)

        elif isclass(obj):
            assert issubclass(obj, torch.nn.Module)
            # wrap all methods by default
            method_names = set(obj.__dict__.keys())
            # wrap call_module by default
            method_names -= set(torch.nn.Module.__dict__.keys())
            if obj not in cls._wrapped_modules_to_patch:
                cls._wrapped_modules_to_patch.append(obj)

                if not skip_compile:
                    _compilable_modules.add(obj)

            for method_name in method_names:
                maybe_method = getattr(obj, method_name)
                if isfunction(maybe_method):
                    wrapped_method = (obj, method_name)
                    if wrapped_method not in _wrapped_methods_to_patch:
                        _wrapped_methods_to_patch.append(wrapped_method)

                        if not skip_compile:
                            _compilable_methods[method_name].add(obj)
        else:
            raise RuntimeError("Wrap arg must be a str or function or class")

    @classmethod
    def apply_wrap(cls):
        """Apply all recorded wrap actions."""
        while len(cls._wrapped_objs_to_patch) > 0:
            (
                obj,
                frame,
                skip_compile,
            ) = cls._wrapped_objs_to_patch.pop()
            cls._apply_a_wrap(obj, frame, skip_compile)

    @classmethod
    def is_wrapped_module(cls, m: Module) -> bool:
        return isinstance(m, tuple(cls._wrapped_modules_to_patch))

    @classmethod
    def wrap(cls, obj, skip_compile=False, prev_frame=False):
        """Record a wrap action for given object.

        Args:
            obj: Object to be wrapped.
            skip_compile (bool, optional):
                Whether wrapped obj is skipped in compile, used by
                `horizon_plugin_pytorch.quantization.fx.split_compilable_model
                .split_compilable_model`. Defaults to False.
            prev_frame (bool, optional):
                Whether wrapped obj is defined in previous frame.
                Defaults to False.
        """
        if not type(obj) in (str, FunctionType) and not inspect.isclass(obj):
            raise RuntimeError("Wrap arg must be a str or function or class")
        def_frame = inspect.currentframe()
        if prev_frame:
            def_frame = def_frame.f_back
        cls._wrapped_objs_to_patch.add((obj, def_frame, skip_compile))

        return obj


def wrap(skip_compile: bool = False):
    """
    Extend torch.fx.wrap.

    This function can be:

    1. called or used as a decorator on a string to register a builtin
        function as a "leaf function"

    2. called or used as a decorator on a function to register this
        function as a "leaf function"

    3. called or used as a decorator on subclass of torch.nn.Module to
       register this module as a "leaf module", and register all user
       defined method in this class as "leaf method"

    4. called or used as a decorator on a class method to
       register it as "leaf method"

    Args:
        skip_compile (bool, optional):
            Whether wrapped obj is skipped in compile, used by
            `horizon_plugin_pytorch.quantization.fx.split_compilable_model.split_compilable_model`.
            Defaults to False.

    Returns:
        FxWrapManager.wrap: The actural decorator.
    """  # noqa: E501
    if not isinstance(skip_compile, bool):
        logger.warning(
            "wrap usage has been changed, please pass necessary args",
            extra={"call_times_context": ("message")},
        )
        return FxWrapManager.wrap(skip_compile, prev_frame=True)

    return partial(FxWrapManager.wrap, skip_compile=skip_compile)


def get_supported_method():
    """
    Get a mapping from a class to its registered method names.

    Must be called after `FxWrapManager.apply_wrap`.
    """
    ret = {}
    for cls, method_name in _wrapped_methods_to_patch:
        if cls in ret:
            ret[cls].append(method_name)
        else:
            ret[cls] = [method_name]
    return ret


_torch_horizon_op_mapping = {}
_torch_horizon_nn_op_mapping = {}


def replace_torch_op(name, is_nn_op=False):
    """
    Mark a module replaceable with given torch function.

    Args:
        name(str): The name of torch function.
        is_nn_op(bool): Whether the op is under torch.nn.functional
    """

    def wrapper(mod):
        if is_nn_op:
            _torch_horizon_nn_op_mapping[name] = mod
        else:
            _torch_horizon_op_mapping[name] = mod
        return mod

    return wrapper


def convert_fx_node_name(node_name):
    """Convert node name to access the member directly."""
    return node_name.replace("_", ".")


def get_fx_node_input_output(node):
    """Find direct parents and children of a fx node."""
    input_name = []
    if len(node.all_input_nodes) > 0:
        for item in node.all_input_nodes:
            input_name.append(convert_fx_node_name(item.name))

    output_name = []
    if len(list(node.users.keys())) > 0:
        for item in list(node.users.keys()):
            output_name.append(convert_fx_node_name(item.name))

    return input_name, output_name


def is_fx_node_name_match_module_name(node_name, module_name):
    """Check for a match between graph node name and module name."""
    # process: remove all punctuation, everything to lower case
    def process(name):
        return "".join(c for c in name if c not in string.punctuation).lower()

    processed_node_name = process(node_name)
    processed_module_name = process(module_name)

    # module names start with 'module.' in distributed data-parallel
    # training, but fx graph node names don't; check for both
    distributed_node_name = "module." + node_name
    distributed_processed_node_name = "module" + processed_node_name

    return (
        (node_name == module_name)
        or (distributed_node_name == module_name)
        or (processed_node_name == processed_module_name)
        or (distributed_processed_node_name == processed_module_name)
    )


def get_node_used_count(node):
    """Count node used times as users' args.

    Example:
    .. code-block:: python
        x = conv(x)
        x = add(x, x)
        assert get_node_used_count(conv) == 2
    """
    use_count = 0
    for u in node.users:
        flat, _ = tree_flatten((u.args, u.kwargs))
        flat = list(filter(lambda x: isinstance(x, torch.fx.Node), flat))
        use_count += flat.count(node)
    return use_count


def match_node_operation(node, op_list):
    # skip constant operation
    if not any(
        isinstance(arg, Node) and arg.op != "get_attr"
        for arg in tree_flatten((node.args, node.kwargs))[0]
    ):
        return None

    # torch.xxx
    if node.op == "call_function":
        func_name = getattr(node.target, "__name__", None)
        # replace torch.add
        if func_name in op_list:
            return func_name
        # replace torch.add_
        if func_name.strip("_") in op_list:
            return func_name.strip("_")

    # Tensor.xxx and skip FloatFunctional.xxx
    if node.op == "call_method" and node.args[0].op != "get_attr":
        method_name = node.target
        # replace x.add(y)
        if method_name in op_list:
            return method_name
        # replace x.add_(y)
        if method_name.strip("_") in op_list:
            return method_name.strip("_")

    return None
