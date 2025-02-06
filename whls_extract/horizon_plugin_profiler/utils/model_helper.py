import builtins
import copy
import importlib
import inspect
import logging
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from distutils.version import LooseVersion
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Union

from horizon_plugin_profiler.utils.hbdk4_optional import HbirModule
from horizon_plugin_profiler.utils.location_info import LocationManager
from horizon_plugin_profiler.utils.typeguard import typechecked

import torch
from torch import Tensor, nn
from torch.fx.node import Node
from torch.nn import Module
from torch.utils._pytree import tree_flatten
from torch.utils.hooks import RemovableHandle

from horizon_plugin_pytorch import nn as nnf
from horizon_plugin_pytorch._torchvision_wrapper import ops as nnv
from .location_info import TorchLocationInfo

try:
    from horizon_plugin_pytorch.tensor_dispatch_wrapper import (
        DispatchedTensorWrapper,
    )
except ImportError:
    DispatchedTensorWrapper = None

__all__ = [
    "call_with_hooks",
    "find_leaf_modules",
    "get_device",
    "swap_ff_with_horizonff",
    "is_leaf_module",
    "register_hook_on_leaf",
    "has_submodule",
    "_as_tuple",
    "apply_to_collection",
    "attach_qualified_name",
    "remove_qualified_name",
    "pytree_convert",
    "HookAndTorchFunctionHelper",
    "HbirModuleWrapper",
    "get_model_training_state",
    "set_model_training_state",
]

logger = logging.getLogger(__name__)


def get_model_training_state(model: torch.nn.Module):
    """Get the training state of the model.

    Args:
        model: torch.nn.Module
    Returns:
        state: a dict of training state of the model.
    """
    state = {}
    for name, module in model.named_modules():
        state[name] = module.training
    return state


def set_model_training_state(model: torch.nn.Module, state: dict):
    """Set the training state of the model.

    Args:
        model: torch.nn.Module
        state: a dict of training state of the model.
    """
    for name, module in model.named_modules():
        if name in state:
            module.train(state[name])
        else:
            raise ValueError(f"module {name} not found in state")


def call_with_hooks(func):
    r"""Call Module method with hooks."""
    from torch.utils import hooks

    @wraps(func)  # retain owner information
    def _call_impl(mod, *input, **kwargs):
        mod._last_called_method_name = func.__name__

        # copy from module._call_impl
        # Do not call functions when jit is used
        full_backward_hooks = []
        if len(mod._backward_hooks) > 0:
            full_backward_hooks, _ = mod._get_backward_hooks()

        # forward pre
        for hook_id, hook in mod._forward_pre_hooks.items():
            if hasattr(
                mod, "_forward_pre_hooks_with_kwargs"
            ) and mod._forward_pre_hooks_with_kwargs.get(hook_id, False):
                result = hook(mod, input, kwargs)
                if result is not None:
                    if isinstance(result, tuple) and len(result) == 2:
                        input, kwargs = result
                    else:
                        raise RuntimeError(
                            "forward pre-hook must return None or a tuple "
                            f"of (new_args, new_kwargs), but got {result}."
                        )
            else:
                result = hook(mod, input)
                if result is not None:
                    if not isinstance(result, tuple):
                        result = (result,)
                    input = result

        bw_hook = None
        if len(full_backward_hooks) > 0:
            bw_hook = hooks.BackwardHook(mod, full_backward_hooks)
            input = bw_hook.setup_input_hook(input)

        # call func
        result = func(mod, *input, **kwargs)

        for hook_id, hook in mod._forward_hooks.items():
            if hasattr(
                mod, "_forward_hooks_with_kwargs"
            ) and mod._forward_hooks_with_kwargs.get(hook_id, False):
                hook_result = hook(mod, input, kwargs, result)
            else:
                hook_result = hook(mod, input, result)
            if hook_result is not None:
                result = hook_result

        if bw_hook:
            result = bw_hook.setup_output_hook(result)
        return result

    return _call_impl


try:
    from horizon_plugin_pytorch.utils._swap_horizon_float_nn import (
        swap_nn_with_horizonnn,
    )
except ImportError:

    def swap_ff_with_horizonff(model: torch.nn.Module) -> None:
        r"""Refine this docstring in the future.

        Swap torch.nn.quantized.FloatFunctional with
        horizon_plugin_pytorch.nn.quantized.FloatFunctional,
        which is wrapped with horizon fx wrapper
        """
        import horizon_plugin_pytorch

        modules_to_swap = []
        for name, module in model.named_children():
            if isinstance(
                module,
                torch.nn.quantized.FloatFunctional,
            ):
                modules_to_swap.append(name)
            else:
                swap_ff_with_horizonff(module)

        for name in modules_to_swap:
            setattr(
                model,
                name,
                horizon_plugin_pytorch.nn.quantized.FloatFunctional(),
            )


else:
    swap_ff_with_horizonff = swap_nn_with_horizonnn


def get_supported_module_classes():
    from horizon_plugin_pytorch.quantization import quantization_mappings

    supported_module_classes = (
        set(quantization_mappings.get_qat_module_mappings().keys())
        | set(quantization_mappings.get_qat_module_mappings().values())
        | set(quantization_mappings.get_quantized_operator_mappings().keys())
        | set(quantization_mappings.get_quantized_operator_mappings().values())
    )
    joint_qat_ops = set(
        quantization_mappings.get_qat_module_mappings().values()
    ) - set(quantization_mappings.get_quantized_operator_mappings().keys())
    # remove joint qat ops to step into inter small ops
    supported_module_classes -= joint_qat_ops
    return supported_module_classes


# ops like setitem/where is not joint qat op but also not in quantized mapping,
# so add here
def check_qat_leaf_module(module):
    from horizon_plugin_pytorch.quantization import quantization_mappings

    qat_ops = set(quantization_mappings.get_qat_module_mappings().values())
    quantized_ops = set(
        quantization_mappings.get_quantized_operator_mappings().keys()
    )

    if type(module) in qat_ops:
        return (
            False
            if any(
                [
                    type(submod) in quantized_ops
                    for _, submod in module.named_modules()
                ]
            )
            else True
        )

    return False


def is_leaf_module(module: torch.nn.Module) -> bool:
    """Check if a module is leaf."""
    if type(module) in get_supported_module_classes():
        return True

    if check_qat_leaf_module(module):
        return True

    if module.__module__.startswith("torch.nn") and not isinstance(
        module,
        (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict),
    ):
        # unsupported float module should be treated as leaf
        return True

    return False


def find_leaf_modules(
    model: torch.nn.Module,
    check_leaf_module: callable = None,
    prefix: str = "",
):
    if check_leaf_module is None:
        check_leaf_module = is_leaf_module
    leaf_modules = set()

    for name, module in model.named_children():
        if check_leaf_module(module):
            leaf_modules.add(prefix + name)
        else:
            leaf_modules |= find_leaf_modules(
                module, check_leaf_module, prefix + name + "."
            )

    return leaf_modules


def get_device(device):
    """Process multi type of device."""
    if isinstance(device, torch.device) or device is None:
        return device
    elif isinstance(device, (str, int)):
        return torch.device(device)
    else:
        raise ValueError(f"Invalid device {device}")


def register_hook_on_leaf(
    model: torch.nn.Module,
    forward_hook: callable = None,
    forward_pre_hook: callable = None,
    backward_hook: callable = None,
    check_leaf_module: callable = None,
    prefix: str = "",
    to_reg_prefixes: Tuple = (),
    to_reg_types: Tuple = (),
) -> Dict[str, Tuple[RemovableHandle, RemovableHandle]]:
    """
    Register forward_hook and forward_pre_hook on all leaf modules in a model.

    Args:
        model: The input model.
        forward_hook: forward_hook to register. Defaults to None.
        forward_pre_hook: forward_pre_hook to register. Defaults to None.
        backward_hook: backward_hook to register. Defaults to None.
        check_leaf_module: A function to check if a module is leaf. Pass None
            to use pre-defined `is_leaf_module`. Defaults to None.
        prefix: The name of root module, only for internal use. Defaults to "".
        to_reg_prefixes: modules with prefixes names to register hooks.
            Defaults to ().
        to_retg_types: which module types to register hooks. Defaults to ()

    Returns:
        Dict:A mapping from module's qualified name to the handler of
            registered hooks.
    """
    if check_leaf_module is None:
        check_leaf_module = is_leaf_module

    handler_dict = {}

    for name, module in model.named_children():
        if check_leaf_module(module):
            # use type not isinstance to avoid subclass judgement error
            if (not to_reg_prefixes and not to_reg_types) or (
                type(module) in to_reg_types
                or (prefix + name).startswith(to_reg_prefixes)
            ):
                handler = [None, None, None]
                if (
                    forward_hook is not None
                    and forward_hook not in module._forward_hooks.values()
                ):
                    handler[0] = module.register_forward_hook(forward_hook)
                if (
                    forward_pre_hook is not None
                    and forward_pre_hook
                    not in module._forward_pre_hooks.values()
                ):
                    handler[1] = module.register_forward_pre_hook(
                        forward_pre_hook
                    )
                if (
                    backward_hook is not None
                    and backward_hook not in module._backward_hooks.values()
                ):
                    # do not use full_backward_hook!
                    # it will convert op input QTensor to Tensor
                    handler[2] = module.register_backward_hook(backward_hook)
                handler_dict[prefix + name] = handler
        else:
            handler_dict.update(
                register_hook_on_leaf(
                    module,
                    forward_hook,
                    forward_pre_hook,
                    backward_hook,
                    check_leaf_module=check_leaf_module,
                    prefix=prefix + name + ".",
                    to_reg_prefixes=to_reg_prefixes,
                    to_reg_types=to_reg_types,
                )
            )

    return handler_dict


def has_submodule(module: torch.nn.Module, target: str):
    """Check if module has submodule with prefix target."""
    if target == "":
        return True

    atoms = target.split(".")
    mod = module
    for item in atoms:
        if not hasattr(mod, item):
            return False
        mod = getattr(mod, item)
        if not isinstance(mod, torch.nn.Module):
            return False
    return True


def _as_tuple(inputs):
    # Special case for common case of passing a single Tensor
    if isinstance(inputs, (torch.Tensor, dict)):
        inputs = (inputs,)
    # done primarily so that weird iterables fail here and not pybind11 code
    elif not isinstance(inputs, tuple):
        inputs = tuple(inputs)
    return inputs


def apply_to_collection(
    data: Any,
    dtype: Union[type, tuple],
    function: Callable,
    *args,
    wrong_dtype: Optional[Union[type, tuple]] = None,
    **kwargs,
) -> Any:
    """
    Recursively applies a function to all elements of a certain dtype.

    Migrated from pytorch_lightning.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of
            ``function``)
        wrong_dtype: the given function won't be applied if this type is
            specified and the given collections is of the :attr:`wrong_type`
            even if it is of type :attr`dtype`
        **kwargs: keyword arguments (will be forwarded to calls of
            ``function``)

    Returns:
        the resulting collection
    """
    elem_type = type(data)

    # Breaking condition
    if isinstance(data, dtype) and (
        wrong_dtype is None or not isinstance(data, wrong_dtype)
    ):
        return function(data, *args, **kwargs)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        return elem_type(
            {
                k: apply_to_collection(v, dtype, function, *args, **kwargs)
                for k, v in data.items()
            }
        )

    if isinstance(data, tuple) and hasattr(data, "_fields"):  # named tuple
        return elem_type(
            *(
                apply_to_collection(d, dtype, function, *args, **kwargs)
                for d in data
            )
        )

    if isinstance(data, Sequence) and not isinstance(data, str):
        return elem_type(
            [
                apply_to_collection(d, dtype, function, *args, **kwargs)
                for d in data
            ]
        )

    # data is neither of dtype, nor a collection
    return data


def attach_qualified_name(
    model: torch.nn.Module,
    attach_shared_times: bool = False,
) -> None:
    """Attach qualified name to all named modules."""
    for name, module in model.named_modules(remove_duplicate=False):
        if hasattr(module, "_qualified_name"):
            logger.warning(
                f"{module._qualified_name} and {name} refer to the same "
                f"instance, we will use the former one as module name.",
                extra={"call_times_context": ("message")},
            )
        else:
            module._qualified_name = name
            if attach_shared_times:
                module._shared_times = 0


def remove_qualified_name(model: torch.nn.Module) -> None:
    """Remove attached qualified name of all named modules."""
    for _, module in model.named_modules(remove_duplicate=False):
        if hasattr(module, "_qualified_name"):
            del module._qualified_name
        if hasattr(module, "_shared_times"):
            del module._shared_times


def pytree_convert(
    input, convert_type, func, skip_unsupported=True, strict_type=False
):
    """Manipulate the elements in python list/tuple/dict structures.

    Args:
        input: Input structure.
        convert_type: The type of elements to be manipulated.
        func: A function takes a target element and return a manipulated one.
        skip_unsupported: Whether skip unsupported type or raise an exception.
            Defaults to True.
        strict_type: Whether use strict type judjement.
            Defaults to False.

    Returns:
        Same structure as input with manipulated elements.
    """
    if (
        type(input) is convert_type
        if strict_type
        else isinstance(input, convert_type)
    ):
        return func(input)
    elif isinstance(input, (list, tuple)):
        return type(input)(
            pytree_convert(
                x, convert_type, func, skip_unsupported, strict_type
            )
            for x in input
        )
    elif isinstance(input, dict):
        ret = type(input)()
        for k, v in input.items():
            ret[k] = pytree_convert(
                v, convert_type, func, skip_unsupported, strict_type
            )
        for i in dir(input):
            if not i.startswith("__") and not callable(getattr(input, i)):
                # copy attr for dict instance like easydict
                setattr(ret, i, getattr(input, i))
        return ret
    elif skip_unsupported:
        return input
    else:
        raise TypeError("Unsupported input type {}".format(type(input)))


class HookAndTorchFunctionHelper:
    """Helper class of module hook and __torch_function__ of tensor."""

    class TracerTensor(Tensor):
        """Patch Tensor for tracing."""

        def __new__(cls, base: Tensor):
            if isinstance(base, cls):
                base = base._base
            return base.as_subclass(cls)

        def __init__(self, base: Tensor):
            if isinstance(base, type(self)):
                self.__base = base._base
            else:
                self.__base = base

        @classmethod
        def insert_jit_tensor_into_wrapper(cls, dispatched_tensor_wrapper):
            if DispatchedTensorWrapper is not None:
                return DispatchedTensorWrapper.wrap(
                    cls.wrap(
                        DispatchedTensorWrapper.unwrap(
                            dispatched_tensor_wrapper
                        )
                    )
                )
            else:
                return dispatched_tensor_wrapper

        @classmethod
        def del_jit_tensor_in_wrapper(cls, dispatched_tensor_wrapper):
            if DispatchedTensorWrapper is not None:
                return DispatchedTensorWrapper.wrap(
                    cls.unwrap(
                        DispatchedTensorWrapper.unwrap(
                            dispatched_tensor_wrapper
                        )
                    )
                )
            else:
                return dispatched_tensor_wrapper

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}

            ori_args, ori_kwargs = args, kwargs

            func, types, args, kwargs = cls._torch_function_preprocess(
                func, types, args, kwargs
            )

            if DispatchedTensorWrapper is not None and any(
                [
                    isinstance(x, DispatchedTensorWrapper)
                    for x in tree_flatten((args, kwargs))[0]
                ]
            ):
                rewrapped_args = pytree_convert(
                    args,
                    DispatchedTensorWrapper,
                    cls.insert_jit_tensor_into_wrapper,
                )
                rewrapped_kwargs = pytree_convert(
                    kwargs,
                    DispatchedTensorWrapper,
                    cls.insert_jit_tensor_into_wrapper,
                )
                func_ret = DispatchedTensorWrapper.__torch_function__(
                    func, types, rewrapped_args, rewrapped_kwargs
                )
                func_ret = pytree_convert(
                    func_ret,
                    DispatchedTensorWrapper,
                    cls.del_jit_tensor_in_wrapper,
                )
            else:
                func_ret = func(*args, **kwargs)

            func_ret = cls._torch_function_postprocess(
                func, types, ori_args, ori_kwargs, func_ret
            )

            return func_ret

        @property
        def _base(self):
            return self.__base

        @classmethod
        def wrap(cls, tensors):
            return pytree_convert(tensors, Tensor, lambda x: cls(x))

        @classmethod
        def unwrap(cls, trace_tensors):
            return pytree_convert(
                trace_tensors, cls, lambda x: x._base, strict_type=True
            )

        @classmethod
        def _unwrap_to_origin(cls, tensor):
            if tensor.__class__ is cls:
                return cls._unwrap_to_origin(cls.unwrap(tensor))
            if (
                DispatchedTensorWrapper is not None
                and tensor.__class__ is DispatchedTensorWrapper
            ):
                return cls._unwrap_to_origin(
                    DispatchedTensorWrapper.unwrap(tensor)
                )
            return tensor

        @classmethod
        def unwrap_to_origin(cls, tensors):
            return pytree_convert(tensors, Tensor, cls._unwrap_to_origin)

        @classmethod
        def _torch_function_preprocess(cls, func, types, args, kwargs):
            """Preprocess of __torch_function__."""
            args = cls.unwrap(args)
            kwargs = cls.unwrap(kwargs)
            return func, types, args, kwargs

        @classmethod
        def _torch_function_postprocess(
            cls, func, types, args, kwargs, func_ret
        ):
            """Postprocess of __torch_function__."""
            return cls.wrap(func_ret)

        def __deepcopy__(self, memo):
            new_tensor = self.__class__(copy.deepcopy(self._base))
            new_tensor.__dict__ = copy.deepcopy(self.__dict__, memo)
            return new_tensor

        @classmethod
        @contextmanager
        def hack_isinstance(cls):
            builtin_isinstance = builtins.isinstance
            try:

                def jit_isinstance(obj, class_or_tuple):
                    if builtin_isinstance(obj, class_or_tuple):
                        return True
                    elif builtin_isinstance(obj, cls):
                        return builtin_isinstance(obj._base, class_or_tuple)
                    return False

                builtins.isinstance = jit_isinstance
                yield
            finally:
                builtins.isinstance = builtin_isinstance

    def __init__(self, with_stack_info=True):
        self.with_stack_info = with_stack_info

    def _is_leaf_module(self, module: Module):
        """Whether a given Module is a leaf module."""
        return is_leaf_module(module)

    def _forward_pre_hook(self, mod, args, kwargs):
        """Implement module forward pre hook."""
        # Caution: do not get location from this hook!
        return self.TracerTensor.unwrap(args), self.TracerTensor.unwrap(kwargs)

    def _forward_hook(self, mod, args, kwargs, output):
        """Implement module forward hook."""
        return self.TracerTensor.wrap(output)

    def _forward_pre_hook_without_kwargs(self, mod, args):
        """Implement module forward pre hook without kwargs."""
        # Caution: do not get location from this hook!
        rets = self._forward_pre_hook(mod, args, {})
        if rets is not None:
            return rets[0]

    def _forward_hook_without_kwargs(self, mod, args, output):
        """Implement module forward hook without kwargs."""
        return self._forward_hook(mod, args, {}, output)

    def _example_inputs_preprocess(self, example_inputs):
        """Preprocess example inputs before running forward."""
        return self.TracerTensor.wrap(example_inputs)

    @typechecked
    def _register_hook_and_forward(
        self,
        model: Module,
        example_inputs: Any,
    ) -> Any:
        """
        Register hook and run forward.

        Args:
            example_inputs (Any): Inputs used to run model forward.

        """
        self.model = model

        if not isinstance(example_inputs, (list, tuple)):
            example_inputs = (example_inputs,)

        with self:
            logger.debug(
                f"[{self.__class__.__name__}] Running model forward ..."
            )
            ret = model(*example_inputs)

        return ret

    def __enter__(self, *args, **kwargs):
        self.hook_handles = []
        modules_in_leaf_module = {}
        model = self.model
        self.loc_manager = LocationManager(model, self.with_stack_info)

        swap_ff_with_horizonff(model)
        logger.debug(
            f"[{self.__class__.__name__}] Attaching qualified name ..."
        )
        attach_qualified_name(model)

        logger.debug(f"[{self.__class__.__name__}] Registering hook ...")
        # register hook to record module in graph
        for name, mod in model.named_modules():
            if self._is_leaf_module(mod) and mod not in modules_in_leaf_module:
                for _, sub_mod in mod.named_modules():
                    modules_in_leaf_module[sub_mod] = None
                logger.debug(
                    f"[{self.__class__.__name__}] Registering hook on {name}"
                )
                # hook with kwargs is supported after torch 2.0
                if LooseVersion(torch.__version__) >= LooseVersion("2.0"):
                    hook_handle = (
                        mod.register_forward_pre_hook(
                            self._forward_pre_hook,
                            with_kwargs=True,
                            prepend=True,
                        ),
                        mod.register_forward_hook(
                            self._forward_hook,
                            with_kwargs=True,
                        ),
                    )
                else:
                    hook_handle = (
                        mod.register_forward_pre_hook(
                            self._forward_pre_hook_without_kwargs,
                        ),
                        mod.register_forward_hook(
                            self._forward_hook_without_kwargs,
                        ),
                    )
                    # move forward pre hook to the first
                    mod._forward_pre_hooks.move_to_end(
                        hook_handle[0].id, last=False
                    )
                self.hook_handles.extend(hook_handle)

        def input_preprocess_hook(mod, args):
            return self._example_inputs_preprocess(args)

        self.hook_handles.append(
            model.register_forward_pre_hook(input_preprocess_hook)
        )

        # Location hook is registered after the hook of self,
        # so the location in self._forward_pre_hook is not correct!!!
        self.loc_manager.__enter__()

        self.hack_isinstance_contextmanager = (
            self.TracerTensor.hack_isinstance()
        )
        self.hack_isinstance_contextmanager.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self.hack_isinstance_contextmanager.__exit__(None, None, None)
        self.loc_manager.__exit__()
        logger.debug(f"[{self.__class__.__name__}] Removing hook ...")
        for handle in self.hook_handles:
            handle.remove()

        logger.debug(f"[{self.__class__.__name__}] Deleting qualified name...")
        remove_qualified_name(self.model)


class HbirModuleWrapper(Module):
    @typechecked
    def __init__(self, hbir_model: HbirModule) -> None:
        super().__init__()

        from horizon_plugin_pytorch.quantization import hbdk4 as hb4

        self.hbir = hbir_model
        # Always use this func, because hbir.__getitem__ always return new func
        # Callback must be registered on this func
        self.func = self.hbir.functions[0]
        self.native_pytree = self.func.support_pytree
        if not self.native_pytree:
            self.input_flattener = hb4.get_hbir_input_flattener(hbir_model)
            self.output_unflattener = hb4.get_hbir_output_unflattener(
                hbir_model
            )

    def convert_to_cpu(self, input):
        device = [None]

        def to_cpu(x: torch.Tensor):
            device[0] = x.device
            return x.detach().cpu()

        return (
            pytree_convert(input, torch.Tensor, to_cpu, skip_unsupported=True),
            device[0],
        )

    def convert_to_device(self, input, device):
        return pytree_convert(
            input,
            torch.Tensor,
            lambda x: x.to(device),
            skip_unsupported=True,
        )

    def hbdk4_pytree_forward(self, *inputs):
        inputs, device = self.convert_to_cpu(inputs)
        hbir_rets = self.func(inputs)
        return self.convert_to_device(hbir_rets, device)

    def plugin_pytree_forward(self, *inputs):
        inputs, device = self.convert_to_cpu(inputs)
        flat_input = self.input_flattener(inputs)
        hbir_rets = self.func(*flat_input)
        return self.output_unflattener(
            self.convert_to_device(hbir_rets, device)
        )

    def forward(self, *inputs):
        if self.native_pytree:
            return self.hbdk4_pytree_forward(*inputs)
        else:
            return self.plugin_pytree_forward(*inputs)

    def __deepcopy__(self, memo):
        new_hbir = self.hbir.clone()
        new_func = new_hbir[0]

        cls = self.__class__
        new_mod = cls.__new__(cls)
        memo[id(self)] = new_mod
        for k, v in self.__dict__.items():
            if k == "hbir":
                setattr(new_mod, k, new_hbir)
            elif k == "func":
                setattr(new_mod, k, new_func)
            else:
                setattr(new_mod, k, copy.deepcopy(v, memo))
        return new_mod


def get_float_functional_classes():
    from torch.nn.quantized import FloatFunctional as TorchFloatFunctional

    from horizon_plugin_pytorch.nn.qat import FloatFunctional as QATFunctional
    from horizon_plugin_pytorch.nn.quantized import (
        FloatFunctional,
        QFunctional,
    )

    return (TorchFloatFunctional, FloatFunctional, QATFunctional, QFunctional)


def is_ff_node(node: Node, model: torch.nn.Module) -> bool:
    if (
        node.op == "call_method"
        and len(node.args) > 0
        and isinstance(node.args[0], Node)
        and node.args[0].op == "get_attr"
    ):
        try:
            mod = model.get_submodule(node.args[0].target)
            return isinstance(mod, get_float_functional_classes())
        except AttributeError:
            return False
    else:
        return False


def get_fused_classes():
    return tuple(
        cls
        for _, cls in inspect.getmembers(
            importlib.import_module("horizon_plugin_pytorch.nn.intrinsic"),
            inspect.isclass,
        )
    )


def is_fused_ff_node(node: Node, model: torch.nn.Module) -> bool:
    if (
        node.op == "call_method"
        and len(node.args) > 0
        and isinstance(node.args[0], Node)
        and node.args[0].op == "get_attr"
    ):
        try:
            mod = model.get_submodule(node.args[0].target)
            return isinstance(mod, get_fused_classes())
        except AttributeError:
            return False
    else:
        return False


def get_qconfig_useless_modules():
    return (
        nn.Identity,
        nnf.Identity,
        nnf.Interpolate,
        nn.ConstantPad1d,
        nn.ConstantPad2d,
        nn.ConstantPad3d,
        nn.ZeroPad2d,
        nn.ReplicationPad1d,
        nn.ReplicationPad2d,
        nn.ReplicationPad3d,
        nn.ReLU,
        nnv.RoIAlign,
        nnf.GridSample,
        nnf.LookUpTable,
        nn.Upsample,
        nn.UpsamplingNearest2d,
        nn.UpsamplingBilinear2d,
        nnf.MultiScaleRoIAlign,
    )


class ModelStage(Enum):
    FLOAT = "float"
    QAT = "qat"
    HBIR = "hbir"


def get_model_stage(model: torch.nn.Module):
    if isinstance(model, (HbirModule, HbirModuleWrapper)) or any(
        isinstance(mod, (HbirModule, HbirModuleWrapper))
        for _, mod in model.named_modules()
    ):
        return ModelStage.HBIR

    for _, mod in model.named_modules():
        op_name = TorchLocationInfo.format_op_name(mod)
        if "horizon_plugin_pytorch.nn.qat" in op_name:
            return ModelStage.QAT

    return ModelStage.FLOAT


# model from other platform may not support deepcopy, like hbir, pl...
def deepcopy(model):
    try:
        if isinstance(model, HbirModule):
            new_model = model.clone()
        else:
            new_model = copy.deepcopy(model)
    except Exception:
        logger.warning(
            f"Deepcopy {type(model)} failed, use origin model. Please check if"
            " has inplace modification in model."
        )
        new_model = model
    return new_model


# get/set attr in model. Used in find_bad_case and sensitivity
def _get_attr(model, attr):
    attr = attr.split(".")
    target = model
    for name in attr:
        target = getattr(target, name)
    return target


def _set_attr(model, attr, value):
    attr = attr.split(".")
    target = model
    for name in attr[:-1]:
        target = getattr(target, name)
    setattr(target, attr[-1], copy.deepcopy(value))
