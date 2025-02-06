import builtins
import copy
import importlib
import inspect
import logging
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from distutils.version import LooseVersion
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.fx.node import Node
from torch.nn import Module
from torch.utils._pytree import tree_flatten
from torch.utils.hooks import RemovableHandle

from horizon_plugin_pytorch import nn as nnf
from horizon_plugin_pytorch._compat import get_unique_devices
from horizon_plugin_pytorch._torchvision_wrapper import ops as nnv
from .location_info import LocationManager
from .misc import pytree_convert
from .typeguard import typechecked

logger = logging.getLogger(__name__)

__all__ = [
    "call_with_hooks",
    "find_leaf_modules",
    "get_device",
    "is_leaf_module",
    "register_hook_on_leaf",
    "has_submodule",
    "_as_tuple",
    "apply_to_collection",
    "attach_qualified_name",
    "remove_qualified_name",
    "HookAndTorchFunctionHelper",
    "get_model_training_state",
    "set_model_training_state",
    "register_forward_hook",
]


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


def get_supported_module_classes(skip_joint_ops=True):
    """Get module classes supported by plugin.

    Args:
        skip_joint_ops (bool, optional): Whether exclude module that composed
            by submodules. Defaults to True.
    """
    from horizon_plugin_pytorch.quantization import quantization_mappings

    supported_module_classes = (
        set(quantization_mappings.get_qat_module_mappings().keys())
        | set(quantization_mappings.get_qat_module_mappings().values())
        | set(quantization_mappings.get_quantized_operator_mappings().keys())
        | set(quantization_mappings.get_quantized_operator_mappings().values())
    )
    if skip_joint_ops:
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


def is_leaf_module(module: torch.nn.Module, skip_joint_ops=True) -> bool:
    """Check if a module is leaf."""
    if type(module) in get_supported_module_classes(skip_joint_ops):
        return True

    if check_qat_leaf_module(module):
        return True

    # to support hybrid device check
    if isinstance(module, torch.quantization.FakeQuantizeBase):
        return True

    if module.__module__.startswith("torch.nn") and not isinstance(
        module,
        (
            torch.nn.Sequential,
            torch.nn.ModuleList,
            torch.nn.ModuleDict,
            torch.nn.DataParallel,
            torch.nn.parallel.DistributedDataParallel,
        ),
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
    registered_ids: List = [],  # noqa B006
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
            if (
                (not to_reg_prefixes and not to_reg_types)
                or (
                    type(module) in to_reg_types
                    or (prefix + name).startswith(to_reg_prefixes)
                )
            ) and id(module) not in registered_ids:
                handler = [None, None, None]
                if forward_hook is not None:
                    handler[0] = module.register_forward_hook(
                        forward_hook, prepend=True
                    )
                if forward_pre_hook is not None:
                    handler[1] = module.register_forward_pre_hook(
                        forward_pre_hook
                    )
                if backward_hook is not None:
                    # do not use full_backward_hook!
                    # it will convert op input QTensor to Tensor
                    handler[2] = module.register_backward_hook(backward_hook)
                handler_dict[prefix + name] = handler
                registered_ids.append(id(module))
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
                    registered_ids=registered_ids,
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
                "{} and {} refer to the same instance, "
                "we will use the former one as module name".format(
                    module._qualified_name, name
                ),
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
        def get_main_type(cls, args, kwargs):
            ret = Tensor
            for x in tree_flatten((args, kwargs))[0]:
                from horizon_plugin_pytorch.tensor_dispatch_wrapper import (
                    DispatchedTensorWrapper,
                )

                if isinstance(x, DispatchedTensorWrapper):
                    return DispatchedTensorWrapper
                if isinstance(x, cls) and type(x._base) is not Tensor:
                    ret = type(x._base)
            return ret

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}

            main_type = cls.get_main_type(args, kwargs)

            from horizon_plugin_pytorch.tensor_dispatch_wrapper import (
                DispatchedTensorWrapper,
            )

            if main_type is DispatchedTensorWrapper:
                return DispatchedTensorWrapper.__torch_function__(
                    func, types, args, kwargs
                )

            ori_args, ori_kwargs = args, kwargs

            func, types, args, kwargs = cls._torch_function_preprocess(
                func, types, args, kwargs
            )

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
            return pytree_convert(trace_tensors, cls, lambda x: x._base)

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

    @classmethod
    def _is_leaf_module(cls, mod):
        """Whether a given Module is a leaf module."""
        return is_leaf_module(mod)

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

    def _example_inputs_preprocess(self, example_inputs, example_kw_inputs):
        """Preprocess example inputs before running forward."""
        return self.TracerTensor.wrap(example_inputs), self.TracerTensor.wrap(
            example_kw_inputs
        )

    @typechecked
    def _register_hook_and_forward(
        self,
        model: Module,
        example_inputs: Any = None,
        example_kw_inputs: Any = None,
    ) -> Any:
        """
        Register hook and run forward.

        Args:
            example_inputs (Any): Inputs used to run model forward.

        """
        if example_inputs is None and example_kw_inputs is None:
            raise ValueError(
                "example_inputs or example_kw_inputs must be provided"
            )

        if example_inputs is None:
            example_inputs = ()
        if example_kw_inputs is None:
            example_kw_inputs = {}

        if not isinstance(example_inputs, (list, tuple)):
            example_inputs = (example_inputs,)

        logger.debug(
            f"[{self.__class__.__name__}] Attaching qualified name ..."
        )
        attach_qualified_name(model)

        logger.debug(f"[{self.__class__.__name__}] Registering hook ...")
        hook_handles = []
        modules_in_leaf_module = {}
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

                hook_handles.extend(hook_handle)

        loc_manager = LocationManager(model)
        # Location hook is registered after the hook of self,
        # so the location in self._forward_pre_hook is not correct!!!
        with loc_manager, torch.no_grad():
            LocationManager.push("")
            logger.debug(f"[{self.__class__.__name__}] Processing inputs ...")
            (
                example_inputs,
                example_kw_inputs,
            ) = self._example_inputs_preprocess(
                example_inputs, example_kw_inputs
            )
            LocationManager.pop()

            logger.debug(
                f"[{self.__class__.__name__}] Running model forward..."
            )
            with self.TracerTensor.hack_isinstance():
                ret = model(*example_inputs, **example_kw_inputs)

        logger.debug(f"[{self.__class__.__name__}] Removing hook ...")
        for handle in hook_handles:
            handle.remove()

        logger.debug(f"[{self.__class__.__name__}] Deleting qualified name...")
        remove_qualified_name(model)

        return ret


class ModelState:
    """A class to store and recover model device and training state."""

    def __init__(self, training, device) -> None:
        self.training = training
        self.device = device

    @classmethod
    def record(cls, model: torch.nn.Module):
        """Record model state as a ModelState instance."""
        devices = get_unique_devices(model)
        if len(devices) != 1:
            device = None
        else:
            device = next(iter(devices))
        return cls(model.training, device)

    def apply(self, model: torch.nn.Module):
        """Apply recorded state to input model."""
        model.train(self.training)
        if self.device is not None:
            model.to(self.device)

        return model


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
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
    )


def register_forward_hook(
    mod: nn.Module,
    hook: Callable,
    *,
    prepend: bool = False,
    with_kwargs: bool = False,
):
    """Support prepend on torch < 2.0."""
    if LooseVersion(torch.__version__) >= LooseVersion("2.0"):
        return mod.register_forward_hook(
            hook, prepend=prepend, with_kwargs=with_kwargs
        )
    else:
        assert not with_kwargs, "with_kwargs is unsupported in torch < 2.0"
        handle = mod.register_forward_hook(hook)
        if prepend:
            mod._forward_hooks.move_to_end(handle.id, last=False)
        return handle
