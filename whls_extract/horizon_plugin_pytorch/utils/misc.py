import os
import traceback
from copy import deepcopy

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten

from horizon_plugin_pytorch.march import (  # noqa: F401
    March,
    get_march,
    set_march,
)
from horizon_plugin_pytorch.qtensor import QTensor


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
    input_list, spec = tree_flatten(input)
    output_list = []
    for x in input_list:
        if (
            type(x) is convert_type
            if strict_type
            else isinstance(x, convert_type)
        ):
            output_list.append(func(x))
        elif skip_unsupported:
            output_list.append(x)
        else:
            raise TypeError("Unsupported input type {}".format(type(x)))

    return tree_unflatten(output_list, spec)


def tensor_struct_repr(input):
    """Convert Tensor structures to str with Tensor meta and ignore values."""
    input_list, spec = tree_flatten(input)
    repr_list = []
    for x in input_list:
        if isinstance(x, Tensor):
            repr_list.append(
                "{}(shape={}, dtype={}, device={})".format(
                    x.__class__.__name__,
                    tuple(x.shape),
                    x.dtype,
                    x.device,
                )
            )
        else:
            repr_list.append(str(x))
    return str(tree_unflatten(repr_list, spec))


def to_device(data, device="cpu"):
    def _to_device(x: Tensor):
        if isinstance(x, QTensor):
            return QTensor(
                x.as_subclass(Tensor).to(device),
                x.q_scale().to(device) if x.q_scale() is not None else None,
                x.dtype,
                x.per_channel_axis,
            )
        elif isinstance(x, Tensor):
            return x.to(device)
        else:
            raise NotImplementedError

    return pytree_convert(
        data,
        (Tensor,),
        _to_device,
        skip_unsupported=True,
    )


def copy_module_attrs(
    from_module: torch.nn.Module,
    to_module: torch.nn.Module,
    deep: bool = False,
):
    for attr, value in from_module.__dict__.items():
        to_module.__setattr__(attr, deepcopy(value) if deep else value)


def is_called_by_plugin(step_back=2) -> bool:
    """Whether current func is called internal.

    This is implemented by trace back through currnet stack, and judje if
    the code of any frame is inside plugin install path.

    Args:
        step_back (int, optional): Igore the first n frame in stack trace back.
            Defaults to 2.

    Example:
    .. code-block:: python

        # step_back = 2 to judje if func_a is called inside plugin
        def func_a():
            is_called_by_plugin(step_back)
        # step_back = 3 to judje if func_b is called inside plugin
        def func_b():
            func_a()
        # step_back = 4 to judje if func_c is called inside plugin
        def func_c():
            func_b()

    """
    import horizon_plugin_pytorch

    plugin_path = os.path.abspath(
        os.path.dirname(horizon_plugin_pytorch.__file__)
    )

    stacks = traceback.extract_stack()
    for i, stack in enumerate(reversed(stacks)):
        if i == step_back:
            source_path = os.path.abspath(stack.filename)
            return source_path.startswith(plugin_path)

    return True


def check_march(msg):
    march = get_march()
    assert march is not None, msg
