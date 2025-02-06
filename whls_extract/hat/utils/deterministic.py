import functools
import logging
import os
from decimal import localcontext
from inspect import isfunction

import torch
import torch.nn as nn
from horizon_plugin_pytorch import nn as hnn

try:
    from torch.overrides import TorchFunctionMode
except ImportError:
    TorchFunctionMode = object

from hat.utils.apply_func import to_device
from hat.utils.distributed import get_dist_info
from hat.utils.logger import MSGColor, format_msg
from hat.utils.package_helper import check_packages_available

logger = logging.getLogger(__name__)

__all__ = [
    "set_deterministic_level",
    "deterministic_level",
    "maybe_cast_to_deterministic",
    "get_hooked_modules",
    "get_all_modules",
    "get_hooked_ops",
    "get_all_ops",
    "deterministic_summary",
]


hooked_modules_list = set()
modules_list = set()

hooked_ops_list = set()
ops_list = set()


def _get_non_deterministic_modules():
    # see non-deterministic module and ops:
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html  # noqa E501

    _non_deterministic_modules = [
        # non-deterministic module in torch
        nn.AvgPool3d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
        nn.MaxPool3d,
        nn.AdaptiveMaxPool2d,
        nn.FractionalMaxPool2d,
        nn.FractionalMaxPool3d,
        nn.MaxUnpool1d,
        nn.MaxUnpool2d,
        nn.MaxUnpool3d,
        nn.ReflectionPad1d,
        nn.ReflectionPad2d,
        nn.ReflectionPad3d,
        nn.ReplicationPad1d,
        nn.ReplicationPad2d,
        nn.ReplicationPad3d,
        nn.NLLLoss,
        nn.CTCLoss,
        nn.Upsample,
        # non-deterministic module in horizon-plugin-pytorch
        hnn.MultiScaleRoIAlign,
        hnn.qat.MultiScaleRoIAlign,
        hnn.Interpolate,
        hnn.GridSample,
    ]

    deprecate_non_deterministic_modules = []
    if not check_packages_available(
        "horizon_plugin_pytorch>=2.4.6", raise_exception=False
    ):
        deprecate_non_deterministic_modules.append(hnn.qat.Interpolate)
    if not check_packages_available(
        "horizon_plugin_pytorch>2.4.8", raise_exception=False
    ):
        deprecate_non_deterministic_modules.append(hnn.qat.GridSample)
    _non_deterministic_modules.extend(deprecate_non_deterministic_modules)

    return _non_deterministic_modules


non_deterministic_modules = _get_non_deterministic_modules()


non_deterministic_ops = [
    # non-deterministic op in torch
    "torch.nn.functional.grid_sample",
    "torch.histc",
    "torch.nn.functional.interpolate",
    # "torch.nn.functional.cross_entropy",
    "torch.bincount",
    "torch.cumsum",
    "torch.Tensor.cumsum",
    "torch.Tensor.scatter_reduce",
    "torch.Tensor.resize_",
    "torch.Tensor.put_",
    # non-deterministic op in horizon-plugin-pytorch
    "autocasted_interpolate_outer",
    "horizon_plugin_pytorch.nn.interpolate.autocasted_interpolate_outer",
    "horizon_plugin_pytorch.nn.grid_sample.autocasted_grid_sample_outer",
]


def deterministic_level():
    return int(os.getenv("HAT_DETERMINISTIC_LEVEL", "0"))


def set_deterministic_level(level: int = None):
    """Set deterministic level.

    Args:
        level: Deterministic level. Defaults to None.
    """
    if level:
        assert level in [0, 1, 2], (
            f"`deterministic_level` should be one of [0, 1, 2], "
            f"but get {level}."
        )
        os.environ["HAT_DETERMINISTIC_LEVEL"] = str(level)
    else:
        level = deterministic_level()

    if level > 0:
        torch.use_deterministic_algorithms(True)

    if level == 2:
        # set CUBLAS_WORKSPACE_CONFIG for `torch.mm`, `torch.mv`, `torch.bmm`
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html  # noqa E501
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def reset_deterministic_level():
    if os.getenv("HAT_DETERMINISTIC_LEVEL", None):
        del os.environ["HAT_DETERMINISTIC_LEVEL"]


def format_op_name(op):
    if isinstance(op, str):
        return op
    if isinstance(op, torch.nn.Module):
        return "{}.{}".format(op.__module__, op.__class__.__name__)
    if getattr(torch.Tensor, op.__name__, None) is op:
        return "torch.Tensor.{}".format(op.__name__)
    elif getattr(torch, op.__name__, None) is op:
        return "torch.{}".format(op.__name__)
    elif isfunction(op):
        return "{}.{}".format(op.__module__, op.__name__)
    else:
        return str(op)


class DeterministicFunctionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args, kwargs=None):

        global ops_list
        global hooked_ops_list

        func_name = format_op_name(func)

        if kwargs is None:
            kwargs = {}

        if "getset_descriptor" not in str(func):
            ops_list.add(func_name)

        if func_name in non_deterministic_ops:
            hooked_ops_list.add(func_name)

            # move to cpu
            args = to_device(args, device="cpu")
            kwargs = to_device(kwargs, device="cpu")
            ret = func(*args, **kwargs)

            # move back to cuda
            rank, _ = get_dist_info()
            ret = to_device(ret, device=rank)
            return ret
        else:
            return func(*args, **kwargs)


def register_hooks_to_non_deterministic_modules(model: nn.Module):
    """Add hooks to non-deterministic module.

    Note:
        The hook will put non-deterministic modules running on cpu.

    Args:
        model: Model to add hooks.

    """
    global hooked_modules_list
    global modules_list

    def _forward_pre_device_hook(module, input):
        # move module and input to cpu
        module.cpu()
        input = to_device(input, device="cpu")
        return input

    def _forward_device_hook(module, input, output):
        # move output to cuda
        rank, _ = get_dist_info()
        output = to_device(output, device=rank)
        return output

    for name, module in model.named_modules():
        module_name = "{}.{}".format(
            module.__module__, module.__class__.__name__
        )

        if not isinstance(
            module, (nn.Sequential, nn.ModuleDict, nn.ModuleList)
        ):
            modules_list.add(module_name)

        if isinstance(module, tuple(non_deterministic_modules)):
            module.register_forward_pre_hook(_forward_pre_device_hook)
            module.register_forward_hook(_forward_device_hook)
            hooked_modules_list.add((name, module_name))


def maybe_cast_to_deterministic(func):
    """Deterministic wrapper."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if deterministic_level() == 2:
            if check_packages_available(
                "torch>=1.13.0", raise_exception=False
            ):
                mode = DeterministicFunctionMode()
            else:
                msg = "`DeterministicFunctionMode` require torch>=1.13.0, but"
                +f"get {torch.__version__}, will skip hook operator."
                logger.warning(
                    format_msg(
                        msg=msg,
                        color=MSGColor.RED,
                    )
                )
                mode = localcontext()
        else:
            mode = localcontext()

        with mode:
            result = func(*args, **kwargs)
        return result

    return wrapper


def cast_model_to_deterministic(model: nn.Module):
    """Cast non-deterministic op and module in model to deterministic.

    Note:
        Here add hooks to non-deterministic op and non-deterministic module.
        In case, the hooked op and module will run on cpu.

    Args:
        model: Model to add hooks.
    """
    register_hooks_to_non_deterministic_modules(model)


def get_hooked_modules():
    """Get hooked modules in model."""
    global hooked_modules_list
    return hooked_modules_list


def get_all_modules():
    """Get all modules in model."""
    global modules_list
    return modules_list


def get_hooked_ops():
    """Get hooked ops in model."""
    return hooked_ops_list


def get_all_ops():
    """Get all ops in model."""
    return ops_list


def deterministic_summary():
    """Summary deterministic hook information."""
    all_ops, hooked_ops = get_all_ops(), get_hooked_ops()
    all_modules, hooked_modules = get_all_modules(), get_hooked_modules()

    msg = "\n" + "=" * 50 + "DETERMINISTIC HOOK SUMMARY" + "=" * 50 + "\n"
    msg += (
        f"\nAll ops in model: \n{all_ops}\n"
        + f"\nHooked the ops to deterministic: \n{hooked_ops}\n"
    )

    msg += (
        f"\nAll modules in model: \n{all_modules}\n"
        + f"\nHooked the modules to deterministic: \n{hooked_modules}\n"
    )

    logger.info(
        format_msg(
            msg=msg,
            color=MSGColor.GREEN,
        )
    )
