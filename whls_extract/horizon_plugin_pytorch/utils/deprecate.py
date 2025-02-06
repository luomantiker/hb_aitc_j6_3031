import logging
from functools import wraps
from inspect import getfullargspec
from typing import Any

import torch.distributed as dist
from packaging import version

import horizon_plugin_pytorch

logger = logging.getLogger(__name__)


def display_msg_and_check_version(
    msg: str, rank_zero_only: bool = False, removal_version: str = None
):
    r"""Display given message and check version.

    Display given message using horizon logger and raise error of version check
    fails.

    Args:
        msg: Message to display.
        rank_zero_only: Whether to log messages only on rank zero.
        removal_version: Document the version at the time of remove. Set to
        None if none removal behavior is expected.

    """

    if rank_zero_only:
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            logger.warning(msg)
    else:
        logger.warning(msg)
    if removal_version and version.parse(
        horizon_plugin_pytorch.__version__
    ) >= version.parse(removal_version):
        raise RuntimeError(msg)


def deprecated_interface_warning(
    deprecation_version: str,
    removal_version: str,
    new_interface: Any = None,
    author: str = "horizon",
    rank_zero_only: bool = False,
):
    r"""Warn the deprecated interface.

    Warn the deprecated interface (could be func or class), and replace it
    with new interface if version < removal_version.

    Args:
        deprecation_version: Version since when the args are deprecated.
        removal_version: Version when the args will be removed.
        new_interface: The new interface (could be func or class).
        None if the interface is fully deprecated.
        author: Author name.
        rank_zero_only: Whether to log messages only on rank zero.

    """

    def interface_warning_wrapper(old_interface):
        @wraps(old_interface)
        def wrapper(*args, **kwargs):
            msg = (
                f"[Deprecated] TODO({author}): `{old_interface.__name__}` "
                f"was deprecated since {deprecation_version} and will be "
                f"removed in {removal_version}."
            )
            if new_interface:
                msg = f"{msg} Please use `{new_interface.__name__}` instead."
            display_msg_and_check_version(msg, rank_zero_only, removal_version)

            if new_interface:
                return new_interface(*args, **kwargs)

            return old_interface(*args, **kwargs)

        return wrapper

    return interface_warning_wrapper


def deprecated_args_warning(
    name_dict: dict,
    deprecation_version: str,
    removal_version: str,
    cls_name: str = None,
    author: str = "horizon",
    rank_zero_only: bool = False,
):
    r"""Warn the deprecated arguments.

    Warn the deprecated arguments (could be from func or method),
    and replace them with new names if version < removal_version.

    Args:
        name_dict:
            key (str): Deprecated argument name.
            val (str): Expected argument name. Set to None if fully
            deprecated.
        deprecation_version: Version since when the args are deprecated.
        removal_version: Version when the args will be removed.
        cls_name: Name of class. Useful when you are deprecating arguments in
        a method of a class.
        author: Author name.
        rank_zero_only: Whether to log messages only on rank zero.

    """

    def args_warning_wrapper(old_func):
        @wraps(old_func)
        def new_func(*args, **kwargs):
            # get name of the function
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f"{cls_name}.{func_name}"
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in kwargs:
                        assert (
                            not dst_arg_name or dst_arg_name not in kwargs
                        ), (
                            f"The expected behavior is to replace "
                            f'the deprecated key "{src_arg_name}" to '
                            f'new key "{dst_arg_name}", but got them '
                            f"in the arguments at the same time, which "
                            f'is confusing. "{src_arg_name}" will be '
                            f"deprecated in the future, please "
                            f'use "{dst_arg_name}" instead.'
                        )

                        msg = (
                            f"[Deprecated] TODO({author}): `{src_arg_name}` "
                            f"was deprecated in `{func_name}` since "
                            f"{deprecation_version} and will be removed "
                            f"in {removal_version}."
                        )
                        if dst_arg_name:
                            msg = f"{msg} Please use `{dst_arg_name}` instead."
                        display_msg_and_check_version(
                            msg, rank_zero_only, removal_version
                        )
                        value = kwargs.pop(src_arg_name)
                        if dst_arg_name:
                            kwargs[dst_arg_name] = value

            # apply converted arguments to the decorated method
            return old_func(*args, **kwargs)

        return new_func

    return args_warning_wrapper


def deprecated_default_value_warning(
    old_value_dict: dict,
    deprecation_version: str,
    removal_version: str,
    cls_name: str = None,
    author: str = "horizon",
    rank_zero_only: bool = False,
):
    r"""Warn the change of the default values.

    Warn the change of the default values (could be func or method),
    and replace them will old values if version < removal_version.

    Args:
        old_value_dict:
            key (str): Involved key name.
            val (str): Old default value.
        deprecation_version: Version when the default values are deprecated.
        removal_version: Version when the new default values will take over.
        cls_name: Name of class. Useful when you are deprecating arguments in
        a method of a class.
        author: Author name.
        rank_zero_only: Whether to log messages only on rank zero.

    """

    def args_warning_wrapper(old_func):
        @wraps(old_func)
        def new_func(*args, **kwargs):
            # get name of the function
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f"{cls_name}.{func_name}"
            args_info = getfullargspec(old_func)
            arg_names = args_info.args[: len(args)]
            for key_name, old_value in old_value_dict.items():
                # only warning when user does not give custom value
                if key_name not in arg_names + list(kwargs.keys()):
                    num_args = len(args_info.args) - len(args_info.defaults)
                    new_value = args_info.defaults[
                        args_info.args.index(key_name) - num_args
                    ]
                    msg = (
                        f"[Deprecated] TODO({author}): Default value "
                        f"`{old_value}` for `{key_name}` was deprecated since "
                        f"{deprecation_version} and will be replaced by "
                        f"`{new_value}` in {removal_version}."
                    )
                    display_msg_and_check_version(msg, rank_zero_only)
                    if version.parse(
                        horizon_plugin_pytorch.__version__
                    ) < version.parse(removal_version):
                        kwargs[key_name] = old_value

            # apply converted arguments to the decorated method
            return old_func(*args, **kwargs)

        return new_func

    return args_warning_wrapper


def deprecated_module_attr_warning(
    mod: Any,
    deprecated_attr: str,
    deprecation_version: str = None,
    removal_version: str = None,
    new_attr: str = None,
    msg: str = None,
    author: str = "horizon",
    rank_zero_only: bool = False,
):
    """Return a wrapped object that warns about deprecated accesses.

    Args:
        mod: Module with deprecated attr.
        deprecated_attr: deprecated attr in the module.
        deprecation_version: Document the version at the time of deprecation.
        removal_version: Document the version at the time of remove. Set to
        None if none removal behavior is expected.
        new_attr: the name of the new attr. None if full deprecated.
        msg: Custom message. None if use default msg
        author: Author name.
        rank_zero_only: Whether to log messages only on rank zero.

    """

    if not msg:
        assert removal_version is not None and deprecated_attr is not None, (
            "`deprecated_attr` and `removal_version` must be set "
            "if use default msg!"
        )
        msg = (
            f"`{deprecated_attr}` was deprecated since {deprecation_version} "
            f"and will be removed in {removal_version}."
        )
        if new_attr:
            msg = f"{msg} Please use `{new_attr}` instead."

    msg = f"[Deprecated] TODO({author}): {msg}"

    class Wrapper(object):
        def __getattr__(self, attr):
            if attr == deprecated_attr:
                display_msg_and_check_version(
                    msg, rank_zero_only, removal_version
                )
            return getattr(mod, attr)

        def __setattr__(self, attr, value):
            if attr == deprecated_attr:
                display_msg_and_check_version(
                    msg, rank_zero_only, removal_version
                )
            return setattr(mod, attr, value)

    return Wrapper()


def mark_interface_state(state: str):
    def wrapper(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            logger.warning(
                "{} is in {} state and is subject to changes, we do not "
                "provide any guarantee".format(func, state)
            )
            return func(*args, **kwargs)

        return wrapped_func

    return wrapper
