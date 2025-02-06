import logging
from typing import Any

import torch.distributed as dist
from packaging import version

import horizon_plugin_pytorch

logger = logging.getLogger(__name__)


def deprecated_warning(
    author: str,
    deprecation_version: str,
    removal_version: str,
    old_name: str,
    new_name: str = None,
    func_name: str = None,
    rank_zero_only: bool = False,
) -> None:
    """Format deprecated warning function.

    Args:
        author: Author name.
        deprecation_version: Document the version at the time of deprecation.
        removal_version: Document the version at the time of remove.
        old_name: the name of func or class or arg that will be deprecated.
        new_name: the name of new func or class or arg can be used to replace
            old one.
        func_name: the name of func that args will be changed.
        rank_zero_only: Whether to log messages only on rank zero.

    """

    if func_name:
        msg = (
            f"[Deprecated] TODO({author}): the arg `{old_name}` of func "
            f"{func_name} was deprecated in {deprecation_version} and will be "
            f"removed in {removal_version}."
        )
    else:
        msg = (
            f"[Deprecated] TODO({author}): `{old_name}` was deprecated in "
            f"{deprecation_version} and will be removed in {removal_version}."
        )
    if new_name:
        msg = f"{msg} Please use `{new_name}` instead."

    if rank_zero_only:
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            logger.warning(msg)
    else:
        logger.warning(msg)
    if version.parse(horizon_plugin_pytorch.__version__) >= version.parse(
        removal_version
    ):
        raise RuntimeError(msg)


def deprecated_access(
    mod: Any,
    deprecated: str,
    author: str,
    deprecation_version: str,
    removal_version: str,
    new_name: str = None,
    rank_zero_only: bool = False,
):
    """Return a wrapped object that warns about deprecated accesses.

    Args:
        mod: Module with deprecated attr.
        deprecated: deprecated attr in the module,
        author: Author name.
        deprecation_version: Document the version at the time of deprecation.
        removal_version: Document the version at the time of remove.
        new_name: the name of new func or class can be used to replace old one.
        new_key: the new key of func or class can be used to replace old one.
        rank_zero_only: Whether to log messages only on rank zero.

    """

    class Wrapper(object):
        def __getattr__(self, attr):
            if attr == deprecated:
                deprecated_warning(
                    author,
                    deprecation_version,
                    removal_version,
                    attr,
                    new_name,
                    None,
                    rank_zero_only,
                )
            return getattr(mod, attr)

        def __setattr__(self, attr, value):
            if attr == deprecated:
                deprecated_warning(
                    author,
                    deprecation_version,
                    removal_version,
                    attr,
                    new_name,
                    None,
                    rank_zero_only,
                )
            return setattr(mod, attr, value)

    return Wrapper()
