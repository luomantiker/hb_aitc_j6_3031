import functools
import logging

import torch.distributed as dist

__all__ = ["deprecated_warning"]


logger = logging.getLogger(__name__)


@functools.lru_cache(128)
def _warn_once(logger, msg: str):
    logger.warning(msg)


def _deprecation_warning(msg):
    _warn_once(logger, "\033[31m[Deprecated] " + msg + "\033[0m")


def deprecated_warning(
    author: str,
    deprecation_version: str,
    removal_version: str,
    old_name: str,
    old_key: str = None,
    new_name: str = None,
    new_key: str = None,
    rank_zero_only: bool = False,
) -> None:
    """Format deprecated warning function.

    Args:
        author: Author name.
        deprecation_version: Document the version at the time of deprecation.
        removal_version: Document the version at the time of remove.
        old_name:  the name of func or class that will be deprecated.
        old_key: the key name of func or class that will be deprecated.
        new_name: the name of new func or class can be used to replace old one.
        new_key: the new key of func or class can be used to replace old one.
        rank_zero_only: Whether to log messages only on rank zero.

    """

    msg = f"TODO({author}): "
    msg_tmp = f"`{old_name}` was deprecated in {deprecation_version}"
    msg_tmp = f"{msg_tmp} and will be removed in {removal_version}."
    msg_old_key = ""
    if old_key:
        msg_old_key = f"the key `{old_key}` of "
    msg = f"{msg}{msg_old_key}{msg_tmp}"
    if new_name:
        msg = f"{msg} Please use `{new_name}` instead."
    elif new_key:
        msg = f"{msg} Please use the key `{new_key}` instead."
    if rank_zero_only:
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            _deprecation_warning(msg)
    else:
        _deprecation_warning(msg)
