# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
import pathlib
import pickle
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Generator, Tuple, Union

import fsspec

logger = logging.getLogger(__name__)

__all__ = [
    "get_filesystem",
    "join_path",
    "file_load",
    "split_file_path",
]


def get_filesystem(path: Union[str, Path]) -> fsspec.AbstractFileSystem:
    """Get fsspec.filesystem from provided path.

       Filesystem can support file, http, hdfs.

    Args:
        path (str or Path): Provided path for filesystem.
    Returns:
        fsspec.filesystem

    """

    path = str(path)
    if "://" in path:
        # use the fileystem from the protocol specified
        # HDFS depends on pyarrow
        # HTTP depends on aiohttp
        return fsspec.filesystem(path.split(":", 1)[0])
    else:
        # use local filesystem
        return fsspec.filesystem("file")


def join_path(
    prefix_path: Union[str, os.PathLike],
    input: Any,
    ignore_key: Union[str, Sequence] = None,
) -> Any:  # noqa: D205,D400
    """
    Join any str objects or os.PathLike objects
    in input with prefix path recursively. When
    ignore_key is not None, stop joining for the
    objects whose key is included in ignore_key.

    Examples::

        >>> import pathlib
        >>> input = dict(
        ...    path1='user1',
        ...    path2=['user2','user3'],
        ...    path3=pathlib.Path('user4'),
        ...    path4={"group1":"user5"},
        ...    v=2)
        ... ignore_key = ["group1"]

        >>> join_path('/home',input,ignore_key)
        OrderedDict([('path1', '/home/user1'),
        ('path2', ['/home/user2', '/home/user3']),
        ('path3', '/home/user4'),
        ('path4', OrderedDict([('group1', 'user5')])),
        ('v', 2)])
    Args:
        prefix_path (str): prefix path.
        input (any): any format path to join.
        ignore_key (Sequence): ignored keys for path join.
    """
    if isinstance(input, (str, os.PathLike)):
        return os.path.join(prefix_path, input)
    elif isinstance(input, Sequence):
        return [join_path(prefix_path, one, ignore_key) for one in input]
    elif isinstance(input, Mapping):
        result = OrderedDict()
        for k, v in input.items():
            if ignore_key is not None and k in ignore_key:
                result[k] = v
            else:
                result[k] = join_path(prefix_path, v, ignore_key)
        return result
    else:
        return input


def pickle_load(data):  # type: ignore
    out = pickle.load(data)
    return out


def file_load(file: str) -> Generator:
    with open(file, "rb") as f:
        while True:
            try:
                yield pickle_load(f)
            except EOFError:
                break


def split_file_path(file_path: str) -> Tuple:
    """Get the dir path, file name and file extension of input file.

    Args:
        file_path: Input file path.

    Returns:
        (dir path, file name and file extension) of input file.
    """
    path = pathlib.Path(file_path)
    return (path.parent, path.name, "".join(path.suffixes))


def check_file_exist_with_warning(
    file_path: str,
    overwrite: bool = True,
):
    """Check whether the file exists and show overwrite result.

    Args:
        file_path: Path of file to check.
        overwrite: Whether to overwrite existed file.
                If True, will delete the old file.
    """
    if os.path.exists(file_path):
        if overwrite:
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(
                    f"Remove `{file_path}` failed, make sure you have "
                    "the permission to manipulate the folder where the files are saved"  # noqa E501
                )
                logger.info(str(e))
        else:
            logger.warning(
                f"File `{file_path}` already exists, set `overwrite=True` "  # noqa E501
                "to delete the old file to avoid unexpected results."
            )
