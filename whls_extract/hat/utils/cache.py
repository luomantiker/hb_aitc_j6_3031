# Copyright (c) Horizon Robotics. All rights reserved.
#
# This file contains core feature cache code.
#
# :class:`Cache` implement read and write io using MXRecord.
# Yet we tried to use Lmdb but found it broken in ddp training, so we
# turn to MXRecord.

import logging
import multiprocessing as mp
import os
from enum import Enum
from typing import Any, Dict, Mapping, Optional

import msgpack
import msgpack_numpy
import torch
import torch.utils.data as data

try:
    from hatbc.workflow.trace import make_traceable
except ImportError:
    make_traceable = property

from hat.utils.apply_func import (
    apply_to_collection,
    convert_numpy,
    convert_tensor,
)
from hat.utils.distributed import get_dist_info
from hat.utils.pack_type import Lmdb, MXRecord
from hat.utils.package_helper import require_packages

__all__ = [
    "get_global_cache",
    "cache_feature_op",
    "Cache",
    "CacheIOType",
]

logger = logging.getLogger(__name__)


# When caching data to file, batch data need to split by batch axis.
# But for some data, it is not layout by batch axis, such as homo_offset,
# where first axis is "view". So skip splitting such data.
KEY_NAME_WITH_NON_BATCH_TENSOR = []


class CacheIOType(Enum):
    Memory = "memory"
    MXRecord = "mxrecord"
    Lmdb = "lmdb"


class Cache(data.Dataset):
    """Cache file io.

    Args:
        method: low level cache io type. Option: memory, rec.
        cache_file: cache file, required when use MXRecord or Lmdb.
        writable: set cache io writable.
        debug: set in debug mode, print key in cache.
        exit_when_write_finish: exit program when write finish.
        batch_length: a dictionary specify the length in one batch
            of each data(along dim 0).
    """

    def __init__(
        self,
        *,
        method: Optional[CacheIOType] = CacheIOType.MXRecord,
        cache_file: Optional[str] = None,
        writable: Optional[bool] = False,
        debug: Optional[bool] = False,
        exit_when_write_finish: Optional[bool] = True,
        batch_length: Optional[Dict] = None,
    ):
        self.debug = debug
        self.exit_when_write_finish = exit_when_write_finish
        self.writable = writable
        self._index = 0
        self.batch_length = batch_length

        if method is CacheIOType.Memory:
            self._cache = {}
        elif method is CacheIOType.MXRecord:
            assert cache_file, "cache_file is required if using MXRecord."
            self._rec = MXRecord(cache_file, writable=writable)
            self._rec.open()
            self._idx = 0
        elif method is CacheIOType.Lmdb:
            # Lmdb cache may broken in ddp. We haven't solve this problem and
            # turn to MXRecord. We keep Lmdb code here for further develop.
            logger.warn("lmdb may broken in ddp training, use rec instead")
            self._lmdb = Lmdb(cache_file, writable=writable)
        else:
            raise NotImplementedError(f"Unknown io type: {method}")

    def write(self, key: int, data: Any) -> None:
        """Write data to cache."""
        if self.debug:
            print(f"writing, key: {key}")

        if hasattr(self, "_cache"):
            self._cache[key] = data
        elif hasattr(self, "_rec"):
            assert self.writable
            assert isinstance(data, dict)
            data = convert_numpy(data)
            self._rec.write(
                key, msgpack.packb(data, default=msgpack_numpy.encode)
            )
        elif hasattr(self, "_lmdb"):
            assert isinstance(data, dict)
            data = convert_numpy(data)
            self._lmdb.write(
                key, msgpack.packb(data, default=msgpack_numpy.encode)
            )
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        return self.read(index)

    def _bytes2str_for_key(self, data):
        """
        Convert bytes to str.

        When get dict data form rec,
        the key is bytes type, we convert to str type recursively.
        """

        if isinstance(data, (list, tuple)):
            return [self._bytes2str_for_key(x) for x in data]

        elif isinstance(data, dict):
            res = {}
            for k, v in data.items():
                if isinstance(k, bytes):
                    res[k.decode("utf-8")] = self._bytes2str_for_key(v)
                else:
                    res[k] = self._bytes2str_for_key(v)
            return res
        else:
            return data

    def read(self, key: int) -> Any:
        """Read data from cache by key."""
        if self.debug:
            print(f"reading, key: {key}")
            print(f"current process: {mp.current_process().name}")

        if hasattr(self, "_cache"):
            assert key in self._cache, f"{key} not found in cache"
            return self._cache[key]
        elif hasattr(self, "_rec"):
            assert not self.writable
            # for rec, key is regard as idx
            idx = key
            data = self._rec.read(idx)
            data = msgpack.unpackb(data, object_hook=msgpack_numpy.decode)
            data = self._bytes2str_for_key(data)
            return data
        elif hasattr(self, "_lmdb"):
            if self._lmdb.env is None:
                self._lmdb.open()
            key_int = hash(key)
            data = self._lmdb.read(key_int)
            data = msgpack.unpackb(data, object_hook=msgpack_numpy.decode)
            data = convert_tensor(data)
            return data
        else:
            raise NotImplementedError

    def _fetch_single_batch_data(self, data, batch_idx, batch_size):
        assert isinstance(data, Mapping)
        res = {}
        for name, value in data.items():
            if value is None:
                continue
            if name in KEY_NAME_WITH_NON_BATCH_TENSOR:
                res[name] = value
            else:
                assert isinstance(value, (Mapping, torch.Tensor))

                if isinstance(value, Mapping):
                    res[name] = self._fetch_single_batch_data(
                        value, batch_idx, batch_size
                    )
                else:
                    if self.batch_length:
                        length_one_batch = self.batch_length.get(name, 1)
                    else:
                        length_one_batch = 1

                    assert value.shape[0] // length_one_batch == batch_size
                    res[name] = value[
                        batch_idx
                        * length_one_batch : (batch_idx + 1)
                        * length_one_batch
                    ]
        return res

    def write_with_index(self, data: Dict[str, Any]) -> None:
        """Save data to self._cache with index.

        "index" is auto count by `self._index`. In future, index can be
        index from dataloader's sampler.

        """
        batch_size = fetch_batch_size(data, self.batch_length)
        for idx in range(batch_size):
            data_i = self._fetch_single_batch_data(data, idx, batch_size)
            key = self._index + idx
            self.write(key, data_i)
        self._index += batch_size

    def is_empty(self) -> bool:
        """Check if cache is empty."""
        if hasattr(self, "_cache"):
            return len(self._cache) == 0
        else:
            raise NotImplementedError

    def clear(self) -> None:
        """Clear cache."""
        if hasattr(self, "_cache"):
            self._cache = {}
        else:
            raise NotImplementedError

    def __contains__(self, key) -> bool:
        """Judge if cache contains key."""
        if hasattr(self, "_cache"):
            return key in self._cache
        else:
            raise NotImplementedError

    def close(self):
        """Close file io."""
        if hasattr(self, "_cache"):
            pass
        elif hasattr(self, "_rec"):
            self._rec.close()
        elif hasattr(self, "_lmdb"):
            self._lmdb.close()
        else:
            raise NotImplementedError

    def __len__(self):
        if hasattr(self, "_cache"):
            return len(self._cache)
        elif hasattr(self, "_rec"):
            rec_io = self._rec.record
            return len(rec_io.idx)
        else:
            raise NotImplementedError


def fetch_batch_size(
    data: Dict[str, Any], batch_length: Optional[dict] = None
) -> int:
    """Fetch batch_size from batch data.

    Some data may stack on batch dimension, so we specify
    the length of one batch of these data. It assumes there
    is at least one 4-dim tensor in data. If not, it will
    raise `NotImplementedError`.

    """
    batch_size = 0

    def find_tensor(item, length_one_batch=1):
        nonlocal batch_size
        if item.ndim == 4:
            batch_size = item.shape[0] // length_one_batch

    assert isinstance(data, Mapping)
    for name, value in data.items():
        if batch_length:
            length_one_batch = batch_length.get(name, 1)
        else:
            length_one_batch = 1
        apply_to_collection(
            value, torch.Tensor, find_tensor, length_one_batch=length_one_batch
        )

    if not batch_size:
        raise NotImplementedError(
            "Unknown data structure, "
            "can't fetch batch_size from it. You need to do some "
            "coding in `fetch_batch_size` function to support it."
        )
    return batch_size


_global_cache = None


_DEBUG = False


def get_global_cache(cache_file, writable, batch_length=None):
    """Get global cache. It will initialize cache at the first time."""
    global _global_cache
    if _global_cache is None:
        # show process ID
        if _DEBUG:
            print(f"[get_global_cache] PID: {mp.current_process().ident}")

        if writable:
            assert not os.path.exists(
                cache_file
            ), f"{cache_file} found, please manually delete first."
            # append rank suffix to cache_file name
            rank, world_size = get_dist_info()
            if world_size > 1:
                cache_file = (
                    os.path.splitext(cache_file)[0] + f"_rank_{rank}.rec"
                )  # noqa
            assert not os.path.exists(
                cache_file
            ), f"{cache_file} found, please manually delete first."

        _global_cache = Cache(
            method=CacheIOType.MXRecord,
            cache_file=cache_file,
            debug=_DEBUG,
            writable=writable,
            batch_length=batch_length,
        )
    return _global_cache


@make_traceable
@require_packages("hatbc")
def cache_feature_op(data, cache_file, batch_length=None):
    """Traceable function to cache feature.

    在ddp模式下，cache_feature_op创建多个cache_file，
    每个进程会独立的将cache写入到对应的cache_file中。
    """
    global_cache = get_global_cache(
        cache_file=cache_file, writable=True, batch_length=batch_length
    )
    global_cache.write_with_index(data)
    return data
