# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
import random
import socket
from datetime import timedelta
from functools import wraps
from typing import Any, List, Tuple

import torch
import torch.distributed as dist
import torch.utils.data as data
from torch.distributed import ProcessGroup

__all__ = [
    "find_free_port",
    "get_dist_info",
    "rank_zero_only",
    "get_device_count",
    "get_global_process_group",
    "get_local_host",
    "set_local_process_group",
    "get_local_process_group",
    "split_process_group_by_host",
    "reduce_mean",
    "get_comm_backend_name",
]

logger = logging.getLogger(__name__)

_GLOBAL_PROCESS_GROUP = None
_LOCAL_PROCESS_GROUP = _GLOBAL_PROCESS_GROUP
_HOST_PROCESS_GROUP_CACHED = {}


def find_free_port(start_port: int = 10001, end_port: int = 19999) -> int:
    """Find free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_port = -1
    max_retry = 300
    for _i in range(0, max_retry):
        port = random.randint(start_port, end_port)
        try:
            sock.bind(("", port))
            free_port = port
            sock.close()
            break
        except socket.error:
            continue
    if free_port == -1:
        raise Exception("can not found a free port")
    return free_port


def rank_zero_only(fn):
    """Migrated from pytorch-lightning."""

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


def dist_initialized():
    initialized = False
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
    return initialized


def get_dist_info(process_group=None) -> Tuple[int, int]:
    """Get distributed information from process group."""
    if dist_initialized():
        rank = dist.get_rank(process_group)
        world_size = dist.get_world_size(process_group)
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_worker_info() -> Tuple[int, int]:
    worker_info = data.get_worker_info()
    if worker_info is not None:
        worker_idx = worker_info.id
        num_workers = worker_info.num_workers
    else:
        num_workers = 1
        worker_idx = 0
    return worker_idx, num_workers


def get_device_count(process_group=None):
    """Return the number of GPUs available."""
    initialized = dist_initialized()
    if initialized:
        _, world_size = get_dist_info(process_group)
        return world_size
    else:
        return torch.cuda.device_count()


def create_process_group(rank_list: Tuple[str]) -> Any:
    """Create a new process group by a list of rank id."""
    group = None
    if dist_initialized():
        group = dist.new_group(
            rank_list,
            timeout=timedelta(
                seconds=int(
                    os.environ.get("HAT_PROCESS_GROUP_TIMEOUT", "1800")
                )
            ),
        )
    return group


def get_global_process_group():
    global _GLOBAL_PROCESS_GROUP
    return _GLOBAL_PROCESS_GROUP


def set_local_process_group(process_group):
    global _LOCAL_PROCESS_GROUP
    _LOCAL_PROCESS_GROUP = process_group


def get_local_process_group():
    global _LOCAL_PROCESS_GROUP
    return _LOCAL_PROCESS_GROUP


def get_local_host():
    """Get local host ip."""
    try:
        hostid = socket.gethostname()
        if hostid is None:
            hostid = socket.getfqdn()
            if hostid == "localhost":
                hostid = socket.gethostbyname(hostid)
                if hostid == "0.0.0.0":
                    raise Exception("get host name failed")
        return hostid
    except Exception as e:
        logger.error(str(e))
        return None


def split_process_group_by_host(
    process_group: ProcessGroup = None,
) -> Tuple[ProcessGroup, bool]:
    """Get process group that only contains localhost rank within process group.

    Args:
        process_group: a process_group which contains local rank.
    """
    if not dist_initialized():
        return process_group, False

    global _HOST_PROCESS_GROUP_CACHED
    if process_group in _HOST_PROCESS_GROUP_CACHED:
        # use cached result for process group
        return _HOST_PROCESS_GROUP_CACHED[process_group], True

    # first get host name/ipaddr for current rank
    hostid = get_local_host()
    if hostid is None:
        # get host failed, fallback to origin process group
        return process_group, False

    # 1. first split process group by host
    current_rank, _ = get_dist_info(None)
    _, local_world_size = get_dist_info(process_group)
    # get all rank and hostid within process_group
    local_data = [current_rank, hostid]
    glob_data = [None for _ in range(local_world_size)]
    all_gather_object(glob_data, local_data, process_group)

    # aggregate ranks by hostid
    glob_host_dict = {}
    for gd in glob_data:
        r, host = gd  # gd is [rank, hostid]
        if host not in glob_host_dict:
            glob_host_dict[host] = [r]
        else:
            glob_host_dict[host].append(r)

    # new group ranks is used to create new groups
    # it is [[r1,r2],[r3,r4]] format.
    # each element will be used to create group
    new_group_ranks = list(glob_host_dict.values())
    if len(new_group_ranks) == 1:
        # the process group all in same host
        # so not need to split
        new_group_ranks = []

    # 2. if process group is not global group, then exchange all split info
    if process_group is not None:
        # exchange group info when process group is not default
        _, world_size = get_dist_info(None)
        glob_data = [None for _ in range(world_size)]
        all_gather_object(glob_data, new_group_ranks, None)
        new_group_ranks = []
        for d in glob_data:
            # d is list of rank-list that on same host
            new_group_ranks.extend(d)

    if len(new_group_ranks) == 0:
        # all process group on one host, not need to create new group
        logger.info(
            f"rank {current_rank} same host {hostid} not need to split"
        )
        return process_group, True

    # create new groups
    result_pg = process_group
    for ranks in new_group_ranks:
        npg = create_process_group(ranks)
        if current_rank in ranks:
            logger.info(f"host: {hostid} create new group for ranks: {ranks}")
            result_pg = npg
    _HOST_PROCESS_GROUP_CACHED[process_group] = result_pg
    return result_pg, True


def all_gather_object(obj_list: List[Any], obj: Any, group=None):
    """Gather object from every ranks in group."""
    if dist_initialized():
        dist.all_gather_object(obj_list, obj, group)
    else:
        assert isinstance(obj_list, list) and len(obj_list) == 1
        obj_list[0] = obj


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device.

    Use 'cpu' for CPU to avoid a deprecation warning.

    When torch version <1.9, `torch.nn.parallel.scatter_gather.gather` does not
    support `namedtuple` outputs, but this one does.
    """
    from hat.utils.apply_func import is_namedtuple

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)  # noqa: F821
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError("All dicts must have the same number of keys")
            return type(out)(
                ((k, gather_map([d[k] for d in outputs])) for k in out)
            )
        if is_namedtuple(out):
            return type(out)._make(map(gather_map, zip(*outputs)))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res


def get_global_out(output, default_value=None):
    global_rank, global_world_size = get_dist_info()
    global_output = [default_value for _ in range(global_world_size)]
    all_gather_object(global_output, output)
    return global_rank, global_output


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def get_comm_backend_name():
    """Get the communication backend name."""
    if dist_initialized():
        return dist.get_backend()
    return ""


def reduce_max(tensor: torch.Tensor) -> torch.Tensor:
    """Obtain the max value of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor
