# Copyright (c) Horizon Robotics. All rights reserved.
import contextlib
import copy
import functools
import logging
import os
from collections import OrderedDict
from distutils.version import LooseVersion
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO, Callable, Dict, Optional, Union

import horizon_plugin_pytorch as horizon
import torch
import torch.nn as nn
from horizon_plugin_pytorch import __version__
from torch.nn.parallel import DataParallel, DistributedDataParallel

from hat.utils.bucket import url_to_local_path

try:
    from hat.utils import aidi

    _AIDI_UTILS_IMPORTED = True
except ImportError:
    _AIDI_UTILS_IMPORTED = False

from hat.utils.hash import get_hash_file_if_hashed_and_local
from hat.utils.hooks import load_state_dict_from_url
from .distributed import get_device_count
from .filesystem import get_filesystem
from .logger import MSGColor, OutputLogger, format_msg

try:
    from horizon_plugin_pytorch.utils.serialization import (
        get_version_from_scriptmodule,
    )
except ImportError:
    get_version_from_scriptmodule = None

try:
    from torch import _dynamo as torch_dynamo
except ImportError:
    torch_dynamo = None

__all__ = [
    "load_checkpoint",
    "load_state_dict",
    "update_state_dict_by_add_prefix",
    "update_state_dict_by_strip_prefix",
    "load_script_module",
    "checkpoint_resumable",
]

logger = logging.getLogger(__name__)


def update_state_dict_by_strip_prefix(
    state_dict: Dict, strip_prefix: str = "module."
) -> Dict:
    """
    Strip prefix in state dict keys, used as default state_dict_update_func.

    Args:
        state_dict: Model state dict.
        strip_prefix: The prefix to strip.

    Return:
        state_dict: Processed state dict.

    """
    if list(state_dict.keys())[0].startswith(strip_prefix):
        prefix_len = len(strip_prefix)
        state_dict = {k[prefix_len:]: v for k, v in state_dict.items()}
    else:
        logger.warning(
            "{} is not at the beginning of state dict".format(strip_prefix)
        )
    return state_dict


def update_state_dict_by_add_prefix(
    state_dict: Dict, prefix: str = "module."
) -> Dict:
    """
    Append prefix in state dict keys.

    Args:
        state_dict: Model state dict.
        strip_prefix: The prefix to strip.

    Return:
        state_dict: Processed state dict.

    """
    new_state_dict = {}
    for k in state_dict.keys():
        new_state_dict[prefix + k] = state_dict[k]
    return new_state_dict


default_update_func = functools.partial(
    update_state_dict_by_strip_prefix, strip_prefix="module."
)


def plugin_version_check(saved_version: str):
    if saved_version is not None:
        saved_plugin_version = saved_version.split("+")[0].split(".")
        current_plugin_version = __version__.split("+")[0].split(".")
        if (
            current_plugin_version[0] == saved_plugin_version[0]
            and current_plugin_version[1] == saved_plugin_version[1]
        ):
            pass
        else:
            logger.warning(
                "plugin version in checkpoint is different from "
                "the current plugin version, this may cause changes "
                "in qat and quantized accuracy."
            )
    else:
        logger.warning(
            "The model has not plugin version information, "
            "we can not check plugin version in model."
        )


def pt_plugin_version_check(pt_file: str):
    if get_version_from_scriptmodule is not None:
        pt_version = get_version_from_scriptmodule(pt_file)
        plugin_version_check(pt_version)
    else:
        logger.warning(
            "Please update your horizon-plugin-pytorch. "
            "Plugin version should be >= 0.14.6. If not, "
            "we can not check plugin version in pt file."
        )


def load_script_module(
    pt_path: Union[str, IO, Path],
    map_location: Optional[str] = None,
    _extra_files: Optional[dict] = None,
    check_plugin_version: bool = False,
) -> torch.jit.ScriptModule:
    """Load torch scriptmodule from path.

    Args:
        pt_path: Provided path for torch scriptmodule.
        map_location: Target device for scriptmodule.
        _extra_files: The extra information given in the map.
        check_plugin_version: Whether to check plugin version.

    Returns:
        pt_model: torch.jit.ScriptModule object.
    """
    if LooseVersion(__version__) >= LooseVersion("1.5.0"):
        # horizon.jit.load check compatibility of pt version and current
        # horizon_plugin_pytorch version. It prints warnings if some ops are
        # incompatible in this two versions.
        pt_model = horizon.jit.load(pt_path, map_location, _extra_files)
        return pt_model

    logger.warning(
        "Please upgrade horizon_plugin_pytorch to v1.5.0 or later, which "
        "gives more detailed description for plugin version compatibility."
    )
    import horizon_plugin_pytorch  # noqa F401

    pt_model = torch.jit.load(pt_path, map_location, _extra_files)
    if check_plugin_version:
        pt_plugin_version_check(pt_path)
    return pt_model


def load_checkpoint(
    path_or_dict: Union[str, IO, Path, dict],
    map_location: Optional[str] = None,
    state_dict_update_func: Optional[Callable] = default_update_func,
    check_hash: bool = True,
    check_plugin_version: bool = False,
    enable_tracking: bool = False,
    load_ema_model: bool = False,
) -> Dict:
    """Load checkpoint from path,url or AIDIModel.

    Args:
        path_or_dict : Provided path for state_dict.
            If you want to load checkpoint from AIDIModel,
            the path_or_dict should be named as
            aidi://model_name/model_version/stage.
        map_location: Target device for checkpoint.
        state_dict_update_func: `state_dict` update function.
        check_hash: Whether to check the file hash.
        check_plugin_version: Whether to check plugin version.
        enable_tracking: Whether to enable tracking aidi checkpoint.
    Returns:
        state_dict: State dict for checkpoint.
    """

    if isinstance(path_or_dict, dict):
        checkpoint = path_or_dict
    else:
        path = str(path_or_dict)
        if path.startswith("http://"):
            with contextlib.redirect_stdout(OutputLogger(logger)):
                checkpoint = load_state_dict_from_url(
                    str(path),
                    map_location=map_location,
                    check_hash=check_hash,
                    max_retry=3,
                )
        elif path.startswith("aidi://"):
            assert (
                _AIDI_UTILS_IMPORTED
            ), "import `hat.utils.aidi` failed, will not support path start with 'aidi://'"  # noqa E501

            client = aidi.get_aidi_client()
            _, _, model_name, model_version, stage = path.split(os.sep)
            with TemporaryDirectory(
                "w", dir=os.path.abspath(".")
            ) as output_dir:
                path = client.model.download(
                    output_dir, model_name, model_version, stage
                )
                path = get_hash_file_if_hashed_and_local(
                    path, check_hash=check_hash
                )
                fs = get_filesystem(path)
                with fs.open(path, "rb") as f:
                    checkpoint = torch.load(f, map_location=map_location)

                # aidi tracking checkpoint
            if enable_tracking:
                aidi.tracking_model_as_input(model_name, model_version, stage)
        elif path.startswith("aidi_artifact://"):
            assert (
                _AIDI_UTILS_IMPORTED
            ), "import `hat.utils.aidi` failed, will not support path start with 'aidi_artifact://'"  # noqa E501
            path = aidi.AIDIExperimentLogger.download_checkpoint_from_artifact(
                path, enable_tracking
            )
            fs = get_filesystem(path)
            with fs.open(path, "rb") as f:
                checkpoint = torch.load(f, map_location=map_location)

        else:
            if path.startswith("dmpv2://"):
                path = url_to_local_path(str(path))
            path = get_hash_file_if_hashed_and_local(
                path, check_hash=check_hash
            )
            fs = get_filesystem(path)
            with fs.open(path, "rb") as f:
                checkpoint = torch.load(f, map_location=map_location)
    if load_ema_model is True and "ema_model" in checkpoint:
        checkpoint = checkpoint["ema_model"]

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if state_dict_update_func:
        assert callable(state_dict_update_func), (
            f"{state_dict_update_func} " "is not callable."
        )
        state_dict = state_dict_update_func(state_dict)

    if "state_dict" in checkpoint:
        checkpoint["state_dict"] = state_dict
    else:
        checkpoint = state_dict
    if check_plugin_version:
        plugin_version_check(checkpoint.get("horizon-plugin-version", None))

    return checkpoint


def is_module_wrapper(module):
    module_wrappers = (DataParallel, DistributedDataParallel)
    return isinstance(module, module_wrappers)


def module_load_state_dict(module, state_dict, strict=False):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = (
            {} if metadata is None else metadata.get(prefix[:-1], {})
        )
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break load->load reference cycle

    return all_missing_keys, unexpected_keys, module


def load_state_dict(
    model: nn.Module,
    path_or_dict: Union[dict, str, Path],
    map_location: Optional[str] = None,
    strip_prefix: str = "module.",
    state_dict_update_func: Optional[Callable] = None,
    check_hash: bool = True,
    allow_miss: bool = False,
    ignore_extra: bool = False,
    ignore_tensor_shape: bool = False,
    verbose: bool = False,
) -> nn.Module:
    """
    Load state_dict from file to model.

    Args:
        model: Model for loading checkpoint.
        path_or_dict : Path of checkpoint or state_dict.
        map_location: Target device for checkpoint.
        strip_prefix: The prefix to strip.
        state_dict_update_func: `state_dict` update function. The input
            of the function is a `state_dict`, The output is a modified
            `state_dict` as you want.
        check_hash: Whether to check the file hash.
        allow_miss: Whether to allow missing while loading state dict.
        ignore_extra: Whether to ignore extra while loading state dict.
        ignore_tensor_shape: Whether to ignore matched key name but
            unmatched shape of tensor while loading state dict.
        verbose: Show unexpect_key and miss_key info.
    Returns:
        model: Model with pretrained checkpoint.
    """
    if state_dict_update_func is None:
        state_dict_update_fn = functools.partial(
            update_state_dict_by_strip_prefix, strip_prefix=strip_prefix
        )
    else:
        state_dict_update_fn = state_dict_update_func
    checkpoint = load_checkpoint(
        path_or_dict,
        map_location,
        state_dict_update_func=state_dict_update_fn,
        check_hash=check_hash,
    )
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # TODO(mengyang.duan): check if it can be deleted in torch2.1
    # Note: In torch2.0, `torch.compile(model).state_dict()` will start
    # with `_orig_mod.`, here is for compatibility with load_state_dict.
    is_compiled_model = torch_dynamo and isinstance(
        model, torch_dynamo.eval_frame.OptimizedModule
    )
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if is_compiled_model and not k.startswith("_orig_mod."):
            # for load common `model.state_dict()` to
            # compiled_model (`torch.compile(model)`):
            # add `_orig_mod.` prefix
            new_state_dict["_orig_mod." + k] = v
        elif k.startswith("_orig_mod.") and not is_compiled_model:
            # for load `torch.compile(model).state_dict()` to common `model`:
            # delete `_orig_mod.` prefix
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v

    # Note: `_metadata` may store version information of torch op. It may
    # cause error when load qat checkpoint if static_dict missing `_metadata`.
    if hasattr(state_dict, "_metadata"):
        new_state_dict._metadata = copy.deepcopy(state_dict._metadata)

    state_dict = new_state_dict

    if ignore_tensor_shape:
        miss_key, unexpect_key, model = module_load_state_dict(
            model, state_dict, strict=False
        )
    else:
        miss_key, unexpect_key = model.load_state_dict(
            state_dict, strict=False
        )

    logger.info("state_dict in checkpoint num: {}".format(len(state_dict)))
    logger.info("state_dict in model num: {}".format(len(model.state_dict())))
    logger.warning("miss_key num: {}".format(len(miss_key)))
    if verbose:
        logger.warning("miss_key: {}".format(" ".join(miss_key)))
    logger.warning("unexpect_key num: {}".format(len(unexpect_key)))
    if verbose:
        logger.warning("unexpect_key: {}".format(" ".join(unexpect_key)))

    if len(miss_key) > 0 and not allow_miss:
        raise ValueError("set allow_miss=True to skip this check")
    if len(unexpect_key) > 0 and not ignore_extra:
        raise ValueError("set ignore_extra=True to skip this check")
    return model


def checkpoint_resumable(checkpoint) -> bool:
    checkpoint_num_devices = checkpoint.get("devices", None)
    if checkpoint_num_devices is None:
        logger.warning(
            format_msg(
                "The number of devices is not found in checkpoint.",
                MSGColor.RED,
            )
        )
        return False
    else:
        cur_num_devices = get_device_count()
        return checkpoint_num_devices == cur_num_devices
