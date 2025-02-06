# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Callable, Optional

from hat.registry import OBJECT_REGISTRY
from hat.utils.checkpoint import load_checkpoint, load_state_dict
from hat.utils.global_var import set_value
from hat.utils.logger import MSGColor, format_msg
from .converters import BaseConverter

__all__ = ["LoadCheckpoint", "LoadMeanTeacherCheckpoint"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class LoadCheckpoint(BaseConverter):
    """Load the checkpoint from file to model and return the checkpoint.

    LoadCheckpoint usually happens before or after BaseConverter.It means
    the model needs to load parameters before or after BaseConverter.

    Args:
        checkpoint_path: Path of the checkpoint file.
        state_dict_update_func: `state_dict` update function. The input
            of the function is a `state_dict`, The output is a modified
            `state_dict` as you want.
        check_hash: Whether to check the file hash.
        allow_miss: Whether to allow missing while loading state dict.
        ignore_extra: Whether to ignore extra while loading state dict.
        ignore_tensor_shape: Whether to ignore matched key name but
            unmatched shape of tensor while loading state dict.
        verbose: Show unexpect_key and miss_key info.
        enable_tracking: whether enable tracking checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: str,
        state_dict_update_func: Optional[Callable] = None,
        check_hash: bool = True,
        allow_miss: bool = False,
        ignore_extra: bool = False,
        ignore_tensor_shape: bool = False,
        verbose: bool = False,
        enable_tracking: bool = False,
        load_ema_model: bool = False,
    ):
        super(LoadCheckpoint, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.state_dict_update_func = state_dict_update_func
        self.check_hash = check_hash
        self.allow_miss = allow_miss
        self.ignore_extra = ignore_extra
        self.ignore_tensor_shape = ignore_tensor_shape
        self.verbose = verbose
        self.enable_tracking = enable_tracking
        self.load_ema_model = load_ema_model

    def __call__(self, model):
        if self.checkpoint_path is None:
            logger.info(
                format_msg(
                    f"Checkpoint path is None, skip.",  # noqa E501
                    MSGColor.RED,
                )
            )
            return model

        model_checkpoint = load_checkpoint(
            path_or_dict=self.checkpoint_path,
            map_location="cpu",
            state_dict_update_func=self.state_dict_update_func,
            check_hash=self.check_hash,
            enable_tracking=self.enable_tracking,
            load_ema_model=self.load_ema_model,
        )
        set_value("model_checkpoint", model_checkpoint)
        model = load_state_dict(
            model,
            path_or_dict=model_checkpoint,
            allow_miss=self.allow_miss,
            ignore_extra=self.ignore_extra,
            ignore_tensor_shape=self.ignore_tensor_shape,
            verbose=self.verbose,
        )
        logger.info(
            format_msg(
                f"Load the checkpoint successfully from {self.checkpoint_path}",  # noqa E501
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class LoadMeanTeacherCheckpoint(BaseConverter):
    """Load the Mean-teacher model checkpoint.

    student and teacher model have same structure.
    LoadMeanTeacherCheckpoint usually happens before or after BaseConverter.
    It means the model needs to load parameters before or after BaseConverter.

    Args:
        checkpoint_path: Path of the checkpoint file.
        state_dict_update_func: `state_dict` update function. The input
            of the function is a `state_dict`, The output is a modified
            `state_dict` as you want.
        check_hash: Whether to check the file hash.
        allow_miss: Whether to allow missing while loading state dict.
        ignore_extra: Whether to ignore extra while loading state dict.
        verbose: Show unexpect_key and miss_key info.
    """

    def __init__(
        self,
        checkpoint_path: str,
        strip_prefix: str = "module.",
        state_dict_update_func: Optional[Callable] = None,
        check_hash: bool = True,
        allow_miss: bool = False,
        ignore_extra: bool = False,
        verbose: bool = False,
    ):
        super(LoadMeanTeacherCheckpoint, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.strip_prefix = strip_prefix
        self.state_dict_update_func = state_dict_update_func
        self.check_hash = check_hash
        self.allow_miss = allow_miss
        self.ignore_extra = ignore_extra
        self.verbose = verbose

    def __call__(self, model):
        if self.checkpoint_path is None:
            logger.info(
                format_msg(
                    f"Checkpoint path is None, skip.",  # noqa E501
                    MSGColor.RED,
                )
            )
            return model

        model_checkpoint = load_checkpoint(
            path_or_dict=self.checkpoint_path,
            map_location="cpu",
            state_dict_update_func=self.state_dict_update_func,
            check_hash=self.check_hash,
        )
        set_value("model_checkpoint", model_checkpoint)
        model = load_state_dict(
            model,
            path_or_dict=model_checkpoint,
            strip_prefix=self.strip_prefix,
            state_dict_update_func=self.state_dict_update_func,
            allow_miss=self.allow_miss,
            ignore_extra=self.ignore_extra,
            verbose=self.verbose,
        )
        logger.info(
            format_msg(
                f"Load the checkpoint successfully from {self.checkpoint_path}",  # noqa E501
                MSGColor.GREEN,
            )
        )
        return model
