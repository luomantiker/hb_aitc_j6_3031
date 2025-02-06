"""Qat mode specify the way that ConvBN* handle the BN operation."""
import logging
from typing import Dict, List

import torch

from horizon_plugin_pytorch.utils.load_state_dict_helper import get_version

__all__ = [
    "QATMode",
    "set_qat_mode",
    "get_qat_mode",
    "ReLUMode",
    "tricks",
    "handle_relu6_trick",
]

logger = logging.getLogger(__name__)


class QATMode(object):

    WithBN = "with_bn"
    FuseBN = "fuse_bn"
    WithBNReverseFold = "with_bn_reverse_fold"


_qat_mode = QATMode.FuseBN


def set_qat_mode(qat_mode):
    global _qat_mode
    assert qat_mode in [
        QATMode.FuseBN,
        QATMode.WithBN,
        QATMode.WithBNReverseFold,
    ]
    _qat_mode = qat_mode


def get_qat_mode():
    global _qat_mode
    return _qat_mode


class ReLUMode(object):

    FORCE_RELU = "relu_mode_force_relu"
    FORCE_RELU6 = "relu_mode_force_relu6"
    INHERIT = "relu_mode_inherit"


class Tricks:
    def __init__(
        self,
        relu6=ReLUMode.INHERIT,
        fx_force_duplicate_shared_convbn: bool = False,
    ):
        """
        Config some custom behaviour.

        Current support:
        1) relu and relu6 in fused modules
        2) whether force duplicate shared conv-bn pattern when using
            prepare_qat_fx and fuse_fx. Default False.

        Args:
            relu6 (ReLUMode, optional):
                Whether automativally replace relu with relu6 in fused modules.
                ReLUMode.INHERIT, automatically use relu6 for op version < 2;
                ReLUMode.FORCE_RELU, ignore op version and use relu;
                ReLUMode.FORCE_RELU6, ignore op version and use relu6.
                Defaults to ReLUMode.INHERIT.
            fx_force_duplicate_shared_convbn:
                Whether to keep shared conv-bn pattern in float model still
                shared in qat model when using fx. Till now, shared conv-bn
                patterns will separate into standalone convbn modules in qat
                model, which unexpectedly makes qat model different from float
                model when using fx . This arg's default value will be changed
                to False after plugin 1.9.0. If you are not loading your old
                checkpoint, please set this arg False to train your new models.
        """
        self.relu6 = relu6
        self.fx_force_duplicate_shared_convbn = (
            fx_force_duplicate_shared_convbn
        )

    @property
    def relu6(self):
        return self._relu6

    @relu6.setter
    def relu6(self, relu6):
        assert relu6 in (
            ReLUMode.FORCE_RELU,
            ReLUMode.FORCE_RELU6,
            ReLUMode.INHERIT,
        )
        if relu6 == ReLUMode.FORCE_RELU6:
            logger.warning(
                "relu6 trick is deprecated, please"
                " use calibration to deal with outliers",
                extra={"call_times_context": ("message")},
            )
        self._relu6 = relu6

    @property
    def fx_force_duplicate_shared_convbn(self):
        return self._fx_force_duplicate_shared_convbn

    @fx_force_duplicate_shared_convbn.setter
    def fx_force_duplicate_shared_convbn(
        self, fx_force_duplicate_shared_convbn
    ):
        assert isinstance(fx_force_duplicate_shared_convbn, bool)
        self._fx_force_duplicate_shared_convbn = (
            fx_force_duplicate_shared_convbn
        )


tricks = Tricks()


def _init_relu6(obj, *args, **kwargs):
    if tricks.relu6 == ReLUMode.FORCE_RELU6:
        obj.use_relu6 = True
        obj._version = 1
    else:
        obj.use_relu6 = False

    return obj


def handle_relu6_trick(cls: torch.nn.Module):
    assert hasattr(cls, "_version")
    assert cls._version > 1

    old_init = cls.__init__

    def _init_with_relu6(obj, *args, **kwargs):
        ret = old_init(obj, *args, **kwargs)
        _init_relu6(obj, *args, **kwargs)

        return ret

    cls.__init__ = _init_with_relu6

    old_from_float = cls.from_float

    def _from_float_with_relu6(cls, *args, **kwargs):
        ret = old_from_float(*args, **kwargs)
        _init_relu6(ret, *args, **kwargs)
        return ret

    cls.from_float = classmethod(_from_float_with_relu6)

    old_load_from_state_dict = cls._load_from_state_dict

    def _load_from_state_dict_with_relu6(
        obj: torch.nn.Module,
        state_dict: Dict,
        prefix: str,
        local_metadata: Dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):

        missing_key_num = len(missing_keys)
        unexpected_key_num = len(unexpected_keys)

        old_load_from_state_dict(
            obj,
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        if len(missing_keys) == missing_key_num:
            if tricks.relu6 == ReLUMode.INHERIT:
                if get_version(obj, prefix, local_metadata) == 1:
                    obj.use_relu6 = True
                    logger.warning(
                        "relu6 trick is enabled in some ops, "
                        "set tricks.relu6 = ReLUMode.FORCE_RELU "
                        "before load_state_dict to use relu",
                        extra={"call_times_context": ("message")},
                    )

        if not strict:
            while len(missing_keys) > missing_key_num:
                missing_keys.pop()
            while len(unexpected_keys) > unexpected_key_num:
                unexpected_keys.pop()

    cls._load_from_state_dict = _load_from_state_dict_with_relu6

    return cls
