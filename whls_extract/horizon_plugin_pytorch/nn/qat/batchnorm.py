import torch

from horizon_plugin_pytorch.utils.load_state_dict_helper import (
    replace_mod_name,
)
from .qat_meta import QATModuleMeta


class BatchNorm1d(torch.nn.BatchNorm1d, metaclass=QATModuleMeta):
    @replace_mod_name("bn", None)
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class BatchNorm2d(torch.nn.BatchNorm2d, metaclass=QATModuleMeta):
    @replace_mod_name("bn", None)
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class BatchNorm3d(torch.nn.BatchNorm3d, metaclass=QATModuleMeta):
    @replace_mod_name("bn", None)
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class SyncBatchNorm(torch.nn.SyncBatchNorm, metaclass=QATModuleMeta):
    @replace_mod_name("bn", None)
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
