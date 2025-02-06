# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from abc import ABC, abstractmethod
from typing import List

import torch.nn as nn
from horizon_plugin_pytorch.qat_mode import QATMode, get_qat_mode
from torch.quantization.fake_quantize import FakeQuantizeBase

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import flatten, is_list_of_type
from hat.utils.elastic import elastic_need_resume
from hat.utils.model_helpers import (
    fuse_norm_recursively,
    get_binding_module,
    has_normalization,
)
from .callbacks import CallbackMixin

__all__ = ["FreezeModule", "FuseBN"]

logger = logging.getLogger(__name__)


class OnlineModelTrick(CallbackMixin, ABC):
    """Base class for dynamic model trick."""

    def __init__(
        self,
        modules: List[List[str]],
        step_or_epoch: List[int],
        update_by: str,
    ):
        self.modules = modules
        self.step_or_epoch = step_or_epoch
        assert is_list_of_type(modules, list), f"{modules} is not list of list"
        assert is_list_of_type(
            step_or_epoch, int
        ), f"{step_or_epoch} is not list of int"
        assert len(modules) == len(step_or_epoch)

        self.update_by = update_by
        assert update_by in ("step", "non_global_step", "epoch")

        self._elastic_resumed = False

    def on_loop_begin(self, model, **kwargs):
        """Check module name."""
        # compatible with DDP/DP
        _model = get_binding_module(model)

        # assert: all names in self.modules should be sub-module of model
        names, _ = flatten(self.modules)
        for name in names:
            _sub_model = _model
            for _sub_name in name.split("."):
                assert hasattr(
                    _sub_model, _sub_name
                ), f"{name} not found in model, please recheck"
                _sub_model = getattr(_sub_model, _sub_name)

    def on_epoch_begin(self, model, epoch_id, **kwargs):
        """Do process on epoch begin if need."""
        if self.update_by != "epoch":
            return

        if elastic_need_resume() and not self._elastic_resumed:
            for epoch_idx in self.step_or_epoch:
                if epoch_id > epoch_idx:
                    self._do_process(model, epoch_idx)
            self._elastic_resumed = True

        if epoch_id not in self.step_or_epoch:
            return

        self._do_process(model, epoch_id)

    def on_step_begin(self, model, step_id, global_step_id, **kwargs):
        """Do process on step begin if need."""
        if self.update_by not in ["step", "non_global_step"]:
            return

        cur_step_id = (
            step_id if self.update_by == "non_global_step" else global_step_id
        )

        if elastic_need_resume() and not self._elastic_resumed:
            for step_idx in self.step_or_epoch:
                if cur_step_id > step_idx:
                    self._do_process(model, step_idx)
            self._elastic_resumed = True

        if cur_step_id not in self.step_or_epoch:
            return

        self._do_process(model, cur_step_id)

    def _do_process(self, model, step_or_epoch_id):
        """Process sub_module by step or epoch id."""

        # compatible with DDP/DP
        _model = get_binding_module(model)

        # get current module index
        index = self.step_or_epoch.index(step_or_epoch_id)

        # do process for each module
        for module_name in self.modules[index]:
            self.process(_model, module_name)

    @abstractmethod
    def process(self, model: nn.Module, name: str) -> None:
        """Process sub_module in model given name."""
        pass


@OBJECT_REGISTRY.register
class FreezeModule(OnlineModelTrick):
    """
    Freeze module parameter while training. Useful in finetune case.

    Args:
        module: sub model names.
        step_or_epoch: when to freeze module, same length as module.
        update_by: by step or by epoch.
        only_batchnorm: Only freeze batchnorm, with valid gradient.
            Default is False.

    Example:
        >>> freeze_module_callback = FreezeModule(
        ...    modules=[['backbone'], ['neck']],
        ...    step_or_epoch=[10000, 15000],
        ...    update_by='step',
        ...    only_batchnorm=True,
        ... )
    """

    def __init__(
        self,
        modules: List[List[str]],
        step_or_epoch: List[int],
        update_by: str,
        only_batchnorm: bool = False,
    ):
        super().__init__(modules, step_or_epoch, update_by)
        self.only_batchnorm = only_batchnorm

        if not only_batchnorm:
            logger.warning(
                "Please use FreezePartialModule to delete grad "
                + "in model_convert. FreezeModule callbacks may "
                + "cause useless grad sync in DDP."
            )

    def process(self, model, name):
        """Freeze module inplace."""
        m = model
        for _sub_name in name.split("."):
            m = getattr(m, _sub_name)

        # set batchnorm and dropout in eval mode
        m.eval()

        if self.only_batchnorm:
            logger.info(f"[FreezeModule] freeze bn in {name}.")
        else:
            # disable grad
            logger.info(f"[FreezeModule] freeze {name} to disable grad.")
            for param in m.parameters():
                param.requires_grad = False


@OBJECT_REGISTRY.register
class FuseBN(OnlineModelTrick):
    """
    Fuse batchnorm layer in float training.

    Usually batchnorm is fused in QAT, but sometimes you can do it
    float training.

    Args:
        module: sub model names to fuse bn.
        step_or_epoch: when to fusebn, same length as module.
        update_by: by step or by epoch.
        inplace: if fuse bn inplace
    Note:
        Only Conv+BN inside nn.Sequential or nn.ModuleList can be merged.

    Example:
        >>> fuse_bn_callback = FuseBN(
        ...    modules=[['backbone'], ['neck']],
        ...    step_or_epoch=[10000, 15000],
        ...    update_by='step',
        ... )
    """

    def __init__(
        self,
        modules: List[List[str]],
        step_or_epoch: List[int],
        update_by: str,
        inplace: bool = False,
    ):
        super().__init__(modules, step_or_epoch, update_by)
        self.inplace = inplace

    def _check_is_qat(self, model):
        if hasattr(model, "activation_post_process") and isinstance(
            model.activation_post_process, FakeQuantizeBase
        ):
            return True
        for mod in model.children():
            is_qat = self._check_is_qat(mod)
            if is_qat:
                return is_qat
        return False

    def on_loop_begin(self, model, **kwargs):
        if self._check_is_qat(model):
            assert get_qat_mode() != QATMode.FuseBN, (
                "Under fuse_bn qat_mode, FuseBN callback is not supported "
                "for qat step, because qat without bn in this mode."
                "Please remove this callback from qat_trainer in config file."
            )
        super().on_loop_begin(model, **kwargs)

    def process(self, model, name, inplace=False):
        """Fuse bn inplace."""
        logger.info(f"[FuseBN] fuse bn in {name}")
        node = getattr(model, name)
        node_fused = fuse_norm_recursively(
            node, fuse_list=["bn"], inplace=self.inplace
        )
        assert not has_normalization(node_fused, check_list=["bn"])
        setattr(model, name, node_fused)
