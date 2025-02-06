# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from math import ceil, cos, pi
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from hat.callbacks import CallbackMixin
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import is_list_of_type, is_list_sorted
from hat.utils.data_helpers import get_dataloader_length, has_len_func

__all__ = [
    "StepDecayOptimParamUpdater",
    "CosOptimParamUpdater",
    "CyclicOptimParamUpdater",
]

logger = logging.getLogger(__name__)


class OptimParamUpdaterBase(CallbackMixin):
    """Base class for Optimizer Parameters updater.

    Compare to Lr Updater, Param updater provide more general parameter update
    functions. OptimParamUpdaterBase has implement warming up param schedule,
    so children class only need to implement function `get_param`, to specific
    formal training param schedule.

    Args:
        param_name: Specified parameter name of parameter in optimizer
            param_groups. eg: "lr", "weight_decay"
        update_by: Among formal training, update specified param on
            'step' begin or on 'epoch' begin.
            If equal to 'step', update param according to 'global_step_id'.
            If equal to 'epoch', update param according to 'epoch_id'.
            Default 'step'.
        warmup_by: Among warmup training, update param on 'step' begin or on
            'epoch' begin, similar to `update_by`.
            Default 'step'.
        warmup_len: Num of warmup steps or epochs.
            If warmup_by == 'step', it means warmup steps.
            If warmup_by == 'epoch', it means warmup epochs.
        warmup_mode: Type of warmup used. It can be 'constant', 'linear' now.
        warmup_begin_param: Beginning param used to calculated warmup param.
            If warmup_mode is 'constant', no-op.
        warmup_param_ratio: Used to calculate warmup ending param though two
            steps:
            (1) Achieve beginning param (init_param) of formal training from
            optimizer.
            (2) warmup_end_param = init_param * warmup_param_ratio.
        step_log_interval: param logging interval on step begin, only work when
            warmup_by == 'step'.
            If warmup_by == 'epoch', logging param value on each epoch begin.
    """

    UPDATE_MODES = ["step", "epoch"]
    WARMUP_MODES = ["constant", "linear"]

    def __init__(
        self,
        param_name: str,
        update_by: str = "step",
        warmup_by: str = "step",
        warmup_len: int = 0,
        warmup_mode: str = "linear",
        warmup_begin_param: float = 0.0,
        warmup_param_ratio: float = 1.0,
        step_log_interval: int = 1,
    ):
        update_by = update_by.lower()
        warmup_by = warmup_by.lower()
        assert (
            update_by in self.UPDATE_MODES
        ), f"{update_by} not in {self.UPDATE_MODES}"

        assert (
            warmup_by in self.UPDATE_MODES
        ), f"{warmup_by} not in {self.UPDATE_MODES}"
        assert warmup_len >= 0, warmup_len
        assert (
            warmup_mode in self.WARMUP_MODES
        ), f"{warmup_mode} not in {self.WARMUP_MODES}"
        assert warmup_begin_param >= 0, warmup_begin_param
        assert 0 <= warmup_param_ratio <= 1.0, warmup_param_ratio

        self.param_name = param_name
        self.update_by = update_by
        self.warmup_by = warmup_by
        self.warmup_len = warmup_len
        self.warmup_mode = warmup_mode
        self.warmup_begin_param = warmup_begin_param
        self.warmup_param_ratio = warmup_param_ratio
        self.step_log_interval = step_log_interval

        if self.warmup_by == "step":
            self.warmup_steps = self.warmup_len
            self.warmup_epochs = None
        else:
            # unknown yet, init using data_loader len on train begin
            self.warmup_steps = None
            self.warmup_epochs = self.warmup_len

        # initial param for each param group, is a list
        self._per_group_init_param = None

    def get_param(self, begin_param: float, num_update: int):
        """Calculate formal training param value for each step or epoch.

        Args:
            begin_param: Beginning param of formal training.
            num_update: Current num of param updates.
        """
        raise NotImplementedError

    def get_warmup_param(self, warmup_end_param: float, num_update: int):
        """Calculate warmup training param for each step or epoch.

        Args:
            warmup_end_param: param value when warmup ending.
            num_update: Current num of param updates.
        """
        assert 0 <= num_update < self.warmup_steps
        num_update = float(num_update)
        if self.warmup_mode == "linear":
            increase = (
                (warmup_end_param - self.warmup_begin_param)
                * num_update
                / self.warmup_steps
            )
            return self.warmup_begin_param + increase

        elif self.warmup_mode == "constant":
            return warmup_end_param

        else:
            raise ValueError("Invalid warmup mode %s" % self.warmup_mode)

    def set_param(
        self,
        optimizer: torch.optim.Optimizer,
        per_group_value: Sequence[float],
    ):
        """Set value for each param group.

        Args:
            optimizer: Optimizer instance.
            per_group_value: value for each param group.
        """
        assert isinstance(optimizer, torch.optim.Optimizer), type(optimizer)
        assert len(optimizer.param_groups) == len(
            per_group_value
        ), f"{len(optimizer.param_groups)} vs. {len(per_group_value)}"

        for param_group, value in zip(optimizer.param_groups, per_group_value):
            param_value = param_group[self.param_name]
            if isinstance(param_value, (list, tuple)):
                value = (value, *param_value[1:])
            param_group[self.param_name] = value

    def set_formal_training_param(
        self, optimizer: torch.optim.Optimizer, num_update: int
    ):
        """Calculate current param value then assign to optimizer.

        Args:
            optimizer: Optimizer instance.
            num_update: Current num of param updates.
        """
        per_group_param = [
            self.get_param(init_param, num_update=num_update)
            for init_param in self._per_group_init_param
        ]
        self.set_param(optimizer, per_group_param)

    def set_warmup_training_param(
        self, optimizer: torch.optim.Optimizer, num_update: int
    ):
        """Calculate current param then assign to optimizer.

        Args:
            optimizer: Optimizer instance.
            num_update: Current num of param updates.
        """
        per_group_wm_param = [
            self.get_warmup_param(
                warmup_end_param=init_param * self.warmup_param_ratio,
                num_update=num_update,
            )
            for init_param in self._per_group_init_param
        ]
        self.set_param(optimizer, per_group_wm_param)

    def log_param(self, optimizer, epoch_id, step_id, global_step_id):
        last_param = ""
        for param_group in optimizer.param_groups:
            last_param += f"{param_group[self.param_name]},"
        logger.info(
            "Epoch[%d] Step[%d] GlobalStep[%d] %s=%s"
            % (
                epoch_id,
                step_id,
                global_step_id,
                self.param_name,
                str(last_param),
            )
        )

    def on_loop_begin(self, optimizer, data_loader, **kwargs):
        """Prepare some vars for param updater."""
        # 1. init warmup_steps
        assert (
            data_loader is not None
        ), "You have to provide a real dataloader when you begin training"
        err_msg = (
            "You can't get the length of data_loader, "
            "We recommand you set warmup_by == 'step' and "
            "update_by == 'step'"
        )
        if has_len_func(data_loader):
            self.step_per_epoch = get_dataloader_length(data_loader)
            assert (
                self.step_per_epoch != float("inf")
                and self.step_per_epoch is not None
            ), err_msg
            if self.warmup_by == "epoch":
                self.warmup_steps = self.warmup_len * self.step_per_epoch
            else:
                self.warmup_epochs = ceil(
                    self.warmup_len / self.step_per_epoch
                )
        else:
            assert (
                self.warmup_by == "step" and self.update_by == "step"
            ), err_msg

        # 2. backup initial param value of optimizer
        # NOTE: when resuming from a checkpoint, if 'initial_param' is not
        # saved, it will be set according to the optimizer params
        if optimizer is not None:
            for group in optimizer.param_groups:
                param_value = group[self.param_name]
                if isinstance(param_value, (list, tuple)):
                    assert len(param_value) >= 1
                    param_value = param_value[0]

                group.setdefault(f"initial_{self.param_name}", param_value)

            self._per_group_init_param = [
                group[f"initial_{self.param_name}"]
                for group in optimizer.param_groups
            ]

    def on_epoch_begin(self, optimizer, epoch_id, global_step_id, **kwargs):
        """Update param on each epoch begin if update by 'epoch'."""
        # noqa: E501
        assert self.warmup_steps is not None
        if self.update_by == "epoch" and global_step_id >= self.warmup_steps:
            # set param for each param group among formal training
            self.set_formal_training_param(optimizer, num_update=epoch_id)
        self.log_param(
            optimizer,
            epoch_id=epoch_id,
            step_id=0,
            global_step_id=global_step_id,
        )

    def on_step_begin(
        self, optimizer, epoch_id, step_id, global_step_id, **kwargs
    ):  # noqa: D205,D400
        """On each step begin, update warmup param or formal training param (if
        update by 'step').
        """
        assert self.warmup_steps is not None
        if global_step_id < self.warmup_steps:
            self.set_warmup_training_param(
                optimizer, num_update=global_step_id
            )
            if (global_step_id + 1) % self.step_log_interval == 0:
                self.log_param(
                    optimizer,
                    epoch_id=epoch_id,
                    step_id=step_id,
                    global_step_id=global_step_id,
                )

        elif self.update_by == "step":
            # set param according to param_name for each param group among
            # formal training
            self.set_formal_training_param(
                optimizer, num_update=global_step_id
            )
            if (global_step_id + 1) % self.step_log_interval == 0:
                self.log_param(
                    optimizer,
                    epoch_id=epoch_id,
                    step_id=step_id,
                    global_step_id=global_step_id,
                )


@OBJECT_REGISTRY.register
class StepDecayOptimParamUpdater(OptimParamUpdaterBase):
    """Optimizer Param Updater Callback for adjusting param with warmup and decay.

    Args:
        param_decay_id: The epoch(step) list for param decay.
        It means the epoch(step) id you want to decay after warmup.
        param_decay_factor: Factor for param decay.
    """

    def __init__(
        self,
        param_name: str,
        update_by: str = "epoch",
        warmup_by: str = "epoch",
        warmup_len: int = 0,
        warmup_mode: str = "linear",
        warmup_begin_param: float = 0.0,
        warmup_param_ratio: float = 1.0,
        step_log_interval: int = 1,
        param_decay_id: List[int] = None,
        param_decay_factor: float = 0.1,
    ):
        super(StepDecayOptimParamUpdater, self).__init__(
            param_name=param_name,
            update_by=update_by,
            warmup_by=warmup_by,
            warmup_len=warmup_len,
            warmup_mode=warmup_mode,
            warmup_begin_param=warmup_begin_param,
            warmup_param_ratio=warmup_param_ratio,
            step_log_interval=step_log_interval,
        )

        if param_decay_id is not None:
            assert is_list_of_type(
                param_decay_id, int
            ), "param_decay_id should be a list of int"
            assert is_list_sorted(
                param_decay_id
            ), "param_decay_id should be sorted ascending"
        assert (
            0 <= param_decay_factor <= 1.0
        ), "param_decay_factor should be in [0.0, 1.0]"
        self.param_decay_id = param_decay_id
        self.param_decay_factor = param_decay_factor

    def get_param(
        self, begin_param: float, num_update: int
    ):  # noqa: D205,D400
        """Calculate new param value after warmup according to the decay epoch
        list `param_decay_id`.

        Args:
            begin_param: Beginning param of formal training, commonly equal to
                optimizer's initial param.
            num_update:
                Current epochs or steps of param updates.
        Returns:
            param
        """
        if self.param_decay_id is None:
            return begin_param
        if self.update_by == "step":
            assert (
                self.param_decay_id[0] >= self.warmup_steps
            ), "StepDecay should be done after warmup steps."
        else:
            assert (
                self.param_decay_id[0] >= self.warmup_epochs
            ), "StepDecay should be done after warmup epochs."

        exp = self._find_exp(num_update)
        return begin_param * self.param_decay_factor ** exp

    def _find_exp(self, num_update):
        exp = len(self.param_decay_id)
        for i, num in enumerate(self.param_decay_id):
            if num_update < num:
                exp = i
                break

        return exp


@OBJECT_REGISTRY.register
class CosOptimParamUpdater(OptimParamUpdaterBase):
    """Optimizer Param Updater Callback for adjusting param with cosine decay.

    Args:
        max_steps: the formal training steps you want set. If it is None,
            max_steps = num_epochs * self.step_per_epoch - self.warmup_steps
        stop_param: the param of last epoch/step.
    """

    def __init__(
        self,
        param_name: str,
        max_steps: int = -1,
        stop_param: float = 0.0,
        decay_only: bool = False,
        warmup_by: str = "step",
        warmup_len: int = 0,
        warmup_mode: str = "linear",
        warmup_begin_param: float = 0.0,
        warmup_param_ratio: float = 1.0,
        step_log_interval: int = 1,
    ):
        super(CosOptimParamUpdater, self).__init__(
            param_name=param_name,
            update_by="step",
            warmup_by=warmup_by,
            warmup_len=warmup_len,
            warmup_mode=warmup_mode,
            warmup_begin_param=warmup_begin_param,
            warmup_param_ratio=warmup_param_ratio,
            step_log_interval=step_log_interval,
        )
        self.max_steps = max_steps
        self.stop_param = stop_param
        self.decay_only = decay_only

    def on_loop_begin(self, optimizer, data_loader, num_epochs, **kwargs):
        super(CosOptimParamUpdater, self).on_loop_begin(
            optimizer, data_loader, **kwargs
        )
        if self.max_steps < 0:
            assert num_epochs is not None, (
                "you should set the num_epochs of the Trainer or "
                "set the max_steps."
            )
            self.max_steps = (
                num_epochs * self.step_per_epoch - self.warmup_steps
            )

    def get_param(self, begin_param: float, num_update: int):
        factor = 1 + cos(
            pi * (num_update - self.warmup_steps) / self.max_steps
        )
        if self.decay_only and begin_param < self.stop_param:
            return begin_param
        new_param = (
            begin_param - self.stop_param
        ) * factor / 2 + self.stop_param
        return new_param


@OBJECT_REGISTRY.register
class CyclicOptimParamUpdater(OptimParamUpdaterBase):
    """Optimizer Param Updater for adjusting param with OneCycle Updater.

    Args:
        target_ratio: Relative ratio of the highest param value
            and the lowest param to the initial param value.
        cyclic_times: Number of cycles during training
        step_ratio_up: The ratio of the increasing process of
            param value in the total cycle.
        step_log_interval: Logging interval.
    """

    def __init__(
        self,
        param_name: str,
        target_ratio: Tuple[float] = (10, 1e-4),
        cyclic_times: Optional[int] = 1,
        step_ratio_up: Optional[float] = 0.4,
        step_log_interval: int = 1,
    ):
        super().__init__(
            param_name=param_name,
            step_log_interval=step_log_interval,
        )
        self.target_ratio = target_ratio
        self.cyclic_times = cyclic_times
        self.step_ratio_up = step_ratio_up

    def on_loop_begin(self, optimizer, data_loader, num_epochs, **kwargs):
        super(CyclicOptimParamUpdater, self).on_loop_begin(
            optimizer, data_loader, **kwargs
        )

        max_steps = num_epochs * self.step_per_epoch
        self.max_update_per_phase = max_steps // self.cyclic_times
        iter_up_phase = int(self.step_ratio_up * self.max_update_per_phase)
        self.phases = []
        self.phases.append(
            [
                0,
                iter_up_phase,
                1,
                self.target_ratio[0],
            ]
        )
        self.phases.append(
            [
                iter_up_phase,
                self.max_update_per_phase,
                self.target_ratio[0],
                self.target_ratio[1],
            ]
        )

    def get_param(self, begin_param: float, num_update: int):
        for start_update, end_update, start_ratio, end_ratio in self.phases:
            num_update_per_phase = num_update % self.max_update_per_phase
            if start_update <= num_update_per_phase < end_update:
                progress = num_update_per_phase - start_update
                pct = progress / (end_update - start_update)
                new_param = CyclicOptimParamUpdater.annealing_cos(
                    begin_param * start_ratio,
                    begin_param * end_ratio,
                    pct,
                )
                return new_param
        return begin_param

    @staticmethod
    def annealing_cos(start, end, pct):
        """Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        cos_out = np.cos(np.pi * pct) + 1
        return end + (start - end) / 2 * cos_out
