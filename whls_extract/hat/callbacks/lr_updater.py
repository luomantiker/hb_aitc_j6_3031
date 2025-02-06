# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from math import ceil, cos, pi
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch

from hat.callbacks import CallbackMixin
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import is_list_of_type, is_list_sorted
from hat.utils.data_helpers import get_dataloader_length, has_len_func

try:
    import deepspeed

    if deepspeed.version >= "0.14.1":
        from deepspeed.runtime.base_optimizer import ZeROOptimizer
    else:
        from deepspeed.runtime import ZeROOptimizer
except ImportError:
    deepspeed = None
    ZeROOptimizer = None

__all__ = [
    "PolyLrUpdater",
    "StepDecayLrUpdater",
    "CosLrUpdater",
    "NoamLrUpdater",
    "CosineAnnealingLrUpdater",
]

logger = logging.getLogger(__name__)


class LrUpdaterBase(CallbackMixin):
    """Base class for learning rate updater.

    LrUpdaterBase has implement warming up lr schedule, so children class only
    need to implement function `get_lr`, to specific formal training lr
    schedule.
    Args:
        update_by: Among formal training, update lr on 'step' begin or on
            'epoch' begin.
            If equal to 'step', update lr according to 'global_step_id'.
            If equal to 'epoch', update lr according to 'epoch_id'.
            Default 'step'.
        warmup_by: Among warmup training, update lr on 'step' begin or on
            'epoch' begin, similar to `update_by`.
            Default 'step'.
        warmup_len: Num of warmup steps or epochs.
            If warmup_by == 'step', it means warmup steps.
            If warmup_by == 'epoch', it means warmup epochs.
        warmup_mode: Type of warmup used. It can be 'constant', 'linear' now.
        warmup_begin_lr: Beginning lr used to calculated warmup lr.
            If warmup_mode is 'constant', no-op.
        warmup_lr_ratio: Used to calculate warmup ending lr though two steps:
            (1) Achieve beginning lr (init_lr) of formal training from
            optimizer.
            (2) warmup_end_lr = init_lr * warmup_lr_ratio.
        step_log_interval: lr logging interval on step begin, only work when
            warmup_by == 'step'.
            If warmup_by == 'epoch', logging lr on each epoch begin.
    """

    UPDATE_MODES = ["step", "epoch"]
    WARMUP_MODES = ["constant", "linear"]

    def __init__(
        self,
        update_by: str = "step",
        warmup_by: str = "step",
        warmup_len: int = 0,
        warmup_mode: str = "linear",
        warmup_begin_lr: float = 0.0,
        warmup_lr_ratio: float = 1.0,
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
        assert warmup_begin_lr >= 0, warmup_begin_lr
        assert 0 <= warmup_lr_ratio <= 1.0, warmup_lr_ratio

        self.update_by = update_by
        self.warmup_by = warmup_by
        self.warmup_len = warmup_len
        self.warmup_mode = warmup_mode
        self.warmup_begin_lr = warmup_begin_lr
        self.warmup_lr_ratio = warmup_lr_ratio
        self.step_log_interval = step_log_interval

        if self.warmup_by == "step":
            self.warmup_steps = self.warmup_len
            self.warmup_epochs = None
        else:
            # unknown yet, init using data_loader len on train begin
            self.warmup_steps = None
            self.warmup_epochs = self.warmup_len

        # initial lr for each param group, is a list
        self._per_group_init_lr = None

    def get_lr(self, begin_lr: float, num_update: int):
        """Calculate formal training lr for each step or epoch.

        Args:
            begin_lr: Beginning lr of formal training, commonly equal to
                optimizer's initial lr.
            num_update: Current num of lr updates.
        """
        raise NotImplementedError

    def get_warmup_lr(self, warmup_end_lr: float, num_update: int):
        """Calculate warmup training lr for each step or epoch.

        Args:
            warmup_end_lr: lr when warmup ending.
            num_update: Current num of lr updates.
        """
        assert 0 <= num_update < self.warmup_steps
        num_update = float(num_update)
        if self.warmup_mode == "linear":
            increase = (
                (warmup_end_lr - self.warmup_begin_lr)
                * num_update
                / self.warmup_steps
            )
            return self.warmup_begin_lr + increase

        elif self.warmup_mode == "constant":
            return warmup_end_lr

        else:
            raise ValueError("Invalid warmup mode %s" % self.warmup_mode)

    @staticmethod
    def set_lr(
        optimizer: torch.optim.Optimizer, per_group_lr: Sequence[float]
    ):
        """Set lr for each param group.

        Args:
            optimizer: Optimizer instance.
            per_group_lr: lr for each param group.
        """
        optimizer_type = (torch.optim.Optimizer,)
        if deepspeed is not None:
            optimizer_type += (ZeROOptimizer,)
        assert isinstance(optimizer, optimizer_type), type(optimizer)
        assert len(optimizer.param_groups) == len(
            per_group_lr
        ), f"{len(optimizer.param_groups)} vs. {len(per_group_lr)}"

        for param_group, lr in zip(optimizer.param_groups, per_group_lr):
            param_group["lr"] = lr

    def set_formal_training_lr(
        self, optimizer: torch.optim.Optimizer, num_update: int
    ):
        """Calculate current lr then assign to optimizer among formal training.

        Args:
            optimizer: Optimizer instance.
            num_update: Current num of lr updates.
        """
        per_group_lr = [
            self.get_lr(init_lr, num_update=num_update)
            for init_lr in self._per_group_init_lr
        ]
        self.set_lr(optimizer, per_group_lr)

    def set_warmup_training_lr(
        self, optimizer: torch.optim.Optimizer, num_update: int
    ):
        """Calculate current lr then assign to optimizer among warmup training.

        Args:
            optimizer: Optimizer instance.
            num_update: Current num of lr updates.
        """
        per_group_wm_lr = [
            self.get_warmup_lr(
                warmup_end_lr=init_lr * self.warmup_lr_ratio,
                num_update=num_update,
            )
            for init_lr in self._per_group_init_lr
        ]
        self.set_lr(optimizer, per_group_wm_lr)

    @staticmethod
    def log_lr(optimizer, epoch_id, step_id, global_step_id):
        last_lr = optimizer.param_groups[-1]["lr"]
        logger.info(
            "Epoch[%d] Step[%d] GlobalStep[%d] lr=%f"
            % (epoch_id, step_id, global_step_id, last_lr)
        )

    def on_loop_begin(self, optimizer, data_loader, **kwargs):
        """Prepare some vars for lr updater."""
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

        # 2. backup initial lr of optimizer
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not
        # saved, it will be set according to the optimizer params
        if optimizer is not None:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])

            self._per_group_init_lr = [
                group["initial_lr"] for group in optimizer.param_groups
            ]

    def on_epoch_begin(self, optimizer, epoch_id, global_step_id, **kwargs):
        """Update formal training lr on each epoch begin if update by 'epoch'."""  # noqa: E501
        assert self.warmup_steps is not None
        if self.update_by == "epoch" and global_step_id >= self.warmup_steps:
            # set lr for each param group among formal training
            self.set_formal_training_lr(optimizer, num_update=epoch_id)

        self.log_lr(
            optimizer,
            epoch_id=epoch_id,
            step_id=0,
            global_step_id=global_step_id,
        )

    def on_step_begin(
        self, optimizer, epoch_id, step_id, global_step_id, **kwargs
    ):  # noqa: D205,D400
        """On each step begin, update warmup lr or formal training lr (if
        update by 'step').
        """
        assert self.warmup_steps is not None
        if global_step_id < self.warmup_steps:
            self.set_warmup_training_lr(optimizer, num_update=global_step_id)
            if (global_step_id + 1) % self.step_log_interval == 0:
                self.log_lr(
                    optimizer,
                    epoch_id=epoch_id,
                    step_id=step_id,
                    global_step_id=global_step_id,
                )

        elif self.update_by == "step":
            # set lr for each param group among formal training
            self.set_formal_training_lr(optimizer, num_update=global_step_id)
            if (global_step_id + 1) % self.step_log_interval == 0:
                self.log_lr(
                    optimizer,
                    epoch_id=epoch_id,
                    step_id=step_id,
                    global_step_id=global_step_id,
                )


@OBJECT_REGISTRY.register
class PolyLrUpdater(LrUpdaterBase):
    """Reduce the learning rate according to a polynomial of given power.

    Calculate the new learning rate, after warmup by::

        if (num_update - self.warmup_steps) < max_update:
            new_lr = final_lr + (begin_lr - final_lr) * (
            1 - (num_update - self.warmup_steps) / max_update
            ) ^ power
        else:
            new_lr = final_lr.

    Args:
        max_update: Times of lr update among formal training. Used to calculate
            formal training lr.

            .. note::

                max_update should not includes wamup steps or epochs, they can
                be specific independently by warmup_len.

        update_by: Among formal training, update lr on 'step' begin or on
            'epoch' begin.
            If equal to 'step', update lr according to 'global_step_id'.
            If equal to 'epoch', update lr according to 'epoch_id'.
            Default 'step'.
        power: Power of the decay term as a function of the current number of
            updates.
        final_lr: Final learning rate after all steps.
        warmup_by: Among warmup training, update lr on 'step' begin or on
            'epoch' begin, similar to `update_by`.
            Default 'step'.
        warmup_len: Num of warmup steps or epochs.
            If `warmup_by=='step'`, it means warmup steps.
            If `warmup_by=='epoch'`, it means warmup epochs.
        warmup_mode: Type of warmup used. It can be 'constant', 'linear' now.
        warmup_begin_lr: Beginning lr used to calculated warmup lr.
            If warmup_mode is 'constant', no-op.
        warmup_lr_ratio: Used to calculate warmup ending lr though two steps:
            (1) Achieve begging lr (`init_lr`) of formal training from
            optimizer.
            (2) warmup_end_lr = init_lr * warmup_lr_ratio.
        step_log_interval: lr logging interval on step begin, only work when
            warmup_by == 'step'.
            If warmup_by == 'epoch', logging lr on each epoch begin.
    """

    def __init__(
        self,
        max_update: int,
        update_by: str = "step",
        power: float = 1.0,
        final_lr: float = 0.0,
        warmup_by: str = "step",
        warmup_mode: str = "linear",
        warmup_len: int = 0,
        warmup_begin_lr: float = 0.0,
        warmup_lr_ratio: float = 1.0,
        step_log_interval: int = 1,
    ):
        super(PolyLrUpdater, self).__init__(
            update_by=update_by,
            warmup_by=warmup_by,
            warmup_mode=warmup_mode,
            warmup_len=warmup_len,
            warmup_begin_lr=warmup_begin_lr,
            warmup_lr_ratio=warmup_lr_ratio,
            step_log_interval=step_log_interval,
        )
        assert max_update > 0

        self.max_update = max_update
        self.power = float(power)
        self.final_lr = float(final_lr)

    def get_lr(self, begin_lr: float, num_update: int):
        """Calculate new lr after warmup according to a polynomial.

        Args:
            begin_lr: Beginning lr of formal training, commonly equal to
                optimizer's initial lr.
            num_update: Current num of lr updates.

        """
        assert num_update >= 0, num_update
        assert self.max_update > 0, self.max_update

        if self.update_by == "step":
            formal_update = num_update - self.warmup_steps
        else:
            formal_update = num_update - self.warmup_epochs

        if formal_update < self.max_update:
            coeff = (
                1.0 - float(formal_update) / self.max_update
            ) ** self.power
            return (begin_lr - self.final_lr) * coeff + self.final_lr
        else:
            return self.final_lr


@OBJECT_REGISTRY.register
class StepDecayLrUpdater(LrUpdaterBase):
    """Lr Updater Callback for adjusting lr with warmup and decay.

    Args:
        lr_decay_id: The epoch(step) list for lr decay. It means
            the epoch(step) id you want to decay after warmup.
        lr_decay_factor: Factor for lr decay.
    """

    def __init__(
        self,
        update_by: str = "epoch",
        warmup_by: str = "epoch",
        warmup_len: int = 0,
        warmup_mode: str = "linear",
        warmup_begin_lr: float = 0.0,
        warmup_lr_ratio: float = 1.0,
        step_log_interval: int = 1,
        lr_decay_id: List[int] = None,
        lr_decay_factor: float = 0.1,
    ):
        super(StepDecayLrUpdater, self).__init__(
            update_by=update_by,
            warmup_by=warmup_by,
            warmup_len=warmup_len,
            warmup_mode=warmup_mode,
            warmup_begin_lr=warmup_begin_lr,
            warmup_lr_ratio=warmup_lr_ratio,
            step_log_interval=step_log_interval,
        )

        if lr_decay_id is not None:
            assert is_list_of_type(
                lr_decay_id, int
            ), "lr_decay_id should be a list of int"
            assert is_list_sorted(
                lr_decay_id
            ), "lr_decay_id should be sorted ascending"
        assert (
            0 <= lr_decay_factor <= 1.0
        ), "lr_decay_factor should be in [0.0, 1.0]"
        self.lr_decay_id = lr_decay_id
        self.lr_decay_factor = lr_decay_factor

    def get_lr(self, begin_lr: float, num_update: int):  # noqa: D205,D400
        """Calculate new lr after warmup according to the decay epoch
        list `lr_decay_id`.

        Args:
            begin_lr: Beginning lr of formal training, commonly equal to
                optimizer's initial lr.
            num_update:
                Current epochs or steps of lr updates.
        Returns:
            lr: learning rate
        """
        if self.lr_decay_id is None:
            return begin_lr
        if self.update_by == "step":
            assert (
                self.lr_decay_id[0] >= self.warmup_steps
            ), "StepDecay should be done after warmup steps."
        else:
            assert (
                self.lr_decay_id[0] >= self.warmup_epochs
            ), "StepDecay should be done after warmup epochs."

        exp = self._find_exp(num_update)
        return begin_lr * self.lr_decay_factor ** exp

    def _find_exp(self, num_update):
        exp = len(self.lr_decay_id)
        for i, num in enumerate(self.lr_decay_id):
            if num_update < num:
                exp = i
                break

        return exp


@OBJECT_REGISTRY.register
class CosLrUpdater(LrUpdaterBase):
    """Lr Updater Callback for adjusting lr with warmup and cos decay.

    Args:
        max_steps: the formal training steps you want set. If it is None,
            max_steps = num_epochs * self.step_per_epoch - self.warmup_steps
        stop_lr: the lr of last epoch/step.
    """

    def __init__(
        self,
        max_steps: int = -1,
        max_epoch: int = -1,
        stop_lr: float = 0.0,
        warmup_by: str = "step",
        warmup_len: int = 0,
        warmup_mode: str = "linear",
        warmup_begin_lr: float = 0.0,
        warmup_lr_ratio: float = 1.0,
        step_log_interval: int = 1,
    ):
        super(CosLrUpdater, self).__init__(
            update_by="step",
            warmup_by=warmup_by,
            warmup_len=warmup_len,
            warmup_mode=warmup_mode,
            warmup_begin_lr=warmup_begin_lr,
            warmup_lr_ratio=warmup_lr_ratio,
            step_log_interval=step_log_interval,
        )
        self.max_steps = max_steps
        self.max_epoch = max_epoch
        self.stop_lr = stop_lr

    def on_loop_begin(self, optimizer, data_loader, num_epochs, **kwargs):
        super(CosLrUpdater, self).on_loop_begin(
            optimizer, data_loader, **kwargs
        )
        if self.max_epoch > 0:
            self.max_steps = (
                self.max_epoch * self.step_per_epoch - self.warmup_steps
            )
        if self.max_steps < 0:
            assert num_epochs is not None, (
                "you should set the num_epochs of the Trainer or "
                "set the max_steps."
            )
            self.max_steps = (
                num_epochs * self.step_per_epoch - self.warmup_steps
            )

    def get_lr(self, begin_lr: float, num_update: int):
        if self.max_steps != 0:
            factor = 1 + cos(
                pi * (num_update - self.warmup_steps) / self.max_steps
            )
            new_lr = (begin_lr - self.stop_lr) * factor / 2 + self.stop_lr
            return new_lr
        return begin_lr


@OBJECT_REGISTRY.register
class NoamLrUpdater(LrUpdaterBase):
    r"""Noam LR Updater.

    NoamLrUpdater 是Transformer训练中常用的学习率管理器.
    论文出处 (Attention Is All You Need)[https://arxiv.org/pdf/1706.03762.pdf].
    学习率更新公式如下:


    .. math::
       lr = lr_{base} * d_{model}^{-0.5} * \min(
           num_update^{-0.5}, num_update * warmup\_step^{-1.5}
       )

    .. note::

        NoamLrUpdater 的 warmup_step 跟 warmup_by 的使用方式略有不同,
        因此使用了不同的参数命名避免混淆.

    Args:
        d_model: 模型的维度数, 参见计算公式指定. 默认是 256.
        warmup_step: 学习率预热的步数. 默认是 4000.
        update_by: 学习率根据 'step' 更新还是根据 'epoch' 更新. 默认是 "step".
        step_log_interval: 每隔多少次num_update打印一次学习率log. 默认是 1.
    """

    def __init__(
        self,
        d_model: Union[int, float] = 256,
        warmup_step: Union[int, float] = 4000,
        update_by: str = "step",
        step_log_interval=1,
    ):
        super(NoamLrUpdater, self).__init__(
            update_by=update_by, step_log_interval=step_log_interval
        )
        self.d_model = d_model
        self.warmup_step = warmup_step

    def get_lr(self, begin_lr: float, num_update: int):
        step_num = num_update + 1
        ret_lr = (
            begin_lr
            * self.d_model ** 0.5
            * min(step_num ** -0.5, step_num * self.warmup_step ** -1.5)
        )
        return ret_lr

    def get_warmup_lr(self, warmup_end_lr: float, num_update: int):
        """get_warmup_lr.

        在NoamLrUpdater中 get_warmup_lr 已经被吸收进 get_lr 过程中.
        这里实现``get_warmup_lr``是避免某些调用异常;
        """
        return self.get_lr(warmup_end_lr, num_update)


@OBJECT_REGISTRY.register
class OneCycleUpdater(LrUpdaterBase):
    """Lr Updater Callback for adjusting lr with OneCycle Updater.

    Args:
        stop_lr: the lr of last epoch/step.
    """

    def __init__(
        self,
        lr_max: float = 0.001,
        div_factor: float = 10.0,
        pct_start: float = 0.4,
        stop_lr: float = 0.0,
        warmup_by: str = "step",
        step_log_interval: int = 1,
    ):
        super(OneCycleUpdater, self).__init__(
            update_by="step",
            warmup_by=warmup_by,
            step_log_interval=step_log_interval,
        )
        self.div_factor = div_factor
        self.max_lr = lr_max
        self.warmup_begin_lr = lr_max / div_factor
        self.pct_start = pct_start
        self.stop_lr = stop_lr
        self.max_steps = None

    @staticmethod
    def annealing_cos(start, end, pct):
        """Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        cos_out = np.cos(np.pi * pct) + 1
        return end + (start - end) / 2 * cos_out

    def get_warmup_lr(self, warmup_end_lr: float, num_update: int):
        """Calculate warmup training lr for each step or epoch.

        Args:
            warmup_end_lr: lr when warmup ending.
            num_update: Current num of lr updates.
        """
        assert 0 <= num_update < self.warmup_steps
        num_update = float(num_update)
        pct = num_update / self.warmup_steps
        return OneCycleUpdater.annealing_cos(
            self.warmup_begin_lr, warmup_end_lr, pct
        )

    def set_warmup_training_lr(
        self, optimizer: torch.optim.Optimizer, num_update: int
    ):
        """Calculate current lr then assign to optimizer among warmup training.

        Args:
            optimizer: Optimizer instance.
            num_update: Current num of lr updates.
        """
        per_group_wm_lr = [
            self.get_warmup_lr(
                warmup_end_lr=init_lr,
                num_update=num_update,
            )
            for init_lr in self._per_group_init_lr
        ]
        self.set_lr(optimizer, per_group_wm_lr)

    def on_loop_begin(self, optimizer, data_loader, num_epochs, **kwargs):
        super(OneCycleUpdater, self).on_loop_begin(
            optimizer, data_loader, **kwargs
        )
        self.warmup_steps = num_epochs * self.step_per_epoch * self.pct_start
        self.max_steps = num_epochs * self.step_per_epoch - self.warmup_steps

    def get_lr(self, begin_lr: float, num_update: int):
        pct = float(num_update - self.warmup_steps) / self.max_steps
        new_lr = OneCycleUpdater.annealing_cos(begin_lr, self.stop_lr, pct)
        return new_lr


@OBJECT_REGISTRY.register
class CyclicLrUpdater(LrUpdaterBase):
    """Lr Updater Callback for adjusting lr with OneCycle Updater.

    Args:
        target_ratio: Relative ratio of the highest LR
            and the lowest LR to the initial LR.
        cyclic_times: Number of cycles during training
        step_ratio_up: The ratio of the increasing process of
            LR in the total cycle.
    """

    def __init__(
        self,
        target_ratio: Tuple[float] = (10, 1e-4),
        cyclic_times: int = 1,
        step_ratio_up: float = 0.4,
        step_log_interval: int = 1,
    ):
        super(CyclicLrUpdater, self).__init__(
            step_log_interval=step_log_interval
        )
        self.target_ratio = target_ratio
        self.cyclic_times = cyclic_times
        self.step_ratio_up = step_ratio_up

    def on_loop_begin(self, optimizer, data_loader, num_epochs, **kwargs):
        super(CyclicLrUpdater, self).on_loop_begin(
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

    def get_lr(self, begin_lr: float, num_update: int):
        for start_update, end_update, start_ratio, end_ratio in self.phases:
            num_update_per_phase = num_update % self.max_update_per_phase
            if start_update <= num_update_per_phase < end_update:
                progress = num_update_per_phase - start_update
                pct = progress / (end_update - start_update)
                new_lr = OneCycleUpdater.annealing_cos(
                    begin_lr * start_ratio, begin_lr * end_ratio, pct
                )
                return new_lr
        return 0.0


@OBJECT_REGISTRY.register
class CosineAnnealingLrUpdater(LrUpdaterBase):
    """Lr Updater Callback for adjusting lr with warmup and cos decay by epoch.

    Args:
        stop_lr: the lr of last epoch.
        min_lr_ratio: the ratio of minimoun lr.
        warmup_by: Among warmup training, update lr on 'step' begin or on
            'epoch' begin, similar to `update_by`.
            Default 'step'.
        warmup_len: Num of warmup steps or epochs.
            If `warmup_by=='step'`, it means warmup steps.
            If `warmup_by=='epoch'`, it means warmup epochs.
        warmup_mode: Type of warmup used. It can be 'constant', 'linear' now.
        warmup_begin_lr: Beginning lr used to calculated warmup lr.
            If warmup_mode is 'constant', no-op.
        warmup_lr_ratio: Used to calculate warmup ending lr though two steps:
            (1) Achieve begging lr (`init_lr`) of formal training from
            optimizer.
            (2) warmup_end_lr = init_lr * warmup_lr_ratio.
        step_log_interval: lr logging interval on step begin, only work when
            warmup_by == 'step'.
            If warmup_by == 'epoch', logging lr on each epoch begin.
        warmup_lr_begin2ratio: If True, the warmup lr linearly increases from
            warmup_begin_lr to initial_lr * warmup_lr_ratio. Otherwise, the
            warmup range is from initial_lr * warmup_lr_ratio to initial_lr.
    """

    def __init__(
        self,
        stop_lr: float = -1.0,
        min_lr_ratio: float = 1e-3,
        warmup_by: str = "step",
        warmup_len: int = 0,
        warmup_mode: str = "linear",
        warmup_begin_lr: float = 0.0,
        warmup_lr_ratio: float = 1.0,
        update_by: str = "epoch",
        step_log_interval: int = 1,
        warmup_lr_begin2ratio: bool = False,
    ):
        super(CosineAnnealingLrUpdater, self).__init__(
            update_by=update_by,
            warmup_by=warmup_by,
            warmup_len=warmup_len,
            warmup_mode=warmup_mode,
            warmup_begin_lr=warmup_begin_lr,
            warmup_lr_ratio=warmup_lr_ratio,
            step_log_interval=step_log_interval,
        )
        self.min_lr_ratio = min_lr_ratio
        self.stop_lr = stop_lr
        self.warmup_lr_begin2ratio = warmup_lr_begin2ratio

    def on_loop_begin(
        self, optimizer, data_loader, num_epochs, num_steps, **kwargs
    ):
        super(CosineAnnealingLrUpdater, self).on_loop_begin(
            optimizer, data_loader, **kwargs
        )
        if self.update_by == "epoch":
            self.max_process = num_epochs
        elif num_epochs is None:
            self.max_process = num_steps
        else:
            self.max_process = num_epochs * self.step_per_epoch

    def get_lr(self, begin_lr: float, num_update: int):
        """Calculate new lr after warmup according to the decay epoch.

        Args:
            begin_lr: Beginning lr of formal training, commonly equal to
                optimizer's initial lr.
            num_update:
                Current epochs of lr updates.
        Returns:
            lr: learning rate.
        """
        if self.stop_lr < 0:
            stop_lr = begin_lr * self.min_lr_ratio
        else:
            stop_lr = self.stop_lr
        new_lr = OneCycleUpdater.annealing_cos(
            begin_lr, stop_lr, num_update / self.max_process
        )
        return new_lr

    def get_warmup_lr(self, warmup_end_lr: float, num_update: int):
        """Calculate warmup training lr for each step or epoch.

        Args:
            warmup_end_lr: lr when warmup ending.
            num_update: Current num of lr updates.
        """
        if self.warmup_mode == "linear":
            if self.warmup_lr_begin2ratio:
                warmup_end_lr = warmup_end_lr * self.warmup_lr_ratio
                increase = (
                    (warmup_end_lr - self.warmup_begin_lr)
                    * num_update
                    / self.warmup_steps
                )
                return self.warmup_begin_lr + increase
            else:
                k = (1 - num_update / self.warmup_steps) * (
                    1 - self.warmup_lr_ratio
                )
                return warmup_end_lr * (1 - k)

        elif self.warmup_mode == "constant":
            return warmup_end_lr * self.warmup_lr_ratio

        else:
            raise ValueError("Invalid warmup mode %s" % self.warmup_mode)

    def set_warmup_training_lr(
        self, optimizer: torch.optim.Optimizer, num_update: int
    ):
        """Calculate current lr then assign to optimizer among warmup training.

        Args:
            optimizer: Optimizer instance.
            num_update: Current num of lr updates.
        """
        per_group_wm_lr = [
            self.get_warmup_lr(
                warmup_end_lr=init_lr,
                num_update=num_update,
            )
            for init_lr in self._per_group_init_lr
        ]
        self.set_lr(optimizer, per_group_wm_lr)
