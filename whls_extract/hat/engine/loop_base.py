# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DistributedSampler

from hat.callbacks import CallbackMixin
from hat.core.event import EventStorage
from hat.profiler.profilers import SimpleProfiler
from hat.utils.apply_func import _as_list
from hat.utils.checkpoint import checkpoint_resumable
from hat.utils.data_helpers import get_dataloader_length
from hat.utils.deterministic import (
    cast_model_to_deterministic,
    deterministic_level,
    deterministic_summary,
)
from hat.utils.distributed import dist_initialized, rank_zero_only
from hat.utils.elastic import ElasticState, elastic_need_resume
from hat.utils.generator import prefetch_iterator
from hat.utils.global_var import get_value, set_value
from hat.utils.logger import LOG_DIR, MSGColor, format_msg
from hat.utils.request import update_job_strategy
from .processors import BatchProcessorMixin

__all__ = ["PipeBase", "LoopBase"]

logger = logging.getLogger(__name__)


class PipeBase(ABC):
    """Base class for callbacks pipeline."""

    def __init__(
        self,
        callbacks: Optional[Sequence[CallbackMixin]] = None,
        profiler: Optional[dict] = None,
    ):

        self.name = self.__class__.__name__
        self.profile_dir = os.path.join(LOG_DIR, "profile")
        if profiler is None:
            profiler = SimpleProfiler(
                dirpath=self.profile_dir,
                filename="simple_profile",
                auto_describe=True,
            )
        self.profiler = profiler
        if hasattr(self.profiler, "set_engine_name"):
            self.profiler.set_engine_name(self.name)

        self.callbacks = []
        if callbacks is not None:
            self.callbacks = _as_list(callbacks)

        self.strategy = []

    def set_callbacks(self, callbacks: Sequence[CallbackMixin] = None):
        self.callbacks = []
        for cb in _as_list(callbacks):
            assert isinstance(cb, CallbackMixin), type(cb)
            self.callbacks.append(cb)

    @rank_zero_only
    def dump_yaml(self):
        yaml_path = os.path.join(self.profile_dir, "tricks.yaml")
        try:
            os.makedirs(self.profile_dir, exist_ok=True)
        except Exception:
            pass

        # overwrite
        if os.path.exists(yaml_path):
            os.remove(yaml_path)

        with open(yaml_path, "w") as f:
            yaml.dump(self.strategy, f)

    @rank_zero_only
    def upload_strategy(self):
        try:
            update_job_strategy(self.strategy)
        except Exception as e:
            logger.warning(e)
            logger.warning(
                "Train strategy upload failed! This does not affect training."
            )

    def on_loop_begin(self, **kwargs):
        # Note: Not support *args, because different callback may have
        # different positional arguments.
        with self.profiler.profile(f"on_{self.name}_loop_begin"):
            for cb in self.callbacks:
                cb.on_loop_begin(**kwargs)

    def on_loop_end(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_loop_end"):
            for cb in self.callbacks:
                cb.on_loop_end(**kwargs)

    def on_epoch_begin(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_epoch_begin"):
            for cb in self.callbacks:
                cb.on_epoch_begin(**kwargs)

    def on_epoch_end(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_epoch_end"):
            for cb in self.callbacks:
                cb.on_epoch_end(**kwargs)

    def on_step_begin(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_step_begin"):
            for cb in self.callbacks:
                cb.on_step_begin(**kwargs)

    def on_step_end(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_step_end"):
            for cb in self.callbacks:
                cb.on_step_end(**kwargs)

    def on_batch_begin(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_batch_begin"):
            for cb in self.callbacks:
                cb.on_batch_begin(**kwargs)

    def on_batch_end(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_batch_end"):
            for cb in self.callbacks:
                cb.on_batch_end(**kwargs)

    def on_backward_begin(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_backward_begin"):
            for cb in self.callbacks:
                cb.on_backward_begin(**kwargs)

    def on_backward_end(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_backward_end"):
            for cb in self.callbacks:
                cb.on_backward_end(**kwargs)

    def on_optimizer_step_begin(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_optimizer_step_begin"):
            for cb in self.callbacks:
                cb.on_optimizer_step_begin(**kwargs)

    def on_optimizer_step_end(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_optimizer_step_end"):
            for cb in self.callbacks:
                cb.on_optimizer_step_end(**kwargs)

    def on_forward_begin(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_forward_begin"):
            for cb in self.callbacks:
                cb.on_forward_begin(**kwargs)

    def on_forward_end(self, **kwargs):
        with self.profiler.profile(f"on_{self.name}_forward_end"):
            for cb in self.callbacks:
                cb.on_forward_end(**kwargs)

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass


class LoopBase(PipeBase):  # noqa: D205,D400
    """LoopBase controls the data flow from `data_loader` to `model`, including
    model forward, loss backward and parameters update.

    It is hardware independent, run on cpu (device is None) or gpu (device is
    int gpu id).

    By setting `stop_by`, you are able to stop loop by counting epoch
    (default) or step.

    Args:
        model: Model config or a `nn.Module` instance.
        data_loader: Training data loader config or a instantiated data loader.
        optimizer: Optimizer config or a optimizer instance.
        batch_processor: Batch processor config or a `BatchProcessorMixin`
            instance.
        device: Int gpu id or None.
            If int, do `model.cuda(device)` and `data.cuda(device)`.
            If None, no-op.
        model_convert_pipeline: Define the process of model convert.
            e.g. convert float model to qat model, convert qat model
            to quantize model.
        resume_optimizer: whether load optimizer dict when resume checkpoint.
        resume_epoch_or_step: whether need to resume epoch_or_step
            when resume checkpoint.
        resume_dataloader: whether to resume dataloader index.
            Only effective when `stop_by=='step'`.
        stop_by: Stop loop by counting epoch or step.
            If equal to 'epoch', stop loop when `epoch_id == num_epochs - 1`.
            If equal to 'step', stop loop when `global_step_id == num_steps - 1`.
            Default 'epoch'.
        num_epochs: Num of loop epochs, should be non-negative integer.
            If stop_by != 'epoch', no-op.
            Set 0 to skip loop epochs and run `self.on_*_loop_begin/end` only.
        start_epoch: Training start epoch, should be non-negative integer.
        num_steps: Num of loop steps, should be non-negative integer.
            If stop_by != 'step', no-op.
            Set 0 to skip loop steps and run `self.on_*_loop_begin/end` only.
        start_step: Training start step, should be non-negative integer.
        callbacks: Callback configs or instances.
        train_metrics: Metrics on training data.
        val_metrics: Metrics on validation data.
        profiler: To profile individual steps during loop and
            assist in identifying bottlenecks.
        log_interval: Logging output frequency.
        compiler: Converter of `torch.compile`.
    """  # noqa

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        batch_processor: BatchProcessorMixin,
        device: Union[int, None],
        model_convert_pipeline: Optional[Union[Dict, List]] = None,
        resume_optimizer: bool = False,
        resume_epoch_or_step: bool = False,
        resume_dataloader: bool = False,
        stop_by: Optional[str] = "epoch",
        num_epochs: Optional[int] = None,
        start_epoch: Optional[int] = 0,
        num_steps: Optional[int] = None,
        start_step: Optional[int] = 0,
        callbacks: Optional[Sequence[Union[dict, CallbackMixin]]] = None,
        train_metrics: Optional[dict] = None,
        val_metrics: Optional[dict] = None,
        profiler: Optional[dict] = None,
        log_interval: int = 0,
        compiler: Optional[Dict] = None,
    ):
        super(LoopBase, self).__init__(callbacks=callbacks, profiler=profiler)
        assert isinstance(device, int) or device is None, (
            "device should be int (gpu id) or None (run on cpu), but get %s"
            % type(device)
        )

        assert start_epoch >= 0, (
            f"{self.name} loop start epoch should be "
            f"non-negative integer, but get {start_epoch}"
        )
        assert start_step >= 0, (
            f"{self.name} loop start step should be "
            f"non-negative integer, but get {start_step}"
        )

        self.model = model
        self.ema_model = None
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.batch_processor = batch_processor
        self.train_metrics = _as_list(train_metrics)
        self.val_metrics = _as_list(val_metrics)
        self.device = device
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.num_steps = num_steps
        self.start_step = start_step
        self.log_interval = log_interval

        if model_convert_pipeline is not None:
            self.model = model_convert_pipeline(self.model)

        if elastic_need_resume():
            self.model = ElasticState.load_checkpoint(self.model)
            resume_epoch_or_step = True
            resume_optimizer = True
            resume_dataloader = True

        if callable(self.optimizer):
            self.optimizer = self.optimizer(self.model)

        stop_by = stop_by.lower()
        self._stop_by_epoch = stop_by == "epoch"
        self._stop_by_step = stop_by == "step"

        self.checkpoint = get_value("model_checkpoint")
        self._resume_from_checkpoint(
            resume_epoch_or_step,
            resume_optimizer,
            resume_dataloader,
        )

        _skip_loop = False

        if stop_by == "epoch":
            assert num_epochs is not None and self.num_epochs >= 0, (
                f"if stop {self.name} loop by counting epoch, num_epochs "
                f"should be non-negative integer, but get {num_epochs}"
            )
            if self.start_epoch >= num_epochs:
                logger.warning(
                    f"Start epoch {self.start_epoch} larger "
                    f"than num epochs {num_epochs}"
                )
                _skip_loop = True

        elif stop_by == "step":
            assert num_steps is not None and self.num_steps >= 0, (
                f"if stop {self.name} loop by counting step, num_steps "
                f"should be non-negative integer, but get {self.num_steps}"
            )
            if self.start_step >= num_steps:
                logger.warning(
                    f"Start step {self.start_step} larger than "
                    f"num steps {num_steps}"
                )
                _skip_loop = True

        else:
            raise ValueError(
                f"stop_by should be `epoch` or `step`, but get {stop_by}"
            )

        self._skip_loop = _skip_loop
        self.storage = EventStorage()

        if hasattr(self.profiler, "set_model"):
            self.profiler.set_model(self.model)

        if self.device is not None:
            with self.profiler.profile("set_device"):
                self.set_device(self.device)

        if compiler:
            self.model = compiler(self.model)

        if deterministic_level() == 2:
            cast_model_to_deterministic(self.model)

    def _resume_from_checkpoint(
        self,
        resume_epoch_or_step,
        resume_optimizer,
        resume_dataloader,
    ):
        """Resume from checkpoint.

        Get start epoch (step), optimizer states, learning rate and grad scalar
        state from checkpoint.
        """

        if self.checkpoint is not None:

            # check if we can resume from checkpoint
            resumable = checkpoint_resumable(self.checkpoint)

            if resume_epoch_or_step or resume_optimizer:
                assert (
                    resumable
                ), "Resume only when number of devices is consistent"

            if resume_optimizer:
                # update optimizer states from checkpoint
                self.optimizer.load_state_dict(self.checkpoint["optimizer"])

                if not resume_epoch_or_step:
                    for group in self.optimizer.param_groups:
                        assert (
                            "lr" in group
                        ), "Not found `lr` in a optimizer.param_groups"
                        group["initial_lr"] = group["lr"]

            if resume_epoch_or_step:
                # update self.start_epoch and self.start_step
                # from checkpoint to resume from stop point
                self.start_epoch = (
                    self.checkpoint["epoch"] + 1
                    if self._stop_by_epoch
                    else self.checkpoint["epoch"]
                )
                self.start_step = (
                    self.checkpoint["step"] + 1
                    if self._stop_by_step
                    else self.checkpoint["step"]
                )
                logger.info(
                    format_msg(
                        "reset starting point to epoch %d and step %d"
                        % (self.start_epoch, self.start_step),
                        MSGColor.GREEN,
                    )
                )

            # resume grad_scaler state_dict for amp
            grad_scaler_state = self.checkpoint.get("grad_scaler", {})
            if len(grad_scaler_state) > 0 and self.batch_processor is not None:
                self.batch_processor.grad_scaler.load_state_dict(
                    grad_scaler_state
                )

            if resume_dataloader and self._stop_by_step:
                set_value("dataloader_batch_size", self.data_loader.batch_size)
                set_value("dataloader_start_iter", self.start_step)
        else:
            if resume_epoch_or_step or resume_optimizer or resume_dataloader:
                msg = "You have set resume=True, but have not set the "
                " checkpoint to resume. By default, the training will"
                " start from scratch. If you want a resume training,"
                " please make sure to set a checkpoint to resume."

                logger.warning(
                    format_msg(
                        msg=msg,
                        color=MSGColor.RED,
                    )
                )

    def set_device(self, device):
        self.device = device
        self.model.cuda(device)

        for m in chain(self.train_metrics, self.val_metrics):
            if m is not None:
                m.to(device)

    def on_epoch_begin(self, epoch_id, **kwargs):
        if hasattr(self.data_loader, "sampler"):
            sampler = self.data_loader.sampler
            if isinstance(sampler, dict):
                sampler = sampler.values()
            else:
                sampler = _as_list(sampler)

            for sa in sampler:
                if isinstance(sa, DistributedSampler):
                    sa.set_epoch(epoch_id)

        # need to set device again in case device has been changed before epoch begins # noqa
        if self.device is not None:
            self.set_device(self.device)
        super(LoopBase, self).on_epoch_begin(epoch_id=epoch_id, **kwargs)

    def on_epoch_end(self, **kwargs):
        if get_value("dataloader_batch_size") is not None:
            set_value("dataloader_batch_size", None)
        if get_value("dataloader_start_iter") is not None:
            set_value("dataloader_start_iter", None)
        super(LoopBase, self).on_epoch_end(**kwargs)

    def fit(self):
        """Do model fitting on data from data_loader.

        `self.batch_processor` responsible for model forward, loss backward and
        parameters update.

        `self.callbacks` responsible for metric update, checkpoint, logging and
        so on.
        """
        if self._skip_loop:
            msg = (
                f"Skip {self.name} loop and only run `on_*_loop_begin` and "
                f"`on_*_loop_end` as num_epochs={self.num_epochs}, "
                f"num_steps={self.num_steps}, one of them is 0"
            )
        else:
            if self._stop_by_epoch:
                msg = (
                    f"Start {self.name} loop from epoch {self.start_epoch}, "
                    f"num_epochs={self.num_epochs}"
                )
            elif self._stop_by_step:
                msg = (
                    f"Start {self.name} loop from step {self.start_step}, "
                    f"num_steps={self.num_steps}"
                )
            else:
                raise NotImplementedError

        logger.info(format_msg(msg, MSGColor.GREEN))

        # local vars
        epoch_id = self.start_epoch
        global_step_id = self.start_step
        end_loop_flag = self._skip_loop

        # TODO(linkai.liang, 0.5), not pass LoopBase to callback, not do resume
        #  in `Checkpoint` callback
        self.on_loop_begin(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            num_epochs=self.num_epochs,
            num_steps=self.num_steps,
            loop=self,
            train_metrics=self.train_metrics,
            val_metrics=self.val_metrics,
            storage=self.storage,
            profiler=self.profiler,
        )

        while not end_loop_flag:
            self.data_loader_pr = self.profiler.profile_iterable(
                enumerate(prefetch_iterator(self.data_loader)),
                f"get_{self.name}_batch_data",
            )
            self.on_epoch_begin(
                model=self.model,
                epoch_id=epoch_id,
                optimizer=self.optimizer,
                global_step_id=global_step_id,
                train_metrics=self.train_metrics,
                val_metrics=self.val_metrics,
                storage=self.storage,
                data_loader=self.data_loader,
                profiler=self.profiler,
            )

            step_id = 0
            while True:
                if hasattr(self.profiler, "step"):
                    self.profiler.step()

                self.on_step_begin(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch_id=epoch_id,
                    step_id=step_id,
                    data_loader=self.data_loader,
                    start_epoch=self.start_epoch,
                    start_step=self.start_step,
                    global_step_id=global_step_id,
                    train_metrics=self.train_metrics,
                    val_metrics=self.val_metrics,
                    storage=self.storage,
                    profiler=self.profiler,
                )

                try:
                    _, batch = next(self.data_loader_pr)
                except StopIteration:
                    break

                if dist_initialized():
                    with self.profiler.profile(
                        "dataloader_distributed_action_barrier"
                    ):
                        torch.distributed.barrier()

                if self.log_interval > 0 and step_id % self.log_interval == 0:
                    logger.info(
                        f"{step_id} / {get_dataloader_length(self.data_loader)}"  # noqa E501
                    )

                self.batch_processor(
                    step_id,
                    batch,
                    self.model,
                    self.device,
                    optimizer=self.optimizer,
                    storage=self.storage,
                    batch_begin_callback=partial(
                        self.on_batch_begin,
                        global_step_id=global_step_id,
                        step_id=step_id,
                        epoch_id=epoch_id,
                        train_metrics=self.train_metrics,
                        val_metrics=self.val_metrics,
                        storage=self.storage,
                    ),
                    batch_end_callback=partial(
                        self.on_batch_end,
                        global_step_id=global_step_id,
                        step_id=step_id,
                        epoch_id=epoch_id,
                        train_metrics=self.train_metrics,
                        val_metrics=self.val_metrics,
                        storage=self.storage,
                    ),
                    backward_begin_callback=partial(
                        self.on_backward_begin,
                        model=self.model,
                        optimizer=self.optimizer,
                        global_step_id=global_step_id,
                        step_id=step_id,
                        epoch_id=epoch_id,
                    ),
                    backward_end_callback=partial(
                        self.on_backward_end,
                        model=self.model,
                        optimizer=self.optimizer,
                        global_step_id=global_step_id,
                        step_id=step_id,
                        epoch_id=epoch_id,
                    ),
                    optimizer_step_begin_callback=partial(
                        self.on_optimizer_step_begin,
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch_id=epoch_id,
                        step_id=step_id,
                        global_step_id=global_step_id,
                    ),
                    optimizer_step_end_callback=partial(
                        self.on_optimizer_step_end,
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch_id=epoch_id,
                        step_id=step_id,
                        global_step_id=global_step_id,
                    ),
                    profiler=self.profiler,
                    forward_begin_callback=partial(
                        self.on_forward_begin,
                        global_step_id=global_step_id,
                        step_id=step_id,
                        epoch_id=epoch_id,
                    ),
                    forward_end_callback=partial(
                        self.on_forward_end,
                        global_step_id=global_step_id,
                        step_id=step_id,
                        epoch_id=epoch_id,
                    ),
                )
                self.on_step_end(
                    epoch_id=epoch_id,
                    step_id=step_id,
                    global_step_id=global_step_id,
                    data_loader=self.data_loader,
                    model=self.model,
                    ema_model=self.ema_model,
                    optimizer=self.optimizer,
                    num_steps=self.num_steps,
                    device=self.device,
                    callbacks=self.callbacks,
                    train_metrics=self.train_metrics,
                    val_metrics=self.val_metrics,
                    storage=self.storage,
                    profiler=self.profiler,
                )

                step_id += 1
                global_step_id += 1
                if self._stop_by_step and global_step_id >= self.num_steps:
                    end_loop_flag = True
                    break

            self.on_epoch_end(
                epoch_id=epoch_id,
                global_step_id=global_step_id,
                model=self.model,
                ema_model=self.ema_model,
                optimizer=self.optimizer,
                num_epochs=self.num_epochs,
                device=self.device,
                callbacks=self.callbacks,
                train_metrics=self.train_metrics,
                val_metrics=self.val_metrics,
                storage=self.storage,
                profiler=self.profiler,
            )

            epoch_id += 1
            if self._stop_by_epoch and epoch_id >= self.num_epochs:
                end_loop_flag = True

        self.on_loop_end(
            model=self.model,
            ema_model=self.ema_model,
            optimizer=self.optimizer,
            epoch_id=epoch_id,
            global_step_id=global_step_id,
            device=self.device,
            train_metrics=self.train_metrics,
            val_metrics=self.val_metrics,
            callbacks=self.callbacks,
            storage=self.storage,
            profiler=self.profiler,
        )
        if deterministic_level() == 2:
            deterministic_summary()
        self.profiler.describe()
        self.profiler.teardown()

        if dist_initialized():
            torch.distributed.barrier()
