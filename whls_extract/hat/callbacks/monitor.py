# Copyright (c) Horizon Robotics. All rights reserved.

import datetime
import logging
import shutil
import time
from typing import Tuple

import numpy as np
import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.data_helpers import get_dataloader_length
from .callbacks import CallbackMixin

__all__ = ["StatsMonitor"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class StatsMonitor(CallbackMixin):  # noqa: D205,D400
    """
    StatsMonitor Callback is used for some monitors of training
    including epoch time, batch time and so on.

    Args:
        log_freq : Freq of monitor whether to output to log.
        batch_size : manually set 'batchsize' to deal with the fact that
                     the batchsize of the dataloader is not practical
        log_profiler: Whether to output profiler bottlenecks to log.
    """

    def __init__(self, log_freq=200, batch_size=None, log_profiler=False):
        self.log_freq = log_freq
        self.log_profiler = log_profiler
        self.gpu_monitor = True
        if shutil.which("nvidia-smi") is None:
            logger.warning(
                "Cannot monitor gpus because NVIDIA driver is not installed"
            )
            self.gpu_monitor = False

        self.step_time = AverageMeter("step_time", fmt=":6.3f")
        self.epoch_time = AverageMeter("epoch_time", fmt=":6.3f")

        self.step_time_begin = None
        self.epoch_time_begin = None

        self.begin_step = None
        self.batch_size = batch_size

        self.num_epochs = None
        self.num_steps = None

    def on_loop_begin(self, num_epochs, num_steps, **kwargs):
        if num_epochs == 0 and num_steps == 0:
            logger.warn(
                "num_epochs and num_steps are both 0, make sure you know "
                "what you are doing."
            )
        self.num_epochs = num_epochs
        self.num_steps = num_steps

    def on_step_begin(
        self,
        epoch_id,
        step_id,
        global_step_id,
        data_loader=None,
        profiler=None,
        **kwargs,
    ):
        if self.step_time_begin:
            step_time = time.time() - self.step_time_begin
            self.step_time_begin = time.time()
            self.step_time.update(step_time)

            should_log = (step_id + 1) % self.log_freq == 0
            if not should_log:
                return

            msg = "Epoch[%d] Step[%d-%d] Cost Time: %.3fs" % (
                epoch_id,
                self.begin_step,
                step_id,
                self.step_time.sum,
            )

            if data_loader is not None:
                speed = self._estimate_speed(data_loader, self.step_time.sum)
                msg += f" Speed: {speed:.2f} samples/sec"

                (
                    remain_training_time,
                    remain_step_percent,
                ) = self._estimate_remain_time_and_step(
                    data_loader,
                    epoch_id,
                    step_id,
                    global_step_id,
                )
                if remain_training_time >= 0:
                    remain_training_time = str(
                        datetime.timedelta(seconds=remain_training_time)
                    )
                    msg += f" Remaining Time: {remain_training_time}"
                if remain_step_percent > 0:
                    msg += (
                        " Remaining step percent"
                        + f": {remain_step_percent * 100:.2f}%"
                    )
            if (
                self.log_profiler
                and profiler is not None
                and hasattr(profiler, "recorded_durations")
            ):
                # only support simple profile
                recorded_durations = profiler.recorded_durations
                report = []
                for a, d in recorded_durations.items():
                    report.append([a, np.sum(d)])

                report.sort(key=lambda x: x[1], reverse=True)
                msg += (
                    f" Most time cost op: {report[0][0]}({report[0][1]:.2f}s);"
                )
                msg += f" {report[1][0]}({report[1][1]:.2f}s);"
                msg += f" {report[2][0]}({report[2][1]:.2f}s);"

            logger.info(msg)
            self.step_time.reset()
            self.begin_step = step_id + 1

        self.step_time_begin = time.time()

    def on_epoch_begin(self, epoch_id, **kwargs):
        self.epoch_time_begin = time.time()
        self.epoch_time.reset()
        self.step_time.reset()
        self.begin_step = 0
        self.step_time_begin = None
        logger.info("Epoch[%d] Begin " % epoch_id + "=" * 50)

    def on_epoch_end(self, epoch_id, **kwargs):
        logger.info("Epoch[%d] End   " % epoch_id + "=" * 50)
        epoch_time = time.time() - self.epoch_time_begin
        self.epoch_time.update(epoch_time)
        logger.info(
            "Epoch[%d] Cost Time: %.3fs" % (epoch_id, self.epoch_time.sum)
        )

    def _estimate_speed(
        self,
        data_loader: torch.utils.data.DataLoader,
        step_time: float,
    ) -> float:
        """
        Estimate training speed.

        Warning:
            If you are using accumulate_gradient (future feature), estimated
            speed may not be accurate. You need to update code here.
        """
        if self.batch_size is None:
            if hasattr(data_loader, "batch_size"):
                self.batch_size = data_loader.batch_size
            else:
                raise ValueError("{data_loader} has no `batch_size` property")
        speed = self.log_freq * self.batch_size / step_time
        return speed

    def _estimate_remain_time_and_step(
        self,
        data_loader: torch.utils.data.DataLoader,
        epoch_id: int,
        step_id: int,
        global_step_id: int,
    ) -> Tuple[int, float]:
        """Estimate remaining training time."""

        remaining_step_percent = -1.0
        if self.num_epochs:
            epoch_size = get_dataloader_length(data_loader)
            if epoch_size == float("inf"):
                return -1, remaining_step_percent

            remaining_steps = (self.num_epochs - epoch_id - 1) * epoch_size + (
                epoch_size - step_id - 1
            )
            remaining_step_percent = remaining_steps / (
                self.num_epochs * epoch_size
            )
        elif self.num_steps:
            remaining_steps = self.num_steps - global_step_id - 1
            remaining_step_percent = remaining_steps / self.num_steps
        else:
            raise ValueError(
                "One of (num_steps, num_epochs) " "should not be None"
            )

        training_time = int(remaining_steps * self.step_time.avg)
        return training_time, remaining_step_percent


class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
