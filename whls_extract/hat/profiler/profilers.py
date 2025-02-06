# Copyright (c) Horizon Robotics. All rights reserved.
"""
This file is modified from pytorch-lightning.

checking if there are any bottlenecks in your code.
"""
import contextlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TextIO,
    Tuple,
    Union,
)

import numpy as np
import plotly.graph_objects as go
import torch
import torch.distributed
from plotly.subplots import make_subplots

from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import (
    all_gather_object,
    get_comm_backend_name,
    get_dist_info,
    get_local_host,
)
from hat.utils.filesystem import get_filesystem
from hat.utils.logger import rank_zero_info

__all__ = [
    "BaseProfiler",
    "PassThroughProfiler",
    "SimpleProfiler",
]

logger = logging.getLogger(__name__)


class ProfilerAction(Enum):
    """Profiler actions that can be taken at the specified intervals."""

    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3


def schedule(
    *,
    active: int,
    warmup: int = 0,
    repeat: int = 0,
    skip_first: int = 0,
) -> Callable:
    """Return a callable func that can be used as profiler ``schedule`` argument.

    Note:
        The execution order is `skip_first` -> `wait` -> `warmup` -> `active`.
        If `repeat>0`, will do `repeat` times cycle.

    Args:
        skip_first: Steps to skip, which will not be profiled.
        warmup: Steps to warmup.
        active: Steps to record.
        repeat: The cycle to do profile. The optional number of cycles is
            specified with the ``repeat`` parameter, the zero value means that
            the cycles will continue until the profiling is finished.
    """

    def schedule_fn(step: int) -> ProfilerAction:
        assert step >= -1

        # corner case
        if skip_first == 0 and step == -1:
            return ProfilerAction.RECORD

        if step < skip_first:
            return ProfilerAction.NONE
        else:
            step -= skip_first
        num_steps = warmup + active
        if repeat > 0 and step / num_steps >= repeat:
            return ProfilerAction.NONE
        mod_step = step % num_steps

        if mod_step < warmup:
            return ProfilerAction.WARMUP
        else:
            return (
                ProfilerAction.RECORD
                if mod_step < num_steps - 1
                else ProfilerAction.RECORD_AND_SAVE
            )

    assert (
        warmup >= 0
        and active > 0
        and repeat >= 0
        and skip_first >= 0  # noqa E501
    ), (
        f"Invalid profiler schedule arguments, require `active>0` and "
        f"`warmup, repeat, skip_first >=0`, but get `active={active}`,"
        f"`warmup={warmup}`, `repeat={repeat}`, `skip_first={skip_first}`."
    )
    if warmup == 0:
        logger.warning(
            "Profiler won't be using warmup, this can skew profiler results"
        )
    return schedule_fn


def _default_schedule_fn(_: int) -> ProfilerAction:  # noqa D401
    """Default profiler behavior, keeps doing it on every profiler step."""
    return ProfilerAction.RECORD


record_actions_list = [ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE]


class AbstractProfiler(ABC):
    """Specification of a profiler."""

    @abstractmethod
    def start(self, action_name: str) -> None:
        """Define how to start recording an action."""

    @abstractmethod
    def stop(self, action_name: str) -> None:
        """Define how to record the duration once an action is complete."""

    @abstractmethod
    def summary(self) -> str:
        """Create profiler summary in text format."""

    @abstractmethod
    def setup(self, **kwargs: Any) -> None:  # noqa: D205,D400
        """Execute arbitrary pre-profiling set-up steps as
        defined by subclass.
        """

    @abstractmethod
    def teardown(self, **kwargs: Any) -> None:  # noqa: D205,D400
        """Execute arbitrary post-profiling tear-down steps as
        defined by subclass.
        """


class BaseProfiler(AbstractProfiler):  # noqa: D205,D400
    """
    If you wish to write a custom profiler, you should inherit
    from this class.
    """

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        auto_describe: bool = False,
        schedule: Optional[Callable[[int], ProfilerAction]] = None,
        summary_interval: int = -1,
        with_sync: bool = False,
    ) -> None:
        self.dirpath = dirpath
        self.filename = filename
        self.start_time = time.time()
        self._setup = False

        if summary_interval is None:
            summary_interval = -1

        assert (
            sum([summary_interval > 0, auto_describe, schedule is not None])
            <= 1
        ), "`auto_describe=True`, `summary_interval>0` and `schedule` can only be set to use one."  # noqa E501

        self.auto_describe = auto_describe
        self.summary_interval = summary_interval
        self._reset_after_summary = False

        if self.dirpath is None:
            self.dirpath = "./tmp_profiler_outputs"
        with contextlib.suppress(FileExistsError):
            os.makedirs(self.dirpath, exist_ok=True)

        self._output_file: Optional[TextIO] = None
        self._write_stream: Optional[Callable] = None
        self._local_rank: Optional[int] = None
        self._log_dir: Optional[str] = None
        self._stage: Optional[str] = None
        self._with_sync = with_sync

        if schedule:
            self._prof_schedule = schedule
            self._reset_after_summary = True
        else:
            self._prof_schedule = _default_schedule_fn
            if self.auto_describe or self.summary_interval > 0:
                self._reset_after_summary = True

        self._prof_step_num = -1
        self._pre_summary_step_num = 0
        self._summary_prefix = None

        self._prof_current_action = self._prof_schedule(self._prof_step_num)

    @contextmanager
    def profile(
        self,
        action_name: str,
    ) -> None:
        """
        Yield a context manager to encapsulate the scope of a profiled action.

        Example::

            with self.profile('load training data'):
                # load training data code

        The profiler will start once you've entered the context and will
        automatically stop once you exit the code block.
        """

        try:
            if self._prof_current_action in record_actions_list:
                self.start(action_name)
            yield action_name
        finally:
            if self._prof_current_action in record_actions_list:
                self.stop(action_name)

            if self._with_sync:
                if not self._action_should_skip_sync(action_name):
                    with self.profile(f"{action_name}_action_stream_sync"):
                        torch.cuda.synchronize()

                if not self._action_should_skip_sync(action_name):
                    with self.profile(f"{action_name}_action_barrier"):
                        torch.distributed.barrier()

    def profile_iterable(self, iterable, action_name: str) -> None:
        iterator = iter(iterable)
        while True:
            try:
                if self._prof_current_action in record_actions_list:
                    self.start(action_name)
                value = next(iterator)
                if self._prof_current_action in record_actions_list:
                    self.stop(action_name)

                if self._with_sync:
                    if not self._action_should_skip_sync(action_name):
                        with self.profile(f"{action_name}_action_stream_sync"):
                            torch.cuda.synchronize()

                    if not self._action_should_skip_sync(action_name):
                        with self.profile(f"{action_name}_action_barrier"):
                            torch.distributed.barrier()
                yield value

            except StopIteration:
                if self._prof_current_action in record_actions_list:
                    self.stop(action_name)

                if self._with_sync:
                    if not self._action_should_skip_sync(action_name):
                        with self.profile(f"{action_name}_action_stream_sync"):
                            torch.cuda.synchronize()

                    if not self._action_should_skip_sync(action_name):
                        with self.profile(f"{action_name}_action_barrier"):
                            torch.distributed.barrier()

                break

    def step(self):

        if self.auto_describe:
            self._summary_by_time_interval()

        if (
            self.summary_interval > 0
            and (self._prof_step_num + 1) % self.summary_interval == 0
        ) or self._prof_current_action == ProfilerAction.RECORD_AND_SAVE:
            self._summary_by_step_interval()
            self._pre_summary_step_num = self._prof_step_num + 1

        self._prof_pre_action = self._prof_schedule(self._prof_step_num)
        self._prof_step_num += 1
        self._prof_current_action = self._prof_schedule(self._prof_step_num)

    def _summary_by_time_interval(self):
        cur_time = time.time()
        cost_time = cur_time - self.start_time
        if (
            cost_time >= float(os.environ.get("HAT_MONITOR_INTERVAL", 600))
            and self._setup
        ):
            start_time = time.strftime(
                "%Y-%m-%d-%H-%M-%S", time.localtime(int(self.start_time))
            )
            end_time = time.strftime(
                "%Y-%m-%d-%H-%M-%S", time.localtime(int(cur_time))
            )
            self._summary_prefix = (
                f"\nTime[{start_time} ~ {end_time}] Profile Results:"
            )
            self.describe()
            self.start_time = time.time()

            if self._reset_after_summary:
                self.reset()

    def _summary_by_step_interval(self):

        self._summary_prefix = f"\nStep[{self._pre_summary_step_num} ~ {self._prof_step_num}] Profile Results:"  # noqa E501

        self.describe()

        if self._reset_after_summary:
            self.reset()

    def _prepare_filename(
        self,
        with_hostname: bool = False,
        extension: str = ".txt",
    ) -> str:
        filename = ""
        if self._stage is not None:
            filename += f"{self._stage}-"
        filename += str(self.filename)
        if with_hostname:
            filename += f"-{get_local_host()}"
        if self._local_rank is not None:
            filename += f"-{self._local_rank}"
        if extension is not None:
            filename += extension
        return filename

    def _prepare_streams(self) -> None:
        if self._write_stream is not None:
            return
        if self.filename:
            filepath = os.path.join(self.dirpath, self._prepare_filename())
            fs = get_filesystem(filepath)
            file = fs.open(filepath, "a")
            self._output_file = file
            self._write_stream = file.write
        else:
            self._write_stream = rank_zero_info

    def describe(self) -> None:
        """Log a profile report after the conclusion of run."""
        # there are pickling issues with open file handles in Python 3.6
        # so to avoid them, we open and close the files within this function
        # by calling `_prepare_streams` and `teardown`
        self._prepare_streams()
        summary = self.summary()
        if summary:
            if self._summary_prefix is not None:
                summary = self._summary_prefix + summary
            self._write_stream(summary)
        if self._output_file is not None:
            self._output_file.flush()

    def _stats_to_str(self, stats: Dict[str, str]) -> str:
        stage = f"{self._stage.upper()} " if self._stage is not None else ""
        output = [stage + "Profiler Report"]
        for action, value in stats.items():
            header = f"Profile stats for: {action}"
            if self._local_rank is not None:
                header += f" rank: {self._local_rank}"
            output.append(header)
            output.append(value)
        return os.linesep.join(output)

    def setup(
        self,
        stage: Optional[str] = None,
        local_rank: Optional[int] = None,
    ) -> None:
        """Execute arbitrary pre-profiling set-up steps."""
        self._setup = True
        self._stage = stage
        self._local_rank = local_rank

    def teardown(self) -> None:
        """
        Execute arbitrary post-profiling tear-down steps.

        Closes the currently open file and stream.
        """
        self._write_stream = None
        if self._output_file is not None:
            self._output_file.close()
            self._output_file = None  # can't pickle TextIOWrapper

    def __del__(self) -> None:
        self.teardown()

    def start(self, action_name: str) -> None:
        raise NotImplementedError

    def stop(self, action_name: str) -> None:
        raise NotImplementedError

    def summary(self) -> str:
        raise NotImplementedError

    def reset(self) -> None:
        if self.reset_after_summary:
            raise NotImplementedError
        else:
            pass

    @property
    def local_rank(self) -> int:
        return 0 if self._local_rank is None else self._local_rank

    @property
    def local_hostname(self) -> Union[str, None]:
        return get_local_host()

    def _match_any_record_func(
        self,
        action_name: str,
        record_funcs: Set[str],
        strict: bool = False,
    ):
        if strict:
            return any(
                action_name == func_name for func_name in list(record_funcs)
            )
        else:
            return any(
                action_name.startswith(func_name)
                for func_name in list(record_funcs)
            )

    def _action_should_skip_sync(self, action_name: str) -> bool:

        skip_sync_funcs = {"_action_stream_sync", "_action_barrier"}
        for action_suffix in skip_sync_funcs:
            if action_name.endswith(action_suffix):
                return True
        return False


@OBJECT_REGISTRY.register
class PassThroughProfiler(BaseProfiler):  # noqa: D205,D400
    """
    This class should be used when you don't want the (small) overhead of
    profiling. The Trainer uses this class by default.
    """

    def start(self, action_name: str) -> None:
        pass

    def stop(self, action_name: str) -> None:
        pass

    def summary(self) -> str:
        return ""

    def reset(self) -> None:
        pass


@OBJECT_REGISTRY.register
class SimpleProfiler(BaseProfiler):  # noqa: D205,D400
    """
    This profiler simply records the duration of actions (in seconds) and
    reports the mean duration of each action and the total time spent over
    the entire training run.
    """

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        warmup_step: int = 1,
        use_real_duration: bool = False,
        auto_describe: bool = False,
        schedule: Optional[Callable[[int], ProfilerAction]] = None,
        summary_interval: int = -1,
        timeline: bool = False,
        with_plot: bool = False,
        with_sync: bool = False,
    ) -> None:  # noqa: D205,D400
        """
        Args:
            dirpath: Directory path for the ``filename``.
            filename: If present, filename where the profiler results will
            be saved instead of printing to stdout. The ``.txt`` extension
            will be used automatically.
            warmup_step: Warmup steps for skipping statistics.
            use_real_duration: whether to use real duration in cal percentages.
                if not use real duration, the sum of all percentages is 100.
            auto_describe: Whether to automatically output summary
                information at regular intervals.
            schedule: A callable func that takes step (int) as a single
                parameter and returns ``ProfilerAction`` value that specifies
                the profiler action to perform at each step. For example:
                `hat.profiler.profilers.schedule(wait=0, warmup=0, active=10)`
            summary_interval: Step interval for print summary.
            timeline: Whether to record timeline info.
            with_plot: Whether to visualize the data.
            with_sync: Whether do sync after each action end.


        Raises:
            ValueError:
                If you attempt to start an action which has already started, or
                if you attempt to stop recording an action which was never
                started.
        """

        super(SimpleProfiler, self).__init__(
            dirpath=dirpath,
            filename=filename,
            auto_describe=auto_describe,
            summary_interval=summary_interval,
            schedule=schedule,
            with_sync=with_sync,
        )

        self.current_actions: Dict[str, float] = {}
        if schedule is not None:
            self.warmup_step = 0
        else:
            self.warmup_step = warmup_step
        self.use_real_duration = use_real_duration
        self.start_time = time.monotonic()
        self.timeline = timeline

        if self.timeline:
            self._enable_timeline()

        self.recorded_step_durations = defaultdict(defaultdict)
        self.with_plot = with_plot
        self._recorded_durations = None

    def start(self, action_name: str) -> None:
        if action_name in self.current_actions:
            raise ValueError(
                f"Attempted to start {action_name} "
                f"which has already started."
            )
        self.current_actions[action_name] = (
            time.monotonic(),
            # multinode training should use time.time()
            # to make sure timeline, "ms"
            time.time() * 1000,
        )

    def stop(self, action_name: str) -> None:
        end_time = time.monotonic()
        if action_name not in self.current_actions:
            raise ValueError(
                f"Attempting to stop recording an action "
                f"({action_name}) which was never started."
            )
        start_time, timeline_start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations[action_name].append(duration)
        if self.timeline:
            self.record_event(
                action_name, timeline_start_time, duration * 1000
            )
        self.recorded_step_durations[self._prof_step_num][
            action_name
        ] = duration

    def _enable_timeline(self):
        os.makedirs(
            os.path.join(self.dirpath, "raw_timeline_info"),
            exist_ok=True,
        )
        os.makedirs(os.path.join(self.dirpath, "timeline_info"), exist_ok=True)

        self.recorded_event = []
        _, world_size = get_dist_info()
        backend = get_comm_backend_name()
        self.comm_backend = backend
        self.world_size = world_size
        self._timeline_write_stream: Optional[Callable] = None
        self._timeline_output_file: Optional[TextIO] = None
        self.merge_list = []
        self.current_flow_event_id = 1
        self.step_id = 0
        self.step_start_time = time.time()
        self.engine_name = None
        self.total_merge_list = None

    def _tag_category(self, action_name: str) -> str:
        category = ""
        if "Trainer" in action_name:
            category += "Trainer,"
        if "Predictor" in action_name:
            category += "Trainer,"
        if "optimizer" in action_name:
            category += "Optimizer,"
        if "backward" in action_name:
            category += "Backward,"
        if "forward" in action_name:
            category += "Forward,"
        if "batch" in action_name:
            category += "Batch,"
        if "step" in action_name:
            category += "Step,"
        if "loop" in action_name:
            category += "Loop,"
        if "epoch" in action_name:
            category += "Epoch,"
        if "model" in action_name:
            category += "Model,"
        return category[:-1]

    def set_engine_name(self, engine_name: str) -> None:
        self.engine_name = engine_name

    def record_event(
        self, action_name: str, start_time: float, duration: float
    ):
        def create_flow_event(name, start_time):
            self.recorded_event.append(
                {
                    "name": name,
                    "ph": "s",
                    "id": self.current_flow_event_id,
                    "pid": self.local_hostname,
                    "tid": str(self.local_rank),
                    "ts": start_time,
                }
            )
            self.recorded_event.append(
                {
                    "name": name,
                    "ph": "f",
                    "bp": "e",
                    "id": self.current_flow_event_id,
                    "pid": self.local_hostname,
                    "tid": str(self.local_rank),
                    "ts": start_time,
                }
            )
            self.current_flow_event_id += 1

        category = self._tag_category(action_name)
        self.recorded_event.append(
            {
                "ph": "X",
                "cat": category,
                "name": action_name,
                "pid": self.local_hostname,
                "tid": str(self.local_rank),
                "ts": start_time,
                "dur": duration,
            }
        )
        if action_name == f"on_{self.engine_name}_step_begin":
            self.step_start_time = start_time
        elif action_name == f"on_{self.engine_name}_step_end":
            self.recorded_event.append(
                {
                    "ph": "X",
                    "cat": "ProfileStep",
                    "name": f"Profile Step {self.step_id}",
                    "pid": self.local_hostname,
                    "tid": str(self.local_rank),
                    "ts": self.step_start_time,
                    "dur": start_time + duration - self.step_start_time,
                }
            )
            self.step_id += 1
        if self.world_size > 1:
            if "model_forward" in action_name:
                create_flow_event("broadcast_forward", start_time)
            elif "model_backward" in action_name:
                create_flow_event("broadcast_backward", start_time)

    def reset(self) -> None:
        self.recorded_step_durations.clear()
        self._recorded_durations = None

        if self.timeline:
            # clear the recorded event
            self.recorded_event = []
            self._timeline_write_stream: Optional[Callable] = None
            self._timeline_output_file: Optional[TextIO] = None
            self.step_id = 0
            self.step_start_time = time.time()

    def _calculate_step_duration(
        self,
        step_data: defaultdict,
        pop_keys: Optional[List[str]] = None,
    ) -> float:
        if pop_keys is not None:
            pop_keys_set = set(pop_keys)
            filtered_values = (
                v for k, v in step_data.items() if k not in pop_keys_set
            )
        else:
            filtered_values = step_data.values()
        duration = sum(filtered_values)
        return duration

    def _get_max_duration_step_data(self, skip_step: int = -1):
        warmup_step = skip_step if skip_step >= 0 else self.warmup_step

        if (
            not self.recorded_step_durations
            or len(self.recorded_step_durations) <= warmup_step
        ):
            return None, 0, None

        filtered_steps = iter(self.recorded_step_durations.items())
        for _ in range(warmup_step):
            try:
                next(filtered_steps)
            except StopIteration:
                return None, 0, None

        # (step_num, max_duration, data)
        max_step_info = max(
            (
                (step, self._calculate_step_duration(data), data)
                for step, data in filtered_steps
            ),
            key=lambda x: x[1],
        )

        return max_step_info

    def _regroup_records(self):

        records = defaultdict(list)
        for _, step_data in self.recorded_step_durations.items():
            for action_name, cost in step_data.items():
                records[action_name].append(cost)

        return records

    def _make_report(self, skip_step: int = -1) -> Tuple[list, float]:
        warmup_step = skip_step if skip_step >= 0 else self.warmup_step
        real_total_duration = time.monotonic() - self.start_time

        report = []
        total_duration = 0
        for a, d in self.recorded_durations.items():
            # skip the warmup iter profile
            if len(d) > warmup_step:
                new_d = d[warmup_step:]
                skip_d = d[:warmup_step]
            else:
                new_d = d
                skip_d = []
            total_duration += np.sum(new_d)
            report.append([a, new_d, 100.0 * np.sum(new_d), skip_d])
            self.recorded_durations[a] = new_d

        if self.use_real_duration:
            total_duration = real_total_duration
        report.sort(key=lambda x: x[2], reverse=True)
        return report, total_duration

    def summary(self) -> str:

        sep = os.linesep
        output_string = ""
        output_string += (
            f"{sep}{str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}"
        )

        if len(self.recorded_durations) > 0:
            max_key = np.max([len(k) for k in self.recorded_durations.keys()])

            def log_row(
                action, mean, num_calls, total, per, skip_calls, skip_total
            ):
                row = f"{sep}{action:<{max_key}s}\t|  {mean:<15}\t|"
                row += f"{num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|"
                row += f"{skip_calls:<15}\t|  {skip_total:<15}\t|"
                return row

            output_string += log_row(
                "Action",
                "Mean duration (s)",
                "Num calls",
                "Total time (s)",
                "Percentage %",
                "Skip num calls",
                "Skip total time (s)",
            )
            output_string_len = len(output_string)
            output_string += f"{sep}{'-' * output_string_len}"
            report, total_duration = self._make_report()
            output_string += log_row(
                "Total", "-", "-", f"{total_duration:.5}", "100 %", "-", "-"
            )
            output_string += f"{sep}{'-' * output_string_len}"
            for action, durations, duration_per, skip_durations in report:
                output_string += log_row(
                    action,
                    f"{np.mean(durations):.5}",
                    f"{len(durations):}",
                    f"{np.sum(durations):.5}",
                    f"{duration_per / total_duration:.5}",
                    f"{len(skip_durations):}",
                    f"{np.sum(skip_durations):.5}",
                )

            output_string += f"{sep}{'-' * output_string_len}"
            (
                max_step_idx,
                max_cost_time,
                max_step_record,
            ) = self._get_max_duration_step_data()

            if max_step_idx and max_step_record:
                output_string += (
                    f"\nMax Step: {max_step_idx}, CostTime: {max_cost_time}s"
                )
                output_string += f"{sep}{'-' * output_string_len}"

                for action, duration in max_step_record.items():
                    output_string += (
                        f"{sep}{action:<{max_key}s}\t|  {duration:.5}\t|"
                    )
                output_string += f"{sep}{'-' * output_string_len}"
        output_string += sep
        if self.with_plot:
            self.plot_and_save_fig()

        # timline summary
        if self.timeline:
            # dump timeline tracing log
            self._prepare_json_streams()
            summary_dict = self.summary_timeline_info()
            if summary_dict:
                self._timeline_write_stream(summary_dict)
            if self._timeline_output_file is not None:
                self._timeline_output_file.flush()

            if self.world_size > 1:
                self.total_merge_list = [None for _ in range(self.world_size)]
                all_gather_object(self.total_merge_list, self.merge_list)
        return output_string

    def _prepare_timeline_filename(
        self,
        with_hostname: bool = False,
        extension: str = ".json",
    ) -> str:
        filename = ""
        if self._stage is not None:
            filename += f"{self._stage}-"
        filename += str(self.filename)
        if with_hostname:
            filename += f"-{get_local_host()}"
        if self.local_rank is not None:
            filename += f"-{self.local_rank}"
        filename += f"-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        if extension is not None:
            filename += extension
        return filename

    def _prepare_json_streams(self) -> None:
        if self.filename:
            filepath = os.path.join(
                self.dirpath,
                "raw_timeline_info",
                self._prepare_timeline_filename(
                    with_hostname=True, extension=".json"
                ),
            )
            self.merge_list.append(filepath)
            fs = get_filesystem(filepath)
            file = fs.open(filepath, "a")
            self._timeline_output_file = file
            self._timeline_write_stream = partial(json.dump, fp=file)
        else:
            self._timeline_write_stream = rank_zero_info

    def summary_timeline_info(self) -> Dict[str, Any]:
        summary = {
            "schemaVersion": 1,
            "distributedInfo": {
                "backend": self.comm_backend,
                "rank": self.local_rank,
                "world_size": self.world_size,
            },
            "summary_prefix": self._summary_prefix
            if self._summary_prefix is not None
            else "",
            "traceEvents": [
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": self.local_hostname,
                    "tid": str(self.local_rank),
                    "args": {"name": self.local_hostname},
                },
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": self.local_hostname,
                    "tid": str(self.local_rank),
                    "args": {"name": f"Rank: {self.local_rank}"},
                },
            ]
            + self.recorded_event,
        }
        return summary

    def _merge_all_timeline_info(self) -> None:
        dump_dirpath = os.path.join(self.dirpath, "timeline_info")
        for i in range(len(self.merge_list)):
            current_step_timeline_info = {
                "schemaVersion": 1,
                "displayTimeUnit": "ms",
                "distributedInfo": {
                    "backend": self.comm_backend,
                    "rank": self.local_rank,
                    "world_size": self.world_size,
                },
                "summary_prefix": "",
                "traceEvents": [],
            }
            current_summary_prefix = ""
            for rank_merge_list in self.total_merge_list:
                merge_file_path = rank_merge_list[i]
                with open(merge_file_path, "r") as f:
                    timeline_info = json.load(f)
                current_step_timeline_info["traceEvents"].extend(
                    timeline_info["traceEvents"]
                )
                current_step_timeline_info["summary_prefix"] = timeline_info[
                    "summary_prefix"
                ]
                current_summary_prefix = timeline_info["summary_prefix"]
            if current_summary_prefix != "":
                filename = os.path.join(dump_dirpath, f"Step-{i}.json")
            else:
                start = (
                    current_summary_prefix.split("~")[0].split("[")[1].strip()
                )
                end = (
                    current_summary_prefix.split("~")[1].split("]")[0].strip()
                )
                filename = os.path.join(
                    dump_dirpath, f"{start}-{end}" + ".json"
                )
            with open(filename, "w") as f:
                json.dump(current_step_timeline_info, f)

    def teardown(self) -> None:
        # only exec in main process rank 0
        if hasattr(self, "total_merge_list"):
            if self.timeline and self.world_size > 1 and self.local_rank == 0:
                self._merge_all_timeline_info()
        return super().teardown()

    def plot_and_save_fig(self):

        output_dir = os.path.join(self.dirpath, "plot")
        os.makedirs(output_dir, exist_ok=True)

        action_names = list(self.recorded_durations.keys())

        num_rows = len(action_names) + 1

        titles = ["Step CostTime"] + action_names

        fig = make_subplots(
            rows=num_rows,
            cols=1,
            subplot_titles=titles,
        )

        step_nums = list(self.recorded_step_durations.keys())
        if len(step_nums) <= 0:
            logger.warning("`recorded_step_durations` is empty, skip plot")
            return
        # per step cost
        fig.add_trace(
            go.Scatter(
                x=step_nums,
                y=list(
                    map(
                        lambda x: round(self._calculate_step_duration(x), 2),
                        list(self.recorded_step_durations.values()),
                    )
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="step num", row=1, col=1)
        fig.update_yaxes(title_text="cost time (s)", row=1, col=1)

        for idx, action_name in enumerate(action_names, 2):
            durations = self.recorded_durations[action_name]
            if len(durations) > self.warmup_step:
                x_data = step_nums[self.warmup_step :]
            else:
                x_data = step_nums
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=durations,
                    showlegend=False,
                ),
                row=idx,
                col=1,
            )
            fig.update_xaxes(title_text="step num", row=idx, col=1)
            fig.update_yaxes(title_text="cost time (s)", row=idx, col=1)

        fig.update_layout(height=300 * num_rows)

        file_name = (
            f"duration-statistics-{self.local_hostname}-Rank{self.local_rank}"
            f"-Step[{step_nums[0]}-{step_nums[-1]}].html"
        )
        pic_path = os.path.join(output_dir, file_name)
        try:
            fig.write_html(pic_path)
        except Exception as e:
            logger.error(f"Error saving {pic_path}: {e}")

    @property
    def recorded_durations(self):
        if self._recorded_durations is None:
            self._recorded_durations = self._regroup_records()

        return self._recorded_durations
