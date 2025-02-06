# Copyright (c) Horizon Robotics. All rights reserved.

"""
Memory Profiling.

Help profiling the GPU or CPU memory bottleneck in the process
of model training.
"""
import contextlib
import io
import logging
import os
import pickle
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Union

import plotly.graph_objects as go

try:
    import memray
except ImportError:
    memray = None
import numpy as np
import psutil
import torch

try:
    from torch.cuda._memory_viz import _write_blocks, format_flamegraph
except ImportError:
    _write_blocks, format_flamegraph = None, None

from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_local_host
from hat.utils.package_helper import check_packages_available, require_packages
from .profilers import BaseProfiler, ProfilerAction

logger = logging.getLogger(__name__)
__all__ = ["GPUMemoryProfiler", "CPUMemoryProfiler", "StageCPUMemoryProfiler"]


DEFAULT_RECORD_FUNCS = {
    "optimizer_zero_grad",
    "batch_transforms",
    "model_forward",
    "model_backward",
    "optimizer_step",
}


def match_any_func(action_name: str, record_funcs: Set[str]):
    return any(
        action_name.startswith(func_name) for func_name in list(record_funcs)
    )


class GPUMemorySnapshot:
    """Wrapper to take gpu memory snapshot.

    Args:
        output_dir: Dir path to save output files.
        record_snapshot: Whether to record snapshot. Defaults to False.
        snapshot_interval: Step interval to save snapshot. Defaults to 1.
        save_to_svg: Whether to convert to svg format, which can be
            visualized with a browser. Defaults to True.
        record_functions: Set of profiles functions to take snapshot.
    """

    def __init__(
        self,
        output_dir: str,
        record_snapshot: bool = False,
        snapshot_interval: int = 1,
        save_to_svg: bool = True,
        record_functions: Optional[Set[str]] = None,
        **kwargs,
    ) -> None:
        if check_packages_available("torch>=1.13.0", raise_exception=False):
            self.record_snapshot = record_snapshot
        else:
            logger.warning(
                f"Memory snapshot require torch >= 1.13.0, but in your "
                f"environment get torch=={torch.__version__}, will set "
                f"`record_snapshot=False` by default."
            )
            self.record_snapshot = False

        self.snapshot_steps = defaultdict()
        self.snapshot_interval = snapshot_interval
        self.snapshots = defaultdict(defaultdict)
        self.output_dir = output_dir
        self.save_to_svg = save_to_svg

        self.record_functions = (
            record_functions if record_functions else DEFAULT_RECORD_FUNCS
        )

        if self.record_snapshot:
            torch.cuda.memory._record_memory_history(True)
            self.resister_oom_observer(output_dir=self.output_dir)

    def start_action(self, action_name: str):
        if self.record_snapshot and match_any_func(
            action_name, self.record_functions
        ):
            step = self.snapshot_steps.get(action_name, -1)
            self.snapshot_steps[action_name] = step + 1

    def stop_action(
        self,
        action_name: str,
        device: Optional[int] = None,
    ):
        snapshot = None
        if self.record_snapshot and match_any_func(
            action_name, self.record_functions
        ):
            step = self.snapshot_steps.get(action_name, -1)

            if (step + 1) % self.snapshot_interval == 0:
                snapshot = torch.cuda.memory._snapshot(device)
                self.snapshots[str(step)][action_name] = snapshot

                prefix = f"snapshot-{get_local_host()}-rank{device}-step{step}-{action_name}"  # noqa E501
                GPUMemorySnapshot.save_as_flamegraph_data(
                    snapshot=snapshot,
                    output_dir=os.path.join(self.output_dir, "snapshots"),
                    device=device,
                    file_prefix=prefix,
                    save_to_svg=self.save_to_svg,
                )
        return snapshot

    def get_snapshots(self):
        return self.snapshots

    @staticmethod
    def save_as_flamegraph_data(
        snapshot: List[Any],
        output_dir: str,
        device: int,
        file_prefix: str = None,
        save_to_svg: bool = False,
        **kwargs,
    ):
        """Convert the snapshot into a format that can be read by frame graph.

        Args:
            snapshot: Snapshot data.
            output_dir: Output dir to save files.
            file_prefix: Prefix name of saved files.
            device: Rank index.
            save_to_svg: Whether to convert svg. Defaults to False.
        """
        assert (
            check_packages_available("torch>=1.13.0", raise_exception=False)
            and _write_blocks
        ), (
            f"Memory snapshot require torch >= 1.13.0 and has "
            f"`torch.cuda._memory_viz._write_blocks`, but in your environment"
            f" get torch=={torch.__version__}."
        )

        if file_prefix is None:
            file_prefix = f"snapshot-{get_local_host()}-rank{device}"

        with contextlib.suppress(FileExistsError):
            os.makedirs(output_dir, exist_ok=True)

        f = io.StringIO()
        if "segments" in snapshot:
            mem_snapshot = snapshot["segments"]
        else:
            mem_snapshot = snapshot

        for seg in mem_snapshot:
            prefix = f'stream_{seg["stream"]}'
            _write_blocks(f, prefix, seg["blocks"])

        # dump to txt file, which can be read by frame graph
        graph_data_file = os.path.join(output_dir, file_prefix + ".txt")

        with open(graph_data_file, "w") as txt_file:
            txt_file.writelines(f.getvalue())

        # convert to svg
        if save_to_svg:
            try:
                # Note: May fail due to network reasons.
                svg_file = os.path.join(output_dir, file_prefix + ".svg")
                flame_graph = format_flamegraph(f.getvalue())
                with open(svg_file, "w") as svg_file:
                    svg_file.write(flame_graph)
            except Exception as e:
                logger.warning(
                    f"Failed to save convert flame graph to svg: {e}"
                )

    def resister_oom_observer(
        self,
        output_dir: str,
        device: Optional[int] = None,
    ):  # noqa D403
        """OutOfMemoryError observer.

        Note:
          From https://zdevito.github.io/2022/08/16/memory-snapshots.html,
          the reason for OutOfMemoryError can be captured by registering the
          observer, but there was a bug in torch1.13.0, so this will not take
          effect in torch1.13.0.

        Args:
            output_dir: Output dir to save files.
            device: Rank.
        """

        def _oom_observer():
            # snapshot right after an OOM happened
            logger.info("saving allocated state during OOM")
            snapshot = torch.cuda.memory._snapshot(device=device)
            omm_snapshot_pkl = os.path.join(
                output_dir,
                f"oom_snapshot_{get_local_host()}-rank{device}.pkl",
            )
            with open(omm_snapshot_pkl, "wb") as f:
                pickle.dump(snapshot, f)

        try:
            torch._C._cuda_attach_out_of_memory_observer(_oom_observer)
        except AttributeError as e:
            # has bug in torch 1.13.0
            logger.warning(f"Failed to register oom observer: {e}")


@OBJECT_REGISTRY.register
class GPUMemoryProfiler(BaseProfiler):
    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        record_snapshot: bool = False,
        snapshot_interval: int = 1,
        record_functions: Set[str] = None,
        auto_describe: bool = False,
        schedule: Optional[Callable[[int], ProfilerAction]] = None,
        summary_interval: int = -1,
        with_plot: bool = False,
    ):
        """
        GPU Memory Profiler.

        Args:
            dirpath: Directory path for the ``filename``.
            filename: If present, filename where the profiler results will
            be saved instead of printing to stdout. The ``.txt`` extension
            will be used automatically.
            record_snapshot: Whether to record snapshot. Defaults to False.
            snapshot_interval: Step interval to save snapshot. Defaults to 1.
            record_functions: Set of profiles functions to take snapshot.
            auto_describe: Whether to automatically output summary
                information at regular intervals.
            schedule: A callable func that takes step (int) as a single
                parameter and returns ``ProfilerAction`` value that specifies
                the profiler action to perform at each step. For example:
                `hat.profiler.profilers.schedule(wait=0, warmup=0, active=10)`
            summary_interval: Step interval for print summary.
            with_plot: Whether to visualize the data.


        Raises:
            ValueError:
                If you attempt to start an action which has already started, or
                if you attempt to stop recording an action which was never
                started.
        """
        super(GPUMemoryProfiler, self).__init__(
            dirpath=dirpath,
            filename=filename,
            auto_describe=auto_describe,
            summary_interval=summary_interval,
            schedule=schedule,
        )
        if check_packages_available("torch>=1.10.2", raise_exception=False):
            self.memory_metric_list = [
                "memory_allocated",
                "memory_reserved",
                "max_memory_allocated",
                "max_memory_reserved",
                "nvidia_smi",
            ]
            self.show_nvidia_smi = True
        else:
            self.memory_metric_list = [
                "memory_allocated",
                "memory_reserved",
                "max_memory_allocated",
                "max_memory_reserved",
            ]
            self.show_nvidia_smi = False
        self.with_plot = with_plot
        self.current_actions = set()
        self.recorded_memory = defaultdict()

        self.reset()

        self.record_functions = (
            record_functions if record_functions else DEFAULT_RECORD_FUNCS
        )

        self.memory_snapshot = GPUMemorySnapshot(
            record_snapshot=record_snapshot,
            snapshot_interval=snapshot_interval,
            output_dir=self.dirpath,
            device=self.local_rank,
            save_to_svg=True,
            record_functions=self.record_functions,
        )

    def start(self, action_name: str) -> None:
        if action_name in self.current_actions:
            raise ValueError(
                f"Attempted to start {action_name} "
                f"which has already started."
            )
        self.current_actions.add(action_name)
        # Resets the starting point in tracking maximum GPU memory occupied
        # by tensors for a given device.
        torch.cuda.reset_peak_memory_stats()

        self.memory_snapshot.start_action(action_name)

    def stop(self, action_name: str) -> None:
        if action_name not in self.current_actions:
            raise ValueError(
                f"Attempting to stop recording an action "
                f"{action_name} which was never started."
            )
        self.current_actions.remove(action_name)
        for memory_metric in self.memory_metric_list:
            if memory_metric == "memory_allocated":
                memory = torch.cuda.memory_allocated() / 1024.0 / 1024.0
            elif memory_metric == "memory_reserved":
                memory = torch.cuda.memory_reserved() / 1024.0 / 1024.0
            elif memory_metric == "max_memory_allocated":
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            elif memory_metric == "max_memory_reserved":
                memory = torch.cuda.max_memory_reserved() / 1024.0 / 1024.0
            elif memory_metric == "nvidia_smi":
                # torch.cuda.mem_get_info() exists when torch_version >= 1.10
                free_memory, total_memory = torch.cuda.mem_get_info()
                memory = (total_memory - free_memory) / 1024.0 / 1024.0
            self.recorded_memory[memory_metric][action_name].append(memory)

        self.memory_snapshot.stop_action(action_name, device=self.local_rank)

    def reset(self):
        self.recorded_memory.clear()
        for memory_metric in self.memory_metric_list:
            self.recorded_memory[memory_metric] = defaultdict(list)

    def _make_report(self):
        action_names = list(self.recorded_memory["memory_allocated"].keys())
        report = [
            [
                action_name,
            ]
            + [
                self.recorded_memory[memory_metric][action_name]
                for memory_metric in self.memory_metric_list
            ]
            for action_name in action_names
        ]
        return report

    def summary(self):
        sep = os.linesep
        output_string = ""
        if self._stage is not None:
            output_string += f"{self._stage.upper()} "
        output_string += f"GPU Memory Profiler Report{sep}"
        if len(self.recorded_memory["memory_allocated"].keys()) > 0:
            max_key = np.max(
                [
                    len(k)
                    for k in self.recorded_memory["memory_allocated"].keys()
                ]
            )

            def log_row(action_name, *args):
                row = f"{sep}{action_name:<{max_key}s}\t|  "
                for arg in args:
                    row += f"{arg:<22}\t|  "
                return row

            if self.show_nvidia_smi:
                first_line = log_row(
                    "Action",
                    "memory_allocated (M)",
                    "memory_reserved (M)",
                    "max_memory_allocated (M)",
                    "max_memory_reserved (M)",
                    "nvidia_smi (M)",
                    "num_calls",
                )
            else:
                first_line = log_row(
                    "Action",
                    "memory_allocated (M)",
                    "memory_reserved (M)",
                    "max_memory_allocated (M)",
                    "max_memory_reserved (M)",
                    "num_calls",
                )
            output_string += first_line
            output_string_len = len(first_line)
            output_string += f"{sep}{'-' * output_string_len}"
            report = self._make_report()

            if self.with_plot:
                self.plot_and_save_fig()

            if self.show_nvidia_smi:
                for (
                    action_name,
                    memory_allocated,
                    memory_reserved,
                    max_memory_allocated,
                    max_memory_reserved,
                    nvidia_smi,
                ) in report:
                    output_string += log_row(
                        action_name,
                        f"{np.mean(memory_allocated)}",
                        f"{np.mean(memory_reserved)}",
                        f"{np.mean(max_memory_allocated)}",
                        f"{np.mean(max_memory_reserved)}",
                        f"{np.mean(nvidia_smi)}",
                        f"{len(memory_allocated)}",
                    )
            else:
                for (
                    action_name,
                    memory_allocated,
                    memory_reserved,
                    max_memory_allocated,
                    max_memory_reserved,
                ) in report:
                    output_string += log_row(
                        action_name,
                        f"{np.mean(memory_allocated)}",
                        f"{np.mean(memory_reserved)}",
                        f"{np.mean(max_memory_allocated)}",
                        f"{np.mean(max_memory_reserved)}",
                        f"{len(memory_allocated)}",
                    )
            output_string += f"{sep}{'-' * output_string_len}"
            output_string += f"{sep}memory_allocated: Returns the current GPU memory occupied by tensors in bytes for a given device."  # noqa: E501
            output_string += f"{sep}memory_reserved: Returns the current GPU memory managed by the caching allocator in bytes for a given device."  # noqa: E501
            output_string += f"{sep}max_memory_allocated: Returns the maximum GPU memory occupied by tensors in bytes for a given device."  # noqa: E501
            output_string += f"{sep}max_memory_reserved: Returns the maximum GPU memory managed by the caching allocator in bytes for a given device."  # noqa: E501
            output_string += f"{sep}Reference: https://pytorch.org/docs/stable/cuda.html#memory-management"  # noqa: E501
        output_string += sep
        self.save_snapshots()
        return output_string

    def plot_and_save_fig(self):
        output_dir = os.path.join(self.dirpath, "plot")
        os.makedirs(output_dir, exist_ok=True)

        for memory_name in self.memory_metric_list:

            fig = go.Figure()
            fig.update_layout(
                title=memory_name,
                xaxis_title="iters",
                yaxis_title="GPU Memory (M)",
            )

            action_names_memory = self.recorded_memory[memory_name]
            for action_name, action_name_memory in action_names_memory.items():
                if len(action_name_memory) > 1:
                    x = list(range(len(action_name_memory)))
                    y = action_name_memory
                    fig.add_trace(go.Scatter(x=x, y=y, name=action_name))
                else:
                    continue

            file_name = (
                f"GPUMemory-{self.local_hostname}-{memory_name}-Rank{self.local_rank}"  # noqa E501
                f"-Step[{self._pre_summary_step_num} - {self._prof_step_num}].html"  # noqa E501
            )

            pic_path = os.path.join(output_dir, file_name)
            try:
                fig.write_html(pic_path)
            except Exception as e:
                logger.error(f"Error saving {pic_path}: {e}")

        action_names = list(self.recorded_memory["memory_allocated"].keys())
        for action_name in action_names:
            if (
                not len(self.recorded_memory["memory_allocated"][action_name])
                > 1
            ):
                continue
            fig = go.Figure()
            fig.update_layout(
                title=action_name,
                xaxis_title="iters",
                yaxis_title="GPU Memory (M)",
            )
            for memory_name in self.memory_metric_list:
                action_name_memory = self.recorded_memory[memory_name][
                    action_name
                ]
                x = list(range(len(action_name_memory)))
                y = action_name_memory
                fig.add_trace(go.Scatter(x=x, y=y, name=memory_name))

            file_name = (
                f"GPUMemory-{self.local_hostname}-{action_name}-Rank{self.local_rank}"  # noqa E501
                f"-Step[{self._pre_summary_step_num} - {self._prof_step_num}].html"  # noqa E501
            )
            pic_path = os.path.join(output_dir, file_name)

            try:
                fig.write_html(pic_path)
            except Exception as e:
                logger.error(f"Error saving {pic_path}: {e}")

    def save_snapshots(self):
        """Dump all snapshots.

        snapshots format (dict(dict)):
            {

                step_id : {

                    action_name: [...]

                },

            }
        """

        pkl_path = os.path.join(
            self.dirpath,
            f"snapshot_{self.local_hostname}-rank{self.local_rank}.pkl",
        )
        if check_packages_available("torch>=2.1.0", raise_exception=False):
            try:
                torch.cuda.memory._dump_snapshot(f"{pkl_path}")
            except Exception as e:
                logger.error(f"Failed to capture memory snapshot: {e}")
        else:
            snapshots = self.memory_snapshot.get_snapshots()
            if len(snapshots) > 0:
                with open(pkl_path, "wb") as f:
                    pickle.dump(snapshots, f)
            else:
                logger.warning("There is no value in snapshots, skip dump.")


@OBJECT_REGISTRY.register
class CPUMemoryProfiler(BaseProfiler):
    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        auto_describe: bool = False,
        schedule: Optional[Callable[[int], ProfilerAction]] = None,
        summary_interval: int = -1,
        with_plot: bool = False,
    ):
        """
        CPU Memory Profiler.

        Args:
            dirpath: Directory path for the ``filename``.
            filename: If present, filename where the profiler results will
            be saved instead of printing to stdout. The ``.txt`` extension
            will be used automatically.
            auto_describe: Whether to automatically output summary
                information at regular intervals.
            schedule: A callable func that takes step (int) as a single
                parameter and returns ``ProfilerAction`` value that specifies
                the profiler action to perform at each step. For example:
                `hat.profiler.profilers.schedule(wait=0, warmup=0, active=10)`
            summary_interval: Step interval for print summary.
            with_plot: Whether to visualize the data.


        Raises:
            ValueError:
                If you attempt to start an action which has already started, or
                if you attempt to stop recording an action which was never
                started.
        """
        super(CPUMemoryProfiler, self).__init__(
            dirpath=dirpath,
            filename=filename,
            auto_describe=auto_describe,
            summary_interval=summary_interval,
            schedule=schedule,
        )
        self.with_plot = with_plot
        self.memory_metric_list = [
            "action_start",
            "action_end",
            "action_delta",
        ]

        self.current_actions = set()
        self.recorded_memory = defaultdict()
        self.reset()

    def get_process_mem_rss(self):
        processes = []
        p = psutil.Process()
        processes.append(p)
        processes.extend(p.children(recursive=True))
        mem_rss = 0.0
        for process in processes:
            cmd = 'cat /proc/{}/status | grep -E "RssAnon"'.format(process.pid)
            status, output = subprocess.getstatusoutput(cmd)
            if status != 0:
                logger.error(output)
                continue
            mem_rss += int(output.split()[1]) / 1024.0
        return mem_rss

    def start(self, action_name: str) -> None:
        if action_name in self.current_actions:
            raise ValueError(
                f"Attempted to start {action_name} "
                f"which has already started."
            )
        self.current_actions.add(action_name)
        # Get cpu rss memory.
        memory_rss = self.get_process_mem_rss()
        self.recorded_memory["action_start"][action_name].append(memory_rss)

    def stop(self, action_name: str) -> None:
        if action_name not in self.current_actions:
            raise ValueError(
                f"Attempting to stop recording an action "
                f"{action_name} which was never started."
            )
        self.current_actions.remove(action_name)
        memory_rss = self.get_process_mem_rss()
        action_delta_rss = (
            memory_rss - self.recorded_memory["action_start"][action_name][-1]
        )
        self.recorded_memory["action_end"][action_name].append(memory_rss)
        self.recorded_memory["action_delta"][action_name].append(
            action_delta_rss
        )

    def reset(self):
        self.recorded_memory.clear()
        for memory_metric in self.memory_metric_list:
            self.recorded_memory[memory_metric] = defaultdict(list)

    def _make_report(self):
        action_names = list(self.recorded_memory["action_start"].keys())
        report = [
            [
                action_name,
            ]
            + [
                self.recorded_memory[memory_metric][action_name]
                for memory_metric in self.memory_metric_list
            ]
            for action_name in action_names
        ]
        return report

    def summary(self):
        sep = os.linesep
        output_string = ""
        if self._stage is not None:
            output_string += f"{self._stage.upper()} "
        output_string += f"CPU Memory Profiler Report{sep}"
        if len(self.recorded_memory["action_start"].keys()) > 0:
            max_key = np.max(
                [len(k) for k in self.recorded_memory["action_start"].keys()]
            )

            def log_row(action_name, *args):
                row = f"{sep}{action_name:<{max_key}s}\t|  "
                for arg in args:
                    row += f"{arg:<22}\t|  "
                return row

            first_line = log_row(
                "Action",
                "action_start (M)",
                "action_end (M)",
                "action_delta (M)",
            )
            output_string += first_line
            output_string_len = len(first_line)
            output_string += f"{sep}{'-' * output_string_len}"
            report = self._make_report()

            if self.with_plot:
                self.plot_and_save_fig()

            max_iter = 0

            for (
                action_name,
                action_start,
                action_end,
                action_delta,
            ) in report:
                output_string += log_row(
                    action_name,
                    f"{np.mean(action_start)}",
                    f"{np.mean(action_end)}",
                    f"{np.mean(action_delta)}",
                )
                max_iter = max(max_iter, len(action_start))
            output_string += f"{sep}{'-' * output_string_len}"
            output_string += (
                f"{sep}action_start: Returns the current CPU "
                + "rss memory size before action start."
            )
            output_string += (
                f"{sep}action_end: Returns the current CPU "
                + "rss memory size after action end."
            )
            output_string += (
                f"{sep}action_delta: Returns the current CPU "
                + "rss memory size change between the start "
                + "and end of the action."
            )

            output_string += sep
            output_string += sep
            for iter_id in range(max_iter):
                output_string += f"{sep}iter_id: {iter_id}"
                for (
                    action_name,
                    action_start,
                    action_end,
                    action_delta,
                ) in report:
                    if len(action_start) > iter_id:
                        output_string += log_row(
                            action_name,
                            f"{action_start[iter_id]}",
                            f"{action_end[iter_id]}",
                            f"{action_delta[iter_id]}",
                        )
                output_string += f"{sep}{'-' * output_string_len}"
                output_string += sep

        output_string += sep
        return output_string

    def plot_and_save_fig(self):
        output_dir = os.path.join(self.dirpath, "plot")
        os.makedirs(output_dir, exist_ok=True)

        for memory_name in self.memory_metric_list:
            fig = go.Figure()
            fig.update_layout(
                title=memory_name,
                xaxis_title="iters",
                yaxis_title="CPU Memory (M)",
            )

            action_names_memory = self.recorded_memory[memory_name]
            for action_name, action_name_memory in action_names_memory.items():
                if len(action_name_memory) > 1:
                    x = list(range(len(action_name_memory)))
                    y = action_name_memory
                    fig.add_trace(go.Scatter(x=x, y=y, name=action_name))

            file_name = (
                f"CPUMemory-{self.local_hostname}-{memory_name}-Rank{self.local_rank}"  # noqa E501
                f"-Step[{self._pre_summary_step_num} - {self._prof_step_num}].html"  # noqa E501
            )
            pic_path = os.path.join(output_dir, file_name)
            try:
                fig.write_html(pic_path)
            except Exception as e:
                logger.error(f"Error saving {pic_path}: {e}")

        action_names = list(self.recorded_memory["action_start"].keys())
        for action_name in action_names:
            if not len(self.recorded_memory["action_start"][action_name]) > 1:
                continue

            fig = go.Figure()
            fig.update_layout(
                title=action_name,
                xaxis_title="iters",
                yaxis_title="CPU Memory (M)",
            )
            for memory_name in self.memory_metric_list:
                action_name_memory = self.recorded_memory[memory_name][
                    action_name
                ]
                x = list(range(len(action_name_memory)))
                y = action_name_memory
                fig.add_trace(go.Scatter(x=x, y=y, name=memory_name))

            file_name = (
                f"CPUMemory-{self.local_hostname}-{action_name}-Rank{self.local_rank}"  # noqa E501
                f"-Step[{self._pre_summary_step_num} - {self._prof_step_num}].html"  # noqa E501
            )
            pic_path = os.path.join(output_dir, file_name)

            try:
                fig.write_html(pic_path)
            except Exception as e:
                logger.error(f"Error saving {pic_path}: {e}")


@OBJECT_REGISTRY.register
class StageCPUMemoryProfiler(BaseProfiler):
    @require_packages("memray")
    def __init__(
        self,
        profile_action_name: str,
        leaks: bool = True,
        dirpath: Union[str, Path] = None,
        filename: str = None,
        auto_describe: bool = False,
    ):
        """
        Stage CPU Memory Profiler.

        Args:
            profile_action_name: Stage name which you want to profile.
            leaks: Whether check memory leaks, default is True.
            dirpath: Directory path for the ``filename``.
            filename: The profiler results will be saved in
            ``memray-flamegraph-{filename}_rank{rank}_{index}.html``.
            auto_describe: Whether to automatically output summary
                information at regular intervals.

        """
        if dirpath is None:
            dirpath = "./"

        if filename is None:
            filename = profile_action_name

        super(StageCPUMemoryProfiler, self).__init__(
            dirpath=dirpath,
            filename=filename,
            auto_describe=auto_describe,
        )
        pymalloc_flag = os.environ.get("PYTHONMALLOC", None)

        if leaks:
            assert pymalloc_flag == "malloc", (
                "Please set PYTHONMALLOC=malloc when using "
                "StageCPUMemoryProfiler to check memory leaks."
            )

        self.leaks = leaks
        self.profile_action_name = profile_action_name
        self.profile_tracker = None
        self.index = 0
        res = re.findall("get_(.*?)_batch_data", self.profile_action_name)

        if len(res) > 0:
            logger.warn(
                "Please make sure you has set numwork=0 in dataloader!"
            )

    def profile(self, action_name: str):
        if self.profile_action_name == action_name:
            filename = (
                self.filename
                + f"_rank{self.local_rank}_{self.index}_{os.getpid()}.bin"
            )
            file_path = os.path.join(self.dirpath, filename)
            self.profile_tracker = memray.Tracker(file_path)
            self.index += 1
            return self.profile_tracker
        else:
            return self

    def profile_iterable(self, iterable, action_name: str) -> None:
        iterator = iter(iterable)
        while True:
            try:
                if self.profile_action_name == action_name:
                    filename = (
                        self.filename
                        + f"_rank{self.local_rank}_{self.index}.bin"
                    )
                    file_path = os.path.join(self.dirpath, filename)
                    with memray.Tracker(file_path):
                        value = next(iterator)
                    self.index += 1
                    yield value
                else:
                    value = next(iterator)
                    yield value

            except StopIteration:
                break

    def describe_midway(self, idx) -> None:
        filename = (
            self.filename + f"_rank{self.local_rank}_{idx}_{os.getpid()}.bin"
        )
        file_path = os.path.join(self.dirpath, filename)
        if self.leaks:
            cmd = "memray flamegraph --leaks {}".format(file_path)
        else:
            cmd = "memray flamegraph {}".format(file_path)
        status, output = subprocess.getstatusoutput(cmd)
        cmd = "rm {}".format(file_path)
        status, output = subprocess.getstatusoutput(cmd)

    def describe(self) -> None:
        """Log a profile report after the conclusion of run."""
        self.teardown()
        self.profile_tracker = None

        for idx in range(self.index):
            self.describe_midway(idx)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
