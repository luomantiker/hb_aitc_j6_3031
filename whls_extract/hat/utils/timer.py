import functools
import inspect
import logging
import os
import sys
import threading
import time
from abc import ABC
from typing import Any, Callable, Optional

from .distributed import get_dist_info
from .logger import LOG_DIR, get_monitor_logger, rank_zero_info, rank_zero_warn

logger = logging.getLogger(__name__)
DEFAULT_LOGGGER_PATH = os.path.join(LOG_DIR, "timer-warning")

__all__ = [
    "Timer",
    "BytesTimer",
    "TimerContextManager",
    "AlarmTimer",
    "AlarmTimerDecorator",
    "AlarmTimerContextManager",
]


class Timer:
    """
    Class for timing execution speed of function.

    Args:
        name: name of timer
        per_iters: cycles of logging speed info.
    """

    def __init__(
        self,
        name: str,
        per_iters: int = 1000,
        logger_name: str = __name__,
        logger_path: str = DEFAULT_LOGGGER_PATH,
    ):
        self.name = name
        self.per_iters = per_iters
        self.reset()
        self.logger_name = logger_name
        self.logger_path = logger_path

    def reset(self):
        self.iter_num = 0
        self.elapsed_time = 0
        self.tic_time = time.time()

    def tic(self):
        self.tic_time = time.time()

    def toc(self):
        self.elapsed_time += time.time() - self.tic_time
        self.iter_num += 1
        if self.iter_num > self.per_iters:
            self._log()
            self.reset()

    def _log(self):
        rank, _ = get_dist_info()
        root_logger = get_monitor_logger(
            self.logger_name,
            self.logger_path,
            logging.INFO,
            rank=rank,
        )
        root_logger.info(
            "speed of {} is {} iters/sec".format(
                self.name, self.iter_num / self.elapsed_time
            )
        )

    def timeit(self, func):
        @functools.wraps(func)
        def with_timer(*args, **kwargs):
            self.tic()
            ret = func(*args, **kwargs)
            self.toc()
            return ret

        return with_timer


class BytesTimer(Timer):
    """
    Class for timing execution speed of read raw data.

    Args:
        name: name of timer
        per_iters: cycles of logging speed info.
    """

    def __init__(
        self,
        name: str = "reading raw data",
        per_iters: int = 1000,
        logger_name: str = __name__,
        logger_path: str = DEFAULT_LOGGGER_PATH,
    ):
        super().__init__(name, per_iters, logger_name, logger_path)

    def reset(self):
        super().reset()
        self.read_mbytes = 0

    def toc(self, mbytes):
        self.read_mbytes += mbytes
        super().toc()

    def timeit(self, func):
        @functools.wraps(func)
        def with_timer(*args, **kwargs):
            self.tic()
            ret = func(*args, **kwargs)
            if isinstance(ret, bytes):
                mbytes = sys.getsizeof(ret) / 1024.0 / 1024.0
            else:
                mbytes = 0
            self.toc(mbytes)
            return ret

        return with_timer

    def _log(self):
        rank, _ = get_dist_info()
        root_logger = get_monitor_logger(
            self.logger_name,
            self.logger_path,
            logging.INFO,
            rank=rank,
        )
        root_logger.info(
            "speed of {} is {} mbytes/sec, {} record/sec".format(
                self.name,
                self.read_mbytes / self.elapsed_time,
                self.iter_num / self.elapsed_time,
            )
        )


class TimerContextManager(Timer):
    def __init__(
        self,
        name: str,
        logger_name: str = __name__,
        logger_path: str = DEFAULT_LOGGGER_PATH,
    ):
        super().__init__(name, 0, logger_name, logger_path)

    def toc(self):
        self.elapsed_time = time.time() - self.tic_time
        self._log()
        self.reset()

    def _log(self):
        rank_zero_info("{} cost {}s".format(self.name, self.elapsed_time))

    def __enter__(self):
        self.tic()
        return self

    def __exit__(self, *args):
        self.toc()


class AlarmTimer(ABC):
    """
    AlarmTimer is a abstract class for alarm when \
        function execute time over alarm time.

    Args:
        alarm_seconds: the alarm time in seconds.
        alarm_in_zero_rank: if throw alarm in zero rank logger.
        open_monitor_thread: if open monitor thread.
        logger_name: logger name.
        logger_path: logger path.
    """

    def __init__(
        self,
        alarm_seconds: float,
        alarm_in_zero_rank: bool = True,
        open_monitor_thread: bool = False,
        logger_name: str = __name__,
        logger_path: str = DEFAULT_LOGGGER_PATH,
    ) -> None:
        self.alarm_seconds = alarm_seconds
        self.alarm_in_zero_rank = alarm_in_zero_rank
        self.logger_name = logger_name
        self.logger_path = logger_path
        self.open_monitor_thread = open_monitor_thread

    def _alarm_log(self, msg: str, log_level=logging.WARN) -> None:
        """
        Print alarm message in logger.

        Args:
            msg: alarm message.
            log_level: logging level.
        """
        rank, _ = get_dist_info()
        root_logger = get_monitor_logger(
            self.logger_name,
            self.logger_path,
            logging.INFO,
            rank=rank,
        )
        if log_level == logging.INFO:
            root_logger.info(msg)
            if self.alarm_in_zero_rank:
                rank_zero_info(msg)
        else:
            # throw warning in local rank and zero rank logger
            root_logger.warning(msg)
            if self.alarm_in_zero_rank:
                rank_zero_warn(msg)

    def generate_alarm_log(self, func: Callable) -> None:
        # get function information
        func_name = func.__name__
        func_source_file_path = inspect.getsourcefile(func)
        func_source_file_path = os.path.abspath(func_source_file_path)
        _, func_line_number = inspect.getsourcelines(func)
        return func_name, func_source_file_path, func_line_number

    def create_alarm_thread(self, func: Callable) -> Callable:
        # get function information
        (
            func_name,
            func_source_file_path,
            func_line_number,
        ) = self.generate_alarm_log(func)
        # add runtime alarm
        alarm_thread = threading.Timer(
            self.alarm_seconds,
            self.alarm_log,
            (func_name, func_source_file_path, func_line_number),
        )
        return alarm_thread

    def alarm_log(
        self, func_name: str, func_source_file_path: str, func_line_number: str
    ) -> str:
        msg = f'"{func_source_file_path}", line {func_line_number}: \
            {func_name} execute time over {self.alarm_seconds:.2f}s !'
        self._alarm_log(msg)


class AlarmTimerDecorator(AlarmTimer):
    """
    AlarmTimerDecorator is a decorator class for alarm \
        when function execute time over alarm time.

    Agrs:
        alarm_seconds: the alarm time in seconds.
        callback_func: callback function when function \
            execute time over alarm time.
        alarm_in_zero_rank: if throw alarm in zero rank logger.
        open_monitor_thread: if open monitor thread.
        logger_name: logger name.
        logger_path: logger path.

    Usage:
    ```python
        def callback_func(ret, duration):
            (res, uri, idx) = ret
            msg = f"LMDB read data from {uri} by
                idx {idx} cost {duration:.2f}s."
            return res, msg

        @AlarmTimerDecorator(alarm_seconds=0.1,
                             callback_func=callback_func)
        def run_func(*agrs, **kwargs):
            ...
    """

    def __init__(
        self,
        alarm_seconds: float,
        callback_func: Optional[Callable[..., Any]] = None,
        alarm_in_zero_rank: bool = True,
        open_monitor_thread: bool = False,
        logger_name: str = __name__,
        logger_path: str = DEFAULT_LOGGGER_PATH,
    ) -> None:
        super().__init__(
            alarm_seconds,
            alarm_in_zero_rank,
            open_monitor_thread,
            logger_name,
            logger_path,
        )
        self.callback_func = callback_func

    def __call__(self, func: callable) -> callable:
        @functools.wraps(func)
        def alarm_timer(*args, **kwargs):
            if self.open_monitor_thread:
                # create alarm thread
                alarm_thread = self.create_alarm_thread(func)
                alarm_thread.start()

            # record exec time
            start_time = time.time()
            ret = func(*args, **kwargs)
            end_time = time.time()

            if self.open_monitor_thread:
                alarm_thread.cancel()

            # callback function
            duration = end_time - start_time
            if self.callback_func is not None:
                final_ret, msg = self.callback_func(ret, duration=duration)
            else:
                final_ret = ret
            # alarm log
            if duration > self.alarm_seconds:
                if not self.open_monitor_thread:
                    (
                        func_name,
                        func_source_file_path,
                        func_line_number,
                    ) = self.generate_alarm_log(func)
                    self.alarm_log(
                        func_name, func_source_file_path, func_line_number
                    )
                if self.callback_func is not None:
                    self._alarm_log(msg)
            return final_ret

        return alarm_timer


class AlarmTimerContextManager(AlarmTimer):
    """
    AlarmTimerContextManager is a context manager class \
        for alarm when function execute time over alarm time.

    Args:
        alarm_seconds: the alarm time in seconds.
        func_desc: function description.
        alarm_in_zero_rank: if throw alarm in zero rank logger.
        open_monitor_thread: if open monitor thread.
        logger_name: logger name.
        logger_path: logger path.

    Usage:
    ```python
        with AlarmTimerContextManager(alarm_seconds=0.1, func_desc="run_func"):
            run_func(*agrs, **kwargs)
    ```
    """

    def __init__(
        self,
        alarm_seconds: float,
        func_desc: Optional[str] = None,
        alarm_in_zero_rank: bool = True,
        open_monitor_thread: bool = False,
        logger_name: str = __name__,
        logger_path: str = DEFAULT_LOGGGER_PATH,
    ) -> None:
        super().__init__(
            alarm_seconds,
            alarm_in_zero_rank,
            open_monitor_thread,
            logger_name,
            logger_path,
        )
        self.func_desc = func_desc

    def alarm_log(self, msg) -> None:
        self._alarm_log(msg)

    def __enter__(self):
        assert (
            self.func_desc is not None
        ), "Function description is required when use contextmanager mode."
        if self.open_monitor_thread:
            # add runtime alarm
            alarm_msg = f"AlarmTimer Function description <{self.func_desc}>, \
                execute time over {self.alarm_seconds:.2f}s !"
            self.alarm_thread = threading.Timer(
                self.alarm_seconds, self.alarm_log, (alarm_msg,)
            )
            self.alarm_thread.start()
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        end = time.time()
        if self.open_monitor_thread:
            self.alarm_thread.cancel()
        if end - self.begin > self.alarm_seconds:
            self._alarm_log(
                f"This {self.func_desc} took {end - self.begin:.2f}s.",
                log_level=logging.WARN,
            )
