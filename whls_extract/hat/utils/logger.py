# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
import sys
import time
import warnings
from typing import Optional

from .distributed import rank_zero_only

__all__ = [
    "init_logger",
    "DisableLogger",
    "MSGColor",
    "format_msg",
    "rank_zero_info",
    "rank_zero_warn",
    "init_rank_logger",
    "LOG_DIR",
    "OutputLogger",
    "get_monitor_logger",
]


is_local_train = not os.path.exists("/running_package")
LOG_DIR = ".hat_logs" if is_local_train else "/job_log/hat_logs/"


class SingleLevelFilter(logging.Filter):
    def __init__(
        self,
        level: int,
        record: bool = False,
        name: str = "",
    ) -> None:
        super().__init__(name)
        self.level = level
        self.record = record

    def filter(self, record: logging.LogRecord) -> bool:
        if self.record:
            return record.levelno == self.level
        else:
            return record.levelno != self.level


def init_logger(
    log_file,
    logger_name=None,
    rank=0,
    level=logging.INFO,
    overwrite=False,
    clean_handlers=False,
    filter_warning=False,
    set_stderr=True,
    set_stdout=True,
):
    head = (
        "%(asctime)-15s %(levelname)s [%(filename)s:%(lineno)d] Node["
        + str(rank)
        + "] %(message)s"
    )
    if rank != 0:
        log_file += "-rank%d.log" % rank
    else:
        log_file += ".log"
    if os.path.exists(log_file) and overwrite:
        os.remove(log_file)
    try:
        # may fail when multi processes do this concurrently
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    except FileExistsError:
        pass

    logger = logging.getLogger(logger_name)
    if clean_handlers:
        # duplicate handlers will cause duplicate outputs
        logger.handlers = []
    formatter = logging.Formatter(head)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # stderr
    if set_stderr:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(formatter)
        stderr_handler.addFilter(SingleLevelFilter(logging.ERROR, True))
        logger.addHandler(stderr_handler)

    # stdout
    if set_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.addFilter(SingleLevelFilter(logging.ERROR, False))
        logger.addHandler(stdout_handler)
        if filter_warning:
            logger.warn(
                "The warning filtering function has been "
                "activated, and all warnings of training launch process "
                "correspond uniquely to each GPU will be filtered"
            )
            warnings.filterwarnings(action="ignore")
            stdout_handler.addFilter(SingleLevelFilter(logging.WARN, False))

    logger.setLevel(level)


def init_rank_logger(
    rank: int,
    save_dir: str,
    cfg_file: str,
    step: str,
    prefix: Optional[str] = "",
    filter_warning: bool = False,
) -> logging.Logger:
    """Init logger of specific rank.

    Args:
        rank: rank id.
        cfg_file: Config file used to build log file name.
        step: Current training step used to build log file name.
        save_dir: Directory to save log file.
        prefix: Prefix of log file.
        filter_warning: Whether to filter warning

    Returns:
        Logger.
    """
    time_stamp = time.strftime(
        "%Y%m%d%H%M%S", time.localtime(int(time.time()))
    )
    cfg_name = os.path.splitext(os.path.basename(cfg_file))[0]
    log_file = os.path.join(
        save_dir, "%s%s-%s-%s" % (prefix, cfg_name, step, time_stamp)
    )
    init_logger(
        log_file=log_file,
        rank=rank,
        clean_handlers=True,
        filter_warning=filter_warning,
    )

    logger = logging.getLogger()
    return logger


class DisableLogger(object):
    """Disable logger to logging anything under this scope.

    Args:
         enable: Whether enable `DisableLogger`.
            If True, `DisableLogger` works, will disable logging.
            If False, `DisableLogger` is no-op.
        level: used to disable less than or equal to the level.

    Examples::

        >>> import logging
        >>> logger = logging.getLogger()
        >>> with DisableLogger(enable=True):
        ...     logger.info('This info will not logging.')
        >>> logger.info('This info will logging after leaving the scope.')

        >>> with DisableLogger(enable=False):
        ...     logger.info(
        ...         'This info will logging as `DisableLogger` is not enable.')

    """

    def __init__(self, enable: bool = True, level: int = logging.WARNING):
        self.enable = enable
        self.level = level

    def __enter__(self):
        if self.enable:
            # disable level that less than or equal to self.level
            logging.disable(self.level)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        if self.enable:
            logging.disable(logging.NOTSET)


class MSGColor(object):
    BLACK = 30  # default
    RED = 31  # Emergency warning
    GREEN = 32  # Harmless, just for notice


def format_msg(msg, color):
    return "\033[%dm%s\033[0m" % (color, msg)


def _info(*args, **kwargs):
    logger = logging.getLogger(__name__)
    logger.info(*args, **kwargs)


def _warn(*args, **kwargs):
    logger = logging.getLogger(__name__)
    logger.warn(*args, **kwargs)


rank_zero_info = rank_zero_only(_info)

rank_zero_warn = rank_zero_only(_warn)


class OutputLogger(object):
    """Should be used together with redirect_stdout."""

    def __init__(self, logger):
        self.logger = logger

    def write(self, msg):
        if msg and not msg.isspace():
            self.logger.info(msg)

    def flush(self):
        pass


class ExperimentLogger(object):
    def __init__(
        self,
        enable_tracking=False,
        logger_type="aidi",
    ) -> None:

        self.logger_logger_typename = logger_type
        self.enable_tracking = enable_tracking
        if logger_type == "aidi":
            try:
                from hat.utils.aidi import AIDIExperimentLogger

                self.logger = AIDIExperimentLogger()
            except ImportError:
                logging.warning(
                    "import `hat.utils.aidi.AIDIExperimentLogger` failed, will set `experiment_logger=None`."  # noqa E501
                )
                self.logger = None
        else:
            self.logger = None

    def log_config(self, config):
        if self.enable_tracking and self.logger:
            self.logger.log_config(config)

    def init_group(self, group_name):
        if self.enable_tracking and self.logger:
            self.logger.init_group(group_name)

    @classmethod
    def log_exception(
        cls,
        exception: Exception,
        prefix: str = None,
        enable_tracking: bool = False,
        logger_type: str = "aidi",
    ):
        if prefix:
            exception_content = f"{prefix}: {str(exception)}"
        else:
            exception_content = str(exception)
        logging.error(exception_content)
        exp_logger = cls(enable_tracking, logger_type)
        exp_logger.logger.log_exception(exception)


def get_monitor_logger(logger_name, log_file, level, rank):
    logger = logging.getLogger(logger_name)
    if not logger.hasHandlers():
        init_logger(
            log_file=log_file,
            logger_name=logger_name,
            rank=rank,
            level=level,
            set_stderr=False,
            set_stdout=False,
        )
    logger.propagate = False
    return logger
