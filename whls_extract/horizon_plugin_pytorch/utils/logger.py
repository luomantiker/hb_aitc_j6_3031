import logging
import os
import time
from collections import Counter

import torch.distributed as dist
from termcolor import colored

__all__ = [
    "set_logger",
]


class _ColorfulFormatter(logging.Formatter):
    """Format log with different colors according to the log level."""

    def __init__(self, *args, **kwargs):
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):  # noqa: N802
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.INFO:
            log = colored(log, "green")
        elif record.levelno == logging.WARNING:
            log = colored(log, "yellow")
        elif (
            record.levelno == logging.ERROR
            or record.levelno == logging.CRITICAL
        ):
            log = colored(log, "red")
        else:
            return log
        return log


class _ParentHandlerFilter(logging.Filter):
    """
    Filter log records that have a higher level handler in the parent logger.

    It will prevent the child logger and the parent logger from processing
    record at the same time.
    """

    def __init__(self, cur_logger):
        super().__init__()
        self.cur_logger = cur_logger

    def filter(self, record):
        return not self.cur_logger.propagate or not _has_higher_level_handler(
            self.cur_logger.parent
        )


class _CallTimesFilter(logging.Filter):
    """
    Filter log records by the number of times they have been logged.

    It will prevent propagation to the parent logger.
    """

    def __init__(self, cur_logger):
        super().__init__()
        self.context_counter = Counter()
        self.cur_logger = cur_logger

    def filter(self, record):
        context = getattr(record, "call_times_context", None)
        limit = getattr(record, "call_times_limit", 1)

        if context is None:
            self.cur_logger.propagate = True
            return True

        if isinstance(context, str):
            context = (context,)

        hash_key = ()
        if "location" in context:
            hash_key = hash_key + (f"{record.pathname}_{record.lineno}",)
        if "message" in context:
            hash_key = hash_key + (record.getMessage(),)

        self.context_counter[hash_key] += 1

        if self.context_counter[hash_key] <= limit:
            self.cur_logger.propagate = True
            return True

        self.cur_logger.propagate = False
        return False


class _RankFilter(logging.Filter):
    """Filter log records by rank."""

    def __init__(self, rank=0):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        rank = dist.get_rank() if dist.is_initialized() else 0
        return rank == self.rank


class _LazyFileHandler(logging.FileHandler):
    """Create the log file only when the first log record is emitted."""

    def __init__(self, filename, mode="a", encoding=None, delay=True):
        # Initialize with delay=True to avoid creating the file immediately
        super().__init__(filename, mode, encoding, delay)
        self._initialized = False

    def _open(self):
        if not self._initialized:
            os.makedirs(os.path.dirname(self.baseFilename), exist_ok=True)
            self._initialized = True
        return super()._open()


def _has_higher_level_handler(logger):
    cur_logger = logger
    while cur_logger:
        if cur_logger.hasHandlers():
            return True
        if cur_logger.propagate:
            cur_logger = cur_logger.parent
        else:
            return False
    return False


def set_logger(name, level=logging.INFO, file_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logging._acquireLock()
    try:
        logger.handlers = []
    finally:
        logging._releaseLock()

    default_format = _ColorfulFormatter(
        "%(asctime)s %(levelname)s [%(pathname)s:%(lineno)d] %(message)s"
    )

    default_stream_handler = logging.StreamHandler()
    default_stream_handler.setLevel(level)
    default_stream_handler.setFormatter(default_format)
    # _CallTimesFilter has the highest priority
    # if a record is filtered by other filter before _CallTimesFilter,
    # _CallTimesFilter can't block its propagation to the parent logger.
    default_stream_handler.addFilter(_CallTimesFilter(logger))
    default_stream_handler.addFilter(_ParentHandlerFilter(logger))
    default_stream_handler.addFilter(_RankFilter())
    logger.addHandler(default_stream_handler)

    if file_dir is not None:
        time_stamp = time.strftime(
            "%Y%m%d%H%M%S", time.localtime(int(time.time()))
        )
        filename = f"{name}_{time_stamp}.log"
        filepath = os.path.join(file_dir, filename)

        default_file_handler = _LazyFileHandler(filepath)
        default_file_handler.setLevel(level)
        default_file_handler.setFormatter(default_format)
        default_file_handler.addFilter(_CallTimesFilter(logger))
        default_file_handler.addFilter(_RankFilter())
        logger.addHandler(default_file_handler)

    return logger
