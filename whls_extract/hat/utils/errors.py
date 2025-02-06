import contextlib
import faulthandler
import os
from typing import Callable, Optional, TypeVar

from torch.distributed.elastic.multiprocessing.errors import (
    ErrorHandler as _ErrorHandler,
)
from torch.distributed.elastic.multiprocessing.errors import record as _record

from hat.utils.distributed import get_local_host

__all__ = ["record"]

T = TypeVar("T")

IS_LOCAL = not os.path.exists("/running_package")
ERR_LOG_DIR = (
    ".hat_logs/error_records"
    if IS_LOCAL
    else "/job_log/hat_logs/error_records/"
)


class ErrorHandler(_ErrorHandler):

    FILE_NAME_FORMAT = "%s-HOST[%s]-PID[%s]"
    DEFAULT_PREFIX = "SignalStack"

    def __init__(
        self,
        output_dir: Optional[str] = None,
        record_prefix: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.record_prefix = record_prefix

        self.output_dir = output_dir if output_dir is not None else ERR_LOG_DIR
        with contextlib.suppress(FileExistsError):
            os.makedirs(self.output_dir, exist_ok=True)

        self._faulthandler_file = None
        self._py_traceback_file = None
        self._faulthandler_stream = None

    def initialize(self):
        faulthandler.enable(
            file=self.faulthandler_record_stream,
            all_threads=True,
        )

    def record_exception(self, e: BaseException):
        faulthandler.dump_traceback(
            file=self.faulthandler_record_stream,
            all_threads=True,
        )
        super().record_exception(e)

    def _get_error_file_path(self):
        env_err_file = os.getenv("TORCHELASTIC_ERROR_FILE", None)
        if env_err_file is None:
            return self.py_traceback_record
        else:
            return env_err_file

    @property
    def py_traceback_record(self):
        if self._py_traceback_file is None:
            self._py_traceback_file = self._prepare_file(file_ext=".json")

        return self._py_traceback_file

    @property
    def faulthandler_record(self):
        if self._faulthandler_file is None:
            self._faulthandler_file = self._prepare_file(file_ext=".txt")

        return self._faulthandler_file

    @property
    def faulthandler_record_stream(self):
        if self._faulthandler_stream is None:
            self._faulthandler_stream = open(
                self.faulthandler_record,
                mode="w+",
            )

        return self._faulthandler_stream

    def _prepare_file(self, file_ext=".txt"):
        prefix = (
            self.record_prefix if self.record_prefix else self.DEFAULT_PREFIX
        )

        host = get_local_host()
        pid_num = os.getpid()

        filename = (
            self.FILE_NAME_FORMAT
            % (
                prefix,
                host,
                pid_num,
            )
            + file_ext
        )
        file_path = os.path.join(self.output_dir, filename)

        return file_path


def record(
    output_dir: Optional[str] = None,
    record_prefix: Optional[str] = None,
    error_handler: Optional[ErrorHandler] = None,
) -> Callable[..., T]:
    """Record faulthandler stack and torch elastic error.

    Args:
        output_dir: Output dir to save result.
        record_prefix: Prefix of record file.
        error_handler: Handler to record error info.

    Example:
    >>> @record(record_prefix="test")
    ... def main():
    ...     pass

    >>> if __name__=="__main__":
    ...     main()
    """

    if error_handler is None:
        error_handler = ErrorHandler(
            output_dir=output_dir,
            record_prefix=record_prefix,
        )

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        return _record(
            fn=fn,
            error_handler=error_handler,
        )

    return decorator
