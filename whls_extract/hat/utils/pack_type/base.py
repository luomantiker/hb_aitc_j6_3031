# Copyright (c) Horizon Robotics. All rights reserved.
import os
from abc import ABC
from typing import Any, Tuple

from hat.utils.logger import LOG_DIR
from hat.utils.timer import AlarmTimerDecorator, BytesTimer

__all__ = ["PackType"]


def alarm_timer_callback(ret: Any, duration: float) -> Tuple[Any, str]:
    """
    Define a callback function for the alarm timer.

    Args:
        ret: Return value of function.
        duration: Duration of function.
    """
    res, uri, idx = ret
    msg = f"LMDB read data idx [{idx}] from [{uri}] cost {duration:.2f}s."
    return res, msg


class PackType(ABC):
    """
    Data type interface class.

    Args:
        fixed_read_data: If reuse a fixed read data .
        log_speed: If logger the read data speed. The default interval of log \
            speed is call `read()` 1000 times. You can set the interval by \
            setting environment variable `HAT_MONITOR_INTERVAL`.
        kwargs: Kwargs of PackType.
    """

    def __init__(
        self,
        fixed_read_data: bool = False,
        log_speed: bool = True,
        **kwargs,
    ):
        self.fixed_read_data = fixed_read_data
        self.log_speed = log_speed
        self.fixed_data = None
        self.timer = BytesTimer(
            per_iters=int(float(os.getenv("HAT_MONITOR_INTERVAL", 1000))),
            logger_path=os.path.join(LOG_DIR, "data-profiler"),
        )

    def open(self):
        """Open the data file."""
        pass

    def close(self):
        """Close the data file."""
        pass

    def write(self, idx: int, record: bytes):
        """Write record into data file by idx."""
        pass

    @AlarmTimerDecorator(60, callback_func=alarm_timer_callback)
    def read(self, idx):
        if not hasattr(self, "uri"):
            self.uri = None
        res = self.timer.timeit(self._read)(idx)
        return (res, self.uri, idx)

    def _read(self, idx):
        if self.fixed_read_data:
            if self.fixed_data is None:
                self.fixed_data = self.read_idx(idx)
            return self.fixed_data
        else:
            return self.read_idx(idx)

    def read_idx(self, idx):
        """Read the idx-th data."""
        pass

    def reset(self):
        """Reset the data file operator."""
        pass

    def get_keys(self):
        """Get keys for read."""
        pass

    def __del__(self):
        """Recycle resources."""
        self.close()

    def __len__(self):
        """Get the length."""
        pass
