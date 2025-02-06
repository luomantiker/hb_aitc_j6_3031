# Copyright (c) Horizon Robotics. All rights reserved.
from collections.abc import Iterable
from typing import Any, Generator, Tuple

__all__ = ["prefetch_iterator"]


def prefetch_iterator(
    iterable: Iterable,
) -> Generator[Tuple[Any, bool], None, None]:
    """Return an iterator that pre-fetches and caches the next item."""  # noqa: E501
    it = iter(iterable)
    try:
        # the iterator may be empty from the beginning
        last = next(it)
    except StopIteration:
        return

    yield last
    for val in it:
        last = val
        yield last
