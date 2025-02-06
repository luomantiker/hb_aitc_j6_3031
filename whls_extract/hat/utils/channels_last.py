# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from collections.abc import Mapping

import torch

logger = logging.getLogger(__name__)


__all__ = ["convert_memory_format"]


def convert_memory_format(batch, keys, memory_format=torch.channels_last):
    """
    Convert memory format.

    This function will help to convert certain keys in batch dict
    to a channels_last memory_format.

    Args:
        batch: dict of data.
        keys: certain keys for converting.
        memory_format: memory format of converting.

    """

    if isinstance(batch, torch.Tensor) and batch.dim() == 4:
        return batch.to(memory_format=memory_format)
    elif isinstance(batch, (list, tuple)):
        batch_type = type(batch)
        return batch_type(
            [convert_memory_format(b, None, memory_format) for b in batch]
        )
    elif isinstance(batch, Mapping):
        if keys is None:
            for key, value in batch.items():
                v_key = value.keys() if isinstance(value, Mapping) else ()
                batch[key] = convert_memory_format(
                    batch[key], v_key, memory_format
                )
        else:
            for k in keys:
                assert k in batch.keys(), f"Cannot find {k} in input batch."
                batch[k] = convert_memory_format(batch[k], None, memory_format)
        return batch
    return batch
