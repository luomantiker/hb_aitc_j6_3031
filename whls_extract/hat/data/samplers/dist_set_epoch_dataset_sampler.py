# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Optional

from torch.utils.data.distributed import DistributedSampler

from hat.registry import OBJECT_REGISTRY

__all__ = ["DistSetEpochDatasetSampler"]


@OBJECT_REGISTRY.register
class DistSetEpochDatasetSampler(DistributedSampler):  # noqa: D205,D400
    """
    Distributed sampler that supports set epoch in dataset.

    Args:
        dataset: compose dataset
        num_replicas: same as DistributedSampler
        rank: Same as DistributedSampler
        shuffle: if shuffle data
        seed: random seed
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super(DistSetEpochDatasetSampler, self).__init__(
            dataset, num_replicas, rank, shuffle, seed, drop_last
        )

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
