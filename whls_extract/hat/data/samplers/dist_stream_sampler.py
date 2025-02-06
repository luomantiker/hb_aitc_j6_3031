# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import itertools
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_dist_info

__all__ = ["DistStreamBatchSampler"]


def sync_random_seed(seed: int = None, device: str = "cuda") -> int:
    """
    Synchronize a random seed across distributed training processes.

    Args:
    seed : Seed value to synchronize.
           If None, a random seed will be generated. Default is None.
    device : Device type for the seed tensor. Default is "cuda".

    Returns:
        Synchronized random seed across all processes.
    """
    if seed is None:
        seed = np.random.randint(2 ** 31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


@OBJECT_REGISTRY.register
class DistStreamBatchSampler(Sampler):
    """
    Distributed stream batch sampler.

    Args:
        dataset: The dataset from which to sample batches.
        batch_size: Batch size per GPU. Default is 1.
        seed : Random seed for synchronization across processes.
               Default is 0.
        skip_prob : Probability of skipping a sample.
                    Default is 0.5.
        max_skip_num : Maximum number of samples to skip.
                       Default is 1.
        sequence_flip_prob : Probability of flipping the sequence.
                             Default is 0.1.
        keep_consistent_seq_aug : Whether to keep consistent sequence
                                  augmentation across samples.
                                  Default is True.
    """

    def __init__(
        self,
        dataset,
        batch_size: Dataset = 1,
        seed: int = 0,
        skip_prob: float = 0.5,
        max_skip_num: int = 1,
        sequence_flip_prob: float = 0.1,
        keep_consistent_seq_aug: bool = True,
    ) -> None:

        rank, world_size = get_dist_info()

        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)
        self.keep_consistent_seq_aug = keep_consistent_seq_aug

        self.size = len(self.dataset)
        meta = self.dataset.get_meta()
        self.scene_info = meta["scene_info"]

        self.groups_num = len(self.scene_info.keys())
        self.global_batch_size = batch_size * world_size
        assert self.groups_num >= self.global_batch_size

        # Now, for efficiency, make a dict group_idx: List[dataset sample_idxs]
        self.group_idx_to_sample_idxs = {}
        start = 0
        for idx, (_, v) in enumerate(self.scene_info.items()):
            end = start + v
            self.group_idx_to_sample_idxs[idx] = list(range(start, end))
            start = end
        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator

        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(
                self.rank * self.batch_size + local_sample_idx
            )
            for local_sample_idx in range(self.batch_size)
        ]
        # Keep track of a buffer of dataset sample idxs
        # for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]
        self.aug_per_local_sample = [None for _ in range(self.batch_size)]
        self.skip_prob = skip_prob
        self.sequence_flip_prob = sequence_flip_prob
        self.max_skip_num = max_skip_num

    def _infinite_group_indices(self):
        """
        Infinite generator yielding shuffled indices.

        Yields:
            Shuffled group index.
        """
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.groups_num, generator=g).tolist()

    def _group_indices_per_global_sample_idx(self, global_sample_idx: int):
        """
        Generate a stream of group indices for a specific global sample index.

        Args:
            global_sample_idx : Global index across all GPUs.

        Yields:
            Group index for batch sampling.
        """
        yield from itertools.islice(
            self._infinite_group_indices(),
            global_sample_idx,
            None,
            self.global_batch_size,
        )

    def __iter__(self):
        """
        Iterate indefinitely over the dataset to yield batches of samples.

        Yields:
            A batch of samples, each represented as a dictionary
            with keys 'idx' and 'aug'.
        """
        while True:
            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    # skip = False
                    new_group_idx = next(
                        self.group_indices_per_global_sample_idx[
                            local_sample_idx
                        ]
                    )
                    self.buffer_per_local_sample[
                        local_sample_idx
                    ] = copy.deepcopy(
                        self.group_idx_to_sample_idxs[new_group_idx]
                    )
                    flip_prob = np.random.uniform()
                    if flip_prob < self.sequence_flip_prob:
                        self.buffer_per_local_sample[
                            local_sample_idx
                        ] = self.buffer_per_local_sample[local_sample_idx][
                            ::-1
                        ]
                    if self.keep_consistent_seq_aug:
                        self.aug_per_local_sample[
                            local_sample_idx
                        ] = self.get_aug()

                if not self.keep_consistent_seq_aug:
                    self.aug_per_local_sample[
                        local_sample_idx
                    ] = self.get_aug()
                skip_cnt = 0
                while (
                    np.random.uniform() < self.skip_prob
                    and len(self.buffer_per_local_sample[local_sample_idx]) > 1
                    and skip_cnt < self.max_skip_num
                ):
                    self.buffer_per_local_sample[local_sample_idx].pop(0)
                    skip_cnt += 1

                curr_batch.append(
                    {
                        "idx": self.buffer_per_local_sample[
                            local_sample_idx
                        ].pop(0),
                        "aug": self.aug_per_local_sample[local_sample_idx],
                    }
                )
            yield curr_batch

    def __len__(self) -> int:
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Set epoch."""
        self.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def get_aug(self) -> Dict:
        """
        Return the augmentation parameters for the current sample.

        Returns:
            Augmentation parameters for the current sample.
        """
        return self.dataset.get_aug()
