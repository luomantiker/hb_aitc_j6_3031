# Copyright (c) Horizon Robotics. All rights reserved.
import math
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as data

from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_dist_info

__all__ = [
    "RepeatDataset",
    "ComposeDataset",
    "ResampleDataset",
    "ConcatDataset",
    "DistributedComposeRandomDataset",
    "ComposeIterableDataset",
    "CBGSDataset",
    "ChunkShuffleDataset",
]


@OBJECT_REGISTRY.register
class RepeatDataset(data.Dataset):
    """
    A wrapper of repeated dataset.

    Using RepeatDataset can reduce the data loading time between epochs.

    Args:
        dataset: The datasets for repeating.
        times: Repeat times.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, times: int):
        self.dataset = dataset
        self.times = times
        if hasattr(self.dataset, "flag"):
            self.flag = np.tile(self.dataset.flag, times)
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len


@OBJECT_REGISTRY.register
class ComposeDataset(data.Dataset):
    """Dataset wrapper for multiple datasets with precise batch size.

    Args:
        datasets: config for each dataset.
        batchsize_list: batchsize for each task dataset.
    """

    def __init__(self, datasets: List[Dict], batchsize_list: List[int]):
        self.datasets = datasets
        self.batchsize_list = batchsize_list
        self.total_batchsize = sum(batchsize_list)

        self.len_list = [len(dataset) for dataset in self.datasets]
        self.max_len = max(self.len_list)
        self.total_len = sum(self.len_list)
        self.dataset_bounds = []
        flag = 0
        for bachsize in self.batchsize_list:
            self.dataset_bounds.append(flag + bachsize)
            flag += bachsize
        self.iter_time = 0

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        if self.iter_time >= self.total_batchsize:
            self.iter_time = 0
        for i, bound in enumerate(self.dataset_bounds):
            if self.iter_time < bound:
                self.iter_time += 1
                idx = idx % len(self.datasets[i])
                assert idx < len(self.datasets[i]), (
                    f"{idx} exceeds " f"{len(self.datasets[i])}"
                )
                return self.datasets[i][idx]

    def __repr__(self):
        return "ComposeDataset"

    def __str__(self):
        return str(self.datasets)


@OBJECT_REGISTRY.register
class DistributedComposeRandomDataset(data.IterableDataset):
    """

    Dataset wrapper for multiple datasets fair sample weights accross multi workers in a distributed environment.

    Each datsaet is cutted by (num_workers x num_ranks).

    Args:
        datasets: list of datasets.
        sample_weights: sample weights for each dataset.
        shuffle: shuffle each dataset when set to True
        seed: random seed for shuffle
        multi_sample_output: whether dataset outputs multiple samples at the same time.
    """  # noqa

    def __init__(
        self,
        datasets: List[data.Dataset],
        sample_weights: List[int],
        shuffle=True,
        seed=0,
        multi_sample_output=False,
    ):
        self.datasets = datasets
        assert all(
            i > 0 for i in sample_weights
        ), "sample weights must be positive"
        self.total_weights = sum(sample_weights)
        self.sample_weights = [p / self.total_weights for p in sample_weights]
        self.shuffle = shuffle
        self.seed = seed
        self.multi_sample_output = multi_sample_output
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def __len__(self):
        len_list = [len(dataset) for dataset in self.datasets]
        total_len = sum(len_list)
        return total_len

    def __iter__(self):
        all_stopped = [False for _ in self.sample_weights]

        curr_rank = self.rank
        world_size = self.world_size

        worker_info = data.get_worker_info()
        if worker_info is not None:
            curr_worker = worker_info.id
            num_workers = worker_info.num_workers
        else:
            num_workers = 1
            curr_worker = 0

        index_ranges = []
        index_datasets = []
        for dataset in self.datasets:
            step_len = int(
                math.ceil(
                    len(dataset) / (float(world_size) * float(num_workers))
                )
            )
            real_len = step_len * world_size * num_workers
            pad_num = real_len - len(dataset)
            curr_idx = num_workers * curr_rank + curr_worker
            if curr_idx >= pad_num:
                curr_start = (
                    pad_num * (step_len - 1) + (curr_idx - pad_num) * step_len
                )
                curr_end = curr_start + step_len
            else:
                curr_start = curr_idx * (step_len - 1)
                curr_end = curr_start + step_len - 1

            index_range = list(range(curr_start, curr_end))
            pos = 0
            while len(index_range) < step_len:
                index_range.append(curr_start + pos)
                pos += 1

            index_ranges.append(index_range)
            index_datasets.append(list(range(len(dataset))))
        if self.shuffle:
            np.random.seed(self.seed)
            for i in range(len(index_datasets)):
                np.random.shuffle(index_datasets[i])
        cursors = [0] * len(self.datasets)
        while True:
            dataset_idx = np.random.choice(
                len(self.datasets), 1, p=self.sample_weights
            )[0]
            curr_cursor = cursors[dataset_idx]
            if cursors[dataset_idx] >= len(index_ranges[dataset_idx]):
                cursors[dataset_idx] = 0
                curr_cursor = cursors[dataset_idx]
                all_stopped[dataset_idx] = True
                exit_flag = True
                for stopped in all_stopped:
                    if not stopped:
                        exit_flag = False
                        break
                if exit_flag:
                    return
                if self.shuffle:
                    np.random.shuffle(index_ranges[dataset_idx])
            cursors[dataset_idx] += 1
            if not self.multi_sample_output:
                yield self.datasets[dataset_idx][
                    index_datasets[dataset_idx][
                        index_ranges[dataset_idx][curr_cursor]
                    ]
                ]
            else:
                for d in self.datasets[dataset_idx][
                    index_datasets[dataset_idx][
                        index_ranges[dataset_idx][curr_cursor]
                    ]
                ]:
                    yield d

    def __repr__(self):
        return "DistributedComposeRandomDataset"

    def __str__(self):
        return str(self.datasets)


@OBJECT_REGISTRY.register
class ComposeIterableDataset(data.IterableDataset):
    """Dataset wrapper built on ComposeDataset, shuffle, supporting multi workers.

    Args:
        datasets: config for each dataset.
        batchsize_list: batchsize for each dataset.
        multi_sample_output: whether dataset outputs multiple samples at the same time.
    """  # noqa

    def __init__(
        self,
        datasets: List[Dict],
        batchsize_list: List[int],
        multi_sample_output: bool = True,
    ) -> None:

        min_bs = min(batchsize_list)
        batchsize_list = [round(bs / min_bs) for bs in batchsize_list]
        self.datasets = ComposeDataset(datasets, batchsize_list)
        self.multi_sample_output = multi_sample_output

        self.len = len(self.datasets)
        self.idx = np.arange(self.len)

    def __iter__(self):
        worker_info = data.get_worker_info()
        np.random.shuffle(self.idx)

        while True:
            if worker_info is None:
                for idx in self.idx:
                    if self.multi_sample_output:
                        for d in self.datasets[idx]:
                            yield d
                    else:
                        yield self.datasets[idx]
            else:  # in a worker process
                # split workload
                per_worker = int(
                    math.ceil(self.len / float(worker_info.num_workers))
                )
                interval = worker_info.id * per_worker
                for idx in self.idx[interval : interval + per_worker]:
                    if self.multi_sample_output:
                        for d in self.datasets[idx]:
                            yield d
                    else:
                        yield self.datasets[idx]


@OBJECT_REGISTRY.register
class ResampleDataset(data.Dataset):
    """
    A wrapper of resample dataset.

    Using ResampleDataset can resample on original dataset
        with specific interval.

    Args:
        dataset: The datasets for resampling.
        with_flag: Whether to use dataset.flag. If True,
            resampling dataset.flag with resample_interval (
            dataset must has flag attribute in this case.)
        with_pack_flag: Whether to use dataet.pack_flag.
            If True, resampling pack_flag with resample_interval
            (dataset must has flag attribute in this case.)
            Default to False. Pack_flag identities samples belonging
            to different packs. Data belonging to the same pack has
            the same pack_flag and vice versa.
        resample_interval: resample interval.
    """

    def __init__(
        self,
        dataset: Dict,
        with_flag: bool = False,
        with_pack_flag: bool = False,
        resample_interval: int = 1,
    ):
        assert resample_interval >= 1 and isinstance(
            resample_interval, int
        ), "resample interval not valid!"
        self.dataset = dataset
        self.resample_interval = resample_interval
        self._ori_len = len(self.dataset)
        if with_flag:
            assert hasattr(dataset, "flag"), "dataset must has group flag"
            assert isinstance(
                dataset.flag, np.ndarray
            ), "dataset flag must is numpy array instance"
            assert (
                len(dataset) == dataset.flag.shape[0]
            ), "dataset flag length at axis 0 must equal to the dataset length"
            self.flag = dataset.flag[
                np.arange(0, self._ori_len, self.resample_interval)
            ]
        if with_pack_flag:
            assert hasattr(dataset, "pack_flag"), "dataset must has pack flag"
            assert isinstance(
                dataset.pack_flag, np.ndarray
            ), "dataset pack_flag must is numpy array instance"
            assert (
                len(dataset) == dataset.pack_flag.shape[0]
            ), "dataset flag length at axis 0 must equal to the dataset length"
            self.pack_flag = dataset.pack_flag[
                np.arange(0, self._ori_len, self.resample_interval)
            ]

    def __getitem__(self, idx):
        return self.dataset[idx * self.resample_interval]

    def __len__(self):
        return math.ceil(self._ori_len / self.resample_interval)


@OBJECT_REGISTRY.register
class ConcatDataset(data.ConcatDataset):
    """A wrapper of concatenated dataset with group flag.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`,
    addititionally concatenat the group flag of all dataset.

    Args:
        datasets: A list of datasets.
        with_flag: Whether to concatenate datasets flags.
            If True, concatenate all datasets flag (
            all datasets must has flag attribute in this case).
            Default to False.
        with_pack_flag: Whether to concatenate dataset.pack_flag.
            If True, aggregates and concatenates all datasets
            pack_flag (all datasets must has pack_flag attribute
            in this case). Default to False. Pack_flag identities
            data belonging to different packs. Data belonging to
            the same pack has the same pack_flag and vice versa.
        record_index: Whether to record the index. If True,
            record the index. Default to False.
    """

    def __init__(
        self,
        datasets: List,
        with_flag: bool = False,
        with_pack_flag: bool = False,
        record_index: bool = False,
        accumulate_flag: bool = False,
    ):
        super(ConcatDataset, self).__init__(datasets)
        self._record_index = record_index
        if with_flag:
            if accumulate_flag:
                accumulate_sum = 0

            flags = []
            for dataset in datasets:
                assert hasattr(dataset, "flag"), "dataset must has group flag"
                assert isinstance(
                    dataset.flag, np.ndarray
                ), "dataset flag must is numpy array instance"
                assert (
                    len(dataset) == dataset.flag.shape[0]
                ), "dataset flag length at axis 0 must equal to the dataset length"  # noqa: E501
                if accumulate_flag:
                    flag_tmp = dataset.flag + accumulate_sum
                    flags.append(flag_tmp)
                    accumulate_sum += len(np.unique(dataset.flag))
                else:
                    flags.append(dataset.flag)
            self.flag = np.concatenate(flags)

        if with_pack_flag:
            pack_flags = []
            pack_start_index = 0
            for dataset in datasets:
                assert hasattr(
                    dataset, "pack_flag"
                ), "dataset must has pack_flag"
                assert isinstance(
                    dataset.pack_flag, np.ndarray
                ), "dataset pack_flag must is numpy array instance"
                assert (
                    len(dataset) == dataset.pack_flag.shape[0]
                ), "dataset pack_flag length at axis 0 must equal to the dataset length"  # noqa: E501
                current_dataset_pack_flag = (
                    dataset.pack_flag + pack_start_index
                )
                pack_start_index = np.max(current_dataset_pack_flag) + 1
                pack_flags.append(current_dataset_pack_flag)
            self.pack_flag = np.concatenate(pack_flags)
        self.len_list = [len(dataset) for dataset in self.datasets]

    def __getitem__(self, idx) -> Dict:
        res = super().__getitem__(idx)
        if self._record_index:
            assert isinstance(res, dict), "__getitem__ must return a dict"
            res["index"] = idx
        return res


@OBJECT_REGISTRY.register
class CBGSDataset(object):
    """A wrapper of class sampled dataset.

    Implementation of paper
    `Class-balanced Grouping and Sampling for Point Cloud 3D Object
    Detection <https://arxiv.org/abs/1908.09492.>`_.

    Balance the number of scenes under different classes.

    Args:
        dataset: The dataset to be class sampled.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        self.sample_indices = self._get_sample_indices()
        if hasattr(self.dataset, "flag"):
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.sample_indices],
                dtype=np.uint8,
            )

    def _get_sample_indices(self):
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx in range(len(self.dataset)):
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()]
        )
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(
                cls_inds, int(len(cls_inds) * ratio)
            ).tolist()
        return sample_indices

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        ori_idx = self.sample_indices[idx]
        return self.dataset[ori_idx]

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.sample_indices)


@OBJECT_REGISTRY.register
class ChunkShuffleDataset(data.IterableDataset):
    """
    Dataset wrapper for chunk shuffle.

    Chunk shuffle will divide the entire dataset into chunks,
    then shuffle within chunks and shuffle between chunks.

    Args:
        dataset: datasets for shuffle.
        chunk_size_in_worker: Chunk size for shuffle in each worker.
        drop_last: if drop last.
        sort_by_str: whether to sort key by str.
            Str is the sort method of lmdb.
        seed: random seed for shuffle

    """

    def __init__(
        self,
        dataset,
        chunk_size_in_worker=1024,
        drop_last=True,
        sort_by_str=False,
        seed=0,
    ):
        self.dataset = dataset
        self.seed = seed
        self.drop_last = drop_last
        self.chunk_size_in_worker = chunk_size_in_worker
        self.sort_by_str = sort_by_str

        self.rank, self.world_size = get_dist_info()

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is not None:
            cur_worker = worker_info.id
            num_workers = worker_info.num_workers
        else:
            num_workers = 1
            cur_worker = 0

        num_chunks = math.floor(len(self.dataset) / self.chunk_size_in_worker)
        global_padding_size = len(self.dataset) - (
            num_chunks * self.chunk_size_in_worker
        )

        # generate by cursor
        standard_index = list(range(len(self.dataset)))
        if self.sort_by_str:
            standard_index = sorted(standard_index, key=str)
        if global_padding_size == 0:
            global_index = np.array(standard_index)
        else:
            global_index = np.array(standard_index)[:-global_padding_size]

        global_index = global_index.reshape(-1, self.chunk_size_in_worker)

        np.random.seed(self.seed)
        # shuffle between chunks
        np.random.shuffle(global_index)
        # shuffle in chunks
        tmp_index = global_index.T
        np.random.shuffle(tmp_index)
        global_index = tmp_index.T

        global_index = global_index.flatten()

        num_replicas = self.world_size * num_workers
        if self.drop_last and len(global_index) % num_replicas != 0:
            step_len = math.ceil(
                (len(global_index) - num_replicas) / num_replicas
            )
        else:
            step_len = math.ceil(len(global_index) / num_replicas)

        cur_idx = num_workers * self.rank + cur_worker

        global_index = global_index[: step_len * num_replicas].reshape(
            -1, num_replicas
        )
        index_range = global_index[:, cur_idx]

        return self._generator_sample(index_range)

    def _generator_sample(self, index_range):
        for idx in index_range:
            yield self.dataset[idx]

    def __repr__(self):
        return "ChunkShufffleDataset"

    def __str__(self):
        return str(self.dataset)
