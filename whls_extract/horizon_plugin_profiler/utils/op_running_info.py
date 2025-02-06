import os
import shutil
from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Iterable, List, Tuple, Type

from horizon_plugin_profiler.utils.typeguard import typechecked
from horizon_plugin_profiler.utils.version_helper import (
    check_torch_numpy_version,
)

import torch
from tabulate import tabulate
from torch import Tensor
from torch.utils._pytree import tree_flatten
from torch.utils.tensorboard import SummaryWriter

from horizon_plugin_pytorch import QTensor
from .location_info import TorchLocationInfo


class OpRunningInfo:
    """Hold the identification, input, output, w and b of a operation."""

    def __init__(
        self,
        loc: TorchLocationInfo,
        input=None,
        output=None,
        op_states=None,
        **other_infos,
    ) -> None:
        self.loc = loc
        self.input = input
        self.output = output
        if op_states is not None:
            assert isinstance(op_states, dict)
        else:
            op_states = {}
        self.op_states = op_states

        for k, v in other_infos.items():
            setattr(self, k, v)

    def dump(self, path):
        torch.save(self, path)

    @classmethod
    def load(cls, path):
        return torch.load(path)

    def explain(self):
        ret = OrderedDict()
        if self.input is not None:
            ret["input"] = self.input
        for k, v in self.op_states.items():
            ret[k] = v
        if self.output is not None:
            ret["output"] = self.output
        return ret

    def __hash__(self) -> int:
        return str.__hash__(self.loc.pickle())


class OpRunningInfoManager:
    """Hold all `OpRunningInfo` of a model.

    The manager offload `OpRunningInfo` to disk to reduce memory usage,
    user can read all `OpRunningInfo` of a specific module by `get` method.

    Args:
        work_dir (str, optional): A directory to save `OpRunningInfo`.
            If not given, an automatic temp directory will be used.
            Defaults to None.
    """

    _meta_file_name = "op_running_info.meta"
    _static_file_name = "statistic.txt"
    _distribution_out_dir_name = "tensorboard"

    def __init__(self, work_dir: str = None) -> None:
        if work_dir is None:
            self.tmp_dir = TemporaryDirectory()
            work_dir = self.tmp_dir.__enter__()
        else:
            os.makedirs(work_dir, exist_ok=True)
            self.tmp_dir = None
        self.op_infos_path = os.path.join(work_dir, "op_infos")
        os.makedirs(self.op_infos_path, exist_ok=True)

        self.work_dir = work_dir
        self.meta_path = os.path.join(
            work_dir, "op_infos", self._meta_file_name
        )

        self.call_sequence = OrderedDict()
        self.name_to_locs = {}
        self.module_op_appear_times = {}

    def add(self, item: OpRunningInfo):
        # get file name
        module_name = item.loc.mod_name
        idx = self.module_op_appear_times.get(module_name, 0)
        self.module_op_appear_times[module_name] = idx + 1
        file_name = "{}_{}.opinfo".format(module_name, idx)

        # dump OpRunningInfo to file
        item.dump(os.path.join(self.op_infos_path, file_name))

        # record location and filename by order
        self.call_sequence[item.loc] = file_name

        # record mapping from module name to locations
        if module_name in self.name_to_locs:
            self.name_to_locs[module_name].append(item.loc)
        else:
            self.name_to_locs[module_name] = [item.loc]

    def get_info_by_loc(self, location: TorchLocationInfo) -> OpRunningInfo:
        file_name = self.call_sequence[location]
        return OpRunningInfo.load(os.path.join(self.op_infos_path, file_name))

    def get_locs_by_mod_name(self, mod_name: str) -> List[TorchLocationInfo]:
        return self.name_to_locs[mod_name]

    def get_mod_names(self) -> Iterable[str]:
        return self.name_to_locs.keys()

    def dump_meta(self):
        torch.save(
            (
                self.call_sequence,
                self.name_to_locs,
                self.module_op_appear_times,
            ),
            self.meta_path,
        )

    @classmethod
    def load_meta(cls, work_dir):
        self = cls(work_dir)
        assert os.path.exists(self.meta_path)
        (
            self.call_sequence,
            self.name_to_locs,
            self.module_op_appear_times,
        ) = torch.load(self.meta_path)
        return self

    def clean(self):
        if self.tmp_dir is not None:
            self.tmp_dir.__exit__()
        self.tmp_dir = None
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)

    def __del__(self):
        if self.tmp_dir is not None:
            self.tmp_dir.__exit__()

    def _get_tensor_statistics(self, tensor: Tensor):
        dtype = tensor.dtype
        if isinstance(tensor, QTensor):
            scale = (
                tensor.q_scale().item()
                if tensor.q_scale() is not None
                and tensor.q_scale().numel() == 1
                else None
            )
            tensor = tensor.dequantize()
        else:
            scale = None
        if tensor.numel() == 0:
            return dtype, scale, None, None, None, None, tensor.shape
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        mean_val = tensor.mean(dtype=torch.float).item()
        var_val = tensor.float().var().item()
        shape = tensor.shape

        return dtype, scale, min_val, max_val, mean_val, var_val, shape

    @typechecked
    def table(
        self,
        out_dir: str = None,
        prefixes: Tuple[str, ...] = None,
        types: Tuple[Type, ...] = None,
        with_stack: bool = False,
    ):
        if out_dir is None:
            out_dir = self.work_dir

        if types is not None:
            types = tuple(TorchLocationInfo.format_op_type(t) for t in types)

        statistics_headers = [
            "Index",
            "Op Name",
            "Mod Name",
            "Attr",
            "Dtype",
            "Scale",
            "Min",
            "Max",
            "Mean",
            "Var",
            "Shape",
        ]
        if with_stack:
            statistics_headers.insert(3, "Stack")
        statistics_data = []
        idx = 0
        for loc, _ in self.call_sequence.items():
            loc: TorchLocationInfo
            if (
                (prefixes is None and types is None)
                or (types is not None and loc.op_name.startswith(types))
                or (prefixes is not None and loc.mod_name.startswith(prefixes))
            ):
                item = self.get_info_by_loc(loc)
                op_id = [
                    idx,
                    item.loc.op_name
                    + (
                        ""
                        if not hasattr(item, "hbir_op_type")
                        else f"[{item.hbir_op_type}]"
                    ),
                    item.loc.mod_name,
                ]
                if with_stack:
                    op_id.append(item.loc.user_stack)
                op_id = tuple(op_id)
                for attr, value in item.explain().items():
                    value = tree_flatten(value)[0]
                    if len(value) == 1:
                        if isinstance(value[0], Tensor):
                            statistics_data.append(
                                (
                                    op_id
                                    + (attr,)
                                    + self._get_tensor_statistics(value[0])
                                )
                            )
                    else:
                        for i, v in enumerate(value):
                            if isinstance(v, Tensor):
                                statistics_data.append(
                                    (
                                        op_id
                                        + ("{}_{}".format(attr, i),)
                                        + self._get_tensor_statistics(v)
                                    )
                                )

                idx += 1

        with open(os.path.join(out_dir, self._static_file_name), "w") as f:
            f.write(
                tabulate(
                    statistics_data,
                    headers=statistics_headers,
                    tablefmt="psql",
                    floatfmt=".7f",
                    numalign="left",
                )
            )

    def _record_tensor_distribute(
        self,
        tag: str,
        writer: SummaryWriter,
        tensor: Tensor,
        force_per_channel,
    ):
        if not isinstance(tensor, Tensor):
            return
        if tensor.numel() <= 1:
            return
        if isinstance(tensor, QTensor):
            ch_axis = tensor.q_per_channel_axis()
            tensor = tensor.as_subclass(Tensor)
        elif force_per_channel:
            ch_axis = 1
        else:
            ch_axis = -1

        if ch_axis < 0 or tensor.ndim < 2:
            writer.add_histogram(tag, tensor)
        else:
            for i in range(tensor.size(ch_axis)):
                writer.add_histogram(tag, tensor.select(ch_axis, i), i)

    @typechecked
    def tensorboard(
        self,
        out_dir: str = None,
        prefixes: Tuple[str, ...] = None,
        types: Tuple[Type, ...] = None,
        force_per_channel: bool = False,
    ):
        check_torch_numpy_version()

        if out_dir is None:
            out_dir = self.work_dir

        tensorboard_dir = os.path.join(
            out_dir, self._distribution_out_dir_name
        )
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)

        if types is not None:
            types = tuple(TorchLocationInfo.format_op_type(t) for t in types)

        for loc, _ in self.call_sequence.items():
            loc: TorchLocationInfo
            if (
                (prefixes is None and types is None)
                or (types is not None and loc.op_name.startswith(types))
                or (prefixes is not None and loc.mod_name.startswith(prefixes))
            ):
                item = self.get_info_by_loc(loc)
                tag_head = "{}:{}".format(
                    item.loc.mod_name.replace(".", ":"), item.loc.op_name
                )
                for attr, value in item.explain().items():
                    value = tree_flatten(value)[0]
                    if len(value) == 1:
                        self._record_tensor_distribute(
                            "{}:{}".format(tag_head, attr),
                            writer,
                            value[0],
                            force_per_channel,
                        )
                    else:
                        for i, v in enumerate(value):
                            self._record_tensor_distribute(
                                "{}:{}".format(
                                    tag_head, "{}_{}".format(attr, i)
                                ),
                                writer,
                                v,
                                force_per_channel,
                            )

        writer.close()
