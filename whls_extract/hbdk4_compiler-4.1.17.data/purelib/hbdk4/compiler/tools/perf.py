import enum
import functools
import hashlib
import html
import itertools
import json
import math
import operator
import os
import random
import re
import datetime
from typing import Iterable, List, Tuple, Union, Dict, Optional


def to_str(v):
    """helper function like str(), but add comma for integers"""
    if isinstance(v, int):
        return f"{v:,}"
    else:
        return str(v)


def to_escaped_str(s: str):
    """can be used as json key/value"""
    return s.replace('"', '\\"').replace("\n", "<br>")


def to_kmgtp(size: int, use_1024: bool):
    factor = 1024 if use_1024 else 1000
    n = 0
    while size > factor:
        size /= factor
        n += 1

    factor_labels = (
        ("", "Ki", "Mi", "Gi", "Ti", "Pi")
        if use_1024
        else ("", "K", "M", "G", "T", "P")
    )
    return size, factor_labels[n]


def get_common_prefix(strings: List[str]) -> str:
    common_prefix = ""
    for letters in zip(*strings):
        if len(set(letters)) == 1:
            common_prefix += letters[0]
        else:
            break
    return common_prefix


def make_html_table(table: List[Iterable[str]], merge_column: bool):
    """helper function to make an html table.  table[i] is a row"""
    for row_id, row in enumerate(table):
        cells = []  # each cell contains html element (th, td)
        format_str = (
            "<th%s>\n      %s\n    </th>"
            if row_id == 0
            else "<td%s>\n      %s\n    </td>"
        )

        if merge_column:
            col_id = 0
            while col_id < len(row):
                cell = row[col_id]
                col_id_of_different_content = col_id + 1
                while (
                    col_id_of_different_content < len(row)
                    and cell == row[col_id_of_different_content]
                ):
                    col_id_of_different_content += 1
                if col_id_of_different_content == col_id + 1:  # only one cell
                    cells.append(format_str % ("", cell))
                else:
                    cells.append(
                        format_str
                        % (" colspan=%d" % (col_id_of_different_content - col_id), cell)
                    )
                col_id = col_id_of_different_content
        else:
            cells = [format_str % ("", cell) for cell in row]

        table[row_id] = "\n    ".join(cells)

    s = "\n  ".join("<tr>\n    %s\n  </tr>" % row for row in table)

    return "\n<table>\n  %s\n</table>\n" % s


def make_html_text_with_tooltip(
    text: str, tooltip: Union[str, List[str], Tuple[str]], is_icon=True, color=None
):
    """helper function to make html `tooltip` on `text`, tooltip can be str or a list of strings"""
    if isinstance(tooltip, (list, tuple)):
        tooltip = "&#10;".join(html.escape(x) for x in tooltip)
    else:
        tooltip = html.escape(tooltip)
    assert isinstance(tooltip, str)

    color_text = ""
    if color:
        assert isinstance(color, str)
        color_text = ', style="color: %s;"' % color

    if is_icon:
        return '<span class="hbdktooltip", title="%s" %s> %s </span>' % (
            tooltip,
            color_text,
            text,
        )
    return '<div title="%s" %s> %s </div>' % (tooltip, color_text, text)


def is_dict_of(d, key_type, value_type):
    b1 = isinstance(d, dict)
    b2 = all(
        isinstance(k, key_type) and isinstance(v, value_type) for k, v in d.items()
    )
    return b1 and b2


@enum.unique
class PerfType(enum.Enum):  # NOTE-perf-type: search this if the definition changes
    sequential = 0
    pessimistic_estimation = 10
    moderate_estimation = 11
    optimistic_estimation = 12
    tiling_estimation = 13
    actual_insts = 20

    def is_estimation(self):
        return self != PerfType.actual_insts


class MlirType:
    """Value type in MLIR"""

    def __init__(self, text: str):
        self.long_str = text

        tensor_type_regex = r"tensor|memref"
        shape_regex = r"([0-9]+x)*"  # '224x224x3x' or ''
        element_type_regex = r"(f|si|ui|i)[0-9]+|!hbir.bool8"  # f32
        attributes_regex = r"[^>]*"
        regex = re.compile(
            r"(%s)<(%s)(%s)(%s)>"
            % (tensor_type_regex, shape_regex, element_type_regex, attributes_regex)
        )
        mo = regex.match(text)
        assert mo, 'cannot parse MLIR Type "%s"' % text
        self.tensor_type = mo.group(1)  # tensor or memref
        self.shape_str = mo.group(2)  # '224x224x3x' or ''
        self.shape_str = self.shape_str[:-1] if self.shape_str else "1"
        self.element_type = mo.group(4)  # f32
        self.attributes = mo.group(6)  # layout block, etc.

        self.short_str = (
            self.shape_str + "x" + self.element_type
        )  # 1x224x224x3xf32, without attributes
        self.element_bits = int(re.match(r".+?(\d+)", self.element_type).group(1))
        self.element_bytes = int(math.ceil(self.element_bits / 8))
        self.shape = tuple(int(x) for x in self.shape_str.split("x"))
        self.batch = 1 if len(self.shape) <= 3 else self.shape[0]
        self.elements = functools.reduce(operator.mul, self.shape)
        self.bytes = self.elements * self.element_bytes


class TensorInfo:
    """representing a tensor, only for estimated perf"""

    def __init__(self, tid: int, kind: str, mlir_type: Union[MlirType, str]):
        self.tid = tid
        self.kind = kind
        self.mlir_type = (
            mlir_type if isinstance(mlir_type, MlirType) else MlirType(mlir_type)
        )
        self.definer: Optional["OpInfo"] = None  # may have no definer (function args)
        self.users: List["OpInfo"] = []

        self.fixed_color = random.sample(range(64, 192), 3)
        self.fixed_color = "#" + "".join("%02X" % x for x in self.fixed_color)

    @property
    def has_far_user(self) -> bool:
        if self.definer:
            good_cid = self.definer.computing_id + 1  # immediately follows definer
            return any(user.computing_id != good_cid for user in self.users)
        return False

    @property
    def desc(self) -> str:
        return f"id=t{self.tid}, kind={self.kind}, type={self.mlir_type.long_str}"


class OpInfo:
    """for an op (usually a layer before SRAM allocation) in model"""

    def __init__(
        self,
        op_info: dict,
        stage_status: "StageStat",
        tensor_infos: Dict[int, TensorInfo],
    ):
        self.parent = stage_status
        self.map = op_info
        self.op_name = op_info["op_name"]
        self.block_id = op_info.get("block id", 0)  # NOTE-op-table-block-id
        self.op_id = op_info.get("op id", None)  # NOTE-profile-chart-op-id
        self.tile_id = op_info.get("tile id", 0)  # NOTE-profile-chart-tile-id
        self.computing_id = op_info.get("computing id", None)
        self.source_op_id = op_info.get("source op id", None)
        self.layer_group_size = op_info.get("layer_group_size", None)
        self.start = op_info["start"]  # cycles
        self.cycles = op_info["cycles"]
        self.macs = max(
            op_info.get(x + "_macs", 0) for x in ("conv", "matmul", "linear")
        )
        self.categories = tuple(
            k[:-7] for k in op_info if k.endswith("_cycles")
        )  # may belong to a few categories
        self.end = self.start + self.cycles
        self.instance_ids = None
        if "instance_id" in op_info:
            self.instance_ids = (op_info["instance_id"],)
        if "instance_num" in op_info:  # e.g., "3" means 0&1&2
            assert not self.instance_ids, "cannot specify both instance id and num"
            self.instance_ids = tuple(x for x in range(op_info["instance_num"]))

        # l0 inf
        self.l0m_info = {
            "max": op_info.get("op_max_used_l0_size", 0),
            "min": op_info.get("op_min_used_l0_size", 0),
            "real": op_info.get("op_real_used_l0_size", 0),
        }
        self.has_l0m_info = any([v != 0 for v in self.l0m_info.values()])

        self.is_layer_group = "layer_group" in self.categories
        self.is_misc = "misc" in self.categories
        if self.is_layer_group or self.is_misc:
            self.is_load_store = False
            self.is_computing = False
        else:
            self.is_load_store = self.op_name in ("hbdk.load", "hbdk.store")
            self.is_computing = not self.is_load_store

        if self.tile_id != 0 and "tensor id" in self.map:
            tensor_id = self.map["tensor id"]
            if tensor_id in tensor_infos and "tensor info" not in self.map:
                self.map["tensor info"] = tensor_infos[tensor_id].desc

        self.loc = op_info["location"] if "location" in op_info else ""
        self.operands: List[TensorInfo] = []
        self.results: List[TensorInfo] = []
        if self.parent.is_estimation and self.is_computing and self.tile_id == 0:
            # NOTE-estimation-tile-id: only tile 0 contains these info, to save memory
            # operands/results info are deleted from map, because they are too much for tooltip
            keywords = ("id", "kind", "type")

            operand_num = op_info["operand num"]
            for i in range(operand_num):
                temp_new_ti = TensorInfo(
                    *[op_info.pop(f"operand_{i}_{kw}") for kw in keywords]
                )
                if temp_new_ti.tid in tensor_infos:  # has been explicitly defined
                    unique_old_ti = tensor_infos[temp_new_ti.tid]
                    assert unique_old_ti.tid == temp_new_ti.tid
                    assert unique_old_ti.kind == temp_new_ti.kind
                    assert (
                        unique_old_ti.mlir_type.long_str
                        == temp_new_ti.mlir_type.long_str
                    )
                else:  # not explicitly defined (no definer), must be function args or constants
                    assert temp_new_ti.kind in (
                        "input",
                        "input_output",
                        "constant",
                    ), f"{temp_new_ti.kind} (id={temp_new_ti.tid}) tensor has no definer?"
                    unique_old_ti = temp_new_ti

                if self not in unique_old_ti.users:
                    unique_old_ti.users.append(self)
                self.operands.append(unique_old_ti)

            result_num = op_info["result num"]
            for i in range(result_num):
                ti = TensorInfo(*[op_info.pop(f"result_{i}_{kw}") for kw in keywords])
                ti.definer = self

                if ti.tid not in tensor_infos:  # common case
                    tensor_infos[ti.tid] = ti
                    self.results.append(ti)
                else:
                    assert (
                        self.parent.perf_type == PerfType.tiling_estimation
                    ), "multiple definer"
                    assert self.tile_id > 0
                    unique_old_ti = tensor_infos[ti.tid]
                    assert unique_old_ti.tid == ti.tid
                    assert unique_old_ti.kind == ti.kind
                    assert unique_old_ti.mlir_type.long_str == ti.mlir_type.long_str

    def gen_digest(self, only_header: bool, include_l0m_info: bool) -> Tuple:
        """return a tuple of items, generating a row of the op info table"""

        # cycles, with all perf info as tooltip
        perf_infos = [f"{k} = {v}" for k, v in self.map.items()]
        cycles_cell = make_html_text_with_tooltip(
            to_str(self.cycles), perf_infos, False
        )

        # operands/results, with tensor info as tooltip
        def gen_one_ti(ti: TensorInfo):
            t_id, t_kind, t_type = ti.tid, ti.kind, ti.mlir_type
            assert isinstance(t_id, int)
            assert isinstance(t_kind, str)
            assert isinstance(t_type, MlirType)
            text = "t%d, %s, %s" % (t_id, t_kind, t_type.short_str)

            tooltips = [to_str(t_type.elements) + " elements"]
            if ti.definer:  # has definer
                tooltips.append(
                    f"definer: id={ti.definer.op_id}, cid={ti.definer.computing_id}"
                )
            for user in ti.users:
                tooltips.append(f"user: id={user.op_id}, cid={user.computing_id}")
            if t_type.attributes:
                tooltips.append("attributes: " + t_type.attributes)

            c = ti.fixed_color if ti.has_far_user else None
            return make_html_text_with_tooltip(text, tooltips, False, c)

        operands_cell = " ".join(gen_one_ti(ti) for ti in self.operands)
        results_cell = " ".join(gen_one_ti(ti) for ti in self.results)

        l0m_info_cell = " ".join(
            make_html_text_with_tooltip(f"{k}: {v}", "", False)
            for k, v in self.l0m_info.items()
        )

        # location, if exists
        loc_cell = ""
        if self.loc:
            if ":" in self.loc:
                loc_cell = self.loc[
                    self.loc.index(":") + 1 : -1
                ]  # assume mlir location format
            else:
                loc_cell = self.loc
            color = "#" + hashlib.md5(self.loc.encode("utf-8")).hexdigest()[-6:]
            loc_cell = make_html_text_with_tooltip(loc_cell, self.loc, False, color)

        digests = {
            "block id": self.block_id,
            "op id": self.op_id,
            "computing id": self.computing_id,
            "source op id": self.source_op_id,
            "op name": self.op_name,
            "cycles": cycles_cell,
            "MAC num": to_str(self.macs) if self.macs else "",
            "MAC util": self.map.get("mac_util", ""),
            "l0m info": l0m_info_cell,
            "operands": operands_cell,
            "results": results_cell,
            "location": loc_cell,
        }
        if not include_l0m_info:
            digests.pop("l0m info")

        if only_header:
            return tuple(digests.keys())

        return tuple(digests.values())


class StageStat:
    """statistics of 1 function/block's 1 stage"""

    def __init__(self, obj: dict):
        """create StageStat with a dict (json object)"""
        assert isinstance(obj, dict)

        self.parent = None  # to be filled when associating this to a FuncBlockStat

        known_keys = {
            "func_name",
            "signature",
            "setup",
            "details",
            "live_ranges",
            "summary",
        }
        unknown_keys = set(obj.keys()).difference(known_keys)
        assert not unknown_keys, "unknown key in perf json: %s" % unknown_keys
        self.func_name = obj["func_name"]
        self.signature = obj["signature"]

        try:
            self.block_id = int(
                self.signature
            )  # only for block, the index in the function
            self.is_block = True
            self.long_title = "%s (block %s)" % (self.func_name, self.signature)
            self.short_title = self.long_title
        except ValueError:  # is function
            self.block_id = -1
            self.is_block = False
            self.long_title = self.func_name + " " + self.signature
            self.short_title = self.func_name
        self.long_title_escape_quote = self.long_title.replace('"', '\\"')

        self.args = tuple()
        self.args_elements = 0
        self.retvals = tuple()
        self.retvals_elements = 0
        if not self.is_block:
            args_str, retvals_str = re.match("(.+) -> (.+)", self.signature).groups()
            type_regex = re.compile("((tensor|memref)<.+?>)")
            self.args = tuple(MlirType(x[0]) for x in type_regex.findall(args_str))
            self.args_elements = sum(x.elements for x in self.args)
            self.retvals = tuple(
                MlirType(x[0]) for x in type_regex.findall(retvals_str)
            )
            self.retvals_elements = sum(x.elements for x in self.retvals)

        setup = obj["setup"]
        known_keys = {
            "stage",
            "stage_id",
            "perf_type",
            "frequency_mhz",
            "mac_per_cycle",
            "l1m_bytes_per_cycle",  # max speed
            "l1m_bytes",  # capacity
            "l2m_bytes",  # capacity
            "begin_date_time",
            "end_date_time",
            "env_vars",
            "bandwidth",
        }
        unknown_keys = set(setup.keys()).difference(known_keys)
        assert not unknown_keys, 'unknown key in perf json["setup"]: %s' % unknown_keys
        self.stage = setup["stage"]
        self.stage_id = setup["stage_id"]
        self.perf_type = PerfType(setup["perf_type"])
        self.is_estimation = self.perf_type.is_estimation()
        self.mhz = setup["frequency_mhz"]
        self.mac_per_cycle = setup["mac_per_cycle"]
        self.l1m_bytes_per_cycle = setup.get("l1m_bytes_per_cycle", 0)  # max speed
        self.l1m_bytes = setup["l1m_bytes"]
        self.l2m_bytes = setup["l2m_bytes"]
        self.env_vars = setup["env_vars"]  # include default values for unset env vars
        self.begin_date_time = setup["begin_date_time"].strip()
        self.begin_date_time_obj = datetime.datetime.strptime(
            self.begin_date_time, "%c"
        )
        self.begin_date = self.begin_date_time_obj.date()
        self.begin_time = self.begin_date_time_obj.time()
        self.end_date_time = setup["end_date_time"].strip()
        self.end_date_time_obj = datetime.datetime.strptime(self.end_date_time, "%c")
        self.end_date = self.end_date_time_obj.date()
        self.end_time = self.end_date_time_obj.time()
        self.analysis_seconds = (
            self.end_date_time_obj - self.begin_date_time_obj
        ).total_seconds()
        self.bandwidth = setup["bandwidth"]

        self.tensor_infos: Dict[int, "TensorInfo"] = dict()
        self.op_infos = tuple(
            OpInfo(x, self, self.tensor_infos) for x in obj["details"]
        )

        self.live_ranges = obj["live_ranges"]
        assert is_dict_of(self.live_ranges, str, list)
        self.max_mem_usage = dict()
        self.gen_html_chart_of_live_range(0, 0)  # set value to self.max_mem_usage
        self.max_l1m_usage_bytes = sum(
            v for k, v in self.max_mem_usage.items() if k.startswith("l1m")
        )
        self.max_l2m_usage_bytes = sum(
            v for k, v in self.max_mem_usage.items() if k.startswith("l2m")
        )
        self.max_ddr_usage_bytes = sum(
            v for k, v in self.max_mem_usage.items() if k.startswith("ddr")
        )

        # summary, most complicated
        summary = obj["summary"]
        known_keys = {
            "run cycles",
            "all_ops",
            "run cycles (no parallelism)",
        }  # without each op's statistics
        unknown_keys = set(
            x for x in summary.keys() if not isinstance(summary.get(x), dict)
        ).difference(known_keys)
        assert not unknown_keys, (
            'unknown key in perf json["summary"]: %s' % unknown_keys
        )
        self.cycles = max(1, summary["run cycles"])
        self.cycles_without_ilp = summary[
            "run cycles (no parallelism)"
        ]  # ilp = inst-level parallelism

        self.total_stat: Dict[str, int] = summary["all_ops"]
        assert is_dict_of(self.total_stat, str, int)

        self.op_stat: Dict[Dict[str, int]] = dict()
        for op_name, op_stat in summary.items():
            if op_name not in known_keys:  # really an op name
                assert is_dict_of(op_stat, str, int)
                self.op_stat[op_name] = op_stat

        # new statistics, not in json, or not always in json
        self.hz = int(self.mhz * 1e6)
        self.ghz = self.mhz / 1e3
        self.peak_macs = self.mac_per_cycle * self.hz
        self.peak_ops = self.peak_macs * 2
        self.peak_tops = self.peak_ops / 1e12  # float

        self.total_macs = sum(
            self.total_stat.get(x + "_macs", 0) for x in ("conv", "matmul", "linear")
        )
        self.opt_mac_cycles = max(1, math.ceil(self.total_macs / self.mac_per_cycle))
        self.real_mac_cycles = sum(
            self.total_stat.get(x + "_cycles", 0) for x in ("conv", "matmul", "linear")
        )

        self.mac_usage_rate = self.real_mac_cycles / self.cycles
        self.fps = self.hz / self.cycles
        self.util = None  # need to use first StageStat to calculate
        self.real_tops = None  # need to use first StageStat to calculate
        self.latency_us = self.cycles / self.mhz
        self.latency_ms = self.latency_us / 1e3
        self.total_l1m_read_bytes = self.total_stat.get("l1m_read_bytes", 0)
        self.total_l1m_write_bytes = self.total_stat.get("l1m_write_bytes", 0)
        self.total_l0m_read_bytes = self.total_stat.get("l0m_read_bytes", 0)
        self.total_l0m_write_bytes = self.total_stat.get("l0m_write_bytes", 0)

        self.has_l0m_info = any([oi.has_l0m_info for oi in self.op_infos])

        memspaces = ("l1m", "l2m", "ddr")
        pairs = list(itertools.combinations(memspaces, 2))
        transfer_types = ["%s_to_%s" % (a, b) for a, b in pairs] + [
            "%s_to_%s" % (b, a) for a, b in pairs
        ]

        def process_one_type(attr_name: str, stat_keyword: str):
            my_types = list(filter(lambda x: stat_keyword in x, transfer_types))

            # bytes and cycles
            num = sum(self.total_stat.get(x + "_cycles", 0) for x in my_types)
            setattr(self, attr_name + "_cycles", num)
            num = sum(self.total_stat.get(x + "_bytes", 0) for x in my_types)
            setattr(self, attr_name + "_bytes", num)

            # statistics for chunk sizes
            if num == 0:
                return
            d = dict()
            for oi in self.op_infos:
                chunk_size = max(oi.map.get(x + "_bytes", 0) for x in my_types)
                if chunk_size == 0:
                    continue

                aligned_size = 2 ** (int(math.log2(chunk_size)))  # floor
                d[aligned_size] = d.get(aligned_size, 0) + chunk_size
            assert num == sum(d.values())
            setattr(self, attr_name + "_chunk_sizes", d)

        for memspace in memspaces:
            process_one_type("from_" + memspace, memspace + "_to_")
            process_one_type("to_" + memspace, "_to_" + memspace)

        self.total_transfer_cycles = sum(
            self.total_stat.get(x + "_cycles", 0) for x in transfer_types
        )

        self.constant_size = self.total_stat.get("constant size", 0)
        self.constant_elements = self.total_stat.get("constant elements", 0)
        self.other_cycles = (
            self.cycles_without_ilp - self.real_mac_cycles - self.total_transfer_cycles
        )

        for op_info in self.op_infos:
            if op_info.macs and "mac_util" not in op_info.map:
                opt_cycles = int(op_info.macs / self.mac_per_cycle)
                util = opt_cycles * 100 / op_info.cycles
                op_info.map["mac_util"] = "%d%%" % util

        # statistics for tensors size at each computing op
        if self.is_estimation:  # only for estimated perf (because it's before codegen)

            self.computing_op_num = max(
                op_info.computing_id
                for op_info in self.op_infos
                if op_info.is_computing
            )
            self.output_tensor_elements_at_op = [0] * (self.computing_op_num + 1)
            self.live_tensor_elements_at_op = [0] * (self.computing_op_num + 1)

            # ignore simulated tiles[1:] here
            for op_info in filter(
                lambda x: x.is_computing and x.tile_id == 0, self.op_infos
            ):
                for result in op_info.results:
                    size = result.mlir_type.elements
                    self.output_tensor_elements_at_op[op_info.computing_id] += size
                    begin = op_info.computing_id
                    end = (
                        max(x.computing_id for x in result.users)
                        if result.users
                        else begin
                    )
                    for i in range(begin, end + 1):
                        self.live_tensor_elements_at_op[i] += size

    def identifier(self) -> tuple:
        if self.is_block:
            return self.func_name, self.block_id
        else:
            return (self.func_name,)

    def gen_digest(
        self, only_header: bool, arg_num: int, retval_num: int, optional_keys: List[str]
    ) -> List:
        """return a tuple for a table column"""

        stage_cell = "%s (%d)" % (self.stage, self.stage_id)
        metrics = [
            ("stage", stage_cell),
            ("perf type", "%s" % self.perf_type.name),
            ("begin date", "%s" % self.begin_date),
            ("begin time", "%s" % self.begin_time),
            ("end time", "%s" % self.end_time),
            ("analysis time", "%ds" % int(self.analysis_seconds)),
        ]

        for i in range(arg_num):
            metrics.append(
                (
                    "argument #%d" % i,
                    self.args[i].short_str if i < len(self.args) else "",
                )
            )
        for i in range(retval_num):
            metrics.append(
                (
                    "return value #%d" % i,
                    self.retvals[i].short_str if i < len(self.retvals) else "",
                )
            )

        metrics += [
            ("", ""),
            ("MACs", self.total_macs),
            ("MAC cycles (optimal)", self.opt_mac_cycles),
            ("MAC cycles (actual)", self.real_mac_cycles),
            ("run cycles (no parallelism)", self.cycles_without_ilp),
            ("run cycles", self.cycles),
            ("FPS", "%.1f" % self.fps),
            ("utilization", "%.1f%%" % (self.util * 100)),
            ("L1M capacity", "%.1f MiB" % (self.l1m_bytes / 1024 / 1024)),
            ("L1M BW (max)", "%d GB/s" % (self.l1m_bytes_per_cycle * self.hz / 1e9)),
            (
                "L1M BW (actual read)",
                "%d GB/s" % (self.total_l1m_read_bytes * self.fps / 1e9),
            ),
            (
                "L1M BW (actual write)",
                "%d GB/s" % (self.total_l1m_write_bytes * self.fps / 1e9),
            ),
            ("L2M capacity", "%.1f MiB" % (self.l2m_bytes / 1024 / 1024)),
            ("L2M usage", "%.1f MiB" % (self.max_l2m_usage_bytes / 1024 / 1024)),
            (
                "L2M BW (actual read)",
                "%.1f GB/s" % (self.from_l2m_bytes * self.fps / 1e9),
            ),
            (
                "L2M BW (actual write)",
                "%.1f GB/s" % (self.to_l2m_bytes * self.fps / 1e9),
            ),
            (
                "DDR BW (actual read)",
                "%.1f GB/s" % (self.from_ddr_bytes * self.fps / 1e9),
            ),
            (
                "DDR BW (actual write)",
                "%.1f GB/s" % (self.to_ddr_bytes * self.fps / 1e9),
            ),
            ("", ""),
        ]

        if not optional_keys:
            if only_header:
                return [x[0] for x in metrics]
            return [x[1] for x in metrics]

        # with optional keys
        if only_header:
            return [x[0] for x in metrics] + ["stage"] + optional_keys + ["stage"]

        return (
            [x[1] for x in metrics]
            + [stage_cell]
            + [self.total_stat.get(k, 0) for k in optional_keys]
            + [stage_cell]
        )

    def gen_html_table_of_op_infos(self) -> str:
        """return HTML table of op infos"""
        computing_op_infos = [
            x for x in self.op_infos if x.is_computing and x.loc != ""
        ]
        assert computing_op_infos

        table = [computing_op_infos[0].gen_digest(True, self.has_l0m_info)]
        table += [x.gen_digest(False, self.has_l0m_info) for x in computing_op_infos]
        return (
            '\n<h3 style="text-align: center;"> %s %s layers </h3>\n'
            % (self.long_title, self.stage)
        ) + make_html_table(table, False)

    def determine_interval_cycles(self, min_interval_num: int, use_125: bool) -> int:
        """determine proper interval number fulfilling the specified conditions"""

        factors = (2, 2.5, 2) if use_125 else (10,)
        i = 0

        interval_cycles = 100
        while self.cycles / int(interval_cycles * factors[i]) >= min_interval_num:
            interval_cycles *= factors[i]
            i = (i + 1) % len(factors)

        return int(interval_cycles)

    @staticmethod
    def gen_html_chart_of_profile_common() -> str:
        return """
// begin of common part of profile chart
var option = {
    tooltip: {
        formatter: function (params) { return params.marker + params.data.tooltip; }
    },
    title: {
        text: 'to_be_filled',
        left: 'center'
    },
    dataZoom: [{
        type: 'slider',
        filterMode: 'weakFilter',
        showDataShadow: false,
        top: 400,
        labelFormatter: ''
    }, {
        type: 'inside',
        filterMode: 'weakFilter'
    }],
    grid: {
        height: 300
    },
    xAxis: {
        min: 0,
        max: 'to_be_filled',
        scale: true,
        axisLabel: {
            formatter: function (val) {
                return Number(val).toLocaleString() + ' cycles';
            }
        }
    },
    yAxis: {
        data: 'to_be_filled'
    },
    series: [{
        type: 'custom',
        renderItem: renderItem,
        itemStyle: {
            opacity: 0.8
        },
        encode: {
            x: [1, 2],
            y: 0
        },
        data: 'to_be_filled',
        label: {
            show: true,
            formatter: function(params) {
                return params.data.label;
            }
        }
    }]
};  // end of common option of profile chart

function renderItem(params, api) {
    var categoryIndex = api.value(0);
    var start = api.coord([api.value(1), categoryIndex]);
    var end = api.coord([api.value(1) + api.value(2), categoryIndex]);
    var height = api.size([0, 1])[1] * 0.6;

    var rectShape = echarts.graphic.clipRectByRect({
        x: start[0],
        y: start[1] - height / 2,
        width: end[0] - start[0],
        height: height
    }, {
        x: params.coordSys.x,
        y: params.coordSys.y,
        width: params.coordSys.width,
        height: params.coordSys.height
    });

    return rectShape && {
        type: 'rect',
        transition: ['shape'],
        shape: rectShape,
        style: api.style()
    };
}

// end of common part of profile chart
"""

    def gen_html_chart_of_profile(self, serial_id: int) -> str:
        """
        generate profile chart script for this stage
        :param serial_id: the serial id of this func/block in an html
        """

        code = "\n\nvar data_%d = [\n" % serial_id

        # Collect all categories. Each category is a row in chart
        row_names = set()
        for op_info in self.op_infos:
            for category in op_info.categories:  # 1 op may belong to many categories
                row_name = category
                if op_info.instance_ids is None:
                    row_names.add(row_name)
                else:  # add "conv #0", "conv #1", ...
                    row_names.update(
                        set(f"{row_name} #{i}" for i in op_info.instance_ids)
                    )

        # layer group is the first (bottom) row, then load, computing, store
        def sort_keys(name: str):
            d = {
                "layer_group": 0,
                "ddr_to_l2m": 10,
                "ddr_to_l1m": 11,
                "l2m_to_l1m": 12,
                # computing ops are 20
                "l1m_to_l2m": 30,
                "l1m_to_ddr": 31,
                "l2m_to_ddr": 32,
            }
            for k in d:
                if name.startswith(k):
                    return d[k], name
            return 20, name

        # determine row ids
        row_names = list(row_names)
        row_names.sort(key=sort_keys)
        row_name_to_id = {}
        for row_name in row_names:
            row_name_to_id[row_name] = len(row_name_to_id)

        keyword = "tensor id" if self.is_estimation else "source op id"
        keyword_to_color = {}
        layer_group_accumulated_macs = 0
        for op_info in self.op_infos:
            layer_group_accumulated_macs += op_info.macs
            for category in op_info.categories:  # 1 op may belong to many categories
                this_row_names = [category]
                if op_info.instance_ids is not None:
                    this_row_names = [f"{category} #{i}" for i in op_info.instance_ids]
                for row_name in this_row_names:  # draw block for all these names
                    row_id = row_name_to_id[row_name]

                    # generate descriptive tooltip (pop-up window when mouse over)
                    assert (
                        op_info.end <= self.cycles
                    ), "%d ~ %d exceeds run time (%d)" % (
                        op_info.start,
                        op_info.end,
                        self.cycles,
                    )
                    tooltip = "%s <br> op time: <b>%s ~ %s (%s)</b>" % (
                        op_info.op_name,
                        to_str(op_info.start),
                        to_str(op_info.end),
                        to_str(op_info.cycles),
                    )
                    for key in sorted(op_info.map):
                        # no 'xx_cycles' in tooltip because they are shown explicitly.
                        # no 'asm' in tooltip because they are too long
                        if key not in (
                            "op_name",
                            "start",
                            "cycles",
                            category + "_cycles",
                            "asm",
                        ):
                            tooltip += "<br> %s: <b>%s</b>" % (
                                key,
                                to_str(op_info.map[key]),
                            )

                    # determine color and label (shown on the rectangle)
                    label = ""
                    if self.is_estimation and op_info.is_layer_group:
                        # NOTE-pseudo-lg-category
                        opt_cycles = layer_group_accumulated_macs / self.mac_per_cycle
                        util = int(opt_cycles * 100 / max(1, op_info.cycles))
                        this_color = [max(224 - util * 2, 24)] * 3
                        label = f"{op_info.layer_group_size} ({util}%)"
                        layer_group_accumulated_macs = 0
                    elif keyword in op_info.map:  # same keyword -> same color
                        v = op_info.map[keyword]
                        if v not in keyword_to_color:
                            keyword_to_color[v] = random.sample(range(32, 224), 3)
                        this_color = keyword_to_color[v]
                    else:
                        this_color = random.sample(range(32, 224), 3)

                    s = "{ "
                    s += "value: [%d, %d, %d], " % (
                        row_id,
                        op_info.start,
                        op_info.cycles,
                    )
                    s += 'label: "%s", ' % to_escaped_str(label)
                    s += 'tooltip: "%s", ' % to_escaped_str(tooltip)
                    s += (
                        'itemStyle: {normal : {color:"#%s", borderColor:"black"}}'
                        % "".join("%02X" % x for x in this_color)
                    )
                    s += " }"

                    code += "  %s,\n" % s
        code += "];\n"

        row_names = [(b, a) for a, b in row_name_to_id.items()]
        row_names.sort()
        row_names = [x[1] for x in row_names]

        code += """
option.title.text = "{stage} Profile";
option.xAxis.max = {run_cycles};
option.yAxis.data = {row_names};
option.series[0].data = data_{serial_id};
var dom_{serial_id} = document.getElementById("profile_chart_{serial_id}");
var myChart_{serial_id} = echarts.init(dom_{serial_id});
if (option && typeof option === 'object') {{
    myChart_{serial_id}.setOption(option);
}}

    """.format(
            serial_id=serial_id,
            stage=self.stage,
            row_names=repr(row_names),
            run_cycles=self.cycles,
        )

        return code

    @staticmethod
    def gen_html_chart_of_line_common() -> str:
        return """
// begin of common part of line chart
var option = {
  tooltip: {
    trigger: 'axis',
    axisPointer: { type: 'cross' },
  },
  title: {
    text: 'to_be_filled',
    left: 'center'
  },
  legend: {
    top: '3%%',
    data: 'to_be_filled'
  },
  xAxis: {
    type: 'category',
    boundaryGap: false,
    axisPointer: {
      type: 'shadow',
      label: { formatter: 'to_be_filled' }
    },
    axisLabel: {
      formatter: function (val) {
          return Number(val).toLocaleString() + ' cycles';
      }
    },
    data: 'to_be_filled',
  },
  yAxis: [
    {
      type: 'value',
      name: 'percentage',
      axisPointer: { label: {formatter: '{value}%'} },
      axisLabel: { formatter: '{value}%' }
    },
  ],
  series: 'to_be_filled'
};
// end of common part of line chart
"""

    def gen_html_chart_of_util(self, serial_id: int, min_interval_num: int):
        """
        generate util chart script for this stage
        :param serial_id: the serial id of this func/block in an html
        :param min_interval_num: the x-axis must have at least this number of values
        """

        code = "\n\nvar data_%d = [\n" % serial_id

        # determine interval number
        interval_cycles = self.determine_interval_cycles(min_interval_num, True)
        interval_num = math.ceil(self.cycles / interval_cycles)
        interval_names = [interval_cycles * i for i in range(interval_num + 1)]

        # for each category, calculate its values (cycle num) in each interval
        category_interval_values = dict()
        for op_info in self.op_infos:
            if op_info.is_layer_group or op_info.is_misc:
                # NOTE-pseudo-lg-category: do not show layer group in line chart
                continue

            # lambda function to update interval values of category `c`
            def update(c: str, factor: float = 1):
                """
                The increment of each interval is the cycles within this interval.
                The cycles can be enlarged by `factor`
                """

                # decide category name and initialize if newly met
                instances = [c]
                if op_info.instance_ids is not None:  # update for all these instances
                    instances = [f"{c} #{i}" for i in op_info.instance_ids]
                for c2 in instances:
                    if c2 not in category_interval_values:
                        category_interval_values[c2] = [0] * interval_num
                    this_category_interval_values = category_interval_values[c2]

                    # add to corresponding intervals
                    start_interval_index = int(op_info.start // interval_cycles)
                    end_interval_index = math.ceil(op_info.end / interval_cycles)
                    for i in range(start_interval_index, end_interval_index):
                        this_start = max(op_info.start, interval_cycles * i)
                        this_end = min(op_info.end, interval_cycles * (i + 1))
                        assert this_start < this_end

                        increment = int((this_end - this_start) * factor)
                        this_category_interval_values[i] += increment

            # update intervals of cycles
            for category in op_info.categories:
                update(category)

            # update intervals of macs
            if op_info.macs:
                update("mac", op_info.macs / (op_info.cycles * self.mac_per_cycle))

            # update intervals of L1M read/write
            if self.l1m_bytes_per_cycle:
                for rw in ("read", "write"):
                    value = op_info.map.get(f"l1m_{rw}_bytes", 0)
                    if value:
                        factor = value / (op_info.cycles * self.l1m_bytes_per_cycle)
                        update(f"l1m_{rw}", factor)

            # update intervals for conv average
            conv_avg_values = None
            conv_num = 0
            for c, values in category_interval_values.items():
                if c.startswith("conv #"):
                    conv_num += 1
                    if conv_avg_values is None:
                        conv_avg_values = values
                    else:
                        assert len(conv_avg_values) == len(values)
                        conv_avg_values = [sum(x) for x in zip(conv_avg_values, values)]
            if conv_num:
                category_interval_values["conv average"] = [
                    x / conv_num for x in conv_avg_values
                ]

        category_interval_values = dict(sorted(category_interval_values.items()))

        # convert to percentage
        for category, interval_values in category_interval_values.items():
            interval_values = [int(x * 100 / interval_cycles) for x in interval_values]

            s = "{ "
            s += 'name: "%s", ' % to_escaped_str(category)
            s += 'type: "line", '
            s += "smooth: true, "
            # s += 'areaStyle: {}, '
            # s += 'emphasis: {focus: "series"}, '
            # s += 'yAxisIndex: 1, '
            s += "data: " + repr(interval_values)
            s += "}"
            code += "  %s,\n" % s
        code += "];\n"

        code += """
option.title.text = "{stage} Hardware Usage";
option.legend.data = {category_names};
option.xAxis.data = {interval_names};
option.xAxis.axisPointer.label.formatter = function (params) {{
  return Number(params.value).toLocaleString() + ' ~ '
    + (Number(params.value) + {interval_cycles}).toLocaleString() + ' cycles';
}};
option.yAxis[0].name = "usage";
option.series = data_{serial_id};
var dom_{serial_id} = document.getElementById("util_chart_{serial_id}");
var myChart_{serial_id} = echarts.init(dom_{serial_id});
if (option && typeof option === 'object') {{
    myChart_{serial_id}.setOption(option);
}}

    """.format(
            serial_id=serial_id,
            stage=self.stage,
            category_names=repr(list(category_interval_values.keys())),
            interval_names=repr(interval_names),
            interval_cycles=interval_cycles,
        )

        return code

    def gen_html_chart_of_completion(self, serial_id: int, min_interval_num: int):
        """
        generate completion chart script for this stage
        :param serial_id: the serial id of this func/block in an html
        :param min_interval_num: the x-axis must have at least this number of values
        """

        code = "\n\nvar data_%d = [\n" % serial_id

        # determine interval number
        interval_cycles = self.determine_interval_cycles(min_interval_num, True)
        interval_num = math.ceil(self.cycles / interval_cycles)
        interval_names = [interval_cycles * i for i in range(interval_num + 1)]

        # for each category, calculate its values (cycle num) in each interval
        category_interval_values = dict()
        for op_info in self.op_infos:
            if op_info.is_layer_group or op_info.is_misc:
                # NOTE-pseudo-lg-category: do not show layer group in line chart
                continue

            # lambda function to update interval values of category `c`
            def update(c: str, factor: float = 1):
                """
                The increment of each interval is the cycles within this interval.
                The cycles can be enlarged by `factor`
                """

                # decide category name and initialize if newly met
                if c not in category_interval_values:
                    category_interval_values[c] = [0] * interval_num
                this_category_interval_values = category_interval_values[c]

                # add to corresponding intervals
                start_interval_index = int(op_info.start // interval_cycles)
                end_interval_index = math.ceil(op_info.end / interval_cycles)
                for i in range(start_interval_index, end_interval_index):
                    this_start = max(op_info.start, interval_cycles * i)
                    this_end = min(op_info.end, interval_cycles * (i + 1))
                    assert this_start < this_end

                    increment = (this_end - this_start) * factor
                    this_category_interval_values[i] += increment

            # update intervals of cycles
            for category in op_info.categories:
                update(category)

            # update intervals of macs
            if op_info.macs:
                update("mac", op_info.macs / op_info.cycles)

            # update intervals of L1M read/write
            if self.l1m_bytes_per_cycle:
                for rw in ("read", "write"):
                    value = op_info.map.get(f"l1m_{rw}_bytes", 0)
                    if value:
                        factor = value / (op_info.cycles * self.l1m_bytes_per_cycle)
                        update(f"l1m_{rw}", factor)
        category_interval_values = dict(sorted(category_interval_values.items()))

        # convert to accumulated percentage
        for category, interval_values in category_interval_values.items():
            total = sum(interval_values)
            current = 0
            for i, v in enumerate(interval_values):
                current += v
                interval_values[i] = int(current * 100 / total)

            s = "{ "
            s += 'name: "%s", ' % to_escaped_str(category)
            s += 'type: "line", '
            s += "smooth: true, "
            # s += 'areaStyle: {}, '
            # s += 'emphasis: {focus: "series"}, '
            # s += 'yAxisIndex: 1, '
            s += "data: " + repr(interval_values)
            s += "}"
            code += "  %s,\n" % s
        code += "];\n"

        code += """
option.title.text = "{stage} Degree of Completion";
option.legend.data = {category_names};
option.xAxis.data = {interval_names};
option.xAxis.axisPointer.label.formatter = function (params) {{
  return Number(params.value).toLocaleString() + ' ~ '
    + (Number(params.value) + {interval_cycles}).toLocaleString() + ' cycles';
}};
option.yAxis[0].name = "accumulated percentage";
option.series = data_{serial_id};
var dom_{serial_id} = document.getElementById("completion_chart_{serial_id}");
var myChart_{serial_id} = echarts.init(dom_{serial_id});
if (option && typeof option === 'object') {{
    myChart_{serial_id}.setOption(option);
}}

    """.format(
            serial_id=serial_id,
            stage=self.stage,
            category_names=repr(list(category_interval_values.keys())),
            interval_names=repr(interval_names),
            interval_cycles=interval_cycles,
        )

        return code

    @staticmethod
    def gen_html_chart_of_live_range_common() -> str:
        return """
// begin of common part of live range chart
var option = {
  tooltip: {
    trigger: 'axis',
    axisPointer: { type: 'cross' }
  },
  title: {
    text: 'to_be_filled',
    left: 'center'
  },
  legend: {
    top: '3%%',
    data: 'to_be_filled'
  },
  xAxis: {
    type: 'category',
    boundaryGap: false,
    axisPointer: {
      type: 'shadow',
      label: { formatter: 'to_be_filled' }
    },
    axisLabel: {
      formatter: function (val) {
          return Number(val).toLocaleString() + ' cycles';
      }
    },
    data: 'to_be_filled',
  },
  yAxis: [
    {
      type: 'value',
      name: 'Memory Usage',
      axisPointer: { label: {formatter: '{value} MiB'} },
      axisLabel: { formatter: '{value} MiB' }
    },
  ],
  series: 'to_be_filled'
};
// end of common part of live range chart
"""

    def gen_html_chart_of_live_range(self, serial_id: int, min_interval_num: int):
        """
        generate live range chart script for this stage
        :param serial_id: the serial id of this func/block in an html
        :param min_interval_num: the x-axis must have at least this number of values
        """

        if not self.live_ranges:
            return ""

        code = "\n\nvar data_%d = [\n" % serial_id

        category_names = list(self.live_ranges.keys())  # for chart legend
        interval_cycles = (
            self.determine_interval_cycles(min_interval_num, False)
            if min_interval_num
            else 1e10
        )
        interval_num = (
            math.ceil(self.cycles / interval_cycles) if min_interval_num else 1
        )
        interval_names = [interval_cycles * i for i in range(interval_num + 1)]

        for category_id, (category_name, ranges) in enumerate(self.live_ranges.items()):
            escaped_category_name = to_escaped_str(category_name)
            s = "{ "
            s += 'name: "%s", ' % escaped_category_name
            s += 'type: "line", '
            s += (
                'stack: "%s", '
                % escaped_category_name.lower()[: escaped_category_name.find("_")]
            )
            s += "smooth: true, "
            s += "areaStyle: {}, "
            s += 'emphasis: {focus: "series"}, '
            # s += 'yAxisIndex: 1, '

            # find turning points
            x_points = {0}
            x_points = x_points.union(r[2] for r in ranges)
            x_points = x_points.union(r[3] for r in ranges)
            x_points = sorted(list(x_points))

            # find max value in each interval
            max_bytes_in_intervals = [0]
            ranges_by_ascending_start = sorted(
                self.live_ranges[category_name], key=lambda x: x[2]
            )
            ranges_by_ascending_end = sorted(
                ranges_by_ascending_start, key=lambda x: x[3]
            )
            range_num = len(ranges_by_ascending_end)
            old_size = si = ei = 0
            for x_point in x_points:
                interval_id = int(
                    x_point // interval_cycles
                )  # should have enough size (interval_id + 1)
                if len(max_bytes_in_intervals) < interval_id + 1:
                    max_bytes_in_intervals += [old_size] * (
                        interval_id + 1 - len(max_bytes_in_intervals)
                    )
                new_size = old_size
                while si < range_num and ranges_by_ascending_start[si][2] == x_point:
                    new_size += ranges_by_ascending_start[si][1]
                    si += 1
                while ei < range_num and ranges_by_ascending_end[ei][3] == x_point:
                    new_size -= ranges_by_ascending_end[ei][1]
                    ei += 1

                max_bytes_in_intervals[interval_id] = max(
                    max_bytes_in_intervals[interval_id], new_size
                )
                old_size = new_size
            if len(max_bytes_in_intervals) < interval_num:
                max_bytes_in_intervals += [old_size] * (
                    interval_num - len(max_bytes_in_intervals)
                )

            s += "data: [%s]" % ",".join(
                "%.2f" % (x / 1e6) for x in max_bytes_in_intervals
            )
            s += "}"
            code += "  %s,\n" % s
            if min_interval_num == 0:  # called from __init__
                assert category_name not in self.max_mem_usage
                self.max_mem_usage[category_name] = max(max_bytes_in_intervals)
        code += "];\n"

        code += """
option.title.text = "{stage} Memory Usage";
option.legend.data = {category_names};
option.xAxis.data = {interval_names};
option.xAxis.axisPointer.label.formatter = function (params) {{
  return Number(params.value).toLocaleString() + ' ~ '
    + (Number(params.value) + {interval_cycles}).toLocaleString() + ' cycles';
}};
option.series = data_{serial_id};
var dom_{serial_id} = document.getElementById("live_range_chart_{serial_id}");
var myChart_{serial_id} = echarts.init(dom_{serial_id});
if (option && typeof option === 'object') {{
    myChart_{serial_id}.setOption(option);
}}

    """.format(
            serial_id=serial_id,
            stage=self.stage,
            category_names=repr(category_names),
            interval_names=repr(interval_names),
            interval_cycles=interval_cycles,
        )

        return code


class FuncBlockStat:
    """statistics of 1 function/block's all stages"""

    def __init__(self, list_of_ss: List[StageStat]):
        """create FuncBlockStat with its StageStat objects"""

        # parse my StageStat objects
        assert list_of_ss, "FuncBlockStat must have at least 1 StageStat"
        self.stage_stats = tuple(sorted(list_of_ss, key=lambda x: x.stage_id))
        for my_ss in self.stage_stats:
            assert (
                my_ss.parent is None
            ), "StageStat must belong to exactly 1 FuncBlockStat"
            my_ss.parent = self

        def get_common_value(key: str, value_type):
            v = getattr(self.stage_stats[0], key)
            assert isinstance(v, value_type), "self.stage[i].%s is not %s" % (
                key,
                value_type,
            )
            assert all(
                getattr(ss2, key) == v for ss2 in self.stage_stats
            ), "different %s in StageStats: %s" % (
                key,
                [getattr(ss2, key) for ss2 in self.stage_stats],
            )
            return v

        # model-dependent setup
        self.long_title = self.stage_stats[
            0
        ].long_title  # may be different across stages, e.g., tensor v.s. memref
        self.args = self.stage_stats[0].args
        self.args_elements = self.stage_stats[0].args_elements
        self.retvals = self.stage_stats[0].retvals
        self.retvals_elements = self.stage_stats[0].retvals_elements
        self.constant_size = self.stage_stats[0].constant_size
        self.constant_elements = self.stage_stats[0].constant_elements
        self.short_title = get_common_value("short_title", str)
        self.func_name = get_common_value("func_name", str)
        self.is_block = get_common_value("is_block", bool)

        # model-independent setup
        self.hz = get_common_value("hz", int)  # cycle per second
        self.mhz = get_common_value("mhz", int)  # cycle per us
        self.ghz = get_common_value("ghz", float)  # cycle per ns
        self.peak_macs = get_common_value("peak_macs", int)
        self.mac_per_cycle = get_common_value("mac_per_cycle", int)
        self.peak_ops = get_common_value("peak_ops", int)
        self.peak_tops = get_common_value("peak_tops", float)
        self.l1m_bytes_per_cycle = get_common_value("l1m_bytes_per_cycle", int)
        self.l1m_bytes = get_common_value("l1m_bytes", int)
        self.l2m_bytes = get_common_value("l2m_bytes", int)
        self.env_vars = get_common_value("env_vars", dict)
        self.bandwidth = get_common_value("bandwidth", dict)

        # statistics
        self.baseline_macs = self.stage_stats[0].total_macs
        self.baseline_opt_mac_cycles = self.stage_stats[0].opt_mac_cycles
        self.baseline_cycles = (
            self.baseline_opt_mac_cycles
        )  # only mac (conv/matmul/linear)
        self.baseline_fps = (
            self.hz / self.baseline_cycles
        )  # only mac (conv/matmul/linear)
        self.baseline_latency_us = (
            self.baseline_cycles / self.mhz
        )  # only mac (conv/matmul/linear)
        self.baseline_latency_ms = (
            self.baseline_latency_us / 1e3
        )  # only mac (conv/matmul/linear)

        for ss in self.stage_stats:
            real_ops = self.baseline_macs * 2 * ss.fps
            ss.real_tops = real_ops / 1e12
            ss.util = real_ops / ss.peak_ops
            another_util = self.baseline_opt_mac_cycles / ss.cycles
            assert abs(ss.util - another_util) < 0.01, (
                "bad function/block " + self.func_name
            )

        self.final_stat = self.stage_stats[-1]  # may be estimated or actual_insts
        self.stats_by_perf_type = dict()  # categorized by perf type
        for ss1 in self.stage_stats:
            self.stats_by_perf_type.setdefault(ss1.perf_type, []).append(ss1)

        estimated_ss = [x for x in self.stage_stats if x.is_estimation]
        if estimated_ss:
            # has estimated StageStat
            self.final_estimated_stat = estimated_ss[-1]
            estimated_fps = self.final_estimated_stat.fps
            self.score_to_estimation = self.final_stat.fps / estimated_fps
        else:
            # NOTE-codegen-new-bb: (1) such stage_stat should have been skipped in caller; (2) provide only 1 json file
            print(f'"{self.long_title}" has no estimated StageStat')
            self.final_estimated_stat = None
            self.score_to_estimation = 1.0

    def get_us(self, cycles: int) -> float:
        return cycles / self.mhz

    def get_ms(self, cycles: int) -> float:
        return cycles / self.mhz / 1e3

    def __gen_digest_setups_and_values(self) -> List[tuple]:
        """return a list of (name, unit, format, value), len(list) == len(columns)"""
        return [
            ("GMACs", "", "{:.1f}", self.baseline_macs / 1e9),
            # ('dws1 GMACs', '', '{:.2f}', self.final_stat.total_stat.get('tmp_dws1_macs', 0) / 1e9),
            # ('dws2 GMACs', '', '{:.2f}', self.final_stat.total_stat.get('tmp_dws2_macs', 0) / 1e9),
            ("baseline", "ms", "{:.2f}", self.baseline_latency_ms),
            ("MAC", "ms", "{:.2f}", self.get_ms(self.final_stat.real_mac_cycles)),
            # ('dws1', 'ms', '{:.2f}', self.get_ms(self.final_stat.total_stat.get('tmp_dws1_cycles', 0))),
            # ('dws2', 'ms', '{:.2f}', self.get_ms(self.final_stat.total_stat.get('tmp_dws2_cycles', 0))),
            ("to L1M", "ms", "{:.2f}", self.get_ms(self.final_stat.to_l1m_cycles)),
            ("from L1M", "ms", "{:.2f}", self.get_ms(self.final_stat.from_l1m_cycles)),
            ("other", "ms", "{:.2f}", self.get_ms(self.final_stat.other_cycles)),
            ("final", "ms", "{:.2f}", self.final_stat.latency_ms),
            ("util", "%", "{:.0f}", self.final_stat.util * 100),
            ("TOPs", "", "{:.1f}", self.final_stat.real_tops),
            ("in", "M", "{:.2f}", self.args_elements / 1e6),
            ("const", "M", "{:.2f}", self.constant_elements / 1e6),
            ("out", "M", "{:.2f}", self.retvals_elements / 1e6),
            # ('to DDR', 'MB', '{:.1f}', self.final_stat.to_ddr_bytes / 1e6),
            # ('from DDR', 'MB', '{:.1f}', self.final_stat.from_ddr_bytes / 1e6),
            # ('to L2M', 'MB', '{:.1f}', self.final_stat.to_l2m_bytes / 1e6),
            # ('from L2M', 'MB', '{:.1f}', self.final_stat.from_l2m_bytes / 1e6),
            # ('L2M size', 'MB', '{:.1f}', self.final_stat.max_l2m_usage_bytes / 1e6),
            # ('to DDR', 'GB/s', '{:.1f}', self.final_stat.to_ddr_bytes * self.final_stat.fps / 1e9),
            # ('from DDR', 'GB/s', '{:.1f}', self.final_stat.from_ddr_bytes * self.final_stat.fps / 1e9),
            # ('to L2M', 'GB/s', '{:.1f}', self.final_stat.to_l2m_bytes * self.final_stat.fps / 1e9),
            # ('from L2M', 'GB/s', '{:.1f}', self.final_stat.from_l2m_bytes * self.final_stat.fps / 1e9),
            (
                "#stage",
                "",
                "{:s}",
                str(len(self.stage_stats))
                + ("(est)" if self.final_stat.is_estimation else ""),
            ),
            ("score", "", "{:.2f}", self.score_to_estimation),
            ("shape", "", "{:s}", self.args[0].shape_str if self.args else ""),
            ("name", "", "{:s}", self.short_title.strip().split()[0]),
        ]

    def gen_digest_setups(self) -> List[Tuple[str, str, str]]:
        """return a list of (name, unit, format), len(list) == len(columns)"""
        return [x[:-1] for x in self.__gen_digest_setups_and_values()]

    def gen_digest_row(self) -> List:
        """return a list of formats, ret_val[i] is for the i-th column"""
        return [x[-1] for x in self.__gen_digest_setups_and_values()]

    def gen_html_table_of_setup(self) -> str:
        """generate the table of BPU setup and bandwidth"""

        div = ""

        table_setup = [
            ["Metric", "Value"],
            ["frequency", "%.1f GHz" % self.ghz],
            ["peak MAC/s", "%.1f T" % (self.peak_macs / 1e12)],
            ["peak OP/s", "%.1f T" % self.peak_tops],
            ["L1M size", "%.1f MiB" % (self.l1m_bytes / 2**20)],
            ["L2M size", "%.1f MiB" % (self.l2m_bytes / 2**20)],
        ]
        for k, v in self.env_vars.items():
            table_setup.append(['env var "%s"' % k, str(v)])

        s = ", ".join(" ".join(row) for row in table_setup)
        h = hashlib.md5(s.encode()).hexdigest()
        div += '\n<h2 style="text-align: center;"> Setup (hash=%s) </h2>\n' % h
        div += make_html_table(table_setup, False)

        # example: self.bandwidth["l1m_to_l2m_GBps"]["1000"] = 5
        byte_numbers = set(
            itertools.chain.from_iterable(d.keys() for d in self.bandwidth.values())
        )
        byte_numbers = [int(x) for x in byte_numbers]
        byte_numbers.sort()  # list of int
        table_bandwidth = [
            ["category"] + ["%d %sB" % to_kmgtp(x, False) for x in byte_numbers]
        ]
        for bw, d in self.bandwidth.items():
            table_bandwidth.append(
                [bw] + ["%.1f" % d[str(x)] if str(x) in d else "" for x in byte_numbers]
            )

        div += (
            '\n<h2 style="text-align:center;"> Memory Bandwidth (GB/s)</h2>\n'
            + make_html_table(table_bandwidth, False)
        )
        return div

    def gen_html_table_of_stages(self) -> str:
        """generate the table of all stages"""

        # generate title
        title = '\n<h3 style="text-align: center;"> %s stages </h3>\n' % self.long_title

        # collect arg/retval number from all stages
        arg_num = max(len(ss.args) for ss in self.stage_stats)
        retval_num = max(len(ss.retvals) for ss in self.stage_stats)

        # collect keys from all stages (some keys may be absent in some stage)
        optional_keys = set(
            itertools.chain.from_iterable(
                ss.total_stat.keys() for ss in self.stage_stats
            )
        )
        optional_keys = sorted(optional_keys)

        # create the header (1st) column, many of them have tooltip
        first_column = self.stage_stats[0].gen_digest(
            True, arg_num, retval_num, optional_keys
        )
        tooltips = {
            "MAC cycles (optimal)": "mac_num / bpu_peak_mac",
            "MAC cycles (actual)": [
                "actually needs more time because:",
                "&bull; hardware util (depth-wise, small group conv)",
                "&bull; alignment",
                "&bull; redundant computing (for less memory access)",
            ],
            "run cycles (no parallelism)": "run cycles without instruction-level parallelism",
            "run cycles": "actual run cycles at this stage",
            "L2M usage": "may exceed L2M limit in estimated stages",
        }
        for i, name in enumerate(first_column):
            name = name.replace("_", " ")
            if name in tooltips:
                name += make_html_text_with_tooltip("?", tooltips[name])
            first_column[i] = name

        # create the table, each obj (stage) has one column
        table = [first_column] + [
            ss.gen_digest(False, arg_num, retval_num, optional_keys)
            for ss in self.stage_stats
        ]
        table = list(zip(*table))  # transpose

        # process numbers: add comma and percentage
        merge_column = True
        new_table = []
        for row in table:
            new_row = []
            for i, this_v in enumerate(row):
                text = to_str(this_v)  # add comma for number

                # add increase/decrease percentage
                prev_v = row[i - 1] if i >= 2 else None
                if this_v == prev_v and merge_column:
                    # will merge later, so use same text
                    text = new_row[-1]
                elif (
                    isinstance(this_v, int) and isinstance(prev_v, int) and prev_v != 0
                ):
                    div = int((this_v - prev_v) * 100 / prev_v)
                    if div != 0:
                        text += ' <span style="color:%s">(%+d%%)</span>' % (
                            "pink" if div > 0 else "lightgreen",
                            div,
                        )
                new_row.append(text)
            new_table.append(new_row)

        # process run cycles: add tooltip
        opt_row_id = [
            i for i, x in enumerate(table) if x[0].startswith("MAC cycles (optimal)")
        ][0]
        final_cycle_row_id = [
            i
            for i, x in enumerate(table)
            if x[0].startswith("run cycles") and "(" not in x[0]
        ][0]
        final_fps_row_id = [
            i for i, x in enumerate(table) if x[0].startswith("FPS") and "(" not in x[0]
        ][0]
        assert table[opt_row_id][1] == self.baseline_cycles, "%d != %d" % (
            table[opt_row_id][1],
            self.baseline_cycles,
        )
        for row_id in range(opt_row_id, len(table)):
            row = table[row_id]
            if not row[0]:
                break  # process until the next blank line
            if "cycles" in row[0]:
                for col_id, cycles in enumerate(row):
                    if isinstance(cycles, int) and cycles != 0:
                        tooltip = [
                            "%d &#181;s" % int(cycles / self.mhz),
                            "%.1f FPS" % (self.hz / cycles),
                            "%d%% util (relative to baseline)"
                            % int(100 * self.baseline_cycles / cycles),
                        ]
                        new_table[row_id][col_id] = make_html_text_with_tooltip(
                            new_table[row_id][col_id], tooltip, False
                        )
        new_table[opt_row_id][1] = "<b>%s</b> (baseline %s)" % (
            to_str(new_table[opt_row_id][1]),
            make_html_text_with_tooltip("?", "User expects to get this performance"),
        )
        new_table[final_cycle_row_id][-1] = "<b>%s</b> (final %s)" % (
            new_table[final_cycle_row_id][-1],
            make_html_text_with_tooltip("?", "User finally gets this performance"),
        )
        new_table[final_fps_row_id][-1] = "<b>%s</b> (final %s)" % (
            new_table[final_fps_row_id][-1],
            make_html_text_with_tooltip("?", "User finally gets this performance"),
        )

        new_table = [row + [row[0]] for row in new_table]
        return title + make_html_table(new_table, merge_column=merge_column)


class ModelStat:
    def __init__(self, model_name: str, dir_name: str, group_id: int):
        """
        create ModelStat (containing a few FuncBlockStat objects) from json files in the directory
        :param dir_name: must contain a few json files (corresponds to a few stages)
        """

        assert os.path.isdir(dir_name), dir_name + " is not a valid path"

        self.model_name = model_name
        self.dir_name = dir_name  # containing json files
        self.group_id = group_id

        # parse functions and blocks
        identifier_to_ss = dict()
        regex = re.compile(r"tmp_\d+_.+_perf\.json")  # NOTE-perf-json-name
        for filename in os.listdir(dir_name):
            if not regex.fullmatch(filename):
                continue  # not perf json file name

            with open(os.path.join(dir_name, filename), "r") as f:
                a_few_ss_dict = json.load(f)
            if not isinstance(a_few_ss_dict, (list, tuple)):
                print(filename, "is not a perf json, skip")
                continue  # not perf json format

            for ss_dict in a_few_ss_dict:
                ss = StageStat(ss_dict)
                if not ss.is_block:
                    # NOTE-codegen-new-bb: codegen pass may create new blocks, no corresponding original ones. so skip
                    identifier_to_ss.setdefault(ss.identifier(), []).append(ss)

        if len(identifier_to_ss) == 0:
            self.is_valid = False
            self.func_block_stats = tuple()
            return

        self.is_valid = True
        self.func_block_stats = tuple(
            FuncBlockStat(pair[1]) for pair in sorted(identifier_to_ss.items())
        )

        # select representative function
        for fbs in self.func_block_stats:
            if not fbs.is_block:
                self.representative_func_name = fbs.func_name
                self.representative_args = fbs.args
                self.representative_retvals = fbs.retvals
                return
        else:
            self.representative_func_name = ""
            self.representative_args = tuple()
            self.representative_retvals = tuple()

    def gen_html(self):
        """generate html for all functions/blocks in this model, writing to the same directory as json files"""
        assert self.is_valid, "cannot generate html for invalid ModelStat"
        gen_profile_chart = True
        gen_util_chart = True
        gen_completion_chart = True
        gen_lr_chart = False

        # `d` is used to format `html_template`
        d = dict()

        # model-independent part
        with open(os.path.join(os.path.dirname(__file__), "echarts.min.js")) as f:
            d["echart_script"] = f.read()
        d["profile_chart_common_script"] = (
            StageStat.gen_html_chart_of_profile_common() if gen_profile_chart else ""
        )
        d["util_chart_common_script"] = (
            StageStat.gen_html_chart_of_line_common() if gen_util_chart else ""
        )
        d["completion_chart_common_script"] = (
            StageStat.gen_html_chart_of_line_common() if gen_completion_chart else ""
        )
        d["lr_chart_common_script"] = (
            StageStat.gen_html_chart_of_live_range_common() if gen_lr_chart else ""
        )

        # model-dependent part
        d["divs"] = '\n<h1 style="text-align: center;"> %s (%s) </h1>\n' % (
            self.model_name,
            self.dir_name,
        )
        d["divs"] += (
            "\n<div> %s </div>\n" % self.func_block_stats[0].gen_html_table_of_setup()
        )  # same for funcs
        d["profile_chart_all_ss_script"] = ""
        d["util_chart_all_ss_script"] = ""
        d["completion_chart_all_ss_script"] = ""
        d["lr_chart_all_ss_script"] = ""
        serial_id = 0
        for fbs in self.func_block_stats:
            d["divs"] += "\n        <div> %s </div>" % fbs.gen_html_table_of_stages()

            # for each stage stat, gen table or charts
            for ss in fbs.stage_stats:
                # gen table, no charts
                d["divs"] += "\n        <div> %s </div>" % (
                    ss.gen_html_table_of_op_infos()
                )

                # gen charts
                serial_id += 1
                if gen_profile_chart:
                    d["divs"] += (
                        '\n        <div id="profile_chart_%d" style="height: 60%%;"> </div>'
                        % serial_id
                    )
                    d["profile_chart_all_ss_script"] += ss.gen_html_chart_of_profile(
                        serial_id
                    )

                if gen_util_chart:
                    d["divs"] += (
                        '\n        <div id="util_chart_%d" style="height: 60%%;"> </div>'
                        % serial_id
                    )
                    d["util_chart_all_ss_script"] += ss.gen_html_chart_of_util(
                        serial_id, 100
                    )

                if gen_completion_chart:
                    d["divs"] += (
                        '\n        <div id="completion_chart_%d" style="height: 60%%;"> </div>'
                        % serial_id
                    )
                    d[
                        "completion_chart_all_ss_script"
                    ] += ss.gen_html_chart_of_completion(serial_id, 100)

                if gen_lr_chart:
                    d["divs"] += (
                        '\n        <div id="live_range_chart_%d" style="height: 60%%;"> </div>'
                        % serial_id
                    )
                    d["lr_chart_all_ss_script"] += ss.gen_html_chart_of_live_range(
                        serial_id, 100
                    )

        html_template = """
    <!DOCTYPE html>
    <html style="height: 100%%">
        <head>
            <meta charset="utf-8">
            <style>
            table, th, td {
                border: 1px solid black;
                border-collapse: collapse;
                padding: 2px;
                vertical-align: center;
                text-align: center;
                margin-left: auto;
                margin-right: auto;
            }

            .hbdktooltip {
              background-color: lightyellow;
              text-align: center;
              border-radius: 6px;
              padding: 0 4px;
              border: 1px;
              border-style: solid;
              border-color: silver;
            }
            </style>
        </head>
        <body style="height: 100%%; margin: 0">
    %(divs)s

            <script type="text/javascript">
            %(echart_script)s
            </script>

            <script type="text/javascript">
            %(profile_chart_common_script)s
            %(profile_chart_all_ss_script)s
            </script>

            <script type="text/javascript">
            %(util_chart_common_script)s
            %(util_chart_all_ss_script)s
            </script>

            <script type="text/javascript">
            %(completion_chart_common_script)s
            %(completion_chart_all_ss_script)s
            </script>

            <script type="text/javascript">
            %(lr_chart_common_script)s
            %(lr_chart_all_ss_script)s
            </script>

        </body>
    </html>
    """

        html_file_name = os.path.join(self.dir_name, self.model_name + ".html")
        if os.path.exists(html_file_name):
            print("Perf html generated to %s (overwriting)" % html_file_name)
        with open(html_file_name, "w") as f:
            f.write(html_template % d)

    def gen_digest_setups(self) -> List[Tuple[str, str, str]]:
        """return a list of (name, unit, format), len(list) == len(columns)"""
        assert self.is_valid, "only valid model should call this"
        setups = self.func_block_stats[0].gen_digest_setups()
        return setups + [("gid", "", "{:.0f}"), ("dir name", "", "{:s}")]

    def gen_digest_rows(self, column_num: int) -> List[List]:
        """return len(func_block) rows, rows[i][j] is for i-th func/block, j-th column"""
        if self.is_valid:
            rows = [fbs.gen_digest_row() for fbs in self.func_block_stats]
        else:
            rows = [[None] * (column_num - 2)]
        rows = [x + [self.group_id, self.dir_name] for x in rows]
        assert all(len(row) == column_num for row in rows)
        return rows


class ModelZooStat:
    def __init__(self, test_groups: List[List[str]]):
        """create ModelStat for all groups, each group contains a few models (dirs)"""

        model_stats = []
        for group_id, tests_dirs in enumerate(test_groups):
            tests_dirs = sorted(tests_dirs)
            for test_dir in tests_dirs:
                if not os.path.isdir(test_dir):
                    print('cannot find test case "%s"' % test_dir)
                    continue

                model_name = os.path.basename(
                    os.path.normpath(test_dir)
                )  # last dir name as model name
                if model_name == ".":
                    model_name = "unnamed"
                model_name = "model_" + model_name
                model_stat = ModelStat(model_name, test_dir, group_id)
                model_stats.append(
                    model_stat
                )  # invalid (failed) models are still collected

        self.group_num = len(test_groups)
        self.valid_model_num = len([x for x in model_stats if x.is_valid])
        self.invalid_model_num = len(model_stats) - self.valid_model_num
        self.has_valid_model = self.valid_model_num > 0
        self.model_stats = tuple(model_stats)  # default order, in case any model failed

        self.env_vars: Dict[str, str] = dict()
        for model_stat in self.model_stats:
            if not model_stat.is_valid:
                continue
            for fbs in model_stat.func_block_stats:
                if not self.env_vars:
                    self.env_vars = fbs.env_vars
                else:
                    assert (
                        self.env_vars == fbs.env_vars
                    ), "inconsistent env vars in model zoo"

    def gen_htmls(self):
        """generate html for all models, writing to the same subdirectory as json files"""
        for ms in self.model_stats:
            if ms.is_valid:
                ms.gen_html()

    def gen_tsv(self):
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tsv_name = "tmp_%s_%d_models.tsv" % (time_str, len(self.model_stats))
        with open(tsv_name, "w") as fp:
            fp.write("\n".join("\t".join(row) for row in self.gen_digest(True)))

        print("Perf digest generated to", tsv_name)

    def gen_digest_setups(self) -> List[tuple]:
        """return a list of (name, unit, format), len(list) == len(columns)"""
        assert self.has_valid_model
        for ms in self.model_stats:
            if ms.is_valid:
                return ms.gen_digest_setups()
        assert False, "must contain at least 1 valid model"

    def gen_digest_for_one_group(self, group_id: int) -> List[List]:
        """return rows, rows[i][j] is for a func/block, j-th column"""
        column_names = [x[0] for x in self.gen_digest_setups()]

        # collect rows in this group
        rows = []
        for ms in self.model_stats:
            if ms.group_id == group_id:
                rows += ms.gen_digest_rows(len(column_names))

        # add summary row
        columns = [list(x) for x in zip(*rows)]
        for column_id, column in enumerate(columns):
            numbers = [cell for cell in column if isinstance(cell, (int, float))]
            strings = [cell for cell in column if isinstance(cell, str)]

            if numbers:  # geomean value
                column.append(
                    functools.reduce(lambda x, y: x * y, numbers, 1)
                    ** (1.0 / len(numbers))
                )
            elif column_names[column_id] == "shape":  # 'geomean' text
                column.append("geomean")
            elif strings and "name" in column_names[column_id]:  # common name
                column.append(get_common_prefix(strings) + "*")
            else:
                column.append(None)

        return [list(x) for x in zip(*columns)] + [[None] * len(column_names)]

    def gen_digest(self, header_with_unit: bool) -> List[List]:
        """return rows, rows[i][j] is for a func/block, j-th column"""
        setups = self.gen_digest_setups()
        header_row = [x[0] for x in setups]
        column_units = [x[1] for x in setups]
        column_formats = [x[2] for x in setups]

        if header_with_unit:
            for i, header in enumerate(header_row):
                if column_units[i]:
                    header_row[i] += " " + column_units[i]

        rows = []
        for group_id in range(self.group_num):
            rows += self.gen_digest_for_one_group(group_id)

        for row in rows:
            for column_id, cell in enumerate(row):
                if cell is None:
                    cell = ""
                else:
                    cell = column_formats[column_id].format(cell)
                    if not header_with_unit:
                        cell += column_units[column_id]
                row[column_id] = cell

        return [header_row] + rows

    def print_digest_table(self):
        assert self.has_valid_model
        rows = self.gen_digest(False)
        if len(rows) > 10:
            rows.append(rows[0])  # header line also at the bottom

        # make same width for each column
        columns = list(zip(*rows))
        for col_id, column in enumerate(columns):
            if col_id == len(columns) - 1:
                break  # do not pad spaces for the last row

            width = max(len(x) for x in column)
            if "name" in column[0]:  # this row is xxx_name, left-align
                new_column = [x + " " * (width - len(x)) for x in column]
            else:
                new_column = [" " * (width - len(x)) + x for x in column]
            columns[col_id] = new_column

        # print out
        rows = list(zip(*columns))
        print("\n".join("|".join(row) for row in rows), "\n")


def sub_command_perf(subparsers):
    """as a sub-command of `hbdk-view` tool"""

    parser = subparsers.add_parser("perf", help="convert model perf json to html")
    parser.add_argument(
        "directories",
        nargs="+",
        type=str,
        help="directories containing perf json files",
    )

    def runner(args):
        mzs = ModelZooStat([args.directories])  # all as 1 group
        if mzs.has_valid_model:
            mzs.print_digest_table()
            mzs.gen_htmls()
            mzs.gen_tsv()

            message = "%d passed" % mzs.valid_model_num
            if mzs.invalid_model_num:
                message += ", %d failed" % mzs.invalid_model_num
            print(message)
        else:
            print("Cannot find any valid model perf result in provided path(s)")

    parser.set_defaults(func=runner)


if __name__ == "__main__":
    print("this script should not be invoked")
