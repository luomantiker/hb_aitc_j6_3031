from __future__ import annotations
import json
import argparse
from tabulate import tabulate
from enum import Enum

min_iou = 0.9


class PerfSourceType(Enum):
    REPLAY_JSON = "Replay"
    ESTIMATION_JSON = "Estimation"


def get_iou(op_range1, op_range2):
    assert len(op_range1) == 2
    assert len(op_range2) == 2
    i = min(op_range1[1], op_range2[1]) - max(op_range1[0], op_range2[0])
    u = max(op_range1[1], op_range2[1]) - min(op_range1[0], op_range2[0])
    return i / u


class PerfInfo:
    def init_replay(self, input_file):
        with open(input_file) as f:
            replay_json = json.load(f)
        for layer_group in replay_json:
            op_range = (layer_group["begin layer id"], layer_group["end layer id"])
            cycle = layer_group["statistics"]["run cycles"]
            self.layer_group_cycles[op_range] = cycle

    def init_estimation(self, input_file):
        with open(input_file) as f:
            est_json = json.load(f)
        for block in est_json:
            for layer_group in block:
                op_range = (layer_group["begin_op_id"], layer_group["end_op_id"])
                cycle = layer_group["cycles"]
                self.layer_group_cycles[op_range] = cycle

    def __init__(self, input_file: str, source_type: PerfSourceType):
        self.layer_group_cycles = {}
        self.source_type = source_type

        if source_type == PerfSourceType.REPLAY_JSON:
            self.init_replay(input_file)

        elif source_type == PerfSourceType.ESTIMATION_JSON:
            self.init_estimation(input_file)

        else:
            print(f"PerfInfo: source_type {source_type} invalid")
            exit(-1)

        self.layer_groups = sorted(self.layer_group_cycles.keys())
        self.total_cycle = sum(
            self.layer_group_cycles[op_range] for op_range in self.layer_groups
        )

    def compare(self, ref: PerfInfo):
        table_header = [
            f"{self.source_type.value}\nOp Range",
            f"{self.source_type.value}\nLayer Groups",
            f"{self.source_type.value}\nCycles",
            f"{self.source_type.value}\nProportion",
            f"{ref.source_type.value}\nOp Range",
            f"{ref.source_type.value}\nLayer Groups",
            f"{ref.source_type.value}\nCycles",
            f"{ref.source_type.value}\nProportion",
            "Diff",
        ]
        table_data = []
        # layer group indexes of searching, each point to a layer group in layer_groups
        # [[self_l, self_r], [ref_l, ref_r]]
        index = [[0, 0], [0, 0]]
        self_layer_group_cnt = len(self.layer_groups)
        ref_layer_group_cnt = len(ref.layer_groups)

        while index[0][1] < self_layer_group_cnt and index[1][1] < ref_layer_group_cnt:
            self_range = (
                self.layer_groups[index[0][0]][0],
                self.layer_groups[index[0][1]][1],
            )
            ref_range = (
                ref.layer_groups[index[1][0]][0],
                ref.layer_groups[index[1][1]][1],
            )
            if get_iou(self_range, ref_range) >= min_iou:
                cycle = sum(
                    self.layer_group_cycles[self.layer_groups[idx]]
                    for idx in range(index[0][0], index[0][1] + 1)
                )
                ref_cycle = sum(
                    ref.layer_group_cycles[ref.layer_groups[idx]]
                    for idx in range(index[1][0], index[1][1] + 1)
                )
                table_row = [
                    list(self_range),
                    [
                        list(self.layer_groups[idx])
                        for idx in range(index[0][0], index[0][1] + 1)
                    ],
                    cycle,
                    f"{cycle/self.total_cycle:.02%}",
                    list(ref_range),
                    [
                        list(ref.layer_groups[idx])
                        for idx in range(index[1][0], index[1][1] + 1)
                    ],
                    ref_cycle,
                    f"{ref_cycle/ref.total_cycle:.02%}",
                    f"{(cycle-ref_cycle)/ref_cycle:+.2%}",
                ]
                table_data.append(table_row)
                index[0][0] = index[0][1] = index[0][1] + 1
                index[1][0] = index[1][1] = index[1][1] + 1
            elif self.layer_groups[index[0][1]][1] > ref.layer_groups[index[1][1]][1]:
                index[1][1] += 1
            else:
                index[0][1] += 1

        print(
            tabulate(
                table_data,
                headers=table_header,
                colalign=["center"] * len(table_header),
                tablefmt="grid",
                maxcolwidths=25,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "replay_path",
        help="path to replay json, located in codegen_replay_* log directory",
    )
    parser.add_argument(
        "estimation_path",
        help="path to estimation json, generated by extract_layer_group_from_estimation.py",
    )
    parser.add_argument(
        "--min_iou",
        type=float,
        default=0.9,
        help="minimum IOU (Intersection over Union) value that can be tolerate when comparing layer groups, default to 0.9",
    )
    args = parser.parse_args()
    print(args)
    min_iou = args.min_iou
    replay_info = PerfInfo(args.replay_path, PerfSourceType.REPLAY_JSON)
    estimation_info = PerfInfo(args.estimation_path, PerfSourceType.ESTIMATION_JSON)
    replay_info.compare(estimation_info)
