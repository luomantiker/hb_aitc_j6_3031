import argparse
import json
import re


def extract_layer_group_from_estimation(estimation_json_path, layer_group_json_path):
    details = None
    try:
        with open(estimation_json_path, "r") as file:
            details = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        exit()

    map_from_position_to_space_id = {"L1M": 1, "L2M": 2, "DDR": 3}
    layer_groups = []
    valid_op_cnt = 0
    for block_details in details:
        block_layer_groups = []
        for op_detail in block_details["details"]:
            if op_detail["op_name"] != "layer_group_size":
                continue

            # Parse layer group id(begin and end)
            begin_op_id = valid_op_cnt
            end_op_id = begin_op_id + op_detail["layer_group_valid_size"] - 1
            valid_op_cnt += op_detail["layer_group_valid_size"]

            # Parse layer group space ids(input and output
            input_space_ids = []
            output_space_ids = []
            is_supported = True
            for key, value in op_detail.items():
                if key.startswith("input_") or key.startswith("output_"):
                    position = value.split("positions=")[1]
                    if "/" in position:  # Value in two spaces is not supported
                        space_id = None
                    else:
                        space_id = map_from_position_to_space_id[position]
                    if key.startswith("input_"):
                        input_space_ids.append(space_id)
                    else:
                        output_space_ids.append(space_id)

            if is_supported:
                # Parse layer group cycles
                cycles = op_detail["cycles"]
                layer_group = {
                    "begin_op_id": begin_op_id,
                    "end_op_id": end_op_id,
                    "input_space_ids": input_space_ids,
                    "output_space_ids": output_space_ids,
                    "cycles": cycles,
                }
                block_layer_groups.append(layer_group)
        layer_groups.append(block_layer_groups)

    # Write layer groups out
    with open(layer_group_json_path, "w") as file:
        json.dump(layer_groups, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "estimation_json_path", help="path of estimation perf result(json)"
    )
    parser.add_argument(
        "layer_group_json_path", help="path of layer group result(json)"
    )
    args = parser.parse_args()

    extract_layer_group_from_estimation(
        args.estimation_json_path, args.layer_group_json_path
    )
