from typing import List
from hbdk4.compiler._mlir_libs import _hbdk as _hbdk_cext
from hbdk4.compiler import ir as mlir, hbtl
from hbdk4.compiler.dialects._ods_common import get_default_loc_context
import math
import numpy as np
import os


def deinit_llvm():
    _hbdk_cext.deinit_llvm()


def register_dialects(ctx: mlir.Context):
    return _hbdk_cext.register_dialects(ctx)


def register_llvm_translations(ctx: mlir.Context):
    return _hbdk_cext.register_llvm_translations(ctx)


def parse_enum(e, e_name=""):
    return _hbdk_cext.parse_enum(e, e_name, get_default_loc_context())


def is_dynamic_dim(value, dim):
    return _hbdk_cext.is_dynamic_dim(value, dim)


def has_dynamic_dim(value):
    return _hbdk_cext.has_dynamic_dim(value)


def infer_shape(op):
    _hbdk_cext.infer_shape(op)


def eval_conv_util(
    march: str,
    fout_shape: List[int],
    weight_shape: List[int],
    stride: int,
    pad: int,
    fin_bit: int,
    fout_bit: int,
    sumin_bit: int,
):
    _hbdk_cext.eval_conv_util(
        march, fout_shape, weight_shape, stride, pad, fin_bit, fout_bit, sumin_bit
    )


def trans_layout_util(
    march: str,
    shape: List[int],
    src: str,
    dst: str,
    bit: int,
):
    _hbdk_cext.trans_layout_util(march, shape, src, dst, bit)


def average_hamming_distance(
    values: List[int],
    bit_width: int,
) -> float:
    return _hbdk_cext.average_hamming_distance(values, bit_width)


def compress_ratio(
    march: str,
    comp: str,
):
    # March.b25. block size, channel num, channel byte size, enXor, bus alignment, ddr alignment
    params = [256, 32, 8, False, 16, 16]

    if march == "b30":
        params = [256, 16, 16, True, 32, 64]
    elif march == "b30g":
        params = [128, 8, 16, True, 32, 64]
    elif march == "b30p":
        params = [128, 8, 16, True, 32, 64]

    file_list = [comp]
    dir_path = ""
    if os.path.isdir(comp):
        print(
            f"\nTo read all the given binary files in the folder {comp}. Then view each binary data as 'int8' 'int16' and 'int32' respectively, and calculate its compress ratio....."
        )
        file_list = os.listdir(comp)
        dir_path = comp + "/"
    else:
        print(
            "\nView the given binary data as 'int8' 'int16' and 'int32' respectively, and calculate its compress ratio....."
        )

    def calculate_entropy(data):
        unique_values, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        entropy = np.sum(probabilities * np.log2(probabilities))

        if abs(entropy) < 1e-10:
            return entropy

        return -entropy

    def calculate_compress_ratio(file_name):
        original_data = np.fromfile(dir_path + file_name, dtype=np.int8)
        value_len = len(original_data)
        pad_len = params[0] - value_len % params[0]

        data_int8 = np.pad(
            original_data, (0, pad_len), mode="constant", constant_values=0
        ).astype(np.int8)
        data_int16 = data_int8.view(np.int16)
        data_int32 = data_int8.view(np.int32)

        # Packet
        total_bytes = 0
        for lane in range(0, 4):
            result = hbtl.ops.b30.CompressPacketLane(
                data_int8, lane, 0, 4, params[0], params[1], params[2], params[3]
            )
            total_bytes = (
                total_bytes + math.ceil(result[0] / params[4]) * params[4] + params[5]
            )
        packet_ratio = total_bytes / len(data_int8)

        # Rle
        length_int8, out = hbtl.ops.b25.RunLengthEncode(data_int8)
        length_int16, out = hbtl.ops.b25.RunLengthEncode(data_int16)
        rle_ratio_int8 = length_int8[0] * 2 / len(data_int8)
        rle_ratio_int16 = length_int16[0] * 2 / len(data_int16)

        # Entropy
        entropy_int8 = calculate_entropy(data_int8)
        entropy_int16 = calculate_entropy(data_int16)
        entropy_int32 = calculate_entropy(data_int32)
        entropy_ratio = [entropy_int8 / 8, entropy_int16 / 16, entropy_int32 / 32]

        # Avg Hamming Distance
        distance = [0, 0, 0]
        distance[0] = average_hamming_distance(data_int8.tolist(), 8)
        distance[1] = average_hamming_distance(data_int16.tolist(), 16)
        distance[2] = average_hamming_distance(data_int32.tolist(), 32)

        cur_file_name = file_name
        if len(file_name) > 18:
            cur_file_name = "..." + file_name[-15:]

        print(
            f"{cur_file_name:>18} {len(data_int8):>13} {packet_ratio:>13.2%} {rle_ratio_int8:>13.2%} {rle_ratio_int16:>13.2%} {entropy_ratio[0]:>13.2%} {entropy_ratio[1]:>13.2%} {entropy_ratio[2]:>13.2%} {entropy_int8:>13.2f} {entropy_int16:>13.2f} {entropy_int32:>13.2f} {distance[0]:>13.2f} {distance[1]:>13.2f} {distance[2]:>13.2f}"
        )

    print("\nCompress ratio = (Compressed bytes / Original bytes).")
    print(
        "Compressed bytes: The space need after the compress algorithm. e.g. RLE compress bytes is all the pairs (count, data) occupied space."
    )
    print(
        "Info ratio = (Entropy / Bits num of one data). Which represent the proportion of information in one data. e.g. If we view the input data as type int8, after calculation the entropy is 2.8, then the Info ratio is 2.8/8 = 35%(0.35)."
    )
    print("Entropy(H): The entropy of the given data.")
    print(
        "Average Hamming distance(AvgHD): Is calculate by (the Hamming distance of each pair of data which next to each other) / (total data size - 1).\n"
    )
    print(
        f"{'File Name':>18} {'Origin Bytes':>13} {'Packet:int8':>13} {'RLE:int8':>13} {'RLE:int16':>13} {'Info:int8':>13} {'Info:int16':>13} {'Info:int32':>13} {'H:int8':>13} {'H:int16':>13} {'H:int32':>13} {'AvgHD:int8':>13} {'AvgHD:int16':>13} {'AvgHD:int32':>13}"
    )
    print("-" * 200)
    for name in file_list:
        calculate_compress_ratio(name)
    print("\n")
