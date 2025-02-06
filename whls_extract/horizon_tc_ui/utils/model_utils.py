# Copyright (c) 2022 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

from enum import Enum


def list_to_str(origin_list):
    if not origin_list:
        return ""
    res_string = ""
    for item in origin_list:
        res_string += str(item)
        res_string += ";"
    return res_string


class InputDataType(Enum):
    UNDEFINED = 0
    S8 = 1
    U8 = 2
    S32 = 3
    U32 = 4
    F32 = 5
    Gray = 6
    NV12 = 7
    YUV444 = 8
    BGR = 9
    RGB = 10
    BGRP = 11
    RGBP = 12
    NV12_SEPARATE = 13
    S64 = 14
    U64 = 15
    F64 = 16
    S16 = 17
    U16 = 18
    F16 = 19
    S4 = 20
    U4 = 21


class DataType(Enum):
    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    FLOAT16 = 9
    DOUBLE = 10
    UINT32 = 11
    UINT64 = 12
    BFLOAT16 = 13
    BOOL = 14


tensor_datatype_to_input_datatype = {
    DataType.INT8: InputDataType.S8,
    DataType.UINT8: InputDataType.U8,
    DataType.UINT16: InputDataType.U16,
    DataType.INT16: InputDataType.S16,
    DataType.FLOAT16: InputDataType.F16,
    DataType.INT32: InputDataType.S32,
    DataType.UINT32: InputDataType.U32,
    DataType.FLOAT: InputDataType.F32,
    DataType.INT64: InputDataType.S64,
    DataType.UINT64: InputDataType.U64,
    DataType.DOUBLE: InputDataType.F64
}
