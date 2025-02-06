# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
import torch

try:
    import tensorrt as trt
except ImportError:
    trt = None


np2torch_dtype_dict = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}

onnx2torch_dtype_dict = {
    "tensor(bool)": torch.bool,
    "tensor(uint8)": torch.uint8,
    "tensor(int8)": torch.int8,
    "tensor(int16)": torch.int16,
    "tensor(int32)": torch.int32,
    "tensor(int64)": torch.int64,
    "tensor(float16)": torch.float16,
    "tensor(bfloat16)": torch.bfloat16,
    "tensor(float)": torch.float32,
    "tensor(float64)": torch.float64,
}

if trt is not None:
    trt2torch_dtype_dict = {
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32,
    }
    # types belowed are supportted after version 10.
    if hasattr(trt, "int64"):
        trt2torch_dtype_dict[trt.int64] = torch.int64
    if hasattr(trt, "bfloat16"):
        trt2torch_dtype_dict[trt.bfloat16] = torch.bfloat16
else:
    trt2torch_dtype_dict = {}
