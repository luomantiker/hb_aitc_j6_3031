import torch
from hbdk4.compiler import ir as mlir

from hbdk4.compiler.dialects import hbir


def convert_dtype(dtype) -> mlir.Type:
    if dtype == torch.float16:
        return mlir.F16Type.get()
    if dtype == torch.float32:
        return mlir.F32Type.get()
    if dtype == torch.float64:
        return mlir.F64Type.get()
    if dtype == torch.float16:
        return mlir.F16Type.get()

    if dtype == torch.int8:
        return mlir.IntegerType.get_signed(8)
    if dtype == torch.int16:
        return mlir.IntegerType.get_signed(16)
    if dtype == torch.int32:
        return mlir.IntegerType.get_signed(32)
    if dtype == torch.int64:
        return mlir.IntegerType.get_signed(64)

    if dtype == torch.uint8:
        return mlir.IntegerType.get_unsigned(8)
    if dtype == torch.bool:
        return hbir.Bool8Type.get()

    # error msg: 'torch' has no attribute 'uint16'/'uint32'/'uint64'
    # if dtype == torch.uint16:
    #     return mlir.IntegerType.get_unsigned(16)
    # if dtype == torch.uint32:
    #     return mlir.IntegerType.get_unsigned(32)
    # if dtype == torch.uint64:
    #     return mlir.IntegerType.get_unsigned(64)

    raise ValueError("cannot emit mlir element type for type {}".format(dtype))
