import warnings
from typing import List
from hbdk4.compiler import ir as mlir
from hbdk4.compiler import hbtl
import numpy as np


class Dtype:
    def __init__(self, dtype_str: str):
        assert isinstance(
            dtype_str, str
        ), "The arg dtype should be str, but got {}".format(type(str))
        self.is_signed = False
        self.is_unsigned = False
        self.is_float = False
        self.is_bool = False
        self.bit_width = 0
        self.dtype_str = dtype_str

        # preprocess torch prefix
        if self.dtype_str.startswith("torch."):
            self.dtype_str = self.dtype_str[len("torch.") :]

        if self.dtype_str in ["bool", "torch.bool"]:
            self.is_bool = True
            self.bit_width = 8
            return

        for prefix in ["int", "si"]:
            if self.dtype_str.startswith(prefix):
                index = len(prefix)
                digit = self.dtype_str[index:]
                assert digit in [
                    "8",
                    "16",
                    "32",
                    "64",
                ], "Invalid element type {}".format(self.element_type)
                self.is_signed = True
                self.bit_width = int(digit)
                return
        for prefix in ["uint", "ui"]:
            if self.dtype_str.startswith(prefix):
                index = len(prefix)
                digit = self.dtype_str[index:]
                assert digit in [
                    "8",
                    "16",
                    "32",
                    "64",
                ], "Invalid element type {}".format(self.element_type)
                self.is_unsigned = True
                self.bit_width = int(digit)
                return
        for prefix in ["float", "f"]:
            if self.dtype_str.startswith(prefix):
                index = len(prefix)
                digit = self.dtype_str[index:]
                assert digit in ["16", "32", "64"], "Invalid Element type {}".format(
                    self.dtype_str
                )
                self.is_float = True
                self.bit_width = int(digit)
                return
        assert False, "Invalid dtype type {}".format(self.dtype)

    def __str__(self) -> str:
        if self.is_signed:
            return "int" + str(self.bit_width)
        elif self.is_unsigned:
            return "uint" + str(self.bit_width)
        elif self.is_float:
            return "float" + str(self.bit_width)
        elif self.is_bool:
            return "bool"
        else:
            assert False, f"Invalid element type {self.dtype_str}"

    @property
    def mlir(self) -> mlir.Type:
        if self.is_signed:
            return mlir.IntegerType.get_signed(self.bit_width)
        elif self.is_unsigned:
            return mlir.IntegerType.get_unsigned(self.bit_width)
        elif self.is_float:
            float_constructor = {
                16: mlir.F16Type.get,
                32: mlir.F32Type.get,
            }
            assert (
                self.bit_width in float_constructor
            ), f"Unknown float type of bit_width {self.bit_width}"

            return float_constructor[self.bit_width]()
        # TODO： mlir do not support bool type now
        else:
            assert False, "Cannot get valid mlir element type from {}".format(
                self.dtype_str
            )

    @property
    def numpy(self) -> np.dtype:
        return np.dtype(str(self))

    @property
    def torch(self) -> "torch.dtype":
        import torch

        return getattr(torch, str(self))


class TensorType:
    def __init__(self, shape: List[int], dtype: Dtype):
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"shape should be list or tuple, but it is {type(shape)}")

        if not isinstance(dtype, Dtype):
            raise TypeError(
                "The arg dtype should be Dtype, but it is {}".format(type(shape))
            )

        if len(shape) > 0 and not all(isinstance(x, (int)) for x in shape):
            raise ValueError("shape must be list of ints")

        self.shape = shape
        self.dtype = dtype

    def __str__(self):
        shape_str = "x".join([str(s) for s in self.shape])
        return f"tensor<{shape_str}x{self.dtype.mlir}>"

    @classmethod
    def from_mlir_value(cls, mlir_value):
        assert isinstance(
            mlir_value, mlir.Value
        ), "The arg mlir_value should be a mlir.Value, but it is {}".format(
            type(mlir_value)
        )
        shaped_type = mlir.ShapedType(mlir_value.type)
        return cls(shaped_type.shape, Dtype(str(shaped_type.element_type)))

    @classmethod
    def from_hbtl_tensor_type(cls, var):
        return cls(var.shape, Dtype(str(var.dtype)))

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def hbtl_tensor_type(self):
        ele_type = hbtl.ElementType.from_format(self.dtype.numpy.char)
        return hbtl.Variable.tensor_type(self.shape, ele_type)

    @property
    def mlir_tensor_type(self):
        return mlir.RankedTensorType.get(self.shape, self.dtype.mlir)

    def __hash__(self):
        return hash((self.shape, str(self.dtype)))

    def __eq__(self, other):
        return self.shape == other.shape and str(self.dtype) == str(other.dtype)
