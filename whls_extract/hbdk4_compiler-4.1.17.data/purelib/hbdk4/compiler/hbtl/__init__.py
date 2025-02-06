from typing import Iterable

import numpy as np
from hbdk4.compiler._mlir_libs._hbdk import (
    Variable,
    ElementType,
    get_matched_schema,
    infer_result_types,
    invoke_kernel,
    init_cvt,
)

# initialize cvtTools
init_cvt()


def is_torch_dtype(x) -> bool:
    """check if x is torch.dtype without `import torch`

    Args:
        x: torch.dtype or others

    Returns:
        bool: True if x is torch.dtype
    """
    return (
        hasattr(x, "__module__")
        and x.__module__ == "torch"
        and type(x).__name__ == "dtype"
    )


def is_np_dtype(x) -> bool:
    """check if x is np.dtype

    Args:
        x: np.dtype or others

    Returns:
        bool: True if x is np.dtype
    """
    return isinstance(x, np.dtype)


def is_torch_tensor(x) -> bool:
    """check if x is torch.Tensor without `import torch`

    Args:
        x: torch.Tensor or others

    Returns:
        bool: True if x is torch.Tensor
    """
    return (
        hasattr(x, "__module__")
        and x.__module__ == "torch"
        and type(x).__name__ == "Tensor"
    )


def is_np_array(x) -> bool:
    """check if x is np.array

    Args:
        x: np.array or others

    Returns:
        bool: True if x is np.array
    """
    return isinstance(x, np.ndarray)


def is_tensor_like(x) -> bool:
    """check f x is a tensor like object. Following constraints are quired:
    1. class member of `shape` and `dtype`
    2. x.dtype should be torch.dtype or np.dtype
    3. x.shape should be list of ints

    Args:
        x : torch.Tensor, np.array or others

    Returns:
        bool: True if x is a tensor like object
    """
    if hasattr(x, "dtype") and hasattr(x, "shape"):
        is_valid_dtype = is_np_dtype(x.dtype) or is_torch_dtype(x.dtype)
        is_valid_shape = isinstance(x.shape, Iterable) and isinstance(x.shape[0], int)
        return is_valid_dtype and is_valid_shape
    return False


def is_tensor(x) -> bool:
    return is_torch_tensor(x) or is_np_array(x) or is_tensor_like(x)


def get_dtype(dtype) -> ElementType:

    if is_torch_dtype(dtype):
        return ElementType.from_torch(str(dtype))
    elif is_np_dtype(dtype):
        return ElementType.from_numpy(dtype.char)
    return None


def get_tensor(x, strict=False) -> Variable:

    if is_np_array(x):
        return Variable.from_numpy(x)

    elif is_torch_tensor(x):
        dtype = ElementType.from_torch(str(x.dtype))

        # handling legacy torch <= 1.13
        if hasattr(x, "untyped_storage"):
            storage = x.untyped_storage()
        else:
            storage = x.storage()

        return Variable.from_torch(
            storage.data_ptr(),
            x.storage_offset(),
            storage.nbytes(),
            tuple(x.shape),
            x.stride(),
            dtype,
            x.is_cuda,
        )
    elif strict:
        raise ValueError("get hbtl tensor from unknown tensor")
    return x


def get_tensor_type(x, strict=False) -> Variable:
    if is_tensor(x):
        dtype = get_dtype(x.dtype)
        return Variable.tensor_type(tuple(x.shape), dtype)
    elif strict:
        raise ValueError("get hbtl tensor from unknown tensor")
    return x


def _emplace_tensor(tensor_type, use_numpy):
    if use_numpy:
        return np.empty(tensor_type.shape, np.dtype(tensor_type.dtype.format))
    else:
        import torch

        numpy_to_torch_dtype_dict = {
            "bool": torch.bool,
            "uint8": torch.uint8,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        dtype = numpy_to_torch_dtype_dict[str(np.dtype(tensor_type.dtype.format))]
        return torch.empty(size=tensor_type.shape, dtype=dtype)


def get_schema_and_infer_type(ns, op, config_args):
    schema = get_matched_schema(ns, op, config_args)
    result_types = infer_result_types(schema, config_args)
    return schema, result_types


def execute_kernel(ns, op, args):

    # 1. infer result type uses kernel config impl
    infer_args = [get_tensor_type(x) for x in args]
    schema, result_types = get_schema_and_infer_type(ns, op, infer_args)

    # 2. emplace result tensor using numpy.empty or torch.empty
    use_numpy = is_np_array(args[0]) if len(args) else False
    result_values = [_emplace_tensor(a, use_numpy) for a in result_types]

    # 3. convert np.array or torch.tensor into variable
    invoke_inargs = [get_tensor(a) for a in args]
    invoke_outargs = [get_tensor(a, True) for a in result_values]

    # 4. run kernel invoke impl
    invoke_kernel(schema, invoke_outargs, invoke_inargs)
    return result_values


class HBTLModuleFinder:
    def __getattr__(self, module_name):
        class HBTLOpFinder:
            def __getattr__(self, op_name):
                def find_op(*args):
                    return execute_kernel(module_name, op_name, args)

                return find_op

        return HBTLOpFinder()


ops = HBTLModuleFinder()
