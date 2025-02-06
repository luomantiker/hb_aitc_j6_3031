from typing import Tuple, List, Dict, Any, Optional

from hbdk4.compiler import ir as mlir
from hbdk4.compiler.dialects.func import FuncOp
from hbdk4.compiler.dialects.hbir import TrackAttr
from hbdk4.compiler.utils.remove_io_op import (
    run_remove_io_op,
    get_removable_io_op,
)
from hbdk4.compiler.utils.pytree import TreeLikeFuncBase, pickle_object, unpickle_object
from hbdk4.compiler._mlir_libs import _hbdk as _hbdk_cext
from abc import abstractmethod
from hbdk4.compiler._mlir_libs._hbdk import XqSession
from hbdk4.compiler import hbtl
from hbdk4.compiler.utils.default import handle_diagnostic
import numpy as np
import re
import importlib.util
import warnings
from functools import wraps
from numpy.lib.stride_tricks import as_strided


def clear_cached_xq_results(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if isinstance(args[0], Argument):
            args[0].func.cached_xq_results = None
        elif isinstance(args[0], Function):
            args[0].cached_xq_results = None
        else:
            pass
        return result

    return wrapper


class TensorType:
    def __init__(self, tensor: mlir.Type):
        if mlir.NoneType.isinstance(tensor):
            self.tensor = None
        else:
            self.tensor = mlir.ShapedType(tensor)

    @property
    def shape(self) -> Tuple[int]:
        return tuple(self.tensor.shape)

    @property
    def dims(self) -> Tuple[int]:
        return tuple(self.tensor.shape)

    @property
    def rank(self) -> int:
        return self.tensor.rank

    @property
    def dtype(self) -> str:
        return self.tensor.element_type.__str__()

    @property
    def quant_info(self):
        dtype = self.tensor.element_type
        if _hbdk_cext.QntUniformQuantizedType.is_a(dtype):
            return _hbdk_cext.QntUniformQuantizedType(dtype)
        return None

    @property
    def np_dtype(self) -> np.dtype:
        if self.quant_info is not None:
            ele_type = self.quant_info.storage_type
        else:
            ele_type = self.tensor.element_type

        ele_str = ele_type.__str__()
        if ele_str == "!hbir.bool8":
            return np.bool_
        if ele_str.startswith("si"):
            width = int(ele_str[2:])
            switch = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
            return switch.get(width, "unknown signed integer")
        if ele_str.startswith("ui"):
            width = int(ele_str[2:])
            switch = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
            return switch.get(width, "unknown unsigned integer")

        if ele_str == "f16":
            return np.float16
        if ele_str == "f32":
            return np.float32
        if ele_str == "f64":
            return np.float64
        raise TypeError("unknown type {} to np.dtype".format(ele_str))

    @property
    def torch_dtype(self):
        import torch

        if self.quant_info is not None:
            ele_type = self.quant_info.storage_type
        else:
            ele_type = self.tensor.element_type

        ele_type = self.tensor.element_type
        ele_str = ele_type.__str__()
        if ele_str == "!hbir.bool8":
            return torch.bool
        if ele_str.startswith("si"):
            width = int(ele_str[2:])
            switch = {8: torch.int8, 16: torch.int16, 32: torch.int32, 64: torch.int64}
            return switch.get(width, "unknown signed integer")
        if ele_str.startswith("ui"):
            width = int(ele_str[2:])
            switch = {8: torch.uint8}
            return switch.get(width, "unknown unsigned integer")

        if ele_str == "f32":
            return torch.float32
        if ele_str == "f64":
            return torch.float64
        if ele_str == "f16":
            return torch.float16
        raise TypeError("unknown type {} to np.dtype".format(ele_str))

    def __str__(self) -> str:
        if self.tensor is not None:
            return "tensor<{}x{}>".format(
                "x".join([str(s) for s in self.shape]), str(self.tensor.element_type)
            )
        else:
            return "none"


class ValueBase:
    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError

    @property
    def type(self) -> TensorType:
        if TensorType(self.value.type):
            return TensorType(self.value.type)
        raise TypeError(f'unknown argument type "{self.value.type}"')


class Value(ValueBase):
    def __init__(self, value: mlir.Value):
        self._v = value

    @property
    def value(self) -> mlir.Value:
        return self._v

    @property
    def name(self) -> str:
        return _hbdk_cext.get_name(self.value)

    @value.setter
    def value(self, v):
        self._v = v

    def __str__(self) -> str:
        return "{} defined by {}".format(self.type, self.value.__str__())


class Schema:
    def __init__(self, raw):
        self.raw = raw

        sig_match = re.search(r"^([^()]+)", self.raw)
        assert sig_match.span()[0] == 0
        self.namespace, self.signature = sig_match.group(1).split("::")

    def __str__(self) -> str:
        return self.raw


class Operation:
    def __init__(self, op: mlir.Operation):
        if not isinstance(op, mlir.Operation):
            op = op.operation

        self.op = op

    @property
    def type(self) -> str:
        return self.op.name

    @property
    def attributes(self) -> Dict:
        native_attr = {}
        for attr in self.op.attributes:
            if isinstance(attr.attr, mlir.StringAttr):
                native_attr[attr.name] = attr.attr.value
            else:
                native_attr[attr.name] = attr.attr.__str__()
        return native_attr

    @property
    def name(self) -> str:
        name = _hbdk_cext.get_op_name(self.op)
        if name is None:
            return self.op.location.__str__()
        return name

    @property
    def inputs(self) -> List[Value]:
        return [Value(val) for val in self.op.operands]

    @property
    def outputs(self) -> List[Value]:
        return [Value(arg) for arg in self.op.results]

    @property
    def schema(self) -> Optional[Schema]:
        """Return the schema if operation is partition to UDE(Unified kernel Dispatch and Execution)

        Returns:
            Optional[Schema]:
        """
        if self.type == "hbtl.call":
            return Schema(self.attributes["signature"])
        return None

    @property
    def track_attr(self):
        attr = _hbdk_cext.get_track_attr(self.op)
        if attr is None:
            return None
        return TrackAttr(attr)

    def __str__(self) -> str:
        return self.op.get_asm(
            enable_debug_info=True, pretty_debug_info=True, large_elements_limit=4
        )


class Schema:
    def __init__(self, raw):
        self.raw = raw

        sig_match = re.search(r"^([^()]+)", self.raw)
        assert sig_match.span()[0] == 0
        self.namespace, self.signature = sig_match.group(1).split("::")


class Argument(ValueBase):
    def __init__(self, func: "Function", idx: int, is_arg: bool):
        self.func = func
        self.idx = idx
        self.is_arg = is_arg
        self._version_id = _hbdk_cext.get_version_id(self.mlir_args[self.idx])

    def check_version(self, v: "Value"):
        if self._version_id != _hbdk_cext.get_version_id(v):
            raise RuntimeError("this argument is no longer valid")

    @property
    def mlir_args(self):
        if self.is_arg:
            return self.func.opview.body.blocks[0].arguments
        else:
            return list(list(self.func.opview.body.blocks)[-1].operations)[-1].operands

    def _get_version_id_tree(self):
        if self.func.support_pytree:

            def map_arg_to_version_id(item):
                return item._version_id

            from torch.utils._pytree import tree_map

            if self.is_arg:
                return tree_map(map_arg_to_version_id, self.func.inputs)
            else:
                return tree_map(map_arg_to_version_id, self.func.outputs)
        else:
            return None

    def _update_tree_spec(self, tree_nodes, new_node_num):
        if self.func.support_pytree:

            def replace(item):
                if item == self._version_id:
                    return [0] * new_node_num
                else:
                    return item

            from torch.utils._pytree import tree_flatten, tree_map

            result_tree = tree_map(replace, tree_nodes)
            _, new_spec = tree_flatten(result_tree)
            if self.is_arg:
                self.func._in_tree_spec = new_spec
            else:
                self.func._out_tree_spec = new_spec

    @property
    def value(self):
        self.check_version(self.mlir_args[self.idx])
        return self.mlir_args[self.idx]

    @property
    def name(self) -> str:
        return _hbdk_cext.get_name(self.value)

    @name.setter
    def name(self, name: str):
        _hbdk_cext.set_name(self.value, name)

    @property
    def desc(self) -> str:
        return _hbdk_cext.get_desc(self.value)

    @desc.setter
    def desc(self, desc: str):
        _hbdk_cext.set_desc(self.value, desc)

    @property
    def quant_info(self):
        qnt_type = _hbdk_cext.get_qnt_type(self.value)
        if qnt_type is None:
            return None
        return _hbdk_cext.QntUniformQuantizedType(qnt_type)

    @property
    def is_removable(self) -> Tuple:
        """Check if the attached operation is removable. The operation should be single input and single output. For input argument, it should only be used by the operation. For output argument, the operation input should only be used by the operation.

        Returns:
            Tuple: The first element is bool indicating the removable flag. The second element is the diagnostic if it cannot be removed.
        """
        return _hbdk_cext.is_removable(self.value)

    @property
    def _get_attached_op(self):
        if self.is_arg:
            return [op.owner for op in self.value.uses]
        else:
            return [self.value.owner]

    @property
    def get_attached_op(self) -> List[Operation]:
        """Get the argument attached operations. For input argument, return operations uses the argument. For output argument, return the operation defining the argument.

        Returns:
            List[Operation]: _description_
        """
        return [self.func._operations[op] for op in self._get_attached_op]

    @clear_cached_xq_results
    def remove_attached_op(self):
        """Remove the only attached operation

        Returns:
            Tuple: The first element is True when the removal done. The second element is the diagnostic if it cannot be removed.

        Note:
            Quantize and Dequantize op should be removed after convert
        """
        attached_op = self._get_attached_op
        new_id_start = (
            self.func.version_id_max + 1
            if self.is_arg
            else self.func.version_id_min - 1
        )
        status, diag = _hbdk_cext.remove_attached_op(self.value)
        if status:
            for op in attached_op:
                del self.func._operations[op]  # delete the map to maintain correctness
            _hbdk_cext.set_version_id(self.mlir_args[self.idx], new_id_start)
        return status, diag

    @handle_diagnostic
    @clear_cached_xq_results
    def erase(self):
        """Remove the argument from function argument

        Returns:
            Tuple: The first element is True when the removal done. The second element is the diagnostic if it cannot be removed.
        """
        version_tree = self._get_version_id_tree()
        if self.is_arg:
            diag = _hbdk_cext.remove_function_argument(self.func.opview, self.idx)
            if diag[0]:
                self._update_tree_spec(version_tree, 0)
            return diag
        else:
            diag = _hbdk_cext.remove_function_result(self.func.opview, self.idx)
            if diag[0]:
                self._update_tree_spec(version_tree, 0)
            return diag

    @handle_diagnostic
    @clear_cached_xq_results
    def insert_transpose(self, permutes: List[int]):
        """Insert transpose op. Change input/output parameter dimension order.

        Args:
            * permutes (List): Dimension transformation arrangement. Must contain all dimensions of the original input parameters, starting from 0
        Returns:
            List of newly inserted function arguments which is also the inputs/outputs of inserted transpose op

        Raises:
            ValueError when this argument is no longer valid

        Note:
            To avoid the new insertion operator not running in some conversion passes, it is recommended to call the insert_xxx api before the convert stage

        Example:
            module = load("model.bc")
            func = module[0]
            res = func.inputs[0].insert_transpose([3, 1, 2, 0])
        """
        new_id_start = (
            self.func.version_id_max + 1
            if self.is_arg
            else self.func.version_id_min - 1
        )
        desc = self.desc
        if _hbdk_cext.insert_transpose(self.value, permutes):
            _hbdk_cext.set_version_id(self.mlir_args[self.idx], new_id_start)
            _hbdk_cext.set_desc(self.mlir_args[self.idx], desc)
            if self.is_arg:
                return self.func.flatten_inputs[self.idx]
            else:
                return self.func.flatten_outputs[self.idx]
        return None

    @handle_diagnostic
    @clear_cached_xq_results
    def insert_rle(self):
        """Insert rle op. Run length encode on output.

        Returns:
            List of newly inserted function arguments which is the outputs of inserted rle op.

        Raises:
            ValueError when this argument is no longer valid.

        Note:
            The insert_rle api needs to be called after convert.
            If the output is dequantize op, dequantize op should be removed and then call insert_rle.

        Example:
            module = load("model.bc")
            func = module[0]
            res = func.inputs[0].insert_rle()
        """
        new_id_start = self.func.version_id_min - 1
        desc = self.desc
        if _hbdk_cext.insert_rle(self.value):
            _hbdk_cext.set_version_id(self.mlir_args[self.idx], new_id_start)
            _hbdk_cext.set_desc(self.mlir_args[self.idx], desc)
            if desc is not None:
                _hbdk_cext.set_desc(self.mlir_args[self.idx], desc)
            return self.func.flatten_outputs[self.idx]
        return None

    @handle_diagnostic
    def insert_image_convert(self, mode: str = "nv12"):
        """Insert image_convert op. Change input parameter type.

        Args:
            * mode (str): Specify conversion mode, optional values are "nv12"(default) and "gray".

        Returns:
            List of newly inserted function arguments which is also the inputs of inserted image convert op

        Raises:
            ValueError when this argument is no longer valid

        Note:
            To avoid the new insertion operator not running in some conversion passes, it is recommended to call the insert_xxx api before the convert stage

        Example:
            module = load("model.bc")
            func = module[0]
            res = func.inputs[0].insert_image_convert("nv12")
        """

        name = self.name
        new_id_start = self.func.version_id_max + 1
        desc = self.desc
        version_tree = self._get_version_id_tree()
        if _hbdk_cext.insert_image_convert(self.value, mode):
            y_name = name + "_y" if name is not None else None
            uv_name = name + "_uv" if name is not None else None
            if mode == "nv12":
                self._update_tree_spec(version_tree, 2)
                _hbdk_cext.set_name(self.mlir_args[self.idx], y_name)
                _hbdk_cext.set_name(self.mlir_args[self.idx + 1], uv_name)
                _hbdk_cext.set_version_id(self.mlir_args[self.idx], new_id_start)
                _hbdk_cext.set_version_id(
                    self.mlir_args[self.idx + 1], new_id_start + 1
                )
                if desc is not None:
                    for offset in range(2):
                        _hbdk_cext.set_desc(self.mlir_args[self.idx + offset], desc)
                return (
                    self.func.flatten_inputs[self.idx],
                    self.func.flatten_inputs[self.idx + 1],
                )
            else:
                _hbdk_cext.set_name(self.mlir_args[self.idx], y_name)
                _hbdk_cext.set_version_id(self.mlir_args[self.idx], new_id_start)
                if desc is not None:
                    _hbdk_cext.set_desc(self.mlir_args[self.idx], desc)
                return self.func.flatten_inputs[self.idx]
        return None

    @handle_diagnostic
    def insert_image_preprocess(
        self,
        mode: str,
        divisor: int,
        mean: List[float],
        std: List[float],
        is_signed: bool = True,
    ):
        """Insert image_convert op. Change input parameter type.

        Args:
            * mode (str): Specify conversion mode, optional values are "skip"(default, same as None), "yuvbt601full2rgb", "yuvbt601full2bgr", "yuvbt601video2rgb" and "yuvbt601video2bgr".

        Returns:
            List of newly inserted function arguments which is also the inputs of inserted image preprocess op

        Raises:
            ValueError when this argument is no longer valid

        Note:
            To avoid the new insertion operator not running in some conversion passes, it is recommended to call the insert_xxx api before the convert stage

        Example:
            module = load("model.bc")
            func = module[0]
            res = func.inputs[0].insert_image_preprocess("yuvbt601full2rgb", 255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)
        """

        name = self.name
        new_id_start = self.func.version_id_max + 1
        desc = self.desc
        if _hbdk_cext.insert_image_preprocess(
            self.value, mode, divisor, mean, std, is_signed
        ):
            _hbdk_cext.set_name(self.mlir_args[self.idx], name)
            _hbdk_cext.set_version_id(self.mlir_args[self.idx], new_id_start)
            if desc is not None:
                _hbdk_cext.set_desc(self.mlir_args[self.idx], desc)
            return self.func.flatten_inputs[self.idx]
        return None

    @handle_diagnostic
    def insert_roi_resize(
        self,
        mode: str,
        interp_mode="bilinear",
        pad_mode="constant",
        pad_value: Optional[tuple] = (0, -128),
    ):
        """Insert roi_resize op. Change input parameter type.

        Args:
            * mode (str): Specify conversion mode, optional values are "nv12" and "gray".
            * interp_mode (str): Specify interpolation mode, optional values are "bilinear"(default) and "nearest".
            * pad_mode (str): Specify fill mode, optional values are "constant"(default) and "border".
            * pad_value (tuple): Specify the padding value for Y and UV in custom pad_mode, default values are (0, -128).

        Returns:
            List of newly inserted function arguments which is also the inputs of inserted roi resize op

        Raises:
            ValueError when this argument is no longer valid

        Note:
            To avoid the new insertion operator not running in some conversion passes, it is recommended to call the insert_xxx api before the convert stage

        Example:
            module = load("model.bc")
            func = module[0]
            res = func.inputs[0].insert_roi_resize(
                mode = "nv12",
                interp_mode = "nearest",
                pad_mode = "constant",
                pad_value = (66, 77)
            )
        """

        name = self.name
        new_id_start = self.func.version_id_max + 1
        desc = self.desc
        version_tree = self._get_version_id_tree()
        if _hbdk_cext.insert_roi_resize(
            self.value, mode, interp_mode, pad_mode, pad_value
        ):
            block = self.func.opview.body.blocks[0]
            roi_name = name + "_roi" if name is not None else None
            y_name = name + "_y" if name is not None else None
            uv_name = name + "_uv" if name is not None else None

            if mode == "nv12":
                self._update_tree_spec(version_tree, 3)
                _hbdk_cext.set_name(self.mlir_args[self.idx], y_name)
                _hbdk_cext.set_name(self.mlir_args[self.idx + 1], uv_name)
                _hbdk_cext.set_name(self.mlir_args[self.idx + 2], roi_name)
                _hbdk_cext.set_version_id(self.mlir_args[self.idx], new_id_start)
                _hbdk_cext.set_version_id(
                    self.mlir_args[self.idx + 1], new_id_start + 1
                )
                _hbdk_cext.set_version_id(
                    self.mlir_args[self.idx + 2], new_id_start + 2
                )
                if desc is not None:
                    for offset in range(3):
                        _hbdk_cext.set_desc(self.mlir_args[self.idx + offset], desc)
                return (
                    self.func.flatten_inputs[self.idx],
                    self.func.flatten_inputs[self.idx + 1],
                    self.func.flatten_inputs[self.idx + 2],
                )
            else:
                self._update_tree_spec(version_tree, 2)
                _hbdk_cext.set_name(self.mlir_args[self.idx], y_name)
                _hbdk_cext.set_name(self.mlir_args[self.idx + 1], roi_name)
                _hbdk_cext.set_version_id(self.mlir_args[self.idx], new_id_start)
                _hbdk_cext.set_version_id(
                    self.mlir_args[self.idx + 1], new_id_start + 1
                )
                if desc is not None:
                    for offset in range(2):
                        _hbdk_cext.set_desc(self.mlir_args[self.idx + offset], desc)
                return (
                    self.func.flatten_inputs[self.idx],
                    self.func.flatten_inputs[self.idx + 1],
                )
        return None

    @handle_diagnostic
    @clear_cached_xq_results
    def insert_split(self, dim: int):
        """Insert split op.
        Split a single input/output parameter into multiple input/output parameters with a specified dimension of 1.

        Args:
            * dim (int): Dimension along which to split the tensor.

        Returns:
            List of newly inserted function arguments which is also the inputs/outputs of inserted concat/slice op

        Note:
            To avoid the new insertion operator not running in some conversion passes, it is recommended to call the insert_xxx api before the convert stage

        Raises:
            ValueError when this argument is no longer valid


        Example:
            module = load("model.bc")
            func = module[0]
            res = func.inputs[0].insert_split(0)
        """

        name = self.name
        desc = self.desc
        out_num = Value(self.value).type.shape[dim]
        if out_num > 256:
            raise ValueError("cannot split when dim size larger than 256")
        new_id_start = (
            self.func.version_id_max + 1
            if self.is_arg
            else self.func.version_id_min - 1
        )
        version_tree = self._get_version_id_tree()
        if _hbdk_cext.insert_split(self.value, dim):
            if self.is_arg:
                for i in range(out_num):
                    new_name = name + "_" + str(i) if name is not None else None
                    _hbdk_cext.set_name(self.mlir_args[self.idx + i], new_name)
                    _hbdk_cext.set_version_id(
                        self.mlir_args[self.idx + i], new_id_start + i
                    )
                    if desc is not None:
                        _hbdk_cext.set_desc(self.mlir_args[self.idx + i], desc)
                self._update_tree_spec(version_tree, out_num)
                return tuple(
                    self.func.flatten_inputs[self.idx + o] for o in range(out_num)
                )
            else:
                for i in range(out_num):
                    new_name = name + "_" + str(i) if name is not None else None
                    _hbdk_cext.set_name(self.mlir_args[self.idx + i], new_name)
                    _hbdk_cext.set_version_id(
                        self.mlir_args[self.idx + i], new_id_start - i
                    )
                    if desc is not None:
                        _hbdk_cext.set_desc(self.mlir_args[self.idx + i], desc)
                self._update_tree_spec(version_tree, out_num)
                return tuple(
                    self.func.flatten_outputs[self.idx + o] for o in range(out_num)
                )
        return None

    def __str__(self) -> str:
        name = self.name if self.name is not None else "_" + str(self.idx)
        return "{} {}".format(self.type, name)


class Function(TreeLikeFuncBase):
    def __init__(self, func: FuncOp, session=None, parent: "Module" = None):
        if isinstance(func, mlir.Operation):
            func = func.opview

        if not isinstance(func, FuncOp):
            raise TypeError("func is not a mlir.FuncOp")

        self.parent = parent  # keep parent alive

        self.opview = func
        self.session = session if session is not None else XqSession()
        if importlib.util.find_spec("hbdnn") is not None:
            import hbdnn

            self.session.cuda_library = hbdnn.__path__[0] + "/libhbdnn.so"

        self._operations = dict()
        for block in self.opview.body.blocks:
            for op in block:
                self._operations[op] = Operation(op)

        # Set default version ids
        for idx, v in enumerate(self.opview.arguments):
            _hbdk_cext.set_version_id(v, idx + 1)

        for idx, v in enumerate(
            list(list(self.opview.body.blocks)[-1].operations)[-1].operands
        ):
            _hbdk_cext.set_version_id(v, -idx - 1)

        # Set default names
        for idx, v in enumerate(self.flatten_inputs):
            if v.name is None:  # assign name it not provided
                v.name = "_input_{}".format(idx)

        for idx, v in enumerate(self.flatten_outputs):
            if v.name is None:  # assign name it not provided
                v.name = "_output_{}".format(idx)
        self.cached_xq_results = None

    @property
    def operations(self) -> List[Operation]:
        return [self._operations[key] for key in self._operations]

    def _emplace_results(self, use_numpy):
        if use_numpy:
            return [
                np.ndarray(v.type.shape, v.type.np_dtype) for v in self.flatten_outputs
            ]
        else:
            import torch

            return [
                torch.empty(v.type.shape, dtype=v.type.torch_dtype)
                for v in self.flatten_outputs
            ]

    def _convert_variable(self, use_numpy, arrays):
        return [hbtl.get_tensor(v) for v in arrays]

    def _sync_variable(self, use_numpy, vars, arrays):
        if use_numpy:
            res = []
            for v, a in zip(vars, arrays):
                # can not modify kernel stride
                if not all(
                    v_stride == 0 or v_stride == a_stride
                    for v_stride, a_stride in zip(v.strides, a.strides)
                ):
                    raise ValueError(
                        f"Strides do not match, can not modify kernel stride: v.strides={v.strides}, a.strides={a.strides}"
                    )
                res.append(as_strided(a, shape=v.shape, strides=a.strides))
            return res
        else:
            import torch

            res = []
            for v, a in zip(vars, arrays):
                torch_stride = [s * a.element_size() for s in a.stride()]
                # can not modify kernel stride
                if not all(
                    v_stride == 0 or v_stride == a_stride
                    for v_stride, a_stride in zip(v.strides, torch_stride)
                ):
                    raise ValueError(
                        f"Strides do not match, can not modify kernel stride: v.strides={v.strides}, a.strides={torch_stride}"
                    )
                res.append(torch.as_strided(a, size=v.shape, stride=a.stride()))
            return res

    def _launch(self, *args: Any, **kwargs) -> Any:
        use_numpy = isinstance(args[0], np.ndarray)

        # emplace function results if:
        #   a. first time to launch
        #   b. function io has changed
        #   c. input tensor type has changed
        if self.cached_xq_results is None or (
            len(self.cached_xq_results) > 0
            and isinstance(self.cached_xq_results[0], np.ndarray) != use_numpy
        ):
            self.cached_xq_results = self._emplace_results(use_numpy)

        invars = self._convert_variable(use_numpy, args)
        outvars = self._convert_variable(use_numpy, self.cached_xq_results)

        self.session.launch(
            self.opview.operation, tuple(outvars), tuple(invars), **kwargs
        )
        return self._sync_variable(use_numpy, outvars, self.cached_xq_results)

    def __call__(self, *args: Any, **kwargs) -> Any:
        if self.support_pytree:
            from torch.utils._pytree import tree_flatten, tree_unflatten

            input_list, input_spec = tree_flatten(args[0])
            if input_spec != self._in_tree_spec:
                raise ValueError(
                    "tree spec of function does not match given pytree input"
                )
            output_list = self._launch(*input_list)
            return tree_unflatten(output_list, self._out_tree_spec)
        else:
            return self._launch(*args, **kwargs)

    def _dict_to_args(self, feed_dict: Dict[str, Any]):
        args = []
        for arg in self.flatten_inputs:
            if arg.name not in feed_dict.keys():
                raise ValueError("cannot find argument named {}".format(arg))
            args.append(feed_dict[arg.name])
        return args

    def _results_to_dict(self, results) -> Dict[str, Any]:
        return {arg.name: results[idx] for idx, arg in enumerate(self.flatten_outputs)}

    def _dict_feed(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        args = self._dict_to_args(feed_dict)
        results = self._launch(*args)
        return self._results_to_dict(results)

    def feed(self, inputs):
        return self._dict_feed(inputs)

    @clear_cached_xq_results
    def append_output(self, v: Value):
        return _hbdk_cext.insert_function_return(v.value)

    @property
    def version_id_max(self) -> int:
        version_ids = [_hbdk_cext.get_version_id(v) for v in self.opview.arguments]
        return max(version_ids)

    @property
    def version_id_min(self) -> int:
        version_ids = [
            _hbdk_cext.get_version_id(v)
            for v in list(list(self.opview.body.blocks)[-1].operations)[-1].operands
        ]
        return min(version_ids)

    @property
    def name(self) -> str:
        return self.opview.sym_name.value

    @property
    def desc(self) -> str:
        return _hbdk_cext.get_func_desc(self.opview)

    @desc.setter
    def desc(self, desc: str):
        _hbdk_cext.set_func_desc(self.opview, desc)

    @property
    def internal_desc(self) -> str:
        return _hbdk_cext.get_func_internal_desc(self.opview)

    @internal_desc.setter
    def internal_desc(self, desc: str):
        _hbdk_cext.set_func_internal_desc(self.opview, desc)

    @TreeLikeFuncBase._in_tree_spec.getter
    def _in_tree_spec(self):
        tree_dict = unpickle_object(_hbdk_cext.get_func_tree_info(self.opview))
        if tree_dict is not None and "in" in tree_dict.keys():
            return unpickle_object(tree_dict["in"])
        else:
            return None

    @_in_tree_spec.setter
    def _in_tree_spec(self, in_tree_spec):
        """
        Internal tree_spec info method, do not use!
        """
        old_tree_str = _hbdk_cext.get_func_tree_info(self.opview)
        new_tree_dict = {} if old_tree_str is None else unpickle_object(old_tree_str)
        if in_tree_spec is not None:
            new_tree_dict["in"] = pickle_object(in_tree_spec)
        else:
            del new_tree_dict["in"]
        new_tree_info = pickle_object(new_tree_dict) if new_tree_dict else None
        _hbdk_cext.set_func_tree_info(self.opview, new_tree_info)

    @TreeLikeFuncBase._out_tree_spec.getter
    def _out_tree_spec(self):
        tree_dict = unpickle_object(_hbdk_cext.get_func_tree_info(self.opview))
        if tree_dict is not None and "out" in tree_dict.keys():
            return unpickle_object(tree_dict["out"])
        else:
            return None

    @_out_tree_spec.setter
    def _out_tree_spec(self, in_tree_spec):
        """
        Internal tree_spec info method, do not use!
        """
        old_tree_str = _hbdk_cext.get_func_tree_info(self.opview)
        new_tree_dict = {} if old_tree_str is None else unpickle_object(old_tree_str)
        if in_tree_spec is not None:
            new_tree_dict["out"] = pickle_object(in_tree_spec)
        else:
            del new_tree_dict["out"]
        new_tree_info = pickle_object(new_tree_dict) if new_tree_dict else None
        _hbdk_cext.set_func_tree_info(self.opview, new_tree_info)

    @TreeLikeFuncBase.flatten_inputs.getter
    def flatten_inputs(self):
        args = self.opview.arguments
        return [Argument(self, idx, True) for idx, arg in enumerate(args)]

    @TreeLikeFuncBase.flatten_outputs.getter
    def flatten_outputs(self):
        results = list(list(self.opview.body.blocks)[-1].operations)[-1].operands
        return [Argument(self, idx, False) for idx, arg in enumerate(results)]

    def statistics(self):
        op_static = dict()
        for op in self.operations:
            op_type = op.type
            if op_type == "hbtl.call":
                op_type += ": "
                op_type += op.schema.namespace
                op_type += "::"
                op_type += op.schema.signature

            if op_type in ["hbir.conv", "hbir.convtranspose"]:
                op_type += str(op.inputs[1].type.rank - 2)
                op_type += "d"

            if op_type in ["hbir.max_pool", "hbir.avg_pool"]:
                op_type += str(op.inputs[0].type.rank - 2)
                op_type += "d"

            if op_type in op_static:
                op_static[op_type] += 1
            else:
                op_static[op_type] = 1

        print("ops encountered in", self.__str__())
        key_len = len(max(op_static.keys(), key=len))
        for k in sorted(op_static.keys()):
            spaces = " " * (key_len - len(k))
            print("\t", k, spaces, ":", op_static[k])

        return op_static

    @clear_cached_xq_results
    def remove_io_op(self, op_types=None, op_names=None):
        """Experimental function to remove nodes from the model based on types or names

        Note:
            Quantize and Dequantize op should be removed after convert

        Args:
            op_types(list[str]|tuple[str]): a list/tuple of types to remove
            op_names(list[str]|tuple[str]): a list/tuple of names to remove

        Example:
            module = load("model.bc")
            func = module[0]
            func.remove_io_op(['Dequantize','Transpose','Cast'])
        """
        removable_res = get_removable_io_op(self, op_types, op_names)
        if len(removable_res) == 0:
            return []
        removable_res = list(dict.fromkeys(removable_res))
        run_remove_io_op(self, op_types, op_names)
        removable_res += self.remove_io_op(op_types, op_names)
        return removable_res

    def __str__(self) -> str:
        ret_repr = [str(arg) for arg in self.flatten_outputs]
        arg_repr = [str(arg) for arg in self.flatten_inputs]
        return "func @{}({}) -> {}".format(
            self.name, ", ".join(arg_repr), ", ".join(ret_repr)
        )

    def register_callback(self, user_callback, enable_track=True):
        def callback(op, results, operands) -> bool:
            return user_callback(Operation(op), results, operands)

        self.session.register_post_hook(callback)
        self.session.enable_track = enable_track


class Module:
    def __init__(
        self,
        module: mlir.Module,
        input_specs=None,
        output_specs=None,
    ):
        if isinstance(module, mlir.Module):
            module = module.operation

        if isinstance(module, mlir.Operation):
            module = module.opview

        # FIXME: when interpreter refactor
        _hbdk_cext._canonicalize(module.operation, module.context)

        self.session = XqSession()
        self.module = module
        if input_specs is not None and output_specs is not None:
            if len(self.functions) == len(input_specs) and len(input_specs) == len(
                output_specs
            ):
                warnings.warn(
                    "input_specs and output_specs would be removed in next version",
                    DeprecationWarning,
                )
                for op, in_spec, out_spec in zip(
                    self.functions, input_specs, output_specs
                ):
                    op._in_tree_spec = in_spec
                    op._out_tree_spec = out_spec
            else:
                raise ValueError("specs number not match function number")

    @property
    def _legacy_round_key(self) -> str:
        return "hbdk.legacy_round"  # must keep same key in QntSupport.h

    @property
    def _legacy_round(self) -> bool:
        if self._legacy_round_key in self.module.operation.attributes:
            return self.module.operation.attributes[self._legacy_round_key]
        return False

    @_legacy_round.setter
    def _legacy_round(self, value: bool):
        self.module.operation.attributes[self._legacy_round_key] = mlir.Attribute(
            mlir.BoolAttr.get(value)
        )

    @staticmethod
    def parse(asm: str) -> "Module":
        return Module(mlir.Module.parse(asm))

    def clone(self) -> "Module":
        return Module(self.module.operation.clone())

    def infer_type(self) -> None:
        _hbdk_cext._infer_type(self.module.operation, self.module.context)

    def replace_index_tensor_type(self) -> None:
        _hbdk_cext._replace_tensor_dtype(
            "si64", "si32", self.module.operation, self.module.context
        )
        _hbdk_cext._const_fold(self.module.operation, self.module.context)

    def replace_f32_tensor_to_f16(self) -> None:
        _hbdk_cext._replace_tensor_dtype(
            "f32", "f16", self.module.operation, self.module.context
        )
        _hbdk_cext._const_fold(self.module.operation, self.module.context)

    @property
    def functions(self) -> List[Function]:
        """return all functions in module

        Returns:
            List[FunctionHelper]: function are wrapped in FunctionHelper in pair with its symbol name
        """
        ret = []
        for op in self.module.body.operations:
            if isinstance(op, (mlir.Operation, FuncOp)):
                ret.append(Function(op, self.session, self))
        return ret

    @property
    def graphs(self) -> List[Function]:
        """return all functions in module

        Returns:
            List[FunctionHelper]: function are wrapped in FunctionHelper in pair with its symbol name
        """
        return self.functions

    def __getitem__(self, index_or_name):
        if isinstance(index_or_name, int):
            assert index_or_name < len(self.graphs)
            return self.graphs[index_or_name]
        elif isinstance(index_or_name, str):
            for graph in self.graphs:
                if index_or_name == graph.name:
                    return graph
            raise ValueError(f'module has no function "{index_or_name}"')
        else:
            raise TypeError(f"{index_or_name} has wrong type")

    def __str__(self) -> str:
        txt = ["{}:".format(self.__repr__())]
        for func in self.functions:
            txt.append("  {}".format(str(func)))
        return "\n".join(txt)
