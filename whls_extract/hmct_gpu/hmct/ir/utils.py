from enum import Enum
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Union,
)

import numpy as np
import torch

from .onnx_utils import (
    AttributeProto,
    TensorProto,
    TypeProto,
    ValueInfoProto,
    helper,
    numpy_helper,
)

if TYPE_CHECKING:
    from .onnx_graph import OnnxGraph
    from .onnx_node import OnnxNode


class DestInfoTuple(NamedTuple):
    # dest_op field
    dest_op: "OnnxNode"
    # dest_idx field
    dest_idx: int

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DestInfoTuple):
            return self.dest_op is other.dest_op and self.dest_idx == other.dest_idx
        return False

    def __ne__(self, other: object) -> bool:
        return not self == other


class TensorMode(Enum):
    """TensorMode defines the storage model for TensorProto value."""

    NUMPY = 0
    TORCH = 1


class TensorDevice:
    """TensorDevice defines the storage device for TensorProto value."""

    _type: str
    _index: Union[None, int]

    def __init__(
        self,
        device: Optional[Union[str, "TensorDevice"]] = None,
        type: Optional[str] = None,  # noqa: A002
        index: Optional[int] = None,
    ) -> None:
        if device is not None:
            assert type is None and index is None, (
                "Init TensorDevice with either TensorDevice(device=...) "
                "or TensorDevice(type=..., index=...)"
            )
            if isinstance(device, str):
                self._from_string(device)
            else:
                self._from_config(type=device.type, index=device.index)
        else:
            assert type in ["cpu", "cuda"], f"Invalid device type: {type}"
            assert index is None or index >= 0, f"Invalid device index: {index}"
            self._from_config(type=type, index=index)

    def _is_valid_device(self, device: str) -> bool:
        pattern = r"^(cuda|cpu)(:0|:[1-9]\d*)?$"
        return re.match(pattern, device) is not None

    def _from_string(self, device: str) -> None:
        if self._is_valid_device(device):
            self._type = device.split(":")[0]
            if len(device.split(":")) > 1:
                self._index = int(device.split(":")[1])
            else:
                self._index = None
        else:
            raise ValueError(f"Invalid device: {device}")

    def _from_config(self, type: str, index: Optional[int]) -> None:  # noqa: A002
        self._type = type
        self._index = index

    def __repr__(self) -> str:
        if self._index is None:
            return f"TensorDevice(type='{self._type}')"
        return f"TensorDevice(type='{self._type}', index={self._index})"

    def __str__(self) -> str:
        if self._index is None:
            return self._type
        return f"{self._type}:{self._index}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TensorDevice):
            return self.type == other.type and self.index == other.index
        if isinstance(other, str):
            return self == TensorDevice(other)
        return False

    def __ne__(self, other: object) -> bool:
        return not self == other

    @property
    def type(self) -> str:
        return self._type

    @property
    def index(self) -> Union[None, int]:
        return self._index


class DataType(Enum):
    """DataType encapsulates onnx datatypes in a more readable way."""

    UNDEFINED = 0  # onnx.TensorProto.UNDEFINED
    FLOAT32 = 1  # onnx.TensorProto.FLOAT
    UINT8 = 2  # onnx.TensorProto.UINT8
    INT8 = 3  # onnx.TensorProto.INT8
    UINT16 = 4  # onnx.TensorProto.UINT16
    INT16 = 5  # onnx.TensorProto.INT16
    INT32 = 6  # onnx.TensorProto.INT32
    INT64 = 7  # onnx.TensorProto.INT64
    STRING = 8  # onnx.TensorProto.STRING
    BOOL = 9  # onnx.TensorProto.BOOL
    FLOAT16 = 10  # onnx.TensorProto.FLOAT16
    FLOAT64 = 11  # onnx.TensorProto.DOUBLE
    UINT32 = 12  # onnx.TensorProto.UINT32
    UINT64 = 13  # onnx.TensorProto.UINT64
    COMPLEX64 = 14  # onnx.TensorProto.COMPLEX64
    COMPLEX128 = 15  # onnx.TensorProto.COMPLEX128
    BFLOAT16 = 16  # onnx.TensorProto.BFLOAT16
    FLOAT8E4M3FN = 17  # onnx.TensorProto.FLOAT8E4M3FN
    FLOAT8E4M3FNUZ = 18  # onnx.TensorProto.FLOAT8E4M3FNUZ
    FLOAT8E5M2 = 19  # onnx.TensorProto.FLOAT8E5M2
    FLOAT8E5M2FNUZ = 20  # onnx.TensorProto.FLOAT8E5M2FNUZ
    UINT4 = 21  # onnx.TensorProto.UINT4
    INT4 = 22  # onnx.TensorProto.INT4


def onnx_dtype_to_numpy_dtype(dtype: DataType) -> np.dtype:
    """Convert ONNX dtype to Numpy dtype.

    Args:
        dtype: ONNX data type.

    Returns:
        Corresponding Numpy dtype.
    """
    # https://numpy.org/doc/stable/reference/arrays.dtypes.html
    _dtypes: Dict[DataType, np.dtype] = {
        DataType.FLOAT32: np.dtype("float32"),
        DataType.UINT8: np.dtype("uint8"),
        DataType.INT8: np.dtype("int8"),
        DataType.UINT16: np.dtype("uint16"),
        DataType.INT16: np.dtype("int16"),
        DataType.INT32: np.dtype("int32"),
        DataType.INT64: np.dtype("int64"),
        DataType.STRING: np.dtype("object"),
        DataType.BOOL: np.dtype("bool"),
        DataType.FLOAT16: np.dtype("float16"),
        DataType.FLOAT64: np.dtype("float64"),
        DataType.UINT32: np.dtype("uint32"),
        DataType.UINT64: np.dtype("uint64"),
        DataType.COMPLEX64: np.dtype("complex64"),
        DataType.COMPLEX128: np.dtype("complex128"),
    }
    try:
        return _dtypes[dtype]
    except KeyError as exc:
        raise ValueError(f"dtype={dtype} is not supported") from exc


def numpy_dtype_to_onnx_dtype(dtype: np.dtype) -> DataType:
    """Convert Numpy dtype to ONNX dtype.

    Args:
        dtype: Numpy dtype

    Returns:
        Corresponding ONNX data type.
    """
    # https://numpy.org/doc/stable/reference/arrays.dtypes.html
    _dtypes: Dict[np.dtype, DataType] = {
        np.dtype("float32"): DataType.FLOAT32,
        np.dtype("uint8"): DataType.UINT8,
        np.dtype("int8"): DataType.INT8,
        np.dtype("uint16"): DataType.UINT16,
        np.dtype("int16"): DataType.INT16,
        np.dtype("int32"): DataType.INT32,
        np.dtype("int64"): DataType.INT64,
        np.dtype("object"): DataType.STRING,
        np.dtype("bool"): DataType.BOOL,
        np.dtype("float16"): DataType.FLOAT16,
        np.dtype("float64"): DataType.FLOAT64,
        np.dtype("uint32"): DataType.UINT32,
        np.dtype("uint64"): DataType.UINT64,
        np.dtype("complex64"): DataType.COMPLEX64,
        np.dtype("complex128"): DataType.COMPLEX128,
    }
    try:
        return _dtypes[dtype]
    except KeyError as exc:
        raise ValueError(f"dtype={dtype} is not supported") from exc


def onnx_dtype_to_torch_dtype(dtype: DataType) -> torch.dtype:
    """Convert ONNX dtype to PyTorch dtype.

    Args:
        dtype: ONNX data type.

    Returns:
        Corresponding PyTorch dtype.
    """
    # https://github.com/onnx/onnx/blob/main/onnx/onnx-ml.proto#L485
    _dtypes: Dict[DataType, torch.dtype] = {
        DataType.FLOAT32: torch.float32,
        DataType.UINT8: torch.uint8,
        DataType.INT8: torch.int8,
        # 4: UINT16 is not supported:
        # https://github.com/pytorch/pytorch/issues/58734.
        DataType.INT16: torch.int16,
        DataType.INT32: torch.int32,
        DataType.INT64: torch.int64,
        # 8: STRING is not supported
        DataType.BOOL: torch.bool,
        DataType.FLOAT16: torch.float16,
        DataType.FLOAT64: torch.float64,
        # 12: UINT32 is not supported:
        # https://github.com/pytorch/pytorch/issues/58734.
        # 13: UINT64 is not supported:
        # https://github.com/pytorch/pytorch/issues/58734.
        DataType.COMPLEX64: torch.complex64,
        DataType.COMPLEX128: torch.complex128,
        DataType.BFLOAT16: torch.bfloat16,
    }
    try:
        return _dtypes[dtype]
    except KeyError as exc:
        raise ValueError(f"dtype={dtype} is not supported") from exc


def torch_dtype_to_onnx_dtype(dtype: torch.dtype) -> DataType:
    """Convert PyTorch dtype to ONNX dtype.

    Args:
        dtype: PyTorch data type.

    Returns:
        Corresponding ONNX dtype.
    """
    _dtypes: Dict[torch.dtype, DataType] = {
        torch.float32: DataType.FLOAT32,
        torch.uint8: DataType.UINT8,
        torch.int8: DataType.INT8,
        # 4: UINT16 is not supported:
        # https://github.com/pytorch/pytorch/issues/58734.
        torch.int16: DataType.INT16,
        torch.int32: DataType.INT32,
        torch.int64: DataType.INT64,
        # 8: STRING is not supported
        torch.bool: DataType.BOOL,
        torch.float16: DataType.FLOAT16,
        torch.float64: DataType.FLOAT64,
        # 12: UINT32 is not supported:
        # https://github.com/pytorch/pytorch/issues/58734.
        # 13: UINT64 is not supported:
        # https://github.com/pytorch/pytorch/issues/58734.
        torch.complex64: DataType.COMPLEX64,
        torch.complex128: DataType.COMPLEX128,
        torch.bfloat16: DataType.BFLOAT16,
    }
    try:
        return _dtypes[dtype]
    except KeyError as exc:
        raise ValueError(f"dtype={dtype} is not supported") from exc


def get_attribute(proto: AttributeProto, owning_graph: "OnnxGraph") -> Any:
    # lazy import to avoid circular imports
    from .onnx_graph import OnnxGraph
    from .onnx_variable import OnnxVariable

    if proto.ref_attr_name:
        raise ValueError(f"Cannot get value of reference attribute: {proto}")
    if proto.type == AttributeProto.INT:
        return proto.i
    if proto.type == AttributeProto.FLOAT:
        return proto.f
    if proto.type == AttributeProto.STRING:
        return str(proto.s, "utf-8")
    if proto.type == AttributeProto.TENSOR:
        return OnnxVariable(
            owning_graph=owning_graph,
            proto=proto.t,
            is_attr=True,
        )
    if proto.type == AttributeProto.GRAPH:
        return OnnxGraph(
            owning_model=owning_graph.owning_model,
            owning_graph=owning_graph,
            proto=proto.g,
        )
    if proto.type == AttributeProto.INTS:
        return list(proto.ints)
    if proto.type == AttributeProto.FLOATS:
        return list(proto.floats)
    if proto.type == AttributeProto.STRINGS:
        return [str(s, "utf-8") for s in proto.strings]
    if proto.type == AttributeProto.TENSORS:
        return [
            OnnxVariable(owning_graph=owning_graph, proto=t, is_attr=True)
            for t in proto.tensors
        ]
    if proto.type == AttributeProto.GRAPHS:
        return [
            OnnxGraph(
                owning_model=owning_graph.owning_model,
                owning_graph=owning_graph,
                proto=g,
            )
            for g in proto.graphs
        ]

    raise ValueError(f"Unsupported ONNX attribute: {proto}")


def make_attribute(
    key: str,
    value: Any,
) -> AttributeProto:
    """Makes an AttributeProto based on the value type."""
    # lazy import to avoid circular imports
    from .onnx_graph import OnnxGraph
    from .onnx_variable import OnnxVariable

    if isinstance(value, (OnnxVariable, OnnxGraph)):
        value = value.proto
    if isinstance(value, list):
        for item_idx, item_val in enumerate(value):
            if isinstance(item_val, (OnnxVariable, OnnxGraph)):
                value[item_idx] = item_val.proto

    return helper.make_attribute(key=key, value=value)


def get_shape_from_value_info(
    value_info: ValueInfoProto,
) -> Union[None, List[Union[None, str, int]]]:
    if value_info.type.tensor_type.HasField("shape"):
        shape = []
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            else:
                shape.append(None)
        return shape

    return None


def get_dtype_from_value_info(value_info: ValueInfoProto) -> Union[None, DataType]:
    if value_info.type.tensor_type.HasField("elem_type"):
        return DataType(value_info.type.tensor_type.elem_type)
    return None


def make_tensor_value_info(
    name: str,
    dtype: Optional[DataType] = None,
    shape: Optional[Iterable[Union[None, str, int]]] = None,
) -> ValueInfoProto:
    """Makes a ValueInfoProto based on the dtype and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name

    if dtype is None and shape is None:
        return value_info_proto

    type_proto = TypeProto()
    tensor_type_proto = type_proto.tensor_type
    if dtype is not None:
        tensor_type_proto.elem_type = dtype.value

    tensor_shape_proto = tensor_type_proto.shape
    if shape is not None:
        # You might think this is a no-op (extending a normal Python
        # list by [] certainly is), but protobuf lists work a little
        # differently; if a field is never set, it is omitted from the
        # resulting protobuf; a list that is explicitly set to be
        # empty will get an (empty) entry in the protobuf. This
        # difference is visible to our consumers, so make sure we emit
        # an empty shape!
        tensor_shape_proto.dim.extend([])

        for d in shape:
            dim = tensor_shape_proto.dim.add()
            if d is None:
                pass
            elif isinstance(d, int):
                dim.dim_value = d
            elif isinstance(d, str):
                dim.dim_param = d
            else:
                raise ValueError(f"Invalid item in shape: {d}. Needs to of int or str.")

    value_info_proto.type.CopyFrom(type_proto)
    return value_info_proto


def numpy_array_from_onnx_tensor(onnx_tensor: TensorProto) -> np.ndarray:
    return numpy_helper.to_array(onnx_tensor).copy()


def onnx_tensor_from_numpy_array(
    numpy_array: np.ndarray,
    name: Optional[str] = None,
) -> TensorProto:
    return numpy_helper.from_array(arr=numpy_array, name=name)


def torch_tensor_from_onnx_tensor(onnx_tensor: TensorProto) -> torch.Tensor:
    numpy_array = numpy_array_from_onnx_tensor(onnx_tensor)
    return torch_tensor_from_numpy_array(numpy_array)


def onnx_tensor_from_torch_tensor(
    torch_tensor: torch.Tensor,
    name: Optional[str] = None,
) -> TensorProto:
    numpy_array = numpy_array_from_torch_tensor(torch_tensor)
    return onnx_tensor_from_numpy_array(numpy_array, name=name)


def torch_tensor_from_numpy_array(numpy_array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(numpy_array)


def numpy_array_from_torch_tensor(torch_tensor: torch.Tensor) -> np.ndarray:
    return torch_tensor.detach().cpu().numpy()
