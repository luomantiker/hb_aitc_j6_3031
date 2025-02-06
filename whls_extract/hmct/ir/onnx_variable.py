from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
import torch

from hmct.utility import random_data

from .onnx_utils import TensorProto, ValueInfoProto
from .utils import (
    DataType,
    DestInfoTuple,
    TensorDevice,
    TensorMode,
    get_dtype_from_value_info,
    get_shape_from_value_info,
    make_tensor_value_info,
    numpy_array_from_onnx_tensor,
    numpy_array_from_torch_tensor,
    numpy_dtype_to_onnx_dtype,
    onnx_dtype_to_numpy_dtype,
    onnx_tensor_from_numpy_array,
    onnx_tensor_from_torch_tensor,
    torch_dtype_to_onnx_dtype,
    torch_tensor_from_numpy_array,
    torch_tensor_from_onnx_tensor,
)

if TYPE_CHECKING:
    from .onnx_graph import OnnxGraph
    from .onnx_model import OnnxModel
    from .onnx_node import OnnxNode


class OnnxVariable:
    _owning_graph: "OnnxGraph"
    _proto: Union[None, ValueInfoProto, TensorProto]
    _name: str
    _dtype: Union[None, DataType]
    _shape: Union[None, List[Union[None, str, int]]]
    _value: Union[None, torch.Tensor, np.ndarray]
    _is_param: bool
    _is_dummy: bool
    _is_attr: bool
    _src_op: Union[None, "OnnxNode"]
    _dest_infos: List[DestInfoTuple]
    _sync_trigger: bool

    @overload
    def __init__(
        self,
        owning_graph: "OnnxGraph",
        proto: Union[ValueInfoProto, TensorProto],
        *,
        is_attr: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        owning_graph: "OnnxGraph",
        *,
        name: Optional[str] = None,
        dtype: Optional[DataType] = None,
        shape: Optional[Iterable[Union[None, str, int]]] = None,
        value: Optional[Union[np.ndarray, torch.Tensor]] = None,
        is_param: bool = False,
        is_dummy: bool = False,
        is_attr: bool = False,
    ) -> None: ...

    def __init__(
        self,
        owning_graph: "OnnxGraph",
        proto: Optional[Union[ValueInfoProto, TensorProto]] = None,
        name: Optional[str] = None,
        dtype: Optional[DataType] = None,
        shape: Optional[Iterable[Union[None, str, int]]] = None,
        value: Optional[Union[np.ndarray, torch.Tensor]] = None,
        is_param: bool = False,
        is_dummy: bool = False,
        is_attr: bool = False,
    ):
        """初始化OnnxVariable对象."""
        super().__init__()

        # initialize owning_graph for onnx variable
        self._owning_graph = owning_graph

        # initialize src_op && dest_infos for onnx variable
        self._src_op = None
        self._dest_infos = []

        # init from proto or config
        if proto is None:
            self._init_from_config(
                name=name,
                dtype=dtype,
                shape=shape,
                value=value,
                is_param=is_param,
                is_dummy=is_dummy,
                is_attr=is_attr,
            )
        else:
            self._init_from_proto(proto=proto, is_attr=is_attr)
        # validity check for is_param/is_dummy/is_attr
        if self._is_attr:
            assert self._is_param, "attribute variable must be model parameter."
        if self._is_dummy:
            assert not self._is_param, "dummy variable must be model activation."

        # initialize sync trigger flag for model parameter
        if self._is_param:
            self._sync_trigger = False

        # add initialized variable to owning_graph's variables field
        self._owning_graph._variables.add(self)

    def _init_from_config(
        self,
        name: Optional[str] = None,
        dtype: Optional[DataType] = None,
        shape: Optional[Iterable[Union[None, str, int]]] = None,
        value: Optional[Union[np.ndarray, torch.Tensor]] = None,
        is_param: bool = False,
        is_dummy: bool = False,
        is_attr: bool = False,
    ) -> None:
        """基于参数配置初始化OnnxVariable对象."""
        self._proto = None
        # resolve the variable name
        if name is None:
            name = self.owning_model._get_next_unique_name(namespace="variable")
        assert len(name) > 0, "variable name should be non-empty string."
        self._name = name
        # initialize value/dtype/shape for onnx variable
        if is_param:
            if value is None:
                assert (
                    dtype is not None and shape is not None
                ), "dtype and shape should be given for model param when value is None."
                if any(not isinstance(dim, int) for dim in shape):
                    raise ValueError("Each shape dim for model param must be integer.")
                value = random_data(
                    cast(List[int], list(shape)),
                    dtype=onnx_dtype_to_numpy_dtype(dtype),
                    rtype="normal",
                )
            else:
                value_dtype = DataType(
                    numpy_dtype_to_onnx_dtype(value.dtype)
                    if isinstance(value, np.ndarray)
                    else torch_dtype_to_onnx_dtype(value.dtype)
                )
                value_shape = value.shape
                if dtype is not None:
                    assert (
                        dtype == value_dtype
                    ), f"dtype: {dtype} is different from value's dtype: {value_dtype}."
                if shape is not None:
                    assert list(shape) == list(
                        value_shape
                    ), f"shape: {shape} is different from value's shape: {value_shape}."
                dtype, shape = value_dtype, value_shape
            # initialize value/dtype/shape for model param
            self._value = value
            self._dtype = dtype
            self._shape = list(shape)
        else:
            # initialize dtype/shape for model activation
            assert value is None, "value should not be given to init model activation."
            self._value = value
            self._dtype = dtype
            self._shape = shape if shape is None else list(shape)
        # initialize is_param/is_dummy/is_attr for onnx variable
        self._is_param = is_param
        self._is_dummy = is_dummy
        self._is_attr = is_attr

    def _init_from_proto(
        self, proto: Union[ValueInfoProto, TensorProto], is_attr: bool
    ) -> None:
        if isinstance(proto, ValueInfoProto):
            # case for model activations with ValueInfoProto
            self._proto = proto
            self._name = self._proto.name
            self._dtype = get_dtype_from_value_info(self._proto)
            self._shape = get_shape_from_value_info(self._proto)
            self._value = None
            self._is_param = False
        else:
            # case for model parameters
            self._proto = proto
            self._name = self._proto.name
            self._dtype = DataType(self._proto.data_type)
            self._shape = list(self._proto.dims)
            self._value = None
            self._is_param = True
        # initialize is_dummy && is_attr for onnx variable
        self._is_dummy = False
        self._is_attr = is_attr

    @property
    def owning_model(self) -> "OnnxModel":
        """返回当前variable所属的model.

        属性类型:
            读写属性, 允许原地修改
        """
        return self.owning_graph.owning_model

    @property
    def owning_graph(self) -> "OnnxGraph":
        """返回当前variable直接所属的graph.

        属性类型:
            读写属性, 允许原地修改
        """
        return self._owning_graph

    @property
    def proto(self) -> Union[None, TensorProto, ValueInfoProto]:
        """返回基于当前variable导出的TensorProto或者ValueInfoProto对象.

        当OnnxVariable表示模型参数时, 导出TensorProto对象;
        当OnnxVariable表示模型激活时, 导出ValueInfoProto对象
            若该模型激活的shape和dtype信息都缺失, 则返回None

        属性类型:
            只读属性
        """
        # sync from OnnxVariable to TensorProto or ValueInfoProto
        self._sync()
        return self._proto

    @property
    def name(self) -> str:
        """返回当前variable的name.

        属性类型:
            读写属性, 允许赋值修改
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        assert (
            self._name != "PACKED_HBM_MODEL"
        ), "Illegal to change name for variable PACKED_HBM_MODEL"

        if len(name) > 0:
            self._name = name
        else:
            raise ValueError("variable name should be non-empty string.")

    @property
    def tensor_mode(self) -> TensorMode:
        """返回当前variable的存储模式.

        属性类型:
            只读属性
        """
        return self.owning_model.tensor_mode

    @property
    def tensor_device(self) -> TensorDevice:
        """返回当前variable的存储设备.

        属性类型:
            只读属性
        """
        return self.owning_model.tensor_device

    @property
    def is_param(self) -> bool:
        """返回当前variable是否为模型参数.

        模型参数的取值在推理计算时固定, 对应于onnx中的TensorProto对象

        若该变量为模型参数, 则返回True; 若该变量为模型激活, 则返回False.

        属性类型:
            只读属性
        """
        return self._is_param

    @property
    def is_dummy(self) -> bool:
        """返回当前variable是否为虚拟激活.

        仅当该变量为模型激活, 即is_param=False时, 该属性取值有意义

        若满足以下任意条件:
            1) 该激活为node的可选输入, 且无需外部为该输入提供取值
            2) 该激活为node的可选输出, 且无需计算该输出取值
        该激活为虚拟激活, 则返回True; 否则返回False

        属性类型:
            只读属性
        """
        return self._is_dummy

    @property
    def is_attr(self) -> bool:
        """返回当前variable是否为node属性.

        仅当该变量为模型参数, 即is_param=True时, 该属性取值有意义

        若该模型参数为node属性, 则返回True; 若该模型参数为node输入, 则返回False

        属性类型:
            只读属性
        """
        return self._is_attr

    @property
    def dtype(self) -> Union[None, DataType]:
        """返回当前variable的dtype.

        属性类型:
            读写属性, 模型激活允许赋值修改, 模型参数不允许通过dtype属性修改
        """
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: Union[None, DataType]) -> None:
        if self.is_param:
            raise RuntimeError(
                "Couldn't set dtype for param via dtype attribute setter, "
                "use 'onnx_var.value = ...' to set dtype implicitly",
            )

        self._dtype = dtype

    @property
    def shape(self) -> Union[None, Sequence[Union[None, str, int]]]:
        """返回当前variable的shape.

        属性类型:
            读写属性, 模型激活允许原地修改和赋值修改, 模型参数不允许通过shape属性修改
        """
        if self._shape is None:
            return self._shape
        if self.is_param:
            return tuple(self._shape)
        return self._shape

    @shape.setter
    def shape(self, shape: Union[None, Iterable[Union[None, str, int]]]) -> None:
        if self.is_param:
            raise RuntimeError(
                "Couldn't set shape for param via shape attribute setter, "
                "use 'onnx_var.value = ...' to set shape implicitly",
            )
        self._shape = shape if shape is None else list(shape)

    @property
    def is_shape_dynamic(self) -> bool:
        """返回当前variable的shape是否存在动态维度.

        属性类型:
            只读属性
        """
        # shape不存在也认定为是动态shape
        if self.shape is None:
            return True

        # dim缺失或者为字符串或者为负数, 均表示该维度取值动态
        return any(dim is None or isinstance(dim, str) or dim < 0 for dim in self.shape)

    def _parse_value(self) -> None:
        """根据is_param/tensor_mode/tensor_device等状态信息动态解析出value的取值."""
        # case for model activation
        if not self.is_param and self._value is not None:
            if (
                isinstance(self._value, np.ndarray)
                and self.tensor_mode == TensorMode.TORCH
            ):
                self._value = torch_tensor_from_numpy_array(self._value)
                self._value = self._value.to(str(self.tensor_device))
            if isinstance(self._value, torch.Tensor):
                if self.tensor_mode == TensorMode.NUMPY:
                    self._value = numpy_array_from_torch_tensor(self._value)
                elif self.tensor_mode == TensorMode.TORCH:
                    self._value = self._value.to(str(self.tensor_device))
        # case for model parameter
        if self.is_param and self.name != "PACKED_HBM_MODEL":
            if self._value is None:
                if self.tensor_mode == TensorMode.NUMPY:
                    self._value = numpy_array_from_onnx_tensor(self._proto)
                if self.tensor_mode == TensorMode.TORCH:
                    self._value = torch_tensor_from_onnx_tensor(self._proto)
                    self._value = self._value.to(str(self.tensor_device))
            if (
                isinstance(self._value, np.ndarray)
                and self.tensor_mode == TensorMode.TORCH
            ):
                self._value = torch_tensor_from_numpy_array(self._value)
                self._value = self._value.to(str(self.tensor_device))
            if isinstance(self._value, torch.Tensor):
                if self.tensor_mode == TensorMode.NUMPY:
                    self._value = numpy_array_from_torch_tensor(self._value)
                elif self.tensor_mode == TensorMode.TORCH:
                    self._value = self._value.to(str(self.tensor_device))

    @property
    def value(self) -> Union[np.ndarray, torch.Tensor, None]:
        """返回当前variable的张量取值.

        若当前variable为模型激活且无实际计算的具体取值, 则返回None;
        若当前variable为模型参数且参数name为PACKED_HBM_MODEL, 则返回None

        默认返回数据类型为np.ndarray, 可通过OnnxModel中的`to`接口改变存储模式
        在从而在np.ndarray和torch.Tensor间切换

        属性类型:
            读写属性, 允许赋值修改和原地修改
        """
        self._parse_value()
        if self.is_param and self.name != "PACKED_HBM_MODEL":
            # set update flag for model weights
            self._sync_trigger = True
        return self._value

    @value.setter
    def value(self, value: Union[np.ndarray, torch.Tensor, None]) -> None:
        assert (
            self.name != "PACKED_HBM_MODEL"
        ), "Illegal to assign value for PACKED_HBM_MODEL"
        # validity check for model parameter
        if self.is_param:
            # set update flag for model parameter
            self._sync_trigger = True
            # update dtype and shape implicitly for model param
            if isinstance(value, np.ndarray):
                self._dtype = numpy_dtype_to_onnx_dtype(value.dtype)
            elif isinstance(value, torch.Tensor):
                self._dtype = torch_dtype_to_onnx_dtype(value.dtype)
            else:
                raise TypeError(
                    f"type(value) should be either np.ndarray or "
                    f"torch.Tensor, but got {type(value)}."
                )
            self._shape = list(value.shape)
        # validity check for model activation
        if not self.is_param and value is not None:
            # check dtype consistency
            static_dtype = self.dtype
            # it's free to skip check for missing ValueInfoProto
            if static_dtype is not None:
                if isinstance(value, np.ndarray):
                    dynamic_dtype = numpy_dtype_to_onnx_dtype(value.dtype)
                elif isinstance(value, torch.Tensor):
                    dynamic_dtype = torch_dtype_to_onnx_dtype(value.dtype)
                else:
                    raise TypeError(
                        f"type(value) should be either np.ndarray or "
                        f"torch.Tensor, but got {type(value)}."
                    )
                assert static_dtype.value == dynamic_dtype.value, (
                    f"Mismatch between static dtype({static_dtype.name}) and "
                    f"dynamic dtype({dynamic_dtype.name}) for variable {self.name}"
                )
            # check shape consistency
            static_shape = self.shape
            # it's free to skip check for missing ValueInfoProto
            if static_shape is not None:
                dynamic_shape = value.shape
                assert len(static_shape) == len(dynamic_shape), (
                    f"Mismatch between static shape({static_shape}) and "
                    f"dynamic shape({dynamic_shape}) for variable {self.name}."
                )
                for static_dim, dynamic_dim in zip(static_shape, dynamic_shape):
                    if isinstance(static_dim, int) and static_dim != -1:
                        assert static_dim == dynamic_dim, (
                            f"Mismatch between static shape({static_shape}) and "
                            f"dynamic shape({dynamic_shape}) for variable {self.name}."
                        )

        self._value = value

    @property
    def src_op(self) -> Union[None, "OnnxNode"]:
        """返回当前variable的源node.

        属性类型:
            读写属性, 允许原地修改
        """
        return self._src_op

    @property
    def src_idx(self) -> Union[None, int]:
        """返回当前variable在其源node中的输出索引值.

        属性类型:
            只读属性
        """
        src_idx = None
        if self.src_op is not None:
            src_idx = self.src_op._outputs.index(self)

        return src_idx

    @property
    def dest_infos(self) -> Tuple[DestInfoTuple, ...]:
        """返回将当前variable作为输入的node及该variable在node输入序列索引的序列.

        属性类型:
            读写属性, 仅允许原地修改node
        """
        return tuple(self._dest_infos)

    @property
    def dest_ops(self) -> Tuple["OnnxNode", ...]:
        """返回将当前variable作为输入的node序列.

        由于存在相同variable被某个node的多个输入位置使用的可能,
        dest_ops的序列长度小于等于dest_infos的序列长度

        属性类型:
            读写属性, 允许原地修改node, 不可增加或删除node
        """
        dest_ops: List["OnnxNode"] = []
        for dest_op, _ in self._dest_infos:
            if dest_op not in dest_ops:
                dest_ops.append(dest_op)

        return tuple(dest_ops)

    @property
    def dest_op(self) -> "OnnxNode":
        """返回将当前variable作为输入的node.

        若存在多个node将当前variable作为输入, 则抛出异常

        属性类型:
            读写属性, 允许原地修改node
        """
        dest_ops = self.dest_ops
        assert len(dest_ops) == 1, f"node {self.name} has more than 1 dest_ops."

        return dest_ops[0]

    @property
    def is_used(self) -> bool:
        """返回当前variable是否有被使用.

        variable`被使用`的含义包括作为node输入或者graph输出

        属性类型:
            只读属性
        """
        return len(self._dest_infos) > 0 or self in self.owning_graph._outputs

    def replace_all_uses_with(
        self,
        onnx_var: Union[str, "OnnxVariable"],
    ) -> "OnnxVariable":
        """将所有对当前variable的使用替换为给定variable.

        variable`被使用`的含义包括作为node输入或者graph输出
        """
        if isinstance(onnx_var, str):
            onnx_var = self.owning_model.variable_mappings[onnx_var]

        # validity check for given onnx_var
        assert not onnx_var.is_attr, "variable to replace use could not be attribute."
        assert not onnx_var.is_dummy, "variable to replace use could not be dummy."

        # return directly if self and given onnx_var point to the same variable
        if self is onnx_var:
            return self

        # replace all the uses exist in node inputs
        for dest_info in self._dest_infos:
            dest_op, dest_idx = dest_info
            dest_op._inputs[dest_idx] = onnx_var
            onnx_var._dest_infos.append(dest_info)
        self._dest_infos.clear()
        # replace all the uses exist in graph outputs
        for output_idx, output_var in enumerate(self.owning_graph._outputs):
            if self is output_var:
                assert self.owning_graph is onnx_var.owning_graph, (
                    f"The owning graph of current variable {self.name} and "
                    f"given variable {onnx_var.name} are different. Illegal to"
                    f"specify the given variable {onnx_var.name} as graph output."
                )
                self.owning_graph._outputs[output_idx] = onnx_var

        if not self.is_used and self.src_op is None:
            self._destroy()

        return self

    def __repr__(self) -> str:
        """返回当前variable的字符串表示."""
        return f"{self.name}, {self.shape}, {self.dtype}"

    def _destroy(self) -> None:
        """将当前variable从所属graph中清除.

        仅当前variable不充当任何node输入或者graph输出时, 该操作合法
        """
        # variable to be removed should not exist in node inputs or graph outputs
        assert not self.is_used, (
            f"variable {self.name} has been used as node inputs "
            f"or graph outputs, invalid to remove it from graph."
        )
        # variable to be removed should not have src_op
        assert (
            self.src_op is None
        ), f"variable {self.name} has src_op, invalid to remove it from graph."

        # remove variable from graph inputs
        if self in self.owning_graph._inputs:
            self.owning_graph._inputs.remove(self)
        # remove variable from graph variables
        self.owning_graph._variables.remove(self)

    def _sync(self) -> None:
        if self.is_param:
            # sync from OnnxVariable to TensorProto
            if self._proto is None or self._sync_trigger:
                if isinstance(self._value, np.ndarray):
                    self._proto = onnx_tensor_from_numpy_array(
                        numpy_array=self._value,
                        name=self.name,
                    )
                elif isinstance(self._value, torch.Tensor):
                    self._proto = onnx_tensor_from_torch_tensor(
                        torch_tensor=self._value,
                        name=self.name,
                    )
                else:
                    raise TypeError(
                        "type(value) for onnx variable should be either np.ndarray "
                        f"or torch.Tensor, but got {type(self._value)}."
                    )
                # unset sync trigger flag for model parameters
                self._sync_trigger = False
            else:
                # sync for name
                self._proto.name = self.name
        else:
            # sync from OnnxVariable to ValueInfoProto
            if self.dtype is not None or self.shape is not None:
                value_info_proto = make_tensor_value_info(
                    name=self.name,
                    dtype=self.dtype,
                    shape=self.shape,
                )
                self._proto = value_info_proto
            else:
                self._proto = None
