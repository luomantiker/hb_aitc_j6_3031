from collections import defaultdict, deque
import itertools
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)

from .onnx_calibration_node import CalibrationNode
from .onnx_node import OnnxNode
from .onnx_utils import GraphProto
from .onnx_variable import OnnxVariable
from .utils import (
    DataType,
    DestInfoTuple,
    make_tensor_value_info,
)

if TYPE_CHECKING:
    import numpy as np
    import torch

    from .onnx_model import OnnxModel


class OnnxGraph:
    _owning_model: "OnnxModel"
    _owning_graph: Union["OnnxGraph", None]
    _proto: GraphProto
    _name: str
    _variables: Set[OnnxVariable]
    _nodes: List[OnnxNode]
    _inputs: List[OnnxVariable]
    _outputs: List[OnnxVariable]

    @overload
    def __init__(
        self,
        owning_model: "OnnxModel",
        owning_graph: Union["OnnxGraph", None],
        proto: GraphProto,
    ) -> None: ...

    @overload
    def __init__(
        self,
        owning_model: "OnnxModel",
        owning_graph: Union["OnnxGraph", None],
        *,
        name: Optional[str] = None,
    ) -> None: ...

    def __init__(
        self,
        owning_model: "OnnxModel",
        owning_graph: Union["OnnxGraph", None],
        proto: Optional[GraphProto] = None,
        name: Optional[str] = None,
    ):
        """初始化OnnxGraph对象."""
        super().__init__()

        # initialize owning_model for onnx graph
        self._owning_model = owning_model
        # initialize owning_graph for onnx graph
        self._owning_graph = owning_graph

        # initialize empty variables/nodes/inputs/outputs
        self._variables = set()
        self._nodes = []
        self._inputs = []
        self._outputs = []

        # init from proto or config
        if proto is None:
            self._init_from_config(name=name)
        else:
            self._init_from_proto(proto=proto)

        # append initialized graph to owning_model's graphs field
        self._owning_model._graphs.append(self)

    def _init_from_config(self, name: Optional[str] = None) -> None:
        """基于参数配置初始化OnnxModel对象."""
        self._proto = GraphProto()
        # resolve the graph name
        if name is None:
            name = self._owning_model._get_next_unique_name(namespace="graph")
        assert len(name) > 0, "graph name should be non-empty string."
        self._name = name

    def _init_from_proto(self, proto: GraphProto) -> None:
        """基于GraphProto初始化OnnxGraph对象."""
        # we suppose the given GraphProto:
        #   MUST present name field
        self._proto = proto
        self._name = self._proto.name
        # parse the given graph proto
        self._parse_graph_proto()

    def _parse_graph_proto(self) -> None:
        """解析给定的GraphProto以初始化当前graph."""
        # 1) first, create all the variables
        existing_variable_names: Set[str] = set()
        # 1-1) variables from initializer/value_info/graph_input/graph_output
        for proto in itertools.chain(
            self._proto.initializer,
            self._proto.value_info,
            self._proto.input,
            self._proto.output,
        ):
            assert (
                proto.name != ""
            ), "value_info proto or tensor proto name must be non-empty string."
            if proto.name not in existing_variable_names:
                OnnxVariable(owning_graph=self, proto=proto)
                existing_variable_names.add(proto.name)
        # 1-2) variables from node_input/node_output
        for proto in self._proto.node:
            input_num = len(proto.input)
            for idx, name in enumerate(itertools.chain(proto.input, proto.output)):
                is_dummy = False
                # empty-string name for node_input/output means optional input/output
                if name == "":
                    is_dummy = True
                    name = self.owning_model._get_next_unique_name(
                        namespace="variable",
                        existing_names=existing_variable_names,
                        name_prefix="dummy_variable",
                    )
                    # write unique name to proto for dummy variable
                    if idx < input_num:
                        proto.input[idx] = name
                    else:
                        proto.output[idx - input_num] = name
                if name not in existing_variable_names:
                    OnnxVariable(owning_graph=self, name=name, is_dummy=is_dummy)
                    existing_variable_names.add(name)

        # cache the non-attribute variables to save time cost
        variable_mappings = self.variable_mappings

        # 2) second, create all the nodes
        for proto in self._proto.node:
            if proto.op_type == "HzCalibration":
                CalibrationNode(
                    owning_graph=self, proto=proto, variable_mappings=variable_mappings
                ).append_on()
            else:
                OnnxNode(
                    owning_graph=self, proto=proto, variable_mappings=variable_mappings
                ).append_on()
            # write empty-string name back to proto for dummy variable
            input_num = len(proto.input)
            for idx, name in enumerate(itertools.chain(proto.input, proto.output)):
                if variable_mappings[name].is_dummy:
                    if idx < input_num:
                        proto.input[idx] = ""
                    else:
                        proto.output[idx - input_num] = ""

        # 3) third, create all the inputs/outputs
        for proto in self._proto.input:
            # only append input which is not model parameter
            if not variable_mappings[proto.name].is_param:
                self.append_input(variable_mappings[proto.name])
        for proto in self._proto.output:
            self.append_output(variable_mappings[proto.name])

    def _share_variable_among_graphs(
        self, outer_variables: Optional[Mapping[str, OnnxVariable]] = None
    ) -> None:
        """令上下层级graph间符合规则的同名variable共享同一对象."""
        if outer_variables is None:
            outer_variables = self.variable_mappings
        else:
            outer_variables = MappingProxyType(
                {
                    **outer_variables,
                    **self.variable_mappings,
                }
            )
        for onnx_graph in self.graphs:
            for onnx_var in onnx_graph.variables:
                # 仅对符合下述条件的variable允许在不同graph间共享
                if (
                    onnx_var.name in outer_variables
                    and not onnx_var.is_param
                    and not onnx_var.is_dummy
                    and onnx_var.src_op is None
                ):
                    assert onnx_var not in onnx_graph._inputs, (
                        "var to share from outer graph should not exist in "
                        "inner graph inputs."
                    )
                    assert onnx_var not in onnx_graph._outputs, (
                        "var to share from outer graph should not exist in "
                        "inner graph outputs."
                    )
                    # replace the uses of onnx_var with outer_var
                    onnx_var.replace_all_uses_with(outer_variables[onnx_var.name])

            # share variables among graph recursively
            onnx_graph._share_variable_among_graphs(outer_variables)

    @property
    def owning_model(self) -> "OnnxModel":
        """返回当前graph所属的model.

        属性类型:
            读写属性, 允许原地修改
        """
        return self._owning_model

    @property
    def owning_graph(self) -> Union["OnnxGraph", None]:
        """返回当前graph直接所属的graph.

        属性类型:
            读写属性, 允许原地修改
        """
        return self._owning_graph

    @property
    def proto(self) -> GraphProto:
        """返回基于当前graph导出的GraphProto对象.

        属性类型:
            只读属性
        """
        # sync from OnnxModel to GraphProto
        self._sync()
        return self._proto

    @property
    def name(self) -> str:
        """返回当前graph的name.

        属性类型:
            读写属性, 允许赋值修改
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if len(name) > 0:
            self._name = name
        else:
            raise ValueError("graph name should be non-empty string.")

    @property
    def is_main(self) -> bool:
        """返回当前graph是否为所属model的main graph.

        属性类型:
            只读属性
        """
        return self.owning_graph is None

    @property
    def graphs(self) -> AbstractSet["OnnxGraph"]:
        """返回当前graph的直属graph集合.

        属性类型:
            读写属性, 允许原地修改graph, 不可增加或者删除graph
        """
        graphs: Set["OnnxGraph"] = set()
        for onnx_graph in self.owning_model._graphs:
            if onnx_graph.owning_graph is self:
                graphs.add(onnx_graph)

        return frozenset(graphs)

    @property
    def graph_names(self) -> AbstractSet[str]:
        """返回当前graph的直属graph name集合.

        属性类型:
            只读属性
        """
        graph_names: Set[str] = set()
        for onnx_graph in self.owning_model._graphs:
            if onnx_graph.owning_graph is self:
                graph_names.add(onnx_graph.name)

        return frozenset(graph_names)

    @property
    def graph_mappings(self) -> Mapping[str, "OnnxGraph"]:
        """返回当前graph的直属graph name到graph映射.

        属性类型:
            读写属性, 允许原地修改graph, 不可增加或者删除graph
        """
        graph_mappings: Dict[str, "OnnxGraph"] = {}
        for onnx_graph in self.owning_model._graphs:
            if onnx_graph.owning_graph is self:
                graph_mappings[onnx_graph.name] = onnx_graph

        return MappingProxyType(graph_mappings)

    @property
    def inputs(self) -> Tuple[OnnxVariable, ...]:
        """返回当前graph的输入variable序列.

        属性类型:
            读写属性, 允许原地修改variable, 不可增加或者删除variable
        """
        return tuple(self._inputs)

    @property
    def input_names(self) -> Tuple[str, ...]:
        """返回当前graph的输入variable name序列.

        属性类型:
            只读属性
        """
        input_names: List[str] = [input_var.name for input_var in self._inputs]

        return tuple(input_names)

    @property
    def input_dtypes(self) -> Mapping[str, DataType]:
        """返回当前graph的输入variable name到variable dtype映射.

        属性类型:
            只读属性
        """
        input_dtypes: Dict[str, DataType] = {}
        for input_var in self._inputs:
            assert (
                input_var.dtype is not None
            ), "dtype for graph input could not be None."
            input_dtypes[input_var.name] = input_var.dtype

        return MappingProxyType(input_dtypes)

    @property
    def input_shapes(
        self,
    ) -> Mapping[str, Tuple[Union[None, str, int], ...]]:
        """返回当前graph的输入variable name到variable shape映射.

        属性类型:
            只读属性
        """
        input_shapes: Dict[str, Tuple[Union[None, str, int], ...]] = {}
        for input_var in self._inputs:
            assert (
                input_var.shape is not None
            ), "shape for graph input could not be None."
            input_shapes[input_var.name] = tuple(input_var.shape)

        return MappingProxyType(input_shapes)

    @property
    def input_mappings(self) -> Mapping[str, OnnxVariable]:
        """返回当前graph的输入variable name到variable映射.

        属性类型:
            读写属性, 允许原地修改variable, 不可增加或者删除variable
        """
        input_mappings: Dict[str, OnnxVariable] = {}
        for input_var in self._inputs:
            input_mappings[input_var.name] = input_var

        return MappingProxyType(input_mappings)

    @property
    def outputs(self) -> Tuple[OnnxVariable, ...]:
        """返回当前graph的输出variable序列.

        属性类型:
            读写属性, 允许原地修改variable, 不可增加或者删除variable
        """
        return tuple(self._outputs)

    @property
    def output_names(self) -> Tuple[str, ...]:
        """返回当前graph的输出variable name序列.

        属性类型:
            只读属性
        """
        output_names: List[str] = [output_var.name for output_var in self._outputs]

        return tuple(output_names)

    @property
    def output_dtypes(self) -> Mapping[str, Union[None, DataType]]:
        """返回当前graph的输出variable name到variable dtype映射.

        属性类型:
            只读属性
        """
        output_dtypes: Dict[str, Union[None, DataType]] = {}
        for output_var in self._outputs:
            output_dtypes[output_var.name] = output_var.dtype

        return MappingProxyType(output_dtypes)

    @property
    def output_shapes(
        self,
    ) -> Mapping[str, Union[None, Tuple[Union[None, str, int], ...]]]:
        """返回当前graph的输出variable name到variable shape映射.

        属性类型:
            只读属性
        """
        output_shapes: Dict[str, Union[None, Tuple[Union[None, str, int], ...]]] = {}
        for output_var in self._outputs:
            output_shapes[output_var.name] = (
                output_var.shape
                if output_var.shape is None
                else tuple(output_var.shape)
            )

        return MappingProxyType(output_shapes)

    @property
    def output_mappings(self) -> Mapping[str, OnnxVariable]:
        """返回当前graph的输出variable name到variable映射.

        属性类型:
            读写属性, 允许原地修改variable, 不可增加或者删除variable
        """
        output_mappings: Dict[str, OnnxVariable] = {}
        for output_var in self._outputs:
            output_mappings[output_var.name] = output_var

        return MappingProxyType(output_mappings)

    @property
    def nodes(self) -> Tuple[OnnxNode, ...]:
        """返回当前graph的node序列.

        node序列保证满足拓扑序

        属性类型:
            读写属性, 允许原地修改node, 不可增加或者删除node
        """
        return tuple(self._nodes)

    @property
    def node_names(self) -> Tuple[str, ...]:
        """返回当前graph的node name序列.

        node name序列保证满足拓扑序

        属性类型:
            只读属性
        """
        node_names: List[str] = [node.name for node in self._nodes]

        return tuple(node_names)

    @property
    def node_mappings(self) -> Mapping[str, OnnxNode]:
        """返回当前graph的node name到node映射.

        node name到node映射保证满足拓扑序

        属性类型:
            读写属性, 允许原地修改node, 不可增加或删除node
        """
        node_mappings: Dict[str, OnnxNode] = {}
        for onnx_node in self._nodes:
            node_mappings[onnx_node.name] = onnx_node

        return MappingProxyType(node_mappings)

    @property
    def type2nodes(self) -> Mapping[str, AbstractSet[OnnxNode]]:
        """返回当前graph的op_type到node集合的映射.

        属性类型:
            读写属性, 允许原地修改node, 不可增加或删除node
        """
        type2nodes: Dict[str, Set[OnnxNode]] = defaultdict(set)
        for onnx_node in self._nodes:
            type2nodes[onnx_node.op_type].add(onnx_node)

        return MappingProxyType(
            defaultdict(
                frozenset,
                {op_type: frozenset(type2nodes[op_type]) for op_type in type2nodes},
            )
        )

    @property
    def variables(self) -> AbstractSet[OnnxVariable]:
        """返回当前graph的variable集合.

        属性类型:
            读写属性, 允许原地修改variable, 不可增加或删除variable
        """
        return frozenset(self._variables)

    @property
    def variable_names(self) -> AbstractSet[str]:
        """返回当前graph的variable name集合.

        属性类型:
            只读属性
        """
        variable_names: Set[str] = {onnx_var.name for onnx_var in self._variables}

        return frozenset(variable_names)

    @property
    def variable_mappings(self) -> Mapping[str, OnnxVariable]:
        """返回当前graph的variable name到variable映射.

        属性类型:
            读写属性, 允许原地修改variable, 不可增加或删除variable
        """
        variable_mappings: Dict[str, OnnxVariable] = {}
        for onnx_var in self._variables:
            variable_mappings[onnx_var.name] = onnx_var

        return MappingProxyType(variable_mappings)

    def copy_from(
        self,
        variables: Optional[Iterable[OnnxVariable]] = None,
        graphs: Optional[Iterable["OnnxGraph"]] = None,
        nodes: Optional[Iterable[OnnxNode]] = None,
        inputs: Optional[Iterable[OnnxVariable]] = None,
        outputs: Optional[Iterable[OnnxVariable]] = None,
    ) -> "OnnxGraph":
        """将给定OnnxGraph构成元素拷贝到当前graph中."""
        # copy graph variables
        if variables is not None:
            for onnx_var in variables:
                self.create_variable(
                    name=onnx_var.name,
                    dtype=onnx_var.dtype,
                    shape=onnx_var.shape,
                    value=onnx_var.value,
                    is_param=onnx_var.is_param,
                    is_dummy=onnx_var.is_dummy,
                    is_attr=onnx_var.is_attr,
                )
        # copy graph sub-graphs
        if graphs is not None:
            for onnx_graph in graphs:
                self.create_graph(name=onnx_graph.name).copy_from(
                    variables=onnx_graph.variables,
                    graphs=onnx_graph.graphs,
                    nodes=onnx_graph.nodes,
                    inputs=onnx_graph.inputs,
                    outputs=onnx_graph.outputs,
                )
        # copy graph nodes
        if nodes is not None:
            for onnx_node in nodes:
                attributes: Dict[str, Any] = {}
                for attr_name, attr_val in onnx_node.attributes.items():
                    attr_vec = attr_val if isinstance(attr_val, list) else [attr_val]
                    for item_idx, item_val in enumerate(attr_vec):
                        if isinstance(item_val, OnnxVariable):
                            attr_vec[item_idx] = self.variable_mappings[item_val.name]
                        elif isinstance(item_val, OnnxGraph):
                            attr_vec[item_idx] = self.graph_mappings[item_val.name]
                    attributes[attr_name] = (
                        attr_vec if isinstance(attr_val, list) else attr_vec[0]
                    )
                self.create_node(
                    op_type=onnx_node.op_type,
                    domain=onnx_node.domain,
                    name=onnx_node.name,
                    attributes=attributes,
                    inputs=onnx_node.input_names,
                    outputs=onnx_node.output_names,
                ).append_on()
        # copy graph inputs
        if inputs is not None:
            for input_var in inputs:
                self.append_input(input_var.name)
        # copy graph outputs
        if outputs is not None:
            for output_var in outputs:
                self.append_output(output_var.name)

        return self

    def create_graph(
        self,
        name: Optional[str] = None,
    ) -> "OnnxGraph":
        """根据给定参数创建OnnxGraph实例并将其添加到当前graph的graph集合."""
        return OnnxGraph(owning_model=self.owning_model, owning_graph=self, name=name)

    def create_variable(
        self,
        name: Optional[str] = None,
        dtype: Optional[DataType] = None,
        shape: Optional[Iterable[Union[None, str, int]]] = None,
        value: Optional[Union["np.ndarray", "torch.Tensor"]] = None,
        is_param: bool = False,
        is_dummy: bool = False,
        is_attr: bool = False,
    ) -> OnnxVariable:
        """根据给定参数创建OnnxVariable实例并将其添加到当前graph的variable集合."""
        return OnnxVariable(
            owning_graph=self,
            name=name,
            dtype=dtype,
            shape=shape,
            value=value,
            is_param=is_param,
            is_dummy=is_dummy,
            is_attr=is_attr,
        )

    @overload
    def create_node(
        self,
        op_type: str,
        domain: Optional[str] = None,
        name: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        inputs: Optional[Iterable[Union[str, OnnxVariable]]] = None,
        *,
        outputs: Optional[Iterable[Union[str, OnnxVariable]]] = None,
    ) -> OnnxNode: ...

    @overload
    def create_node(
        self,
        op_type: str,
        domain: Optional[str] = None,
        name: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        inputs: Optional[Iterable[Union[str, OnnxVariable]]] = None,
        *,
        num_outputs: Optional[int] = None,
    ) -> OnnxNode: ...

    def create_node(
        self,
        op_type: str,
        domain: Optional[str] = None,
        name: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        inputs: Optional[Iterable[Union[str, OnnxVariable]]] = None,
        outputs: Optional[Iterable[Union[str, OnnxVariable]]] = None,
        num_outputs: Optional[int] = None,
    ) -> OnnxNode:
        """根据给定参数创建OnnxNode实例.

        输入参数outputs用于指定node输出variable对象,
        输入参数num_outputs用于指定node输出variable数量,
        但是两者不可同时给出
        """
        node_cls = CalibrationNode if op_type == "HzCalibration" else OnnxNode
        return node_cls(
            owning_graph=self,
            op_type=op_type,
            domain=domain,
            name=name,
            attributes=attributes,
            inputs=inputs,
            outputs=outputs,
            num_outputs=num_outputs,
        )

    @overload
    def append_input(self, onnx_var: Union[str, OnnxVariable]) -> "OnnxGraph": ...

    @overload
    def append_input(
        self,
        *,
        name: Optional[str] = None,
        dtype: Optional[DataType] = None,
        shape: Optional[Iterable[Union[None, str, int]]] = None,
    ) -> "OnnxGraph": ...

    def append_input(
        self,
        onnx_var: Optional[Union[str, OnnxVariable]] = None,
        name: Optional[str] = None,
        dtype: Optional[DataType] = None,
        shape: Optional[Iterable[Union[None, str, int]]] = None,
    ) -> "OnnxGraph":
        """根据给定参数找到或者创建相应variable并将其添加到当前graph输入序列的末尾."""
        # We now only allow to append input for main graph
        assert self.is_main, "Only allow to append input for main graph."
        if onnx_var is None:
            onnx_var = self.create_variable(name=name, dtype=dtype, shape=shape)
        else:
            onnx_var = (
                self.variable_mappings[onnx_var]
                if isinstance(onnx_var, str)
                else onnx_var
            )
        # validity check for given or created onnx variable
        assert (
            onnx_var not in self._inputs
        ), f"variable {onnx_var.name} already exists in graph inputs."
        assert onnx_var.owning_graph is self, (
            f"The owning graph of variable {onnx_var.name} "
            f"and this graph are different."
        )
        assert onnx_var.src_op is None, (
            f"variable {onnx_var.name} already has src_op: "
            f"{onnx_var.src_op.name}, illegal to specify it as graph input."
        )
        assert not onnx_var.is_param, "model parameter could not be graph input."
        assert not onnx_var.is_dummy, "dummy variable could not be graph input."
        assert onnx_var.dtype is not None, "dtype for graph input could not be None."
        assert onnx_var.shape is not None, "shape for graph input could not be None."

        # append legal onnx_var to graph input
        self._inputs.append(onnx_var)

        return self

    def extend_inputs(
        self, onnx_vars: Iterable[Union[str, OnnxVariable]]
    ) -> "OnnxGraph":
        """将给定variables扩充到当前graph输入序列的末尾."""
        for onnx_var in onnx_vars:
            self.append_input(onnx_var)

        return self

    def append_output(self, onnx_var: Union[str, OnnxVariable]) -> "OnnxGraph":
        """将给定variable添加到当前graph输出序列的末尾."""
        if isinstance(onnx_var, str):
            onnx_var = self.variable_mappings[onnx_var]
        # validity check for given onnx variable
        assert (
            onnx_var not in self._outputs
        ), f"variable {onnx_var.name} already exists in graph outputs."
        assert onnx_var.owning_graph is self, (
            f"The owning graph of variable {onnx_var.name} "
            f"and this graph are different."
        )
        assert not onnx_var.is_dummy, "dummy variable could not be graph output."
        assert not onnx_var.is_attr, "attr variable could not be graph output."

        # append valid onnx_var to graph output
        self._outputs.append(onnx_var)

        return self

    def extend_outputs(
        self,
        onnx_vars: Iterable[Union[str, OnnxVariable]],
    ) -> "OnnxGraph":
        """将给定variables扩充到当前graph输出序列的末尾."""
        for onnx_var in onnx_vars:
            self.append_output(onnx_var)

        return self

    def remove_output(self, idx_or_var: Union[int, str, OnnxVariable]) -> "OnnxGraph":
        """将通过索引或者OnnxVariable对象指定的variable从当前graph输出序列中删除.

        支持索引值为负数
        """
        if isinstance(idx_or_var, int):
            onnx_var = self._outputs[idx_or_var]
        elif isinstance(idx_or_var, str):
            onnx_var = self.output_mappings[idx_or_var]
        else:
            onnx_var = idx_or_var

        # remove given onnx_ver from graph outputs
        self._outputs.remove(onnx_var)

        if not onnx_var.is_used and onnx_var.src_op is None:
            onnx_var._destroy()

        return self

    def clear_outputs(self) -> "OnnxGraph":
        """将当前graph的输出序列清空."""
        for output_idx in reversed(range(len(self._outputs))):
            self.remove_output(output_idx)

        return self

    def __repr__(self) -> str:
        """返回当前graph的字符串表示."""
        graph_str: List[str] = []
        graph_str.extend(
            (
                "OnnxGraph:",
                2 * " " + f"name: {self.name}",
            )
        )
        graph_str.append(2 * " " + "inputs: name, shape, dtype")
        graph_str.extend(4 * " " + f"{input_var}" for input_var in self._inputs)
        graph_str.append(2 * " " + "outputs: name, shape, dtype")
        graph_str.extend(4 * " " + f"{output_var}" for output_var in self._outputs)
        graph_str.append(
            2 * " " + "nodes: op_type, name, [input_names], [output_names]"
        )
        graph_str.extend(4 * " " + f"{onnx_node}" for onnx_node in self._nodes)

        return "\n".join(graph_str)

    def _sort_topologically(self) -> None:
        """对当前graph的node序列原地进行拓扑排序.

        同时对属于当前graph的子图递归进行拓扑排序
        """
        # first, sort the sub-graphs recursively
        for onnx_graph in self.graphs:
            onnx_graph._sort_topologically()
        # then, sort the current graph
        visited = dict.fromkeys(self._nodes, False)
        sorted_nodes: List[OnnxNode] = []
        to_pop_nodes: Deque[OnnxNode] = deque()
        node2indegree = {node: len(node.prev_ops) for node in self._nodes}

        # initialization
        for node, indegree in node2indegree.items():
            if indegree == 0:
                to_pop_nodes.append(node)

        # topological sort
        while to_pop_nodes:
            node = to_pop_nodes.popleft()
            for next_op in node.next_ops:
                node2indegree[next_op] -= 1
                if node2indegree[next_op] == 0:
                    to_pop_nodes.append(next_op)
            visited[node] = True
            sorted_nodes.append(node)

        assert all(visited.values()), (
            "Topological Sort failed. Some operation can not be sorted "
            "(might due to circular reference):\n"
            "\n".join(str(node) for node in visited if visited[node] is False)
        )

        # replace with topological sorted nodes
        self._nodes = sorted_nodes

    def _destroy(self) -> None:
        """将当前graph从所属model中清除."""
        assert not self.is_main, "main graph could not be destroyed."
        self.owning_model._graphs.remove(self)
        # 当前graph可能会使用到所属graph中的variables,
        # 这里删除graph时需要同时删除这些引用
        for onnx_node in self._nodes:
            for input_idx, input_var in enumerate(onnx_node._inputs):
                if input_var.owning_graph is not self:
                    input_var._dest_infos.remove(
                        DestInfoTuple(dest_op=onnx_node, dest_idx=input_idx)
                    )

    def _sync(self) -> None:
        # sync for name
        self._proto.name = self.name

        # sync for node
        del self._proto.node[:]
        for onnx_node in self._nodes:
            self._proto.node.append(onnx_node.proto)

        # sync for input
        del self._proto.input[:]
        for input_var in self._inputs:
            assert not input_var.is_param, "model param could not be graph input."
            assert not input_var.is_dummy, "dummy variable could not be graph input."
            assert (
                input_var.proto is not None
            ), f"The graph input {input_var.name} lacks both shape and dtype."
            self._proto.input.append(input_var.proto)

        # sync for output
        del self._proto.output[:]
        for output_var in self._outputs:
            if output_var.is_param:
                assert (
                    not output_var.is_attr
                ), "attr variable could not be graph output."
                # case for graph outputs of const value
                self._proto.output.append(
                    make_tensor_value_info(
                        name=output_var.name,
                        dtype=output_var.dtype,
                        shape=output_var.shape,
                    )
                )
            else:
                assert (
                    not output_var.is_dummy
                ), "dummy variable could not be graph output."
                # case for graph outputs of non-const value
                if output_var.proto is None:
                    self._proto.output.append(
                        make_tensor_value_info(name=output_var.name)
                    )
                else:
                    self._proto.output.append(output_var.proto)

        # sync for initializer && value_info
        del self._proto.initializer[:]
        del self._proto.value_info[:]
        for onnx_var in self._variables:
            if onnx_var.is_param:
                if not onnx_var.is_attr:
                    self._proto.initializer.append(onnx_var.proto)
            else:
                if (
                    onnx_var.name not in self.input_names
                    and onnx_var.name not in self.output_names
                    and onnx_var.proto is not None
                ):
                    assert (
                        not onnx_var.is_dummy
                    ), "dummy variable has dtype or shape, it's amazing!"
                    self._proto.value_info.append(onnx_var.proto)
