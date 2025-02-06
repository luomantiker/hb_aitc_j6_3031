from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    overload,
)

from .onnx_utils import NodeProto
from .utils import DataType, DestInfoTuple, get_attribute, make_attribute

if TYPE_CHECKING:
    from .onnx_graph import OnnxGraph
    from .onnx_model import OnnxModel
    from .onnx_variable import OnnxVariable


class OnnxNode:
    _owning_graph: "OnnxGraph"
    _proto: NodeProto
    _op_type: str
    _domain: str
    _name: str
    _attributes: Dict[str, Any]
    _inputs: List["OnnxVariable"]
    _outputs: List["OnnxVariable"]

    @overload
    def __init__(
        self,
        owning_graph: "OnnxGraph",
        proto: NodeProto,
        variable_mappings: Mapping[str, "OnnxVariable"],
    ) -> None: ...

    @overload
    def __init__(
        self,
        owning_graph: "OnnxGraph",
        *,
        op_type: str,
        domain: Optional[str] = None,
        name: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        inputs: Optional[Iterable[Union[str, "OnnxVariable"]]] = None,
        outputs: Optional[Iterable[Union[str, "OnnxVariable"]]] = None,
        num_outputs: Optional[int] = None,
    ) -> None: ...

    def __init__(
        self,
        owning_graph: "OnnxGraph",
        proto: Optional[NodeProto] = None,
        variable_mappings: Optional[Mapping[str, "OnnxVariable"]] = None,
        op_type: Optional[str] = None,
        domain: Optional[str] = None,
        name: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        inputs: Optional[Iterable[Union[str, "OnnxVariable"]]] = None,
        outputs: Optional[Iterable[Union[str, "OnnxVariable"]]] = None,
        num_outputs: Optional[int] = None,
    ):
        """初始化OnnxNode对象."""
        super().__init__()

        # initialize owning_graph for onnx node
        self._owning_graph = owning_graph

        # init from proto or config
        if proto is None:
            assert op_type is not None, "op_type should be given if init_from_config."
            self._init_from_config(
                op_type=op_type,
                domain=domain,
                name=name,
                attributes=attributes,
                inputs=inputs,
                outputs=outputs,
                num_outputs=num_outputs,
            )
        else:
            assert (
                variable_mappings is not None
            ), "variable_mappings should be given if init_from_proto."
            self._init_from_proto(proto=proto, variable_mappings=variable_mappings)

    def _init_from_config(
        self,
        op_type: str,
        domain: Optional[str] = None,
        name: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        inputs: Optional[Iterable[Union[str, "OnnxVariable"]]] = None,
        outputs: Optional[Iterable[Union[str, "OnnxVariable"]]] = None,
        num_outputs: Optional[int] = None,
    ) -> None:
        """基于参数配置初始化OnnxNode对象."""
        self._proto = NodeProto()
        # initialize op_type && domain for onnx node
        self._op_type = op_type
        self._domain = "" if domain is None else domain
        # resolve the node name
        if name is None:
            name = self.owning_model._get_next_unique_name(
                namespace="node", name_prefix=op_type
            )
        assert len(name) > 0, "node name should be non-empty string."
        self._name = name
        # initialize attributes for onnx node
        self._attributes = {}
        if attributes is not None:
            for attr_name, attr_val in attributes.items():
                self.set_attribute(attr_name=attr_name, attr_val=attr_val)
        # initialize inputs for onnx node
        self._inputs = []
        if inputs is not None:
            self.extend_inputs(inputs)
        # initialize outputs for onnx node
        self._outputs = []
        assert (
            outputs is None or num_outputs is None
        ), "outputs and num_outputs could not be given simultaneously."
        if outputs is not None:
            self.extend_outputs(outputs)
        if num_outputs is not None:
            for _ in range(num_outputs):
                self.append_output()

    def _init_from_proto(
        self, proto: NodeProto, variable_mappings: Mapping[str, "OnnxVariable"]
    ) -> None:
        """基于NodeProto初始化OnnxNode对象."""
        self._proto = proto
        # initialize op_type && name && domain && attribute for onnx node
        self._op_type = self._proto.op_type
        self._domain = self._proto.domain
        self._name = self._proto.name
        self._attributes = {
            proto.name: get_attribute(
                proto=proto,
                owning_graph=self._owning_graph,
            )
            for proto in self._proto.attribute
        }
        # initialize inputs for onnx node
        self._inputs = []
        for input_name in self._proto.input:
            self.append_input(variable_mappings[input_name])
        # initialize outputs for onnx node
        self._outputs = []
        for output_name in self._proto.output:
            self.append_output(variable_mappings[output_name])

    @property
    def owning_model(self) -> "OnnxModel":
        """返回当前node所属的model.

        属性类型:
            读写属性, 允许原地修改
        """
        return self.owning_graph.owning_model

    @property
    def owning_graph(self) -> "OnnxGraph":
        """返回当前node直接所属的graph.

        属性类型:
            读写属性, 允许原地修改
        """
        return self._owning_graph

    @property
    def proto(self) -> NodeProto:
        """返回基于当前node导出的NodeProto对象.

        属性类型:
            只读属性
        """
        # sync from OnnxNode to NodeProto
        self._sync()
        return self._proto

    @property
    def op_type(self) -> str:
        """返回当前node的op_type.

        属性类型:
            读写属性, 允许赋值修改
        """
        return self._op_type

    @op_type.setter
    def op_type(self, op_type: str) -> None:
        self._op_type = op_type

    @property
    def domain(self) -> str:
        """返回当前node所属opset的domain.

        属性类型:
            读写属性, 允许赋值修改
        """
        return self._domain

    @domain.setter
    def domain(self, domain: str) -> None:
        self._domain = domain

    @property
    def name(self) -> str:
        """返回当前node的name.

        属性类型:
            读写属性, 允许赋值修改
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if len(name) > 0:
            self._name = name
        else:
            raise ValueError("node name should be non-empty string.")

    @property
    def attributes(self) -> Mapping[str, Any]:
        """返回当前node的属性.

        返回映射中key为属性名, value为属性值

        属性类型:
            读写属性, 允许原地修改属性, 不可增加或者删除属性
        """
        return MappingProxyType(self._attributes)

    @property
    def inputs(self) -> Tuple["OnnxVariable", ...]:
        """返回当前node的输入variable序列.

        属性类型:
            读写属性, 允许原地修改variable, 不可增加或者删除variable
        """
        return tuple(self._inputs)

    @property
    def input_names(self) -> Tuple[str, ...]:
        """返回当前node的输入variable name序列.

        属性类型:
            只读属性
        """
        input_names: List[str] = [input_var.name for input_var in self._inputs]

        return tuple(input_names)

    @property
    def input_dtypes(self) -> Tuple[Union[None, DataType], ...]:
        """返回当前node的输入variable dtype序列.

        属性类型:
            只读属性
        """
        input_dtypes: List[Union[None, DataType]] = [
            input_var.dtype for input_var in self._inputs
        ]

        return tuple(input_dtypes)

    @property
    def input_shapes(
        self,
    ) -> Tuple[Union[None, Tuple[Union[None, str, int], ...]], ...]:
        """返回当前node的输入variable shape序列.

        属性类型:
            只读属性
        """
        input_shapes: List[Union[None, Tuple[Union[None, str, int], ...]]] = [
            input_var.shape if input_var.shape is None else tuple(input_var.shape)
            for input_var in self._inputs
        ]

        return tuple(input_shapes)

    @property
    def outputs(self) -> Tuple["OnnxVariable", ...]:
        """返回当前node的输出variable序列.

        属性类型:
            读写属性, 允许原地修改variable, 不可增加或者删除variable
        """
        return tuple(self._outputs)

    @property
    def output_names(self) -> Tuple[str, ...]:
        """返回当前node的输出variable name序列.

        属性类型:
            只读属性
        """
        output_names: List[str] = [output_var.name for output_var in self._outputs]

        return tuple(output_names)

    @property
    def output_dtypes(self) -> Tuple[Union[None, DataType], ...]:
        """返回当前node的输出variable dtype序列.

        属性类型:
            只读属性
        """
        output_dtypes: List[Union[None, DataType]] = [
            output_var.dtype for output_var in self._outputs
        ]

        return tuple(output_dtypes)

    @property
    def output_shapes(
        self,
    ) -> Tuple[Union[None, Tuple[Union[None, str, int], ...]], ...]:
        """返回当前node的输出variable shape序列.

        属性类型:
            只读属性
        """
        output_shapes: List[Union[None, Tuple[Union[None, str, int], ...]]] = [
            output_var.shape if output_var.shape is None else tuple(output_var.shape)
            for output_var in self._outputs
        ]

        return tuple(output_shapes)

    @property
    def prev_ops(self) -> Tuple["OnnxNode", ...]:
        """返回当前node在直接所属graph中的前序node序列.

        属性类型:
            读写属性, 允许原地修改node, 不可增加或删除node
        """
        prev_ops: List["OnnxNode"] = []
        for input_var in self._inputs:
            if (
                input_var.owning_graph is self.owning_graph
                and input_var.src_op is not None
                and input_var.src_op not in prev_ops
            ):
                prev_ops.append(input_var.src_op)

        return tuple(prev_ops)

    @property
    def prev_op(self) -> "OnnxNode":
        """返回当前node在直接所属graph中的前序node.

        若当前node存在多个前序node, 则抛出异常

        属性类型:
            读写属性, 允许原地修改node
        """
        prev_ops = self.prev_ops
        assert len(prev_ops) == 1, f"node {self.name} has more than 1 prev_ops."

        return prev_ops[0]

    @property
    def next_ops(self) -> Tuple["OnnxNode", ...]:
        """返回当前node在直接所属graph中的后继node序列.

        属性类型:
            读写属性, 允许原地修改node, 不可增加或删除node
        """
        next_ops: List["OnnxNode"] = []
        for output_var in self._outputs:
            for dest_op in output_var.dest_ops:
                if dest_op.owning_graph is self.owning_graph:
                    if dest_op not in next_ops:
                        next_ops.append(dest_op)
                else:
                    # 首先, 我们找到直接隶属于当前node所属graph并且包含dest_op的graph
                    direct_subgraph = dest_op.owning_graph
                    while direct_subgraph.owning_graph is not self.owning_graph:
                        assert direct_subgraph.owning_graph is not None
                        direct_subgraph = direct_subgraph.owning_graph
                    # 然后, 我们确定该graph是当前node所属graph下哪个node的属性,
                    # 从而将该node添加到next_ops中
                    for onnx_node in self.owning_graph._nodes:
                        if any(
                            direct_subgraph in attr_val
                            if isinstance(attr_val, list)
                            else direct_subgraph is attr_val
                            for attr_val in onnx_node._attributes.values()
                        ):
                            if onnx_node not in next_ops:
                                next_ops.append(onnx_node)
                            break

        return tuple(next_ops)

    @property
    def next_op(self) -> "OnnxNode":
        """返回当前node在直接所属graph中的后继node.

        若当前node存在多个后继node, 则抛出异常

        属性类型:
            读写属性, 允许原地修改node
        """
        next_ops = self.next_ops
        assert len(next_ops) == 1, f"node {self.name} has more than 1 next_ops."

        return next_ops[0]

    @property
    def is_used(self) -> bool:
        """返回当前node是否有任意输出variable被使用.

        variable`被使用`的含义包括作为node输入或者graph输出

        属性类型:
            只读属性
        """
        return any(output_var.is_used for output_var in self._outputs)

    def set_attribute(self, attr_name: str, attr_val: Any) -> "OnnxNode":
        """根据给定键值对设置当前node的属性."""
        # lazy import to avoid circular imports
        from .onnx_graph import OnnxGraph
        from .onnx_variable import OnnxVariable

        # validity check for attribute value
        attr_vec = attr_val if isinstance(attr_val, list) else [attr_val]
        for item_val in attr_vec:
            if isinstance(item_val, OnnxVariable):
                assert item_val.is_attr, f"variable {item_val.name} is not a attribute."
                assert (
                    not item_val.is_used and item_val.src_op is None
                ), "attr variable should have no producer and consumer."
                assert item_val.owning_graph is self.owning_graph, (
                    f"The owning graph of the node {self._name} "
                    f"and attribute variable {item_val.name} are different."
                )
            if isinstance(item_val, OnnxGraph):
                assert not item_val.is_main, "attr graph should not be main graph."
                assert item_val.owning_graph is self.owning_graph, (
                    f"The owning graph of the node {self._name} "
                    f"and attr graph {item_val.name} are different."
                )

        self._attributes[attr_name] = attr_val

        return self

    def remove_attribute(self, attr_name: str) -> "OnnxNode":
        """将当前node中给定name的属性删除."""
        # lazy import to avoid circular imports
        from .onnx_graph import OnnxGraph
        from .onnx_variable import OnnxVariable

        attr_val = self._attributes.pop(attr_name)
        attr_vec = attr_val if isinstance(attr_val, list) else [attr_val]
        for item_val in attr_vec:
            if isinstance(item_val, (OnnxVariable, OnnxGraph)):
                item_val._destroy()

        return self

    def clear_attributes(self) -> "OnnxNode":
        """将当前node的所有属性清空."""
        for attr_name in list(self._attributes.keys()):
            self.remove_attribute(attr_name=attr_name)

        return self

    def append_input(self, onnx_var: Union[str, "OnnxVariable"]) -> "OnnxNode":
        """将给定variable添加到当前node输入序列的末尾."""
        if isinstance(onnx_var, str):
            onnx_var = self.owning_model.variable_mappings[onnx_var]
        # validity check for the given onnx_var
        assert not onnx_var.is_attr, "attribute variable could not be node input."

        onnx_var._dest_infos.append(
            DestInfoTuple(dest_op=self, dest_idx=len(self._inputs))
        )
        self._inputs.append(onnx_var)

        return self

    def extend_inputs(
        self, onnx_vars: Iterable[Union[str, "OnnxVariable"]]
    ) -> "OnnxNode":
        """将给定variable序列扩充到当前node输入序列的末尾."""
        for onnx_var in onnx_vars:
            self.append_input(onnx_var)

        return self

    def replace_input(
        self,
        idx_or_var: Union[int, str, "OnnxVariable"],
        fresh_var: Union[str, "OnnxVariable"],
    ) -> "OnnxNode":
        """将当前node中的指定输入variable替换为给定fresh_var.

        支持索引值为负数
        """
        if isinstance(fresh_var, str):
            fresh_var = self.owning_model.variable_mappings[fresh_var]
        # validity check for the given fresh_var
        assert not fresh_var.is_attr, "node input could not be attribute var."

        # collect the idxs of input variable to be replaced
        if isinstance(idx_or_var, int):
            # regularize input idx to non-negative
            idx_or_var = (
                idx_or_var if idx_or_var >= 0 else idx_or_var + len(self._inputs)
            )
            # return directly if idx_or_var and fresh_var point to the same variable
            if self._inputs[idx_or_var] is fresh_var:
                return self
            outdated_idxs = [idx_or_var]
        else:
            if isinstance(idx_or_var, str):
                idx_or_var = self.owning_model.variable_mappings[idx_or_var]
            # variable to be replaced should exist in node inputs
            assert idx_or_var in self._inputs, (
                f"variable {idx_or_var.name} does not exist in "
                f"node inputs, invalid to replace it."
            )
            # return directly if idx_or_var and fresh_var point to the same variable
            if idx_or_var is fresh_var:
                return self
            outdated_idxs = [
                input_idx
                for input_idx, input_var in enumerate(self._inputs)
                if idx_or_var is input_var
            ]

        for outdated_idx in outdated_idxs:
            # clear topological info for variable to be replaced
            outdated_var = self._inputs[outdated_idx]
            outdated_var._dest_infos.remove(
                DestInfoTuple(dest_op=self, dest_idx=outdated_idx)
            )
            # update topological info for node && variable to replace
            self._inputs[outdated_idx] = fresh_var
            fresh_var._dest_infos.append(
                DestInfoTuple(dest_op=self, dest_idx=outdated_idx)
            )

            if not outdated_var.is_used and outdated_var.src_op is None:
                outdated_var._destroy()

        return self

    def remove_input(self, idx_or_var: Union[int, str, "OnnxVariable"]) -> "OnnxNode":
        """将当前node中的指定输入variable移除.

        支持索引值为负数
        """
        # collect input idx of variables to be removed
        if isinstance(idx_or_var, int):
            # regularize input idx to non-negative
            removed_idx = (
                idx_or_var if idx_or_var >= 0 else idx_or_var + len(self._inputs)
            )
        else:
            if isinstance(idx_or_var, str):
                idx_or_var = self.owning_model.variable_mappings[idx_or_var]
            # variable to be removed should exist in node inputs
            assert idx_or_var in self._inputs, (
                f"variable {idx_or_var.name} does not exist in "
                f"node inputs, invalid to remove it."
            )
            while idx_or_var in self._inputs:
                removed_idx = self._inputs.index(idx_or_var)
                self.remove_input(removed_idx)
            return self

        # clear topological info for variable to be removed
        removed_var = self._inputs[removed_idx]
        removed_var._dest_infos.remove(
            DestInfoTuple(dest_op=self, dest_idx=removed_idx)
        )
        # update topological info for the following variables
        for input_idx in range(removed_idx + 1, len(self._inputs)):
            input_var = self._inputs[input_idx]
            input_var._dest_infos.remove(
                DestInfoTuple(dest_op=self, dest_idx=input_idx)
            )
            input_var._dest_infos.append(
                DestInfoTuple(dest_op=self, dest_idx=input_idx - 1)
            )
        # delete variables to be removed
        del self._inputs[removed_idx]

        if not removed_var.is_used and removed_var.src_op is None:
            removed_var._destroy()

        return self

    def clear_inputs(self) -> "OnnxNode":
        """将当前node的输入variable序列清空."""
        for input_idx in reversed(range(len(self._inputs))):
            self.remove_input(idx_or_var=input_idx)

        return self

    @overload
    def append_output(self, onnx_var: Union[str, "OnnxVariable"]) -> "OnnxNode": ...

    @overload
    def append_output(
        self,
        *,
        name: Optional[str] = None,
        dtype: Optional[DataType] = None,
        shape: Optional[Iterable[Union[None, str, int]]] = None,
    ) -> "OnnxNode": ...

    def append_output(
        self,
        onnx_var: Optional[Union[str, "OnnxVariable"]] = None,
        name: Optional[str] = None,
        dtype: Optional[DataType] = None,
        shape: Optional[Iterable[Union[None, str, int]]] = None,
    ) -> "OnnxNode":
        """根据给定参数找到或者创建相应variable并将其添加到当前node输入序列的末尾."""
        if onnx_var is None:
            onnx_var = self.owning_graph.create_variable(
                name=name,
                dtype=dtype,
                shape=shape,
            )
        else:
            onnx_var = (
                self.owning_graph.variable_mappings[onnx_var]
                if isinstance(onnx_var, str)
                else onnx_var
            )

        # validity check for given or created onnx variable
        assert (
            onnx_var not in self._outputs
        ), f"variable {onnx_var.name} already exists in node outputs."
        assert onnx_var.owning_graph is self.owning_graph, (
            f"The owning graph of variable {onnx_var.name} "
            f"and this node are different."
        )
        assert onnx_var.src_op is None, (
            f"variable {onnx_var.name} already has src_op: "
            f"{onnx_var.src_op.name}, illegal to specify it as node output."
        )
        assert not onnx_var.is_param, "model parameter could not be node output."

        onnx_var._src_op = self
        self._outputs.append(onnx_var)

        return self

    def extend_outputs(
        self, onnx_vars: Iterable[Union[str, "OnnxVariable"]]
    ) -> "OnnxNode":
        """将给定variables扩充到当前node输出序列的末尾."""
        for onnx_var in onnx_vars:
            self.append_output(onnx_var)

        return self

    def remove_output(self, idx_or_var: Union[int, str, "OnnxVariable"]) -> "OnnxNode":
        """将当前node中的指定输出variable移除.

        支持索引值为负数
        """
        # collect variable to be removed
        if isinstance(idx_or_var, int):
            removed_var = self._outputs[idx_or_var]
        else:
            if isinstance(idx_or_var, str):
                idx_or_var = self.owning_graph.variable_mappings[idx_or_var]
            # variable to be removed should exist in node outputs
            assert idx_or_var in self._outputs, (
                f"variable {idx_or_var.name} does not exist in "
                f"node outputs, invalid to remove it."
            )
            removed_var = idx_or_var

        assert not removed_var.is_used, (
            f"variable {removed_var.name} has been used as node inputs or "
            f"graph outputs, invalid to remove it from node outputs."
        )
        # delete variable from node outputs
        self._outputs.remove(removed_var)
        removed_var._src_op = None

        if not removed_var.is_used and removed_var.src_op is None:
            removed_var._destroy()

        return self

    def clear_outputs(self) -> "OnnxNode":
        """将当前node的输出variable序列清空."""
        for output_idx in reversed(range(len(self._outputs))):
            self.remove_output(output_idx)

        return self

    def replace_all_uses_with(self, onnx_node: Union[str, "OnnxNode"]) -> "OnnxNode":
        """将所有对当前node的使用替换为给定node.

        这里的`使用`的含义包括该node任意输出variable作为其他node输入或者graph输出
        """
        if isinstance(onnx_node, str):
            onnx_node = self.owning_model.node_mappings[onnx_node]
        # return directly if this node and given node point to the same object
        if self is onnx_node:
            return self
        # self and given node should have the same number of outputs
        assert len(self._outputs) == len(onnx_node._outputs), (
            f"node {self.name} has different number of outputs than "
            f"node {onnx_node.name}, invalid to replace it."
        )

        for output_idx in range(len(self._outputs)):
            self._outputs[output_idx].replace_all_uses_with(
                onnx_node._outputs[output_idx]
            )

        return self

    def prepend_on(self) -> "OnnxNode":
        """将当前node插入到直接所属graph的node序列开端."""
        # this node should not exist in graph
        assert (
            self not in self.owning_graph._nodes
        ), f"node {self.name} already exists in graph, invalid to insert it again."

        self.owning_graph._nodes.insert(0, self)

        return self

    def append_on(self) -> "OnnxNode":
        """将当前node插入到直接所属graph的node序列末尾."""
        # this node should not exist in graph
        assert (
            self not in self.owning_graph._nodes
        ), f"node {self.name} already exists in graph, invalid to insert it again."

        self.owning_graph._nodes.append(self)

        return self

    def insert_before(self, onnx_node: Union[str, "OnnxNode"]) -> "OnnxNode":
        """将当前node插入到直接所属graph的node序列并且拓扑序在给定node之前."""
        if isinstance(onnx_node, str):
            onnx_node = self.owning_graph.node_mappings[onnx_node]
        # this node should not exist in graph
        assert (
            self not in self.owning_graph._nodes
        ), f"node {self.name} already exists in graph, invalid to insert it again."
        # self and given node should have the same owning_graph
        assert self.owning_graph is onnx_node.owning_graph, (
            f"The owning graph of node {self.name} and "
            f"node {onnx_node.name} are different."
        )

        insert_idx = self.owning_graph._nodes.index(onnx_node)
        self.owning_graph._nodes = (
            self.owning_graph._nodes[:insert_idx]
            + [self]
            + self.owning_graph._nodes[insert_idx:]
        )

        return self

    def insert_after(self, onnx_node: Union[str, "OnnxNode"]) -> "OnnxNode":
        """将当前node插入到直接所属graph的node序列并且拓扑序在给定node之后."""
        if isinstance(onnx_node, str):
            onnx_node = self.owning_graph.node_mappings[onnx_node]
        # this node should not exist in graph
        assert (
            self not in self.owning_graph._nodes
        ), f"node {self.name} already exists in graph, invalid to insert it again."
        # self and given node should have the same owning_graph
        assert self.owning_graph is onnx_node.owning_graph, (
            f"The owning graph of node {self.name} and "
            f"node {onnx_node.name} are different."
        )

        insert_idx = self.owning_graph._nodes.index(onnx_node) + 1
        self.owning_graph._nodes = (
            self.owning_graph._nodes[:insert_idx]
            + [self]
            + self.owning_graph._nodes[insert_idx:]
        )

        return self

    def move_before(self, onnx_node: Union[str, "OnnxNode"]) -> "OnnxNode":
        """将当前node在直接所属graph的node序列中的位置移动到给定node之前."""
        if isinstance(onnx_node, str):
            onnx_node = self.owning_graph.node_mappings[onnx_node]
        # self and given node should have the same owning_graph
        assert self.owning_graph is onnx_node.owning_graph, (
            f"The owning graph of node {self.name} and "
            f"node {onnx_node.name} are different."
        )

        self.owning_graph._nodes.remove(self)
        self.insert_before(onnx_node)

        return self

    def move_after(self, onnx_node: Union[str, "OnnxNode"]) -> "OnnxNode":
        """将当前node在直接所属graph的node序列中的位置移动到给定node之后."""
        if isinstance(onnx_node, str):
            onnx_node = self.owning_graph.node_mappings[onnx_node]
        # self and given node should have the same owning_graph
        assert self.owning_graph is onnx_node.owning_graph, (
            f"The owning graph of node {self.name} and "
            f"node {onnx_node.name} are different."
        )

        self.owning_graph._nodes.remove(self)
        self.insert_after(onnx_node)

        return self

    def destroy(self) -> "OnnxNode":
        """将当前node从所属graph中清除.

        仅当前node所有输出不充当任何node输入或者graph输出时, 该操作合法
        """
        # output of node to be removed should not exist in node inputs or graph outputs
        assert not self.is_used, (
            f"At least one output of node {self.name} has been used as "
            f"node inputs or graph outputs, invalid to remove it from graph."
        )

        self.clear_inputs()
        self.clear_outputs()
        self.clear_attributes()
        self.owning_graph._nodes.remove(self)

        return self

    def __repr__(self) -> str:
        """返回当前node的字符串表示."""
        return f"{self._op_type}, {self._name}, {self.input_names}, {self.output_names}"

    def _sync(self) -> None:
        # sync for name
        self._proto.name = self.name
        # sync for domain
        self._proto.domain = self.domain
        # sync for op_type
        self._proto.op_type = self.op_type
        # sync for input names
        del self._proto.input[:]
        for input_var in self._inputs:
            if input_var.is_dummy:
                self._proto.input.append("")
            else:
                self._proto.input.append(input_var.name)
        # sync for output names
        del self._proto.output[:]
        for output_var in self._outputs:
            if output_var.is_dummy:
                self._proto.output.append("")
            else:
                self._proto.output.append(output_var.name)
        # sync for node attributes
        del self._proto.attribute[:]
        attr_protos = [
            make_attribute(key=attr_name, value=attr_val)
            for attr_name, attr_val in self.attributes.items()
            if attr_val is not None
        ]
        self._proto.attribute.extend(attr_protos)
