from collections import defaultdict
from copy import deepcopy
import logging
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Union,
    cast,
    overload,
)

from .onnx_calibration_node import CalibrationNode
from .onnx_graph import OnnxGraph
from .onnx_utils import (
    IR_VERSION,
    ModelProto,
    checker,
    defs,
    helper,
    load_model,
    load_model_from_string,
    shape_inference,
    version_converter,
)
from .utils import TensorDevice, TensorMode

if TYPE_CHECKING:
    from .onnx_node import OnnxNode
    from .onnx_variable import OnnxVariable


class OnnxModel:
    _proto: ModelProto
    _ir_version: int
    _opset_import: Dict[str, int]
    _producer_name: Union[None, str]
    _producer_version: Union[None, str]
    _model_domain: Union[None, str]
    _model_version: Union[None, int]
    _graph: OnnxGraph
    _graphs: List[OnnxGraph]
    _tensor_mode: TensorMode
    _tensor_device: TensorDevice

    @overload
    def __init__(self, proto: Union[str, bytes, ModelProto]) -> None: ...

    @overload
    def __init__(
        self,
        *,
        ir_version: Optional[int] = None,
        opset_import: Optional[Mapping[str, int]] = None,
        producer_name: Optional[str] = None,
        producer_version: Optional[str] = None,
        model_domain: Optional[str] = None,
        model_version: Optional[int] = None,
        graph_name: Optional[str] = None,
    ) -> None: ...

    def __init__(
        self,
        proto: Optional[Union[str, bytes, ModelProto]] = None,
        ir_version: Optional[int] = None,
        opset_import: Optional[Mapping[str, int]] = None,
        producer_name: Optional[str] = None,
        producer_version: Optional[str] = None,
        model_domain: Optional[str] = None,
        model_version: Optional[int] = None,
        graph_name: Optional[str] = None,
    ) -> None:
        """初始化OnnxModel对象."""
        super().__init__()

        # initialize info used to represent tensor
        self._tensor_mode = TensorMode.NUMPY
        self._tensor_device = TensorDevice("cpu")

        # init from proto or config
        if proto is None:
            self._init_from_config(
                ir_version=ir_version,
                opset_import=opset_import,
                producer_name=producer_name,
                producer_version=producer_version,
                model_domain=model_domain,
                model_version=model_version,
                graph_name=graph_name,
            )
        else:
            self._init_from_proto(proto=proto)

    def _init_from_config(
        self,
        ir_version: Optional[int] = None,
        opset_import: Optional[Mapping[str, int]] = None,
        producer_name: Optional[str] = None,
        producer_version: Optional[str] = None,
        model_domain: Optional[str] = None,
        model_version: Optional[int] = None,
        graph_name: Optional[str] = None,
    ) -> None:
        """基于参数配置初始化OnnxModel对象."""
        self._proto = ModelProto()
        self._ir_version = IR_VERSION if ir_version is None else ir_version
        assert self._ir_version >= 4 and self._ir_version <= IR_VERSION, (
            f"Only support ir_version within [4, {IR_VERSION}], "
            f"but got {self._ir_version}."
        )
        self._opset_import = (
            {defs.ONNX_DOMAIN: defs.onnx_opset_version()}
            if opset_import is None
            else dict(opset_import)
        )
        self._producer_name = producer_name
        self._producer_version = producer_version
        self._model_domain = model_domain
        self._model_version = model_version
        # initialize empty graph list
        self._graphs = []
        self._graph = OnnxGraph(
            owning_model=self,
            owning_graph=None,
            name=graph_name,
        )

    def _init_from_proto(self, proto: Union[str, bytes, ModelProto]) -> None:
        """基于ModelProto初始化OnnxModel对象."""
        # load model proto
        if isinstance(proto, str):
            self._proto = load_model(proto)
        elif isinstance(proto, bytes):
            self._proto = load_model_from_string(proto)
        else:
            self._proto = proto

        # we suppose the given model proto:
        #   MUST present fields:
        #       ir_version && opset_import && graph field
        #   MAY present fields:
        #       producer_{name/version} && domain && model_version
        self._ir_version = self._proto.ir_version
        self._opset_import = {
            opset_id.domain: opset_id.version for opset_id in self._proto.opset_import
        }
        self._producer_name = (
            self._proto.producer_name if self._proto.HasField("producer_name") else None
        )
        self._producer_version = (
            self._proto.producer_version
            if self._proto.HasField("producer_version")
            else None
        )
        self._model_domain = (
            self._proto.domain if self._proto.HasField("domain") else None
        )
        self._model_version = (
            self._proto.model_version if self._proto.HasField("model_version") else None
        )
        # initialize empty graph list
        self._graphs = []
        self._graph = OnnxGraph(
            owning_model=self, owning_graph=None, proto=self._proto.graph
        )

        # post-processing for model proto
        # 1) add horizon opset for onnx model
        self._opset_import["horizon"] = 1
        # 2) regularize onnx opset domain
        self._regularize_onnx_opset_domain()
        # 3) check if ir_version is greater than IR_VERSION and
        #    force that ir_version >= 4
        self._check_and_regularize_ir_version()
        # 4) regularize graph names to be different
        self._regularize_graph_names()
        # 5) regularize node names to be different
        self._regularize_node_names()
        # 6) regularize variable names to be different
        self._regularize_variable_names()

    def _regularize_onnx_opset_domain(self) -> None:
        """归一化onnx opset的domain."""
        if "ai.onnx" in self._opset_import:
            # regularize domain name for onnx model
            assert (
                "" not in self._opset_import
            ), "Illegal to have empty-string and ai.onnx domain name simultaneously."
            self._opset_import[""] = self._opset_import["ai.onnx"]
            del self._opset_import["ai.onnx"]
            # regularize domain name for onnx node
            for onnx_graph in self._graphs:
                for onnx_node in onnx_graph._nodes:
                    onnx_node.domain = ""

    def _check_and_regularize_ir_version(self) -> None:
        """检查并归一化模型ir_version."""
        if self._ir_version < 4:
            self._ir_version = 4
            logging.warning(
                f"The ir version of the model is {self._ir_version}, which is "
                "less than the minimum supported ir version of 4. "
                "We will upgrade ir_version to 4 forcibly."
            )
        elif self._ir_version > IR_VERSION:
            raise ValueError(
                f"The ir version of the model is {self._ir_version}, which is "
                f"greater than the maximum supported ir version of {IR_VERSION}."
            )

    def _regularize_graph_names(self) -> None:
        """保证当前model不同graph的name都是非空且唯一的."""
        graph_names: Set[str] = set()
        for onnx_graph in self._graphs:
            if onnx_graph.name == "" or onnx_graph.name in graph_names:
                onnx_graph.name = self._get_next_unique_name(
                    namespace="graph", existing_names=graph_names
                )
            graph_names.add(onnx_graph.name)

    def _regularize_node_names(self) -> None:
        """保证当前model不同graph中node的name都是非空且唯一的."""
        node_names: Set[str] = set()
        for onnx_graph in self._graphs:
            for onnx_node in onnx_graph._nodes:
                if onnx_node.name == "" or onnx_node.name in node_names:
                    onnx_node.name = self._get_next_unique_name(
                        namespace="node",
                        existing_names=node_names,
                        name_prefix=onnx_node.op_type,
                    )
                node_names.add(onnx_node.name)

    def _regularize_variable_names_within_graph(self, onnx_graph: OnnxGraph) -> None:
        """保证相同graph下所有variable的name都是非空且唯一的."""
        variable_names: Set[str] = set()
        # 基于ModelProto初始化会保证相同graph内非属性variable的name非空且唯一
        for onnx_var in onnx_graph._variables:
            if not onnx_var.is_attr:
                variable_names.add(onnx_var.name)
        # 进一步考虑属性类variable后, 保证相同graph下所有variable的name非空且唯一
        for onnx_var in onnx_graph._variables:
            if onnx_var.is_attr:
                if onnx_var.name == "" or onnx_var.name in variable_names:
                    onnx_var.name = self._get_next_unique_name(
                        namespace="variable", existing_names=variable_names
                    )
                variable_names.add(onnx_var.name)

    def _regularize_variable_names(self) -> None:
        """保证当前model不同graph下所有variable的name都是非空且唯一的."""
        # 1) 首先, 我们保证相同graph内所有variable的name都是非空且唯一的
        for onnx_graph in self._graphs:
            self._regularize_variable_names_within_graph(onnx_graph=onnx_graph)
        # 2) 其次, 我们保证具有上下层级关系的graph间,
        # 符合规则的同名variable共享相同的OnnxVariable对象
        self._graph._share_variable_among_graphs()
        # 3) 最后, 我们保证所有variable的name都是非空且唯一的
        variable_names: Set[str] = set()
        for onnx_graph in self._graphs:
            for onnx_var in onnx_graph._variables:
                if onnx_var.name == "" or onnx_var.name in variable_names:
                    onnx_var.name = self._get_next_unique_name(
                        namespace="variable", existing_names=variable_names
                    )
                variable_names.add(onnx_var.name)

    @property
    def proto(self) -> ModelProto:
        """返回基于当前model导出的ModelProto对象.

        属性类型:
            只读属性
        """
        # sync from OnnxModel to ModelProto
        self._sync()
        return self._proto

    @property
    def ir_version(self) -> int:
        """返回当前model的ir版本号.

        属性类型:
            读写属性, 允许赋值修改
        """
        return self._ir_version

    @ir_version.setter
    def ir_version(self, ir_version: int) -> None:
        assert ir_version >= 4 and ir_version <= IR_VERSION, (
            f"Only support ir_version within [4, {IR_VERSION}], "
            f"but got {ir_version}."
        )
        self._ir_version = ir_version

    @property
    def opset_import(self) -> Dict[str, int]:
        """返回当前model的opset domain到opset version字典.

        属性类型:
            读写属性, 允许原地修改和赋值修改
        """
        return self._opset_import

    @opset_import.setter
    def opset_import(self, opset_import: Mapping[str, int]) -> None:
        self._opset_import = dict(opset_import)

    @property
    def producer_name(self) -> Union[None, str]:
        """返回当前model的producer_name.

        属性类型:
            读写属性, 允许赋值修改
        """
        return self._producer_name

    @producer_name.setter
    def producer_name(self, producer_name: Union[None, str]) -> None:
        self._producer_name = producer_name

    @property
    def producer_version(self) -> Union[None, str]:
        """返回当前model的producer_version.

        属性类型:
            读写属性, 允许赋值修改
        """
        return self._producer_version

    @producer_version.setter
    def producer_version(self, producer_version: Union[None, str]) -> None:
        self._producer_version = producer_version

    @property
    def model_domain(self) -> Union[None, str]:
        """返回当前model的domain.

        属性类型:
            读写属性, 允许赋值修改
        """
        return self._model_domain

    @model_domain.setter
    def model_domain(self, model_domain: Union[None, str]) -> None:
        self._model_domain = model_domain

    @property
    def model_version(self) -> Union[None, int]:
        """返回当前model的version.

        属性类型:
            读写属性, 允许赋值修改
        """
        return self._model_version

    @model_version.setter
    def model_version(self, model_version: Union[None, int]) -> None:
        self._model_version = model_version

    @property
    def graph(self) -> OnnxGraph:
        """返回当前model的main graph.

        属性类型:
            读写属性, 允许原地修改
        """
        return self._graph

    @property
    def graphs(self) -> AbstractSet[OnnxGraph]:
        """返回当前model的graph集合.

        属性类型:
            读写属性, 允许原地修改graph, 不可增加或者删除graph
        """
        return frozenset(self._graphs)

    @property
    def graph_names(self) -> AbstractSet[str]:
        """返回当前model的graph name集合.

        属性类型:
            只读属性
        """
        graph_names: Set[str] = {onnx_graph.name for onnx_graph in self._graphs}

        return frozenset(graph_names)

    @property
    def graph_mappings(self) -> Mapping[str, OnnxGraph]:
        """返回当前model的graph name到graph映射.

        属性类型:
            读写属性, 允许原地修改graph, 不可增加或者删除graph
        """
        graph_mappings: Dict[str, OnnxGraph] = {}
        for onnx_graph in self._graphs:
            graph_mappings[onnx_graph.name] = onnx_graph

        return MappingProxyType(graph_mappings)

    @property
    def nodes(self) -> AbstractSet["OnnxNode"]:
        """返回当前model的node集合.

        属性类型:
            读写属性, 允许原地修改node, 不可增加或者删除node
        """
        nodes: Set[OnnxNode] = set()
        for onnx_graph in self._graphs:
            nodes.update(onnx_graph._nodes)

        return frozenset(nodes)

    @property
    def node_names(self) -> AbstractSet[str]:
        """返回当前model的node name集合.

        属性类型:
            只读属性
        """
        node_names: Set[str] = set()
        for onnx_graph in self._graphs:
            node_names.update(onnx_graph.node_names)

        return frozenset(node_names)

    @property
    def node_mappings(self) -> Mapping[str, "OnnxNode"]:
        """返回当前model的node name到node映射.

        属性类型:
            读写属性, 允许原地修改node, 不可增加或删除node
        """
        node_mappings: Dict[str, OnnxNode] = {}
        for onnx_graph in self._graphs:
            node_mappings.update(onnx_graph.node_mappings)

        return MappingProxyType(node_mappings)

    @property
    def type2nodes(self) -> Mapping[str, AbstractSet["OnnxNode"]]:
        """返回当前model的op_type到node集合的映射.

        属性类型:
            读写属性, 允许原地修改node, 不可增加或删除node
        """
        type2nodes: Dict[str, Set[OnnxNode]] = defaultdict(set)
        for onnx_graph in self._graphs:
            for onnx_node in onnx_graph._nodes:
                type2nodes[onnx_node.op_type].add(onnx_node)

        return MappingProxyType(
            defaultdict(
                frozenset,
                {op_type: frozenset(type2nodes[op_type]) for op_type in type2nodes},
            )
        )

    @property
    def variables(self) -> AbstractSet["OnnxVariable"]:
        """返回当前model的variable集合.

        属性类型:
            读写属性, 允许原地修改variable, 不可增加或删除variable
        """
        variables: Set[OnnxVariable] = set()
        for onnx_graph in self._graphs:
            variables.update(onnx_graph._variables)

        return frozenset(variables)

    @property
    def variable_names(self) -> AbstractSet[str]:
        """返回当前model的variable name集合.

        属性类型:
            只读属性
        """
        variable_names: Set[str] = set()
        for onnx_graph in self._graphs:
            variable_names.update(onnx_graph.variable_names)

        return frozenset(variable_names)

    @property
    def variable_mappings(self) -> Mapping[str, "OnnxVariable"]:
        """返回当前model的variable name到variable映射.

        属性类型:
            读写属性, 允许原地修改variable, 不可增加或删除variable
        """
        variable_mappings: Dict[str, OnnxVariable] = {}
        for onnx_graph in self._graphs:
            variable_mappings.update(onnx_graph.variable_mappings)

        return MappingProxyType(variable_mappings)

    @property
    def tensor_mode(self) -> TensorMode:
        """返回当前model的张量存储模式.

        张量存储模式包括TensorMode.NUMPY和TensorMode.TORCH两种

        属性类型:
            只读属性
        """
        return self._tensor_mode

    @property
    def tensor_device(self) -> TensorDevice:
        """返回当前model的张量存储设备.

        张量存储模式为TensorMode.TORCH时, 存储设备可以为cpu或者cuda

        张量存储模式为TensorMode.NUMPY时, 存储设备可以为cpu

        属性类型:
            只读属性
        """
        return self._tensor_device

    def to(
        self,
        tensor_mode: Optional[TensorMode] = None,
        tensor_device: Optional[Union[str, TensorDevice]] = None,
    ) -> "OnnxModel":
        """切换当前model的张量到指定存储模式和存储设备."""
        assert (
            tensor_mode is not None or tensor_device is not None
        ), "Either tensor_mode or tensor_device should not be given."

        if tensor_mode is not None:
            self._tensor_mode = tensor_mode
            if self._tensor_mode != TensorMode.TORCH:
                self._tensor_device = TensorDevice("cpu")

        if tensor_device is not None:
            tensor_device = TensorDevice(tensor_device)
            if self._tensor_mode != TensorMode.TORCH and tensor_device.type != "cpu":
                raise ValueError(
                    f"For tensor_model: {self._tensor_mode}, "
                    f"the only available tensor device is cpu."
                )
            self._tensor_device = tensor_device

        return self

    def infer_shapes(self, clear_shapes: bool = True) -> "OnnxModel":
        """基于当前model的输入shapes/dtypes推导出中间激活的shapes/dtypes."""
        # clear existing dtypes && shapes
        if clear_shapes:
            self.clear_shapes()
        # infer dtypes && shapes via onnx infer_shapes interface
        inferred_model = OnnxModel(
            proto=shape_inference.infer_shapes(
                self.proto,
                strict_mode=True,
            )
        )
        # update dtypes && shapes based on inferred_model
        graph_mappings = inferred_model.graph_mappings
        for onnx_graph in self._graphs:
            variable_mappings = graph_mappings[onnx_graph.name].variable_mappings
            for onnx_var in onnx_graph._variables:
                if not onnx_var.is_param and not onnx_var.is_dummy:
                    onnx_var.shape = variable_mappings[onnx_var.name].shape
                    onnx_var.dtype = variable_mappings[onnx_var.name].dtype

        return self

    def clear_shapes(self) -> "OnnxModel":
        """将当前model中间激活已有的shapes/dtypes清空."""
        # clear value_info && graph output
        for onnx_var in self.variables:
            if (
                not onnx_var.is_param
                and onnx_var.name not in self.graph.input_names
                and onnx_var.shape is not None
                and (
                    onnx_var.src_op is None
                    or (
                        onnx_var.src_op.op_type not in ["PyOp"]
                        and onnx_var.src_op.domain not in ["ai.onnx.contrib"]
                    )
                )
            ):
                onnx_var.shape = [
                    f"{onnx_var.name}_dim_{idx}" for idx in range(len(onnx_var.shape))
                ]

        return self

    def convert_version(self, target_version: int) -> "OnnxModel":
        """将当前model的onnx opset转换到指定版本."""
        self.reset_proto(
            proto=version_converter.convert_version(self.proto, target_version)
        )

        return self

    def check_validity(self) -> None:
        """检查当前model的合法性."""
        # check model ir_version
        assert self.ir_version >= 4 and self.ir_version <= IR_VERSION, (
            f"Only support ir_version within [4, {IR_VERSION}], "
            f"but got {self.ir_version}."
        )
        # check onnx opset domain
        assert "ai.onnx" not in self.opset_import, (
            "The name of onnx opset should be regularized to empty string "
            "instead of ai.onnx."
        )
        # check graph names
        if len(self.graphs) != len(self.graph_names):
            raise RuntimeError("There exist duplicated names for OnnxGraph.")
        if any(name == "" for name in self.graph_names):
            raise RuntimeError("There exist empty-string name for OnnxGraph.")
        # check node names
        if len(self.nodes) != len(self.node_names):
            raise RuntimeError("There exist duplicated names for OnnxNode.")
        if any(name == "" for name in self.node_names):
            raise RuntimeError("There exist empty-string name for OnnxNode.")
        # check variable names
        if len(self.variables) != len(self.variable_names):
            raise RuntimeError("There exist duplicated names for OnnxVariable.")
        if any(name == "" for name in self.variable_names):
            raise RuntimeError("There exist empty-string name for OnnxVariable.")
        # call onnx check_model
        checker.check_model(self.proto)

    def reset_proto(
        self,
        proto: Union[str, bytes, ModelProto],
    ) -> "OnnxModel":
        """基于给定ModelProto重置当前model的相关属性.

        OnnxModel相较于ModelProto特有的属性则保持不变
        """
        orig_model = deepcopy(self)
        self.__init__(proto=proto)  # type: ignore[misc]
        self._update_model(onnx_model=orig_model)

        return self

    def sort_topologically(self) -> "OnnxModel":
        """对当前model各graph的node序列原地进行拓扑排序."""
        self.graph._sort_topologically()

        return self

    def __repr__(self) -> str:
        """返回当前model的字符串表示."""
        model_str = ["OnnxModel:"]
        model_str.extend(
            (
                2 * " " + f"ir_version: {self.ir_version}",
                2 * " " + f"opset_import: {self.opset_import}",
                2 * " " + f"producer_name: {self.producer_name}",
                2 * " " + f"producer_version: {self.producer_version}",
                2 * " " + f"model_domain: {self.model_domain}",
                2 * " " + f"model_version: {self.model_version}",
            )
        )
        graph_str = [2 * " " + line_str for line_str in str(self.graph).split("\n")]
        model_str.extend(graph_str)

        return "\n".join(model_str)

    def __deepcopy__(self, memo: Optional[Dict] = None) -> "OnnxModel":
        """OnnxModel对象深拷贝的自定义实现."""
        cloned_model = OnnxModel(proto=deepcopy(self.proto))
        cloned_model._update_model(self)

        return cloned_model

    def _update_model(self, onnx_model: "OnnxModel") -> None:
        """将OnnxModel相较于ModelProto特有的属性根据给定onnx_model更新到当前model."""
        # update for OnnxModel-level states
        self.to(
            tensor_mode=onnx_model.tensor_mode,
            tensor_device=onnx_model.tensor_device,
        )
        # TODO(zsq): OnnxModel在每次执行完modify_model_by_cpp_func后, 原先保存在
        # CalibrationNode.calibration_thresholds中阈值信息将会被清除.
        # 在perchannel校准时, 会需要调用cpp函数set_calibration_channel_scale更新
        # proto的channel_scales属性, 导致此前保存在calibration_thresholds中的
        # kl, max等阈值信息丢失, 因此增加将calibration_thresholds更新到当前模型的逻辑.
        # 待重构set_calibration_channel_scale为python实现之后, 将该逻辑进行移除.
        # update for CalibrationNode calibration thresholds
        graph_mappings = onnx_model.graph_mappings
        for onnx_graph in self._graphs:
            node_mappings = graph_mappings[onnx_graph.name].node_mappings
            for onnx_node in onnx_graph._nodes:
                if (
                    isinstance(onnx_node, CalibrationNode)
                    and onnx_node.name in node_mappings
                ):
                    ref_node = cast(CalibrationNode, node_mappings[onnx_node.name])
                    onnx_node.update_node(ref_node)

    def _get_next_unique_name(
        self,
        namespace: str,
        existing_names: Optional[AbstractSet[str]] = None,
        name_prefix: Optional[str] = None,
    ) -> str:
        """获取不同namespace的下一个unique name."""
        if namespace == "graph":
            existing_names = (
                self.graph_names if existing_names is None else existing_names
            )
            name_prefix = "graph" if name_prefix is None else name_prefix
        elif namespace == "node":
            existing_names = (
                self.node_names if existing_names is None else existing_names
            )
            name_prefix = "node" if name_prefix is None else name_prefix
        elif namespace == "variable":
            existing_names = (
                self.variable_names if existing_names is None else existing_names
            )
            name_prefix = "variable" if name_prefix is None else name_prefix
        else:
            raise ValueError(f"Invalid given namespace: {namespace}.")

        name_suffix = len(existing_names)
        while True:
            unique_name = f"{name_prefix}_{name_suffix}"
            if unique_name not in existing_names:
                break
            name_suffix += 1

        return unique_name

    def _sync(self) -> None:
        # sync for ModelProto-level modifications
        # sync for ir_version
        self._proto.ir_version = self.ir_version
        # sync for opset_import
        del self._proto.opset_import[:]
        for domain, version in self.opset_import.items():
            opsetid = helper.make_opsetid(domain=domain, version=version)
            self._proto.opset_import.append(opsetid)
        # sync for producer_name
        if self.producer_name is None:
            self._proto.ClearField("producer_name")
        else:
            self._proto.producer_name = self.producer_name
        # sync for producer_version
        if self.producer_version is None:
            self._proto.ClearField("producer_version")
        else:
            self._proto.producer_version = self.producer_version
        # sync for domain
        if self.model_domain is None:
            self._proto.ClearField("domain")
        else:
            self._proto.domain = self.model_domain
        # sync for model_version
        if self.model_version is None:
            self._proto.ClearField("model_version")
        else:
            self._proto.model_version = self.model_version
        # sync for GraphProto-level modifications
        self._proto.graph.CopyFrom(self.graph.proto)
