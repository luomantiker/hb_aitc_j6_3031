from typing import Any, List, Dict, Union

import torch

from hbdk4.compiler.torch.jit.utils import get_graph
from hbdk4.compiler.torch.jit import executor
from hbdk4.compiler.torch.jit import optimizer
from hbdk4.compiler.torch.utils import convert_dtype

from hbdk4.compiler.frontend.adaptor import (
    TypeAdaptor,
    AttributeAdaptor,
    ValueAdaptor,
    NodeAdaptor,
    GraphAdaptor,
)
from hbdk4.compiler.frontend.registry import OpType

from hbdk4.compiler import ir as mlir

from hbdk4.compiler.ops.common import create_const_op


# base class for all torch jit adaptors
class TorchJitAdaptorBase(object):
    def __init__(self, jit: Union[torch.Graph, torch.Block, torch.Node, torch.Value]):
        self.jit = jit


class TorchJitAttributeAdaptor(AttributeAdaptor, TorchJitAdaptorBase):
    def __init__(self, jit: torch.Node, name):
        TorchJitAdaptorBase.__init__(self, jit)
        AttributeAdaptor.__init__(self)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> Any:
        return getattr(self.jit, self.jit.kindOf(self.name))(self.name)


class TorchJitTypeAdaptor(TypeAdaptor, TorchJitAdaptorBase):
    def __init__(self, jit, value: Any):
        TypeAdaptor.__init__(self)
        TorchJitAdaptorBase.__init__(self, jit)
        self.exe_value = value
        self.verify()

    def verify(self):
        jit_type = self.jit.kind()
        if jit_type == "DictType":
            assert isinstance(self.exe_value, dict)
        if jit_type == "TupleType":
            assert isinstance(self.exe_value, (list, tuple))
        if jit_type == "ListType":
            assert isinstance(self.exe_value, (list, tuple))
        if jit_type == "TensorType":
            assert isinstance(
                self.exe_value, (torch.Tensor, int, float)
            )  # buggy torchscript typing system

    @property
    def is_dict(self) -> bool:
        return self.jit.kind() == "DictType"

    @property
    def is_tuple(self) -> bool:
        return self.jit.kind() in ["TupleType", "ListType"]

    @property
    def is_tensor(self) -> bool:
        return self.jit.kind() == "TensorType" and isinstance(
            self.exe_value, torch.Tensor
        )

    @property
    def is_class(self) -> bool:
        return self.jit.kind() == "ClassType"

    def get_sub_types(self) -> List[TypeAdaptor]:
        jit_type = self.jit.kind()
        if jit_type in ["ListType"]:
            return [
                TorchJitTypeAdaptor(self.jit.getElementType(), v)
                for v in self.exe_value
            ]
        if jit_type in ["TupleType"]:
            return [
                TorchJitTypeAdaptor(t, v)
                for t, v in zip(self.jit.elements(), self.exe_value)
            ]

        if jit_type == "DictType":
            return {
                key: TorchJitTypeAdaptor(self.jit.getValueType(), self.exe_value[key])
                for key in sorted(self.exe_value.keys())
            }
        assert False

    def emit_mlir_element_type(self) -> mlir.Type:
        return convert_dtype(self.dtype)

    @property
    def dtype(self) -> torch.dtype:
        if self.jit.kind() == "TensorType":
            if self.jit.dtype():  # if jit has annotated dtype then verify it
                assert self.jit.dtype() == self.exe_value.dtype
            return self.exe_value.dtype
        raise ValueError("non-TensorType cannot access dtype")

    @property
    def shape(self) -> List[int]:
        if self.jit.kind() == "TensorType":
            if self.jit.symbolic_sizes():  # if jit has annotated shape then verify it
                assert self.jit.symbolic_sizes() == list(self.exe_value.shape)
            return list(self.exe_value.shape)


class TorchJitValueAdaptor(ValueAdaptor, TorchJitAdaptorBase):
    def __init__(self, jit: torch.Value, value, name=None):
        TorchJitAdaptorBase.__init__(self, jit)
        ValueAdaptor.__init__(self, self.jit.debugName() if name is None else name)
        self.exe_value = value

    @property
    def type(self) -> TorchJitTypeAdaptor:
        return TorchJitTypeAdaptor(self.jit.type(), self.exe_value)

    def emit_mlir_const_op(self):
        assert isinstance(self.value, torch.Tensor)

        loc = mlir.Location.name(self.name)
        return create_const_op(self.value.contiguous().detach().numpy(), loc).result


class TorchJitNodeAdaptor(NodeAdaptor, TorchJitAdaptorBase):
    def __init__(
        self,
        jit: torch.Node,
        operands: List[ValueAdaptor],
        results: List[ValueAdaptor],
    ):
        TorchJitAdaptorBase.__init__(self, jit)
        NodeAdaptor.__init__(self, operands, results, self.jit.scopeName())

    @property
    def type(self) -> OpType:
        namespace, op = self.jit.kind().split("::")
        return OpType(op, namespace, 0, self.jit)

    @property
    def attributes(self) -> Dict[str, AttributeAdaptor]:
        attrs = dict()
        for name in self.jit.attributeNames():
            attrs[name] = TorchJitAttributeAdaptor(self.jit, name)
        return attrs

    def invoke(self, *args, **kwargs):
        results = executor.run_node(self.jit, *args)

        if len(self.results) == 1:
            results = [results]

        return results


class TorchJitGraphAdaptor(GraphAdaptor, TorchJitAdaptorBase):
    def __init__(
        self,
        jit: torch.jit.ScriptModule,
        *args,
        name=None,
    ):
        if isinstance(jit, torch.jit.ScriptModule):
            jit = jit.eval()

        jit = optimizer.optimize(jit)

        # in torch fe: to propagate intermediate states of the graph (shapes and dtypes), we execute the graph interpretively and create ValueAdaptor according to results
        rets, collects = executor.run(jit, *args)

        graph = get_graph(jit)

        TorchJitAdaptorBase.__init__(self, graph)
        GraphAdaptor.__init__(self, [], [], jit.original_name if name is None else name)

        for v in collects:
            self.stack.insert(TorchJitValueAdaptor(v, collects[v]))

        self.operands = [self.stack.find(i.debugName()) for i in graph.inputs()]
        self.results = [self.stack.find(o.debugName()) for o in graph.outputs()]

        if isinstance(jit, torch.jit.ScriptModule):
            self.operands[0].bind(jit)
            self.operands = self.operands[1:]  # the first argument

        self.build()  # build nodes and def-use affiliations

    def build_nodes(self) -> List[TorchJitNodeAdaptor]:
        node_adaptors = []
        for node in self.jit.nodes():
            operands = [self.stack.find(v.debugName()) for v in node.inputs()]
            results = [self.stack.find(v.debugName()) for v in node.outputs()]
            node_adaptors.append(TorchJitNodeAdaptor(node, operands, results))
        return node_adaptors
