from typing import Any, List, Dict

import torch

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
from hbdk4.compiler.torch.utils import convert_dtype


# base class for all torch export adaptors
class TorchExportAdaptorBase(object):
    def __init__(self, obj):
        self.obj = obj


class TorchExportAttributeAdaptor(AttributeAdaptor, TorchExportAdaptorBase):
    def __init__(self, obj: torch.Node, name):
        TorchExportAttributeAdaptor.__init__(self, obj)
        AttributeAdaptor.__init__(self)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> Any:
        assert False  # should not come here


class TorchExportTypeAdaptor(TypeAdaptor, TorchExportAdaptorBase):
    def __init__(self, fake_tensor: torch._subclasses.fake_tensor.FakeTensor):
        TypeAdaptor.__init__(self)
        TorchExportAdaptorBase.__init__(self, fake_tensor)

    @property
    def is_dict(self) -> bool:
        assert False  # should not come here

    @property
    def is_tuple(self) -> bool:
        return isinstance(self.obj, tuple)

    @property
    def is_tensor(self) -> bool:
        return isinstance(self.obj, torch._subclasses.fake_tensor.FakeTensor)

    @property
    def is_class(self) -> bool:
        assert False  # should not come here

    def get_sub_types(self) -> List[TypeAdaptor]:
        if self.is_tuple:
            return [TorchExportTypeAdaptor(v) for v in self.obj]

    @property
    def dtype(self) -> torch.dtype:
        return self.obj.dtype

    @property
    def shape(self) -> List[int]:
        return [n for n in self.obj.shape]

    def emit_mlir_element_type(self) -> mlir.Type:
        return convert_dtype(self.dtype)


class TorchExportValueAdaptor(ValueAdaptor, TorchExportAdaptorBase):
    def __init__(self, node: torch.fx.node.Node):
        TorchExportAdaptorBase.__init__(self, node)
        ValueAdaptor.__init__(self, node.name if node is not None else "anoymous")

    @property
    def type(self) -> TorchExportTypeAdaptor:
        if self.obj is None:
            return TorchExportTypeAdaptor(None)
        return TorchExportTypeAdaptor(self.obj.meta["val"])

    def emit_mlir_const_op(self):
        assert isinstance(self.value, torch.Tensor)

        loc = mlir.Location.name(self.name)
        return create_const_op(self.value.contiguous().detach().numpy(), loc).result


class TorchExportNodeAdaptor(NodeAdaptor, TorchExportAdaptorBase):
    def __init__(
        self,
        node: torch.fx.node.Node,
        operands: List[ValueAdaptor],
        results: List[ValueAdaptor],
    ):
        TorchExportAdaptorBase.__init__(self, node)
        NodeAdaptor.__init__(self, operands, results, self.obj.name)

    @property
    def type(self) -> OpType:
        if self.obj.op == "placeholder":
            return OpType("placeholder", "aten", 0, self.obj)
        elif self.obj.target.__qualname__ == "getitem":
            return OpType("getitem", "aten", 0, self.obj)
        else:
            namespace, op = self.obj.target.name().split("::")
            return OpType(op, namespace, 0, self.doc)

    @property
    def attributes(self) -> Dict[str, AttributeAdaptor]:
        return {}

    def invoke(self, *args, **kwargs):
        results = self.obj.target.op(*args)

        if len(self.results) == 1:
            results = [results]

        return results

    @property
    def doc(self) -> str:
        return str(self.obj.target._schema)


class TorchExportGraphAdaptor(GraphAdaptor, TorchExportAdaptorBase):
    def __init__(
        self,
        prog: torch.export.ExportedProgram,
        name: str = None,
    ):

        TorchExportAdaptorBase.__init__(self, prog)
        GraphAdaptor.__init__(
            self, [], [], name if name is not None else "ExportedProgram"
        )

        for node in prog.graph.nodes:
            self.stack.insert(TorchExportValueAdaptor(node))

        signature = prog.graph_signature

        self.operands = [self.stack.find(i) for i in signature.user_inputs]
        self.results = [self.stack.find(o) for o in signature.user_outputs]

        self.build()  # build nodes and def-use affiliations

        # bind constant to placeholder result
        for k, v in signature.inputs_to_parameters.items():
            value = self.stack.find(k)
            value.bind(prog.state_dict[v])

        for k, v in signature.inputs_to_buffers.items():
            value = self.stack.find(k)
            value.bind(prog.state_dict[v])

    def build_nodes(self) -> List[TorchExportNodeAdaptor]:
        node_adaptors = []
        for node in self.obj.graph.nodes:
            if node.op == "output":
                continue  # skip the last node
            if node.op == "placeholder":
                continue  # skip the input node

            operands = []
            for v in node.args:
                if isinstance(v, torch.fx.node.Node):
                    operands.append(self.stack.find(v.name))
                else:
                    value = TorchExportValueAdaptor(None)
                    value.bind(v)
                    operands.append(value)
            results = [self.stack.find(node.name)]
            node_adaptors.append(TorchExportNodeAdaptor(node, operands, results))
        return node_adaptors
