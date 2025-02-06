from typing import Any, List, Dict, Union, Optional

import numpy as np

from hbdk4.compiler.frontend.adaptor import (
    TypeAdaptor,
    AttributeAdaptor,
    ValueAdaptor,
    NodeAdaptor,
    GraphAdaptor,
)
from hbdk4.compiler.frontend.registry import OpType


from onnx import (
    AttributeProto,
    GraphProto,
    NodeProto,
    TensorProto,
    TypeProto,
    ValueInfoProto,
)

from onnx.helper import get_attribute_value
from onnx.numpy_helper import to_array

from hbdk4.compiler import ir as mlir
from hbdk4.compiler.dialects import hbir

from hbdk4.compiler.ops.common import create_const_op
from hbdk4.compiler.onnx.helper import tensor_dtype_to_np_dtype


# base class for all onnx adaptors
class OnnxAdaptorBase(object):
    def __init__(self, proto):
        self.proto = proto


class OnnxAttributeAdaptor(AttributeAdaptor, OnnxAdaptorBase):
    def __init__(self, proto: AttributeProto):
        AttributeAdaptor.__init__(self)
        OnnxAdaptorBase.__init__(self, proto)

    @property
    def name(self) -> str:
        return self.proto.name

    @property
    def value(self) -> Any:
        return get_attribute_value(self.proto)


class OnnxTypeAdaptor(TypeAdaptor, OnnxAdaptorBase):
    def __init__(self, proto: Union[TensorProto, TypeProto], is_constant: bool):
        TypeAdaptor.__init__(self)
        OnnxAdaptorBase.__init__(self, proto)
        self.is_constant = is_constant

    @property
    def is_tensor(self) -> bool:
        return True

    @property
    def dtype(self) -> np.dtype:
        if "ValueInfoProto" in str(type(self.proto)):
            type_proto = self.proto.type.tensor_type.elem_type
            return tensor_dtype_to_np_dtype(type_proto)
        if "TensorProto" in str(type(self.proto)):
            type_proto = self.proto.data_type
            return tensor_dtype_to_np_dtype(type_proto)
        raise ValueError(
            "unknown proto {} for OnnxTypeAdaptor".format(type(self.proto))
        )

    def emit_mlir_element_type(self) -> mlir.Type:
        if np.issubdtype(self.dtype, np.signedinteger):
            return mlir.IntegerType.get_signed(self.dtype.itemsize * 8)
        if np.issubdtype(self.dtype, np.unsignedinteger):
            return mlir.IntegerType.get_unsigned(self.dtype.itemsize * 8)
        if self.dtype == np.float16:
            return mlir.F16Type.get()
        if self.dtype == np.float32:
            return mlir.F32Type.get()
        if self.dtype == np.float64:
            return mlir.F64Type.get()
        if self.dtype == np.bool_:
            return hbir.Bool8Type.get()
        raise ValueError("cannot emit mlir element type for type {}".format(self.dtype))

    @property
    def shape(self) -> List[int]:
        if "ValueInfoProto" in str(type(self.proto)):
            dim_proto = self.proto.type.tensor_type.shape.dim
            return [dim.dim_value for dim in dim_proto]
        if "TensorProto" in str(type(self.proto)):
            return self.proto.dims
        raise ValueError(
            "unknown proto {} for OnnxTypeAdaptor".format(type(self.proto))
        )


class OnnxValueAdaptor(ValueAdaptor, OnnxAdaptorBase):
    def __init__(self, proto: ValueInfoProto):
        OnnxAdaptorBase.__init__(self, proto)
        ValueAdaptor.__init__(self, self.proto.name)

    @property
    def type(self) -> TypeAdaptor:
        return OnnxTypeAdaptor(self.proto, self.is_constant)

    def emit_mlir_const_op(self):
        assert isinstance(self.value, np.ndarray)

        loc = mlir.Location.name(self.name)
        return create_const_op(self.value, loc).result


class OnnxNodeAdaptor(NodeAdaptor, OnnxAdaptorBase):
    def __init__(
        self,
        proto: NodeProto,
        namespace: str,
        opset_version: int,
        operands: List[ValueAdaptor],
        results: List[ValueAdaptor],
    ):
        OnnxAdaptorBase.__init__(self, proto)
        NodeAdaptor.__init__(self, operands, results, self.proto.name)
        self.namespace = namespace
        self.opset_version = opset_version

    @property
    def type(self) -> OpType:
        return OpType(
            self.proto.op_type, self.namespace, self.opset_version, self.proto
        )

    @property
    def attributes(self) -> Dict[str, AttributeAdaptor]:
        attr_dict = {}
        for proto in self.proto.attribute:
            adaptor = OnnxAttributeAdaptor(proto)
            attr_dict[adaptor.name] = adaptor
        return attr_dict

    def doc(self) -> str:
        return str(self.type)

    def invoke(self, *args, **kwargs):
        try:
            from onnx.reference import ReferenceEvaluator

            sess = ReferenceEvaluator(self.proto)
            feeds = kwargs
            for array, value_adaptor in zip(args, self.operands):
                feeds[value_adaptor.name] = array

            return sess.run(None, feeds)
        except ImportError as e:
            raise e


class OnnxGraphAdaptor(GraphAdaptor, OnnxAdaptorBase):
    def __init__(self, proto: GraphProto, opset: int, *, name: Optional[str] = None):

        OnnxAdaptorBase.__init__(self, proto)
        self.opset = opset

        resource_names = [i.name for i in self.proto.initializer]
        operand_names = [i.name for i in self.proto.input]
        result_names = [i.name for i in self.proto.output]

        # maybe there is duplicate value in both inputs and outputs, remove it
        useless_results = set(result_names).intersection(set(operand_names))
        useless_results = useless_results.union(
            set(result_names).intersection(set(resource_names))
        )
        for res in useless_results:
            print(
                'Output "{}" is duplicated with graph input or initializer. It will be removed from exported IR.'.format(
                    res
                )
            )
        result_names = [o for o in result_names if o not in useless_results]

        # resource may also present in inputs, remove it
        useless_inputs = set(operand_names).intersection(set(resource_names))
        for i in useless_inputs:
            print(
                'Input "{}" is duplicated with graph initializer. It will be removed from exported IR.'.format(
                    i
                )
            )
        operand_names = [i for i in operand_names if i not in useless_inputs]

        input_protos = [i for i in self.proto.input if i.name in operand_names]
        output_protos = [o for o in self.proto.output if o.name in result_names]
        operands = [OnnxValueAdaptor(iproto) for iproto in input_protos]
        results = [OnnxValueAdaptor(oproto) for oproto in output_protos]
        GraphAdaptor.__init__(
            self, operands, results, self.proto.name if name is None else name
        )

        for operand in self.operands:
            self.stack.insert(operand)
        for result in self.results:
            self.stack.insert(result)

        # insert resource
        for rproto in self.proto.initializer:
            if rproto.name not in resource_names:
                continue
            value_adaptor = OnnxValueAdaptor(rproto)
            value_adaptor.bind(to_array(rproto))
            self.stack.insert(value_adaptor)

        # insert intermediate value
        value_protos = []
        for i in self.proto.value_info:
            if i.name not in result_names and i.name not in operand_names:
                value_protos.append(i)

        for vproto in value_protos:
            self.stack.insert(OnnxValueAdaptor(vproto))

        self.build()  # build nodes and def-use affiliations

    def build_nodes(self) -> List[OnnxNodeAdaptor]:
        node_adaptors = []
        for node_proto in self.proto.node:
            # There could be optional operands
            operands = [
                self.stack.find(name) if name else None for name in node_proto.input
            ]
            results = [self.stack.find(name) for name in node_proto.output]
            node_adaptors.append(
                OnnxNodeAdaptor(node_proto, "onnx", self.opset, operands, results)
            )
        return node_adaptors
