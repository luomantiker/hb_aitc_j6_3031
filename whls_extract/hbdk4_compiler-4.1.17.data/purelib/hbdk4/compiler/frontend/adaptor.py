from abc import abstractmethod, abstractproperty
from typing import Any, Iterable, Tuple, Dict, List

import numpy as np

from hbdk4.compiler.frontend.registry import OpType, OpConvertorRegistry

from hbdk4.compiler import ir as mlir
from hbdk4.compiler.overlay import Function

from hbdk4.compiler.dialects import func
from hbdk4.compiler._mlir_libs import _hbdk as _hbdk_cext


class TypeAdaptor:
    @abstractmethod
    def get_sub_types():
        pass

    @abstractproperty
    def is_dict(self) -> bool:
        pass

    @abstractproperty
    def is_tuple(self) -> bool:
        pass

    @abstractproperty
    def is_tensor(self) -> bool:
        pass

    @abstractproperty
    def is_class(self) -> bool:
        pass

    @abstractproperty
    def dtype(self) -> np.dtype:
        pass

    @abstractproperty
    def shape(self) -> Tuple[int]:
        pass

    @property
    def rank(self) -> int:
        return len(self.shape)

    @abstractmethod
    def emit_mlir_element_type(self) -> mlir.Type:
        pass

    def emit_mlir_type(self) -> mlir.RankedTensorType:
        if self.is_tensor:
            element_type = self.emit_mlir_element_type()
            return mlir.RankedTensorType.get(self.shape, element_type, None)
        elif self.is_tuple:
            return [t.emit_mlir_type() for t in self.get_sub_types()]
        return None  # for non tensor type, return nothing

    def verify_mlir_type(self, mlir_type: mlir.Type):
        mlir_shape = mlir.ShapedType(mlir_type).shape
        if mlir_shape != self.shape:
            raise ValueError(
                "verify shape fail {} vs {}".format(self.shape, mlir_shape)
            )
        element_type = self.emit_mlir_element_type()
        if mlir_type.element_type != element_type:
            raise ValueError(
                "verify element_type fail {} vs {}".format(
                    element_type, mlir_type.element_type
                )
            )


class AttributeAdaptor:
    def __init__(self):
        pass

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractproperty
    def value(self) -> Any:
        pass


def containing_mlir_value(value):
    if isinstance(value, (mlir.Value, mlir.OpView)):
        return True
    if isinstance(value, (list, tuple)):
        for v in value:
            if containing_mlir_value(v):
                return True
    if isinstance(value, dict):
        for k in value:
            if containing_mlir_value(value[k]):
                return True
    return False


class ValueAdaptor:
    def __init__(self, name):
        self.name = name
        self.value = None
        self.definer = None
        self.users = []

    @property
    def is_placeholder(self):
        return containing_mlir_value(self.value)

    @property
    def is_constant(self):
        return not self.is_placeholder

    @property
    @abstractmethod
    def type(self) -> str:
        pass

    def bind(self, value: Any):
        self.value = value

    @abstractmethod
    def emit_mlir_const_op(self):
        pass

    # emit hbir.const op or use exising result.
    def emit_or_get_value(self):
        if self.is_constant:
            if self.type.is_tensor:
                # automatically convert tensor constant to hbir.const
                const_op = self.emit_mlir_const_op()
                _hbdk_cext.set_name(const_op, self.name)
                return const_op
        return self.value


class NodeAdaptor:
    def __init__(
        self, operands: Tuple[ValueAdaptor], results: Tuple[ValueAdaptor], name: str
    ):
        self.operands = operands
        self.results = results
        self.name = name

    @abstractproperty
    def type(self) -> OpType:
        pass

    @abstractproperty
    def attributes(self) -> Dict[str, AttributeAdaptor]:
        pass

    @abstractproperty
    def doc(self) -> str:
        pass

    @abstractproperty
    def invoke(self, *args, **kwargs):
        pass

    def emit_mlir_location(self):
        return mlir.Location.name(self.name)

    def emit_mlir_op(self, registry: OpConvertorRegistry):
        cvt = registry.find(self.type)

        kwargs = {i: self.attributes[i].value for i in self.attributes}

        if cvt.foldable and all(
            [i.is_constant for i in self.operands if i is not None]
        ):
            # for const fold
            args = [i.value for i in self.operands]
            results = self.invoke(*args, **kwargs)
            for value_adaptor, constant in zip(self.results, results):
                value_adaptor.bind(constant)
        else:
            if len(self.results) == 1:
                args = [i.type.emit_mlir_type() for i in self.results]
            else:
                # more than one result, in single variable
                args = [[i.type.emit_mlir_type() for i in self.results]]
            args.extend(
                [
                    i.emit_or_get_value() if i is not None else None
                    for i in self.operands
                ]
            )

            with self.emit_mlir_location():
                try:
                    results = cvt.emit_mlir_op(self, *args, **kwargs)

                    if isinstance(results, mlir.Value):
                        results = [results]

                    for value_adaptor, result in zip(self.results, results):
                        value_adaptor.bind(result)
                        if isinstance(result, (mlir.Value, mlir.OpView)):
                            _hbdk_cext.set_name(result, value_adaptor.name)
                except Exception as e:
                    print(
                        f"\033[91mConvert Node {self.type.namespace}::{self.type.signature} ver {self.type.version} failed!\033[91m"
                    )
                    print("\033[93mDetails of the node...\033[00m")
                    print(f"{self.type.description}")
                    raise e


class Stack:
    def __init__(self):
        self.impl = {}

    def insert(self, value: ValueAdaptor):
        if isinstance(value, ValueAdaptor):
            if value.name in self.impl:
                raise ValueError("value {} already exists".format(value.name))
            self.impl[value.name] = value

    def find(self, key: str) -> ValueAdaptor:
        if key in self.impl:
            return self.impl[key]
        raise ValueError("key {} not found".format(key))


class GraphAdaptor:
    def __init__(
        self,
        operands: Iterable[ValueAdaptor],
        results: Iterable[ValueAdaptor],
        name: str,
    ):
        self.operands = operands
        self.results = results
        self.name = name

        self.stack = Stack()

        self.nodes = []

    @abstractproperty
    def build_nodes(self) -> List[NodeAdaptor]:
        pass

    def build(self):
        self.nodes = self.build_nodes()

        for node in self.nodes:
            for result in node.results:
                result.definer = node
            for operand in node.operands:
                if operand is not None:
                    operand.users.append(node)

    def statistics(self):
        ops = dict()
        for op in self.nodes:
            if op.type.key in ops:
                ops[op.type.key] += 1
            else:
                ops[op.type.key] = 1

        key_len = len(max(ops.keys(), key=len))
        for k in sorted(ops.keys()):
            spaces = " " * (key_len - len(k))
            print("\t", k, spaces, ":", ops[k])

    def emit_mlir_func_op(self, registry: OpConvertorRegistry, lower_non_tensor):
        # dfs TypeAdaptor. when type is tensor then enter given function. we finally
        def dfs_type_adaptor(type_adaptor, func, name):
            if type_adaptor.is_tensor:
                return func(type_adaptor, name)
            if type_adaptor.is_dict:
                sub_types = type_adaptor.get_sub_types()
                return {
                    key: dfs_type_adaptor(
                        sub_types[key], func, "{}[{}]".format(name, key)
                    )
                    for key in sorted(sub_types.keys())
                }
            if type_adaptor.is_tuple:
                return [
                    dfs_type_adaptor(sub_type, func, "{}[{}]".format(name, key))
                    for key, sub_type in enumerate(type_adaptor.get_sub_types())
                ]
            if type_adaptor.is_class:
                return func(type_adaptor, name)

        flatten_operand_types = []
        flatten_operand_names = []
        packed_operand_types = []

        flatten_result_types = []
        flatten_result_names = []

        if not lower_non_tensor:
            # FIXME: when mlir supports tuple and named_tuple, directly convert list/tuple to mlir.tuple_type, dict/named_tuple to mlir.named_tuple_type
            flatten_operand_types = self.operands
            flatten_result_types = self.results
        else:
            # Now we have no direct support for named_tuple and tuple, flatten these constructs in the generated mlir func
            def collect_operand_type(x: TypeAdaptor, name: str):
                flatten_operand_types.append(x)
                flatten_operand_names.append(name)
                return x

            packed_operand_types = [
                dfs_type_adaptor(v.type, collect_operand_type, v.name)
                for v in self.operands
            ]

            # flatten nested operands
            def collect_result_type(x: TypeAdaptor, name: str):
                flatten_result_types.append(x)
                flatten_result_names.append(name)
                return x

            [
                dfs_type_adaptor(v.type, collect_result_type, v.name)
                for v in self.results
            ]

        func_type = mlir.FunctionType.get(
            inputs=[v.emit_mlir_type() for v in flatten_operand_types],
            results=[v.emit_mlir_type() for v in flatten_result_types],
        )

        func_op = func.FuncOp(self.name, func_type, visibility=None)

        with mlir.InsertionPoint(func_op.add_entry_block()):

            def _dfs(x, type, func):
                if isinstance(x, (list, tuple)):
                    return [_dfs(i, type, func) for i in x]
                if isinstance(x, dict):
                    return {k: _dfs(x[k], type, func) for k in sorted(x.keys())}
                if isinstance(x, type):
                    return func(x)
                return x

            flatten_operand_mlir_values = list(func_op.entry_block.arguments)

            def collect_operand_mlir_values(x: TypeAdaptor):
                idx = flatten_operand_types.index(x)
                return flatten_operand_mlir_values[idx]

            packed_mlir_value = _dfs(
                packed_operand_types, TypeAdaptor, collect_operand_mlir_values
            )

            for value_adaptor, mlir_value in zip(self.operands, packed_mlir_value):
                value_adaptor.bind(mlir_value)

            for node_adaptor in self.nodes:
                node_adaptor.emit_mlir_op(registry)

            flatten_result_mlir_values = []

            def collect_result_mlir_values(x: mlir.Value):
                flatten_result_mlir_values.append(x)

            [
                _dfs(
                    v.emit_or_get_value(),
                    (mlir.Value, mlir.OpView),
                    collect_result_mlir_values,
                )
                for v in self.results
            ]

            # emit return
            func.ReturnOp(flatten_result_mlir_values)
            # Recompute the function type.
            inputs_type = [v.emit_mlir_type() for v in flatten_operand_types]
            return_types = [v.type for v in flatten_result_mlir_values]
            function_type = mlir.FunctionType.get(
                inputs=inputs_type, results=return_types
            )
            func_op.attributes["function_type"] = mlir.TypeAttr.get(function_type)

        helper = Function(func_op, None, None)

        for idx, name in enumerate(flatten_operand_names):
            helper.inputs[idx].name = name
        for idx, name in enumerate(flatten_result_names):
            helper.outputs[idx].name = name
