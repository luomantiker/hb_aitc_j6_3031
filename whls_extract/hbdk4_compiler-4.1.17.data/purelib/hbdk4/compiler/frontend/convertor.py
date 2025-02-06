from abc import abstractmethod
from typing import Union, List
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.frontend.registry import OpConvertorRegistry, OpType
from hbdk4.compiler.frontend.adaptor import NodeAdaptor


class OpConvertor(object):
    def __init__(self, signature: str, namespace: str, version: int, foldable: bool):
        self.type = OpType(signature, namespace, version)
        self.foldable = foldable

        self.registry = OpConvertorRegistry()
        self.registry.register(self)

    @abstractmethod
    def emit_mlir_op(
        self, adaptor: NodeAdaptor, *args, **kwargs
    ) -> Union[
        mlir.OpView,
        List[mlir.OpView],
        mlir.Value,
        List[mlir.Value],
        mlir.OpResult,
        mlir.OpResultList,
        List[mlir.OpResult],
    ]:
        raise ValueError("unsupported conversion of node {}".format(adaptor.doc))
