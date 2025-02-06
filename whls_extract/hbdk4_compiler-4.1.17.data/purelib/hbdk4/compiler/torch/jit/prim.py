from hbdk4.compiler.frontend.adaptor import NodeAdaptor, TypeAdaptor
from hbdk4.compiler.frontend.convertor import OpConvertor


class PrimConvertor(OpConvertor):
    def __init__(self, name: str):
        super().__init__(name, "prim", 0, True)


class ConstantConvertor(PrimConvertor):
    def __init__(self):
        super().__init__("Constant")


ConstantConvertor()


class GetAttrConvertor(PrimConvertor):
    def __init__(self):
        super().__init__("GetAttr")


GetAttrConvertor()


class NumToTensorConvertor(PrimConvertor):
    def __init__(self):
        super().__init__("NumToTensor")


NumToTensorConvertor()


class DistConstrcutConvertor(PrimConvertor):
    def __init__(self):
        super().__init__("DictConstruct")

    def emit_mlir_op(self, adaptor: NodeAdaptor, output_type, *args):
        return [{args[idx * 2]: args[idx * 2 + 1] for idx in range(len(args) // 2)}]


DistConstrcutConvertor()


class ListConstrcutConvertor(PrimConvertor):
    def __init__(self):
        super().__init__("ListConstruct")

    def emit_mlir_op(self, adaptor: NodeAdaptor, output_type, *args):
        return [args]


ListConstrcutConvertor()


class TupleConstrcutConvertor(PrimConvertor):
    def __init__(self):
        super().__init__("TupleConstruct")

    def emit_mlir_op(self, adaptor: NodeAdaptor, output_type, *args):
        return [args]


TupleConstrcutConvertor()


class ListUnpackConvetor(PrimConvertor):
    def __init__(self):
        super().__init__("ListUnpack")

    def emit_mlir_op(self, adaptor: NodeAdaptor, output_type, args):
        return args


ListUnpackConvetor()


class TupleUnpackConvetor(PrimConvertor):
    def __init__(self):
        super().__init__("TupleUnpack")

    def emit_mlir_op(self, adaptor: NodeAdaptor, output_type, args):
        return args


TupleUnpackConvetor()
