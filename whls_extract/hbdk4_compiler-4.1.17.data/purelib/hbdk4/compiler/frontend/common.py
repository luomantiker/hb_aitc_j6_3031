from typing import Optional
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.ops import hbir


def nchw_to_nhwc(x: mlir.Value, otype: Optional[mlir.Type] = None):
    if isinstance(x, mlir.OpView):
        x = x.result
    if otype == None:
        otype = mlir.UnrankedTensorType.get(x.type.element_type)
    rank = mlir.ShapedType(x.type).rank
    assert rank > 2 and rank < 6
    return hbir.transpose(x, [0, *[i for i in range(2, rank)], 1], output_type=otype)


def nhwc_to_nchw(x: mlir.Value, otype: Optional[mlir.Type] = None):
    if isinstance(x, mlir.OpView):
        x = x.result
    if otype == None:
        otype = mlir.UnrankedTensorType.get(x.type.element_type)
    rank = mlir.ShapedType(x.type).rank
    assert rank > 2 and rank < 6
    return hbir.transpose(
        x, [0, (rank - 1), *[i for i in range(1, rank - 1)]], output_type=otype
    )


def get_unranked(type: mlir.Type):
    return mlir.UnrankedTensorType.get(type.element_type)
