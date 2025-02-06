from typing import List, Optional  # noqa: F401
from hbdk4.compiler.frontend.adaptor import NodeAdaptor
from hbdk4.compiler.frontend.convertor import OpConvertor
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.ops import hbir


def is_unidirectional_broadcastable(a_shape, b_shape):
    """
    In ONNX, tensor B is unidirectional broadcastable to tensor A if one of the following is true:
    1. Tensor A and B both have exactly the same shape.
    2. Tensor A and B all have the same number of dimensions and the length of each dimensions is either a common length or B's length is 1.
    3. Tensor B has too few dimensions, and B can have its shapes prepended with a dimension of length 1 to satisfy property 2.

    Args:
    a_shape (tuple): Shape of tensor A.
    b_shape (tuple): Shape of tensor B.

    Returns:
    bool: True if tensor B is broadcastable to tensor A, False otherwise.
    """
    len_a = len(a_shape)
    len_b = len(b_shape)

    # If B has fewer dimensions, prepend B's shape with 1s
    if len_b < len_a:
        b_shape = [1,] * (
            len_a - len_b
        ) + list(b_shape)

    # Check the compatibility of each dimension
    for i in range(len(a_shape)):
        if a_shape[i] != b_shape[i] and b_shape[i] != 1:
            return False

    return True


class Opset17(OpConvertor):
    def __init__(self, name: str):
        super().__init__(name, "onnx", 17, True)


class LayerNormalization(Opset17):
    def __init__(self):
        super().__init__("LayerNormalization")

    def emit_mlir_op(
        self,
        adaptor: NodeAdaptor,
        otype: mlir.Type,
        x: mlir.Value,
        scale: mlir.Value,
        bias: Optional[mlir.Value] = None,
        *,
        axis=-1,
        epsilon=1e-05,
        stash_type=1,
    ):
        if stash_type != 1:
            raise ValueError(
                f"LayerNormalization not support for stash_type={stash_type} != 1."
            )
        itype = adaptor.operands[0].type
        rank = len(itype.shape)
        assert (-rank <= axis) and (
            axis < rank
        ), 'invalid "aixs" attr of LayerNormalization op'
        norm_dims = (
            list(range(rank + axis, rank)) if (axis < 0) else list(range(axis, rank))
        )

        # align scale rank to targeted rank
        stype = adaptor.operands[1].type
        expansion = [1] * (itype.rank - stype.rank)
        if not is_unidirectional_broadcastable(itype.shape, stype.shape):
            raise ValueError(
                f"scale shape {stype.shape} can not broadcast to input shape {itype.shape}"
            )
        scale = hbir.reshape(scale, [*expansion, *stype.shape])

        if bias is not None:  # align bias rank to targeted rank
            btype = adaptor.operands[2].type
            if not is_unidirectional_broadcastable(itype.shape, btype.shape):
                raise ValueError(
                    f"bias shape {btype.shape} can not broadcast to input shape {itype.shape}"
                )
            expansion = [1] * (itype.rank - btype.rank)
            bias = hbir.reshape(bias, [*expansion, *btype.shape])

        x = hbir.layernorm(
            x, dims=norm_dims, eps=epsilon, weight=scale, bias=bias, output_type=otype
        )
        return x


LayerNormalization()
