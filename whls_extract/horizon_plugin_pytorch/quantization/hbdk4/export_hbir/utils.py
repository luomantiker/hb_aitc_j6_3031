import base64
import logging
from pickle import dumps, loads

import torch
from hbdk4.compiler import Module, ir
from hbdk4.compiler.dialects.hbir import Bool8Type
from hbdk4.compiler.ops import b25, hbir
from torch import Tensor
from torch.utils._pytree import LeafSpec, TreeSpec  # noqa: F401

from horizon_plugin_pytorch.march import March, get_march

logger = logging.getLogger(__name__)


def to_numpy(x: Tensor):
    return x.as_subclass(Tensor).detach().cpu().contiguous().numpy()


def get_hbir_dtype(dtype):
    dtype_mapping = {
        torch.float16: ir.F16Type.get(),
        torch.float32: ir.F32Type.get(),
        torch.float64: ir.F64Type.get(),
        torch.bfloat16: ir.BF16Type.get(),
        torch.int8: ir.IntegerType.get_signed(8),
        torch.int16: ir.IntegerType.get_signed(16),
        torch.int32: ir.IntegerType.get_signed(32),
        torch.int64: ir.IntegerType.get_signed(64),
        torch.uint8: ir.IntegerType.get_unsigned(8),
        torch.bool: Bool8Type.get(),
    }
    return dtype_mapping[dtype]


def get_hbir_tensor_type(dtype, shape=None):
    if shape is None:
        return ir.UnrankedTensorType.get(get_hbir_dtype(dtype))
    else:
        return ir.RankedTensorType.get(shape, get_hbir_dtype(dtype))


def check_inplace(inplace):
    if inplace:
        logger.warning(
            "Inplaced operation detected, please make sure op input is not"
            " come from other inplaced operation (slice for example)."
        )


def check_training(training, op_name):
    if training:
        msg = "Cannot export {} in training mode".format(op_name)
        logger.error(msg)
        raise ValueError(msg)


class LayoutConverter:
    def __init__(self, force_2d=False) -> None:
        """Convert hbir layout.

        Args:
            force_2d (bool, optional):
                If False, input is treated as [N, C, ...], else treat input
                as [..., C, H, W].
                Defaults to False.
        """
        self.ori_rank = 0
        self.force_2d = force_2d

    def nchw_to_nhwc(self, x: ir.Value):
        if isinstance(x, ir.OpView):
            x = x.result

        shape_type = ir.ShapedType(x.type)
        rank = shape_type.rank

        if rank < 3:
            msg = "Cannot convert layout for rank less than 3"
            logger.error(msg)
            raise ValueError(msg)

        self.ori_rank = rank

        if self.force_2d:
            return hbir.transpose(
                x, [*list(range(rank - 3)), rank - 2, rank - 1, rank - 3]
            )
        else:
            return hbir.transpose(x, [0, *list(range(2, rank)), 1])

    def nhwc_to_nchw(self, x: ir.Value):
        if self.ori_rank == 0:
            msg = "Call nhwc_to_nchw before nchw_to_nhwc"
            logger.error(msg)
            raise RuntimeError(msg)

        if isinstance(x, ir.OpView):
            x = x.result

        shape_type = ir.ShapedType(x.type)
        rank = shape_type.rank

        if self.force_2d:
            return hbir.transpose(
                x, [*list(range(rank - 3)), rank - 1, rank - 3, rank - 2]
            )
        else:
            return hbir.transpose(
                x, [0, (rank - 1), *list(range(1, rank - 1))]
            )


def pickle_treespec(tree_spec: TreeSpec) -> str:
    return str(base64.b32encode(dumps(tree_spec)), encoding="utf-8")


def unpickle_treespec(data: str) -> TreeSpec:
    return loads(base64.b32decode(bytes(data, encoding="utf8")))


def lut_with_march(*args, **kwargs):
    if get_march() in (March.NASH_E, March.NASH_M, March.NASH_P, March.NASH):
        # TODO: change to b30.lut after hbdk expose it
        return b25.lut(*args, **kwargs)
    elif get_march() == March.BAYES:
        return b25.lut(*args, **kwargs)
    else:
        raise ValueError("Unsupported march {}".format(get_march()))


def hbir_module_repr(model: Module):
    return model.module.operation.get_asm(
        enable_debug_info=True, pretty_debug_info=True
    )


def get_hbdk4_version():
    from hbdk4.compiler import version

    return version.VERSION
