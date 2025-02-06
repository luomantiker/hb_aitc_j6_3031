import numpy as np
from typing import Any, Iterable, List, Union

from ..hbtl import is_np_array, is_torch_tensor

from .. import ir
from .._mlir_libs import _hbdk
from ..dialects._ods_common import get_default_loc_context
from ..dialects import hbir
from ..dialects._ods_common import get_op_result_or_value


def create_elements_attr(array: np.ndarray) -> ir.OpView:
    """Create hbir.const from np.array

    Args:
        array (np.ndarray)
        loc (Union[None, ir.Location], optional): Specify the location for hbir.constant_op. Defaults to None.

    Returns:
        ir.OpView: mlir.Operation
    """
    if array.dtype == np.bool_:
        return hbir.create_bool8_attr(array, get_default_loc_context())

    if array.dtype == np.float64:  # no f64 support. convert to f32
        array = array.astype(np.float32)

    return ir.DenseElementsAttr.get(array, False)  # must be signed or unsigned


def create_const_op(
    array: np.ndarray, loc: Union[None, ir.Location] = None
) -> ir.OpView:
    """Create hbir.const from np.array

    Args:
        array (np.ndarray)
        loc (Union[None, ir.Location], optional): Specify the location for hbir.constant_op. Defaults to None.

    Returns:
        ir.OpView: mlir.Operation
    """

    ele_attr = create_elements_attr(array)
    out_type = (
        ele_attr.type
        if array.dtype != np.bool_
        else ir.RankedTensorType.get(array.shape, hbir.Bool8Type.get())
    )
    return hbir.ConstantOp(out_type, ele_attr, loc=loc)


def get_value_or_create_const(
    arg: Union[
        ir.OpView,
        ir.Operation,
        ir.Value,
        ir.OpResultList,
        np.ndarray,
        Iterable,
    ]
) -> Union[ir.Value, List[ir.Value]]:
    """normalize value
    If arg is a variant of mlir.Value then normalize it to mlir.Value.
    If arg is np.ndarray then create hbir.const op.

    Args:
        arg (Union[ir.OpView, ir.Operation, ir.Value, ir.OpResultList, np.ndarray, Iterable])

    Returns:
        Union[ir.Value, List[ir.Value]]
    """
    if arg is None:
        return None

    if isinstance(arg, (int, float)):
        arg = np.array(arg)

    if is_np_array(arg):
        return create_const_op(arg)

    if is_torch_tensor(arg):
        return create_const_op(arg.detach().numpy())

    # dfs visit
    if isinstance(arg, Iterable):
        return [get_value_or_create_const(a) for a in arg]

    return get_op_result_or_value(arg)


def get_type_or_create_unranked_type(mlir_type: Union[None, ir.Type]) -> ir.Type:
    """normalize type
    If mlir_type is given then return the type
    If mlir_type is missing then by default create an unranked tensor of f32. FIXME: support type inference for ops

    Args:
        mlir_type (Union[None, ir.Type])

    Returns:
        ir.Type
    """

    if isinstance(mlir_type, (list, tuple)):
        return [get_type_or_create_unranked_type(t) for t in mlir_type]

    if mlir_type is None:
        return ir.NoneType.get(get_default_loc_context())
    if isinstance(mlir_type, ir.ShapedType):
        return mlir_type
    else:
        return ir.UnrankedTensorType.get(mlir_type)


enable_inspect = False


def set_inspect(en: bool):
    global enable_inspect
    enable_inspect = en


def get_loc_or_create_from_frames(loc: Union[None, ir.Location]) -> ir.Location:
    global enable_inspect

    if enable_inspect and loc is None:
        from inspect import stack, getframeinfo

        cur_stack = stack()

        frame_locs = []
        for i in range(2, len(cur_stack)):
            prev_frame = getframeinfo(cur_stack[i - 1][0])
            callee = ir.Location.name(prev_frame.function, None)

            cur_frame = getframeinfo(cur_stack[i][0])
            fileline = ir.Location.file(cur_frame.filename, cur_frame.lineno, 0)

            frame_locs.append(ir.Location.callsite(callee, [fileline]))
        loc = ir.Location.fused(frame_locs)
    return loc


def create_array_attr(
    values: Union[List[int], List[float], None]
) -> Union[None, ir.ArrayAttr]:
    """create from list of integers

    Args:
        values (List[int])

    Raises:
        ValueError: If values is not list of integers or floating-points

    Returns:
        ir.ArrayAttr
    """
    if values is None:
        return None
    if isinstance(values, Iterable):
        array = []
        for v in values:
            if isinstance(v, bool):
                array.append(ir.BoolAttr.get(v))
            elif isinstance(v, (int, np.int64, np.int32, np.int16, np.int8)):
                array.append(
                    ir.IntegerAttr.get(ir.IntegerType.get_signless(64), int(v))
                )
            elif isinstance(v, (float, np.float32, np.float64)):
                array.append(ir.FloatAttr.get(ir.F64Type.get(), float(v)))
            elif isinstance(v, str):
                array.append(ir.StringAttr.get(v))
            elif isinstance(v, list) or isinstance(v, tuple):
                array.append(create_array_attr(v))
            else:
                raise ValueError(
                    "unknown element type of {} for ArrayAttr".format(type(v))
                )
        return ir.ArrayAttr.get(array)
    raise ValueError("cannot create ArrayAttr from", values)


def create_attr(value: Union[int, float, bool, str, None]) -> Union[None, ir.Attribute]:
    """crete attr from opaque value

    Args:
        value (Union[int, float, bool, str, None])

    Raises:
        ValueError: If type of value is not supported

    Returns:
        ir.Attribute
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return ir.BoolAttr.get(value)
    if isinstance(value, int):
        return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
    if isinstance(value, float):
        return ir.FloatAttr.get(ir.F64Type.get(), value)
    if isinstance(value, str):
        return ir.StringAttr.get(value)
    raise ValueError("unknown element type of {} ".format(type(value)))


# valueName to distinguish duplicate enum
def parse_enum(value: str, valueName: str = "") -> ir.Attribute:
    """parse string to hbdk enum attributes"""
    return _hbdk.parse_enum(value, valueName, get_default_loc_context())
