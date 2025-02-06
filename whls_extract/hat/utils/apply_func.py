# Copyright (c) Horizon Robotics. All rights reserved.
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import partial
from inspect import signature
from typing import Any, Callable, List, Optional, Tuple, Union

import horizon_plugin_pytorch
import numpy as np
import torch
from horizon_plugin_pytorch.qtensor import QTensor
from six.moves import map, zip
from torch import Tensor

try:
    from torch._six import string_classes
except ImportError:
    string_classes = (str, bytes)

__all__ = [
    "_as_list",
    "_as_numpy",
    "convert_numpy",
    "convert_tensor",
    "img_array2tensor",
    "img_tensor2array",
    "_get_keys_from_dict",
    "_is_increasing_sequence",
    "multi_apply",
    "flatten",
    "regroup",
    "apply_to_collection",
    "to_flat_ordered_dict",
    "is_list_of_type",
    "to_cuda",
    "is_namedtuple",
    "check_type",
    "limit_period",
    "_call_as_tensor",
    "is_nan",
    "is_inf",
    "is_non_finite",
    "convert_tensor_v2",
]

# TODO(min.du, 0.1): remove prefix _ in interface name #


def _as_list(obj: Any) -> Sequence:
    """Convert the argument to a list if it is not already."""

    if isinstance(obj, (list, tuple)):
        return obj
    elif isinstance(obj, set):
        return list(obj)
    else:
        return [obj]


def _as_numpy(a):
    """Convert a (list of) numpy into numpy.ndarray.

    # TODO(min.du, 0.1): need refactor #

    """
    if isinstance(a, (list, tuple)):
        out = list(a)
        try:
            out = np.concatenate(out, axis=0)
        except ValueError:
            out = np.array(out)
        return out
    return a


def _call_as_tensor(func, *args, **kwargs):
    """
    Call func as on Tensor.

    Used on Qtensor when it behaves abnormally compared to torch.Tensor.
    For example, the following code

    .. code-block:: python

        _call_as_tensor(torch.Tensor.requires_grad_, tensor)

    Will make tensor.requires_grad == True if type(tensor) is QTensor .
    """
    types = (torch.Tensor for _ in args)
    return torch.Tensor.__torch_function__(func, types, args, kwargs)


def convert_numpy(
    data: Any,
    to_list: bool = False,
    dtype: Optional[str] = None,
) -> Any:
    r"""Convert each Tensor array data field into a numpy, recursively."""
    elem_type = type(data)
    if (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if dtype:
            data = data.astype(dtype)
        data = data.tolist() if to_list else data
        return data
    elif isinstance(data, torch.Tensor):
        scale = None
        if isinstance(data, horizon_plugin_pytorch.qtensor.QTensor):
            scale = data.scale.cpu().numpy()
            data = data.as_subclass(torch.Tensor)
        data = data.detach().cpu().numpy()
        if dtype:
            data = data.astype(dtype)
        if to_list:
            data = data.tolist()
        if scale is not None:
            return (data, scale)
        else:
            return data
    elif isinstance(data, Mapping):
        return {
            key: convert_numpy(data[key], to_list=to_list, dtype=dtype)
            for key in data
        }
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return elem_type(
            *(convert_numpy(d, to_list=to_list, dtype=dtype) for d in data)
        )
    elif isinstance(data, Sequence) and not isinstance(data, string_classes):
        return [convert_numpy(d, to_list=to_list, dtype=dtype) for d in data]
    else:
        return data


def img_array2tensor(np_data: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np_data).unsqueeze(0).permute(0, 3, 1, 2).float()


def img_tensor2array(tensor_data: torch.Tensor) -> np.ndarray:
    return tensor_data.permute(0, 2, 3, 1).squeeze().numpy().astype("uint8")


# many people don't known this useful api, so annotate it here.
convert_tensor = torch.utils.data._utils.collate.default_convert


def convert_tensor_v2(
    data: Union[torch.Tensor, np.ndarray, Sequence, int, float]
) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data: Data to be converted.

    Returns:
        Converted tensor.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data.copy())
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


def _get_keys_from_dict(target_dict, field):
    """
    Get keys from dict recrusively.

    # TODO(min.du, 0.1): need refactor #

    """
    field_found = []
    for k, v in target_dict.items():
        if k == field:
            field_found.append(v)
        elif isinstance(v, dict):
            results = _get_keys_from_dict(v, field)
            for result in results:
                field_found.append(result)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    more_results = _get_keys_from_dict(item, field)
                    for another_result in more_results:
                        field_found.append(another_result)
    return field_found


def flatten(obj):
    """Flatten list/tuple/dict object and get layout.

    you can use `regroup` to get original object.

    Args:
        obj (Union[list, tuple, dict]): Object.

    Returns:
        flat (tuple): The flatten object.
        fmts (tuple): Layout of original object.

    Example::

        >>> data = [0, [1, 2]]
        >>> flatten_data, data_layout = flatten(data)
        >>> print(flatten_data)
        (0, 1, 2)
        >>> print(data_layout)
        (<class 'list'>, (<class 'object'>, (<class 'list'>, (<class 'object'>, <class 'object'>))))
        >>> group_data, left_data = regroup(flatten_data, data_layout)
        >>> print(group_data)
        [0, [1, 2]]
        >>> print(left_data)
        ()

    """  # noqa
    if isinstance(obj, dict):
        flat = []
        fmts = [dict, []]
        for key, value in obj.items():
            obj_i, fmt_i = flatten(value)
            flat.extend(obj_i)
            fmts[1].append((key, fmt_i))
        fmts[1] = tuple(fmts[1])
        return tuple(flat), tuple(fmts)
    elif isinstance(obj, (list, tuple)):
        flat = []
        fmts = [type(obj), []]
        for value in obj:
            obj_i, fmt_i = flatten(value)
            flat.extend(obj_i)
            fmts[1].append(fmt_i)
        fmts[1] = tuple(fmts[1])
        return tuple(flat), tuple(fmts)
    else:
        return (obj,), object


def regroup(
    obj: Union[list, tuple], fmts: Union[list, tuple], obj_idx: int = 0
) -> Tuple[Union[list, tuple, dict], int]:
    """Regroup a list/tuple of objects.

    Args:
        obj: List of flatten objects.
        fmts: Layout of original objects.
        obj_idx: The start idx of obj.
    Returns:
        group_data: The grouped objects.
        obj_stop_idx: The stop idx of obj.
    """
    if fmts is object:
        return obj[obj_idx], obj_idx + 1
    assert isinstance(fmts, (list, tuple))
    obj_type = fmts[0]
    if obj_type is dict:
        ret = {}
        for key, fmt_i in fmts[1]:
            ret[key], obj_idx = regroup(obj, fmt_i, obj_idx=obj_idx)
        return ret, obj_idx
    elif obj_type in (list, tuple):
        ret = []
        for fmt_i in fmts[1]:
            res, obj_idx = regroup(obj, fmt_i, obj_idx=obj_idx)
            ret.append(res)
        if obj_type is tuple:
            ret = tuple(ret)
        return ret, obj_idx
    else:
        raise TypeError(f"Unknown type: {obj_type}")


def apply_to_collection(
    data: Any,
    dtype: Union[type, tuple],
    function: Callable,
    *args,
    wrong_dtype: Optional[Union[type, tuple]] = None,
    **kwargs,
) -> Any:
    """
    Recursively applies a function to all elements of a certain dtype.

    Migrated from pytorch_lightning.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of
            ``function``)
        wrong_dtype: the given function won't be applied if this type is
            specified and the given collections is of the :attr:`wrong_type`
            even if it is of type :attr`dtype`
        **kwargs: keyword arguments (will be forwarded to calls of
            ``function``)

    Returns:
        the resulting collection
    """
    elem_type = type(data)

    # Breaking condition
    if isinstance(data, dtype) and (
        wrong_dtype is None or not isinstance(data, wrong_dtype)
    ):
        return function(data, *args, **kwargs)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        return elem_type(
            {
                k: apply_to_collection(v, dtype, function, *args, **kwargs)
                for k, v in data.items()
            }
        )

    if isinstance(data, tuple) and hasattr(data, "_fields"):  # named tuple
        return elem_type(
            *(
                apply_to_collection(d, dtype, function, *args, **kwargs)
                for d in data
            )
        )

    if isinstance(data, Sequence) and not isinstance(data, str):
        return elem_type(
            [
                apply_to_collection(d, dtype, function, *args, **kwargs)
                for d in data
            ]
        )

    # data is neither of dtype, nor a collection
    return data


def is_namedtuple(obj):  # noqa: D205,D400
    """Check whether `obj` is instance of tuple subclass which was created from
    collections.namedtuple.
    """
    return (
        isinstance(obj, tuple)
        and hasattr(obj, "_fields")
        and hasattr(obj, "_asdict")
    )


def to_flat_ordered_dict(
    obj: Any,
    key_prefix: Optional[str] = "",
    flat_condition: Optional[Callable[[Any], bool]] = None,
):  # noqa: D205,D400
    """Flatten a dict/list/tuple object into an `OrderedDict` object,
    the key of which is automatically generated.

    Args:
        obj: Object to be flattened.
        key_prefix: Prefix of keys of result dict.
        flat_condition: Function with (`key`, `values`) as input,
            return `True/False` means whether flat this `values` or not.

    Examples::

        >>> obj = dict(
        ...     a=[dict(c=1)],
        ...     d=(2, 3)
        ... )

        >>> to_flat_ordered_dict(obj, key_prefix='test')
        OrderedDict([('test_a_0_c', 1), ('test_d_0', 2), ('test_d_1', 3)])

        >>> to_flat_ordered_dict(obj, key_prefix='test',
        ...     flat_condition=lambda k, v: not isinstance(v, tuple))
        OrderedDict([('test_a_0_c', 1), ('test_d', (2, 3))])

    """
    assert isinstance(key_prefix, str), type(key_prefix)
    if flat_condition is not None:
        assert callable(flat_condition)

    def _append(x):
        return "%s_%s" % (key_prefix, x) if key_prefix != "" else x

    def _flat():
        if isinstance(obj, dict):
            for k, v in obj.items():
                name2val.update(
                    to_flat_ordered_dict(v, _append(k), flat_condition)
                )
        elif is_namedtuple(obj):
            for k, v in zip(obj._fields, obj):
                name2val.update(
                    to_flat_ordered_dict(v, _append(k), flat_condition)
                )
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                name2val.update(
                    to_flat_ordered_dict(v, _append(str(i)), flat_condition)
                )
        else:
            name2val[key_prefix] = obj

    assert isinstance(key_prefix, str), type(key_prefix)

    if flat_condition is not None:
        # check flat_condition lambda input args
        assert callable(flat_condition)
        formal_args = sorted(signature(flat_condition).parameters)
        assert len(formal_args) == 2, (
            "flat_condition input should be "
            f"(key, value), found {formal_args}"
        )

    name2val = OrderedDict()
    if flat_condition is None:
        _flat()
    elif flat_condition(key_prefix, obj):
        _flat()
    else:
        name2val[key_prefix] = obj

    return name2val


def _is_increasing_sequence(obj, strict: bool = True) -> bool:
    """Return whether an given sequence is increasing order.

    Args:
        obj: list/tuple of comparable, Sequence to be checked.
        strict: whether allow equal or not.

    Returns:
        flag: True means yes.

    # TODO(min.du, 0.1): input type requiring check #

    """
    obj = _as_list(obj)
    pre = obj[0]
    for x in obj[1:]:
        if strict:
            if x <= pre:
                return False
        elif x < pre:
            return False
        pre = x
    return True


def multi_apply_wrapper(func: Callable, **kwargs) -> Callable:
    _func = partial(func, **kwargs) if kwargs else func

    def multi_func(inputs: List[Any]):
        return [_func(i) for i in inputs]

    return multi_func


def multi_apply(func: Callable, *args, **kwargs) -> Tuple:
    """Use func on different objects and merge public attributes.

    Args:
        func: Function handle
        args: Args of all objects.
        kwargs: Shared on different objects.

    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = list(map(pfunc, *args))

    if isinstance(map_results[0], tuple):
        map_results = map(list, zip(*map_results))
    return tuple(map_results)


def is_list_sorted(lst: object):
    return isinstance(lst, (list, tuple)) and all(
        [lst[i] < lst[i + 1] for i in range(len(lst) - 1)]
    )


def is_list_of_type(lst: object, element_type):  # noqa: D205,D400
    """Check whether `lst` is a list/tuple, as well as it's elements are
    instances of `element_type`.

    Args:
        lst: Object to be check.
        element_type: Target element type.

    Returns:
        Return True if `lst` is a list/tuple of 'element_type', else return
        False.
    """
    return isinstance(lst, (list, tuple)) and all(
        isinstance(elem, element_type) for elem in lst
    )


def to_cuda(
    obj: Any,
    device: Optional[torch.device] = None,
    non_blocking: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
    inplace: bool = False,
) -> Any:
    """
    Move object to cuda.

    Args:
        obj (Any): Any data type containing tensor, such as tensor container,
            list tuple and dict, and also optimizer.
        device (:class:`torch.device`): The destination GPU device.
            Defaults to the current CUDA device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host.
            Otherwise, the argument has no effect. Default: ``False``.
        memory_format (:class:`torch.memory_format`, optional): the desired
            memory format of returned Tensor.
            Default: ``torch.preserve_format``.
        inplace (bool, optional): If `obj` is optimizer, inplace must be True.

    """
    flats, fmt = flatten(obj)
    flats = list(flats)
    tensor_look_up_table = {}
    for i in range(len(flats)):
        if isinstance(flats[i], torch.Tensor):
            if inplace:
                raise NotImplementedError
            if id(flats[i]) in tensor_look_up_table:
                flats[i] = tensor_look_up_table[id(flats[i])]
            else:
                as_cuda = flats[i].cuda(
                    device, non_blocking, memory_format=memory_format
                )
                tensor_look_up_table[id(flats[i])] = as_cuda
                flats[i] = as_cuda
        elif isinstance(flats[i], torch.optim.Optimizer):
            assert inplace, (
                "Please set inplace=True when apply " "`to_cuda` on optimizer"
            )
            optimizer = flats[i]
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(
                            device, non_blocking, memory_format=memory_format
                        )
        else:
            pass
    obj, obj_idx = regroup(tuple(flats), fmt)
    assert len(tuple(flats)) == obj_idx, obj_idx
    return obj


def check_type(obj: Any, allow_types: Union[Any, Tuple[Any]]) -> None:
    """Check whether the type of input obj meets the expectation.

    Args:
        obj: Input object.
        allow_types: Expected type.
    """
    if not isinstance(obj, allow_types):
        raise TypeError(f"Expected {allow_types}, but get {type(obj)}")


def limit_period(
    val: Union[Tensor, np.array],
    offset: float = 0.5,
    period: float = np.pi,
) -> Tensor:
    """Limit the value into a period for periodic function.

    Args:
        val: The value to be converted.
        offset: Offset to set the value range.
            Defaults is 0.5.
        period: Period of the value. Defaults is np.pi.

    Returns:
        Value in the range of [-offset * period, (1-offset) * period]
    """
    if isinstance(val, np.ndarray):
        return val - np.floor(val / period + offset) * period
    elif isinstance(val, torch.Tensor):
        return val - torch.floor(val / period + offset) * period
    else:
        raise NotImplementedError


def pytree_convert(
    input: Any,
    convert_type: Any,
    func: Callable,
    skip_unsupported: bool = True,
    strict_type: bool = False,
):
    """Manipulate the elements in python list/tuple/dict structures.

    Note:
        See `horizon_plugin_pytorch.utils.misc.pytree_convert` in
        horizon_plugin_pytorch >=1.9.1.

    Args:
        input: Input structure.
        convert_type: The type of elements to be manipulated.
        func: A function takes a target element and return a manipulated one.
        skip_unsupported: Whether skip unsupported type or raise an exception.
            Defaults to True.
        strict_type: Whether use strict type judjement.
            Defaults to False.

    Returns:
        Same structure as input with manipulated elements.
    """
    if (
        type(input) is convert_type
        if strict_type
        else isinstance(input, convert_type)
    ):
        return func(input)
    elif isinstance(input, (list, tuple)):
        return type(input)(
            pytree_convert(
                x, convert_type, func, skip_unsupported, strict_type
            )
            for x in input
        )
    elif isinstance(input, dict):
        ret = {}
        for k, v in input.items():
            ret[k] = pytree_convert(
                v, convert_type, func, skip_unsupported, strict_type
            )
        return ret
    elif skip_unsupported:
        return input
    else:
        raise TypeError("Unsupported input type {}".format(type(input)))


def to_device(
    data: Union[Tensor, QTensor],
    device: Union[str, torch.device] = "cpu",
):
    """Move data to target device.

    Note:
        See `horizon_plugin_pytorch.utils.misc.to_device` in
        horizon_plugin_pytorch >=1.10.1.

    Args:
        data: Input data.
        device: Target device.
    """

    def _to_device(x: Tensor):
        if isinstance(x, QTensor):
            return QTensor(
                x.as_subclass(Tensor).to(device),
                x.q_scale().to(device) if x.q_scale() is not None else None,
                x.dtype,
                x.per_channel_axis,
            )
        elif isinstance(x, Tensor):
            return x.to(device)
        else:
            raise NotImplementedError

    return pytree_convert(
        data,
        (Tensor,),
        _to_device,
        skip_unsupported=True,
    )


def is_nan(input: torch.Tensor) -> bool:
    """Return a boolean value representing if there is `nan` in input.

    Args:
        input: Input data (Tensor or QTenor)

    """

    if isinstance(input, QTensor):
        return is_nan(input.dequantize())
    else:
        return torch.isnan(input).any()


def is_inf(input: torch.Tensor) -> bool:
    """Return a boolean value representing if there is `inf` in input.

    Args:
        input: Input data (Tensor or QTenor)

    """

    if isinstance(input, QTensor):
        return is_inf(input.dequantize())
    else:
        return torch.isinf(input).any()


def is_non_finite(input: torch.Tensor) -> bool:
    """Return a boolean value representing if there is `inf` or `nan` in input.

    Args:
        input: Input data (Tensor or QTenor)
    """
    return is_nan(input) or is_inf(input)
