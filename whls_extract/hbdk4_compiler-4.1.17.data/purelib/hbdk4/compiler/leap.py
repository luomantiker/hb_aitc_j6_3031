import functools
import types
import sys
import os
import inspect
import copy
import gc

import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Union, Tuple

from hbdk4.compiler.overlay import Module, Function
from hbdk4.compiler._mlir_libs._mlir.ir import Context
from hbdk4.compiler import ir as mlir, hbtl as hbtl_ops
from hbdk4.compiler._mlir_libs._hbdk import Dispatcher, Status
from hbdk4.compiler.dialects.func import FuncOp, ReturnOp
from hbdk4.compiler.dialects._ods_common import get_default_loc_context
from hbdk4.compiler.utils.types import TensorType, Dtype
from hbdk4.compiler.hbtl import is_tensor
from hbdk4.compiler.ops import hbir, hbtl, qnt
from hbdk4.compiler.dialects.hbir import TrackAttr
from hbdk4.compiler._mlir_libs import _hbdk as _hbdk_cext

__all__ = ["leap_export", "leap_func", "load_library"]

# Leap-provided types
# TODO: maybe it's better define our own type class, for now, simply use numpy's type
# Singed integer
int8 = Dtype("si8")
int16 = Dtype("si16")
int32 = Dtype("si32")
int64 = Dtype("si64")
# Unsigned integer
uint8 = Dtype("ui8")
uint16 = Dtype("ui16")
uint32 = Dtype("ui32")
# Floating-point
float16 = Dtype("f16")
float32 = Dtype("f32")

TORCH_EXPORT_FLAG = "is_horizon_plugin_pytorch_export"
HBDK_EXPORT_FLAG = False


def invoke(module: Module, entry_function_name: str, *args):
    """This function runs the specified function of the module according to func_name

    Args:
        * module (Module): the module that contains the function we'd like to run
        * entry_function_name (str): the name of the function
        * args: input arguments to the function

    RETURNS::
        * output_list: type of np.ndarray or torch.tensor or their list, execution results
    """
    assert isinstance(
        module, Module
    ), "The first arg of leap.invoke() must be Module object"
    assert isinstance(
        entry_function_name, str
    ), "The second arg of leap.invoke() must be str object"

    function = module[entry_function_name]
    assert len(function.inputs) == len(
        args
    ), "Func input number must equals args number"
    global HBDK_EXPORT_FLAG
    if HBDK_EXPORT_FLAG:
        if not _hbdk_cext._is_uninitialized_source_infos(module.module):
            raise RuntimeError("The invoked function should not have been converted")

        func_op = module[entry_function_name].opview
        arguments_mapping = {k: v for k, v in zip(func_op.arguments, args)}
        op_to_cloned_op = {}

        assert len(func_op.regions) == 1, "FuncOp should have only one region"
        assert len(func_op.regions[0].blocks), "FuncOp should have only one block"

        def get_cloned_counterpart(operand):
            # First, find the index of this operand in the owner op's results list
            result_number = -1
            for idx, result in enumerate(operand.owner.results):
                if result == operand:
                    result_number = idx

            # Obtain the cloned counterpart of this operand
            assert operand.owner in op_to_cloned_op, "Non-DAG graph"
            return op_to_cloned_op[operand.owner].results[result_number]

        for op in func_op.regions[0].blocks[0]:
            if not op.OPERATION_NAME == "func.return":
                cloned_op = op.operation.clone()
                op_to_cloned_op[op] = cloned_op
                for i, operand in enumerate(cloned_op.operands):
                    if operand in func_op.arguments:
                        # Set operands that are function arguments
                        cloned_op.operands[i] = arguments_mapping[operand]
                        continue
                    if isinstance(operand.owner, mlir.Operation):
                        # Set operands that are the results of some other ops
                        cloned_op.operands[i] = get_cloned_counterpart(operand)
                    else:
                        assert (
                            False
                        ), "operands can only be either arguments or results of other ops"
            else:
                assert (
                    len(op.operands) > 0
                ), "func.return should have at least one operand"
                if len(op.operands) == 1:
                    return get_cloned_counterpart(op.operands[0])
                else:
                    return [get_cloned_counterpart(operand) for operand in op.operands]

    else:

        assert all(
            is_tensor(arg) for arg in args
        ), "Args type must be torch.Tensor or np.ndarray"
        input_name_list = []
        output_name_list = []
        for idx, input in enumerate(function.inputs):
            input.name = "input" + str(idx)
            input_name_list.append(input.name)
        for idx, output in enumerate(function.outputs):
            output.name = "output" + str(idx)
            output_name_list.append(output.name)
        input_dict = {k: v for k, v in zip(input_name_list, args)}
        output_dict = function.feed(input_dict)
        output_list = [output_dict[k] for k in output_name_list]
        return output_list[0] if len(output_list) == 1 else output_list


def _loc(func, ctx=get_default_loc_context()):
    """For func that generate mlir ops, this function returns the corresponding loc info of those generated ops

    Args:
        func: python function or function name str that generates mlir ops

    Returns:
        mlir.Location: loc info of generated ops
    """
    with ctx:
        # import statement must be placed here or will result in infinite loop
        from inspect import stack

        for i, frame_info in enumerate(stack()):
            if frame_info.filename != os.path.abspath(__file__):
                caller = frame_info
                break

        debug_info = dict()

        def get_loc_list_from(loc: mlir.Location) -> List[mlir.Location]:
            loc_type = _hbdk_cext.get_location_type(loc)
            if loc_type == "fused":
                fused_attr = _hbdk_cext.get_track_attr_from_loc(loc)
                if fused_attr:
                    debug_info.update(TrackAttr(fused_attr).debug_info)
                loc_list = []
                for loc_single in _hbdk_cext.get_locations_from_fused_loc(loc):
                    loc_list += get_loc_list_from(loc_single)
                return loc_list
            else:
                return [loc]

        current_loc_list = get_loc_list_from(mlir.Location.current)
        file_location = None
        name_location = None
        for current_loc in current_loc_list:
            loc_type = _hbdk_cext.get_location_type(current_loc)
            if loc_type == "file":
                file_location = current_loc
            elif loc_type == "name":
                name_location = current_loc

        func_name = func if isinstance(func, str) else func.__name__
        if not file_location:
            file_location = mlir.Location.file(
                caller.filename,
                caller.lineno,
                col=caller.code_context[0].find(func_name + "("),
                context=ctx,
            )
        if not name_location:
            name_location: mlir.Location = mlir.Location.name(func_name)
        fused_location = mlir.Location.fused(
            [file_location, name_location], TrackAttr.get(debug_info)
        )
    return fused_location


def _flatten_list(lst, func):
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from _flatten_list(item, func)
        else:
            yield func(item)


def _is_tensor_type(types):
    if isinstance(types, (list, tuple)):
        return all(_is_tensor_type(type) for type in types)
    return isinstance(types, TensorType)


def _build_mlir_module(
    pyfunc: Callable,
    inputs: List[TensorType],
    name: str,
    hook: Callable,
    default_context=True,
    # binding_keys = []
):
    """Build a mlir function, and pass the function arguments to hook function

    Args:
        pyfunc (Callable): The python function, also can be leap.{op}
        inputs (_type_): The input types of python function
        name (str): The name of builded mlir function
        hook (_type_): hook function
        default_context (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    context = get_default_loc_context() if default_context else Context()

    with context:
        module = mlir.Module.create()
        assert _is_tensor_type(
            inputs
        ), "Invalid inputs to build a mlir module, which inputs should be TensorType or a list of TensorType"

        input_types = list(_flatten_list(inputs, lambda x: x.mlir_tensor_type))

        location = _loc(pyfunc, context)
        # Leave the results to blank because it can only be determined after building the graph
        function = FuncOp(
            name,
            (
                input_types,
                [],  # set result types later
            ),
            loc=location,
        )
        module.body.append(function)
        entry_block = function.add_entry_block()
        insertion_point = mlir.InsertionPoint(entry_block)
        with insertion_point:
            global HBDK_EXPORT_FLAG
            HBDK_EXPORT_FLAG = True
            if isinstance(inputs[0], Iterable):
                return_values = hook([argument for argument in function.arguments])
            else:
                real_inputs = [argument for argument in function.arguments]
                return_values = hook(*real_inputs)
            HBDK_EXPORT_FLAG = False

            results = []
            # Record the results, because the FuncOp will drop the real type of result
            if results is None:
                results = []
            if isinstance(return_values, tuple):
                results = list(return_values)
            elif isinstance(return_values, mlir.Value):
                # Returning a single value is fine, coerce it into a list.
                results = [return_values]
            elif isinstance(return_values, mlir.OpView):
                # Returning a single operation is fine, coerce its results a list.
                results = return_values.operation.results
            elif isinstance(return_values, mlir.Operation):
                # Returning a single operation is fine, coerce its results a list.
                results = return_values.results
            else:
                results = list(return_values)
            return_types = [v.type for v in results]
            # return_types = get_result_types(return_values)
            function.attributes["function_type"] = mlir.TypeAttr.get(
                mlir.FunctionType.get(
                    inputs=input_types,
                    results=return_types,
                )
            )

            ReturnOp(results, loc=_loc(pyfunc))
            return LeapFunction(
                context, Module(module), Function(function), name, return_values
            )


class LeapFunction:
    """A class record all the information about mlir function"""

    def __init__(self, context, module, function, name, results):
        self.context = context
        self.module = module
        self.function = function
        self.name = name
        self.results = results

    def print(self, *args, **kwargs):
        self.function.opview.print(*args, **kwargs)


class FunctionRegistry:
    """The class records all of the function which is wrapped by leap.func

    Returns:
        _type_: _description_
    """

    _registry = {}

    def __init__(self, func):
        self.pyfun = func

    @staticmethod
    def declare(*inputs):
        return lambda f: FunctionRegistry.create(f, *inputs)

    @staticmethod
    def create(f, *inputs, name=None):
        function = FunctionRegistry(f)
        leap_function = _build_mlir_module(f, inputs, name or f.__name__, f)
        FunctionRegistry._registry[id(function)] = leap_function
        return function

    def print(self, *args, **kwargs):
        leap_function = FunctionRegistry._registry[id(self)]
        leap_function.function.opview.print(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.pyfun(*args, **kwargs)

    @property
    def module(self):
        leap_function = FunctionRegistry._registry[id(self)]
        return leap_function.module


def has_mlir_value(value):
    """Whether at least one of the arguments is mlir.Value"""
    if isinstance(value, list) or isinstance(value, tuple):
        return any(has_mlir_value(item) for item in value)
    if isinstance(value, dict):
        return has_mlir_value(value.values())
    return isinstance(value, mlir.Value)


def has_eager_value(value):
    """Whether at least one of the arguments is torch.Tensor or np.ndarray"""
    import torch

    if isinstance(value, list) or isinstance(value, tuple):
        return any(has_eager_value(item) for item in value)
    if isinstance(value, dict):
        return has_eager_value(value.values())
    return isinstance(value, torch.Tensor) or isinstance(value, np.ndarray)


def _torch_export_leap(func, *args, **kwargs):
    import torch

    assert TORCH_EXPORT_FLAG in kwargs and kwargs[TORCH_EXPORT_FLAG], ""
    assert callable(func), "The first argument func should be callable"
    kwargs.pop(TORCH_EXPORT_FLAG)

    arguments = [arg for arg in args if not isinstance(arg, torch.Tensor)]
    key_arguments = {k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}
    return functools.partial(func, *arguments, **key_arguments)


def _is_all_tensor(values):
    if isinstance(values, (list, tuple)):
        return all([is_tensor(value) for value in values])
    return False


class OpDef:
    """The base class, call leap.{op} will come here."""

    def __init__(self, func):
        self.func = func
        self.func_name = func.__name__
        self.signature = inspect.signature(func)

    @property
    def default_arguments(self):
        return {}

    @property
    def inputs(self):
        return ["input"]

    @property
    def optional_inputs(self):
        return []

    def _handle_output_type(self, output_type):
        if isinstance(output_type, (list, tuple)):
            return [self._handle_output_type(type) for type in output_type]
        elif isinstance(output_type, str):
            return Dtype(output_type).mlir
        elif isinstance(output_type, Dtype):
            return output_type.mlir
        elif isinstance(output_type, mlir.Type):
            return output_type
        else:
            raise ValueError(
                f"Invalid output_type, which type should be str or leap.Dtype, but its type is {type(output_type)}"
            )

    def emit_mlir_op(self, *args, **kwargs):
        """Build mlir op

        Returns:
            args: There must be mlir.Value
            kwargs:
        """
        # assert has_mlir_value(args)

        ctx = get_default_loc_context()
        for argument in list(_flatten_list(args, lambda x: x)):
            if isinstance(argument, mlir.Value):
                ctx = argument.type.context
                break

        with ctx:
            # assert has_mlir_value(
            #     args
            # ), "The input should have a mlir.Value in the process of creating mlir"
            if "output_type" in kwargs:
                # User have provided output_type by using leap-provided types, so here we need to map leap type to mlir type and override our default type (if there is one)
                output_type = kwargs.pop("output_type")
                mlir_type = self._handle_output_type(output_type)
                kwargs.update({"output_type": mlir_type})
            key_arguments = self.default_arguments
            key_arguments.update(kwargs)
            loc = _loc(self.func, ctx)

            with loc:
                return self.func(*args, **key_arguments)

    def _hash(self, *args, **kwargs):
        """Generate hash value based on the function name and the arguments value

        Returns:
            _type_: _description_
        """
        arguments = [*args]
        arguments.extend(kwargs.values())

        def _flatten_tensors(tensors):
            if isinstance(tensors, (list, tuple)):
                results = []
                for tensor in tensors:
                    results.extend(_flatten_tensors(tensor))
                return results
            assert is_tensor(
                tensors
            ), f"Should be a tensor, but the type is {type(tensors)}"
            return tensors.flatten().tolist()

        hash_arguments = []
        for arg in arguments:
            if is_tensor(arg) or _is_all_tensor(arg):
                hash_arguments.extend(_flatten_tensors(arg))
            elif isinstance(arg, list):
                hash_arguments.append(tuple(arg))
            else:
                hash_arguments.append(arg)
        return hash(tuple(hash_arguments))

    def __call__(self, *args, **kwargs):
        """Eager run: Call the XqEngine to run.
        Firstly, build a mlir module, and build a function based on the arguments.
        After building, pass the arguments to the mlir function and run it.

        Returns:
            _type_: _description_
        """

        bound_args = self.signature.bind(*args, **kwargs)
        arguments_dict = bound_args.arguments

        inputs = []
        constant_inputs = []
        for name in self.inputs:
            assert (
                name in arguments_dict
            ), f"Invalid 'inputs' property for {self.func_name}"
            arg = arguments_dict[name]
            if not is_tensor(arg) and not _is_all_tensor(arg):
                constant_inputs.append(name)
                continue
            # The multiple inputs, which input is list or tuple of tensor
            if isinstance(arg, (list, tuple)):
                inputs.append(
                    [TensorType(t.shape, dtype=Dtype(str(t.dtype))) for t in arg]
                )
            else:
                inputs.append(TensorType(arg.shape, dtype=Dtype(str(arg.dtype))))

        attributes = copy.deepcopy(bound_args)
        updated_inputs = [
            input for input in self.inputs if input not in constant_inputs
        ]
        for input in updated_inputs:
            attributes.arguments.pop(input)

        for key, value in self.default_arguments.items():
            if key not in attributes.arguments.keys():
                attributes.arguments[key] = value

        if "output_type" in attributes.arguments:
            otype = attributes.arguments["output_type"]
            attributes.arguments["output_type"] = self._handle_output_type(otype)

        name = self.func_name + "_" + str(self._hash(*args, **kwargs))
        leap_function = _build_mlir_module(
            self.func,
            inputs,
            name,
            functools.partial(self.func, *attributes.args, **attributes.kwargs),
        )

        model_inputs = [arguments_dict[name] for name in updated_inputs]
        module = leap_function.module
        if isinstance(self, MultiInputOpdef):
            model_inputs = _flatten_list(model_inputs, lambda x: x)
        outputs = module[name](*model_inputs)

        if isinstance(leap_function.results, mlir.Value):
            assert len(outputs) == 1
            output = outputs[0]
        else:
            output = outputs

        del leap_function.module
        del leap_function.context
        del leap_function
        gc.collect()
        return output


mlir_i8 = mlir.IntegerType.get_signed(8)
mlir_i16 = mlir.IntegerType.get_signed(16)


class FilterOpDef(OpDef):
    @property
    def default_arguments(self):
        return {"maxIndex_type": mlir_i16, "filterCoord_type": mlir_i16}


class TopkOpDef(OpDef):
    @property
    def default_arguments(self):
        return {"indices_type": mlir_i8}


class PointerPillarProcessOpDef(OpDef):
    @property
    def default_arguments(self):
        return {"coords_type": mlir_i8}

    @property
    def inputs(self):
        return ["points"]


class BinaryOpDef(OpDef):
    @property
    def inputs(self):
        return ["lhs", "rhs"]


class WhereOpDef(OpDef):
    @property
    def inputs(self):
        return ["condition", "lhs", "rhs"]


class MultiInputOpdef(OpDef):
    @property
    def inputs(self):
        return ["inputs"]


class ImageConvertOpDef(MultiInputOpdef):
    @property
    def default_arguments(self):
        return {"output_type": mlir_i8}


class RppOpDef(MultiInputOpdef):
    @property
    def inputs(self):
        return ["roi", "inputs"]


class GridSampleOpDef(OpDef):
    @property
    def inputs(self):
        return ["input", "grid"]


class RoiResizeOpDef(OpDef):
    @property
    def inputs(self):
        return ["y", "roi", "uv"]


class OpFactory:
    """
    OpFactory:
    With the same attribute operation will return the the same OpDef instance
    The attributes include 'inputs' and 'default_arguments':abbr:
        'inputs':  parameters which is op input in hbir definition
        'default_arguments': leap will help user to set it, for example: 'output_type' attribute of some leap op may be specific

    """

    @staticmethod
    def create_op(func):
        if func.__name__ == "filter":
            return FilterOpDef(func)
        if func.__name__ == "point_pillar_preprocess":
            return PointerPillarProcessOpDef(func)
        if func.__name__ == "image_convert":
            return ImageConvertOpDef(func)
        if func.__name__ == "topk":
            return TopkOpDef(func)
        if func.__name__ in [
            "add",
            "mul",
            "sub",
            "pow",
            "matmul",
            "bitwise_and",
            "bitwise_or",
            "bitwise_xor",
            "logical_and",
            "logical_or",
            "logical_xor",
            "correlation",
            "less",
            "less_equal",
            "equal",
        ]:
            return BinaryOpDef(func)
        if func.__name__ == "where":
            return WhereOpDef(func)
        if func.__name__ in [
            "concat",
            "dpp",
            "einsum",
            "fpp",
            "make_tuple",
            "roi_align",
            "stack",
        ]:
            return MultiInputOpdef(func)
        if func.__name__ == "roi_resize":
            return RoiResizeOpDef(func)

        return OpDef(func)


def _wrap_imported_func(module, func_names_to_exclude=None, func_names_to_include=None):
    """Decorates imported functions from the given module so that:
    1. The user can set the output_type using leap-provided types (e.g. leap.int8) instead of having to provide mlir.Type
    2. We can provide meaningful default output type for some ops (e.g. si8 for hbir.image_convert)
    3. We can set the loc info for each inserted op

    * If func_names_to_exclude is specified, we do not import those specified functions
    * If func_names_to_include is specified, we only import those specified functions
    * If none of them is specified, we import all functions
    """

    assert not (
        (func_names_to_exclude is not None) and (func_names_to_include is not None)
    ), "arguments func_names_to_exclude and func_names_to_include cannot be specified at the same time"

    def decorator(func):
        op_def = OpFactory.create_op(func)

        @functools.wraps(func.__name__)
        def leap_op_wrapper(*args, **kwargs):
            import torch

            if torch.overrides.has_torch_function(args):
                return torch.overrides.handle_torch_function(
                    leap_op_wrapper, args, *args, **kwargs
                )
            if TORCH_EXPORT_FLAG in kwargs and kwargs[TORCH_EXPORT_FLAG]:
                return _torch_export_leap(func, *args, **kwargs)

            global HBDK_EXPORT_FLAG
            if HBDK_EXPORT_FLAG:
                return op_def.emit_mlir_op(*args, **kwargs)

            return op_def(*args, **kwargs)

        return leap_op_wrapper

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, types.FunctionType):
            if func_names_to_exclude and (name in func_names_to_exclude):
                continue
            if func_names_to_include and (name not in func_names_to_include):
                continue
            setattr(sys.modules[__name__], name, decorator(obj))


class ModuleFinder:
    """This class parses function call as "leap.custom.xx.xx" to get name space and op name of custom op"""

    def __getattr__(self, module_name):
        class OpFinder:
            def __getattr__(self, op_name):
                def cpp_custom_wrapper(*args, **kwargs):
                    """Wrapper for leap.custom.
                    According to different input arguments to execute different function, if the input is mlir or has_torch_function, perform export,
                    if not, eager to execute the custom function by hbdk interpreter.

                    Returns:
                        _type_: _description_
                    """
                    from torch.overrides import (
                        has_torch_function,
                        handle_torch_function,
                    )

                    def export_for_torch():
                        import warnings

                        for k, v in kwargs.items():
                            if k != TORCH_EXPORT_FLAG:
                                warnings.warn(
                                    f"Keyword arguments won't be passed to hbir, which is {k}:{v}"
                                )
                        hbtl_args = []
                        hbtl_call_kwargs = {}
                        for arg in args:
                            if hbtl_ops.is_tensor_like(arg):
                                hbtl_args.append(hbtl_ops.get_tensor_type(arg, True))
                            else:
                                if "parameters" not in hbtl_call_kwargs.keys():
                                    hbtl_call_kwargs["parameters"] = []
                                hbtl_call_kwargs["parameters"].append(arg)
                                hbtl_args.append(arg)

                        schema, hbtl_types = hbtl_ops.get_schema_and_infer_type(
                            module_name, op_name, hbtl_args
                        )
                        outputs_type = [
                            TensorType.from_hbtl_tensor_type(hbtl_type).mlir_tensor_type
                            for hbtl_type in hbtl_types
                        ]
                        hbtl_call_kwargs["outputs_type"] = outputs_type
                        hbtl_call_kwargs["signature"] = str(schema)
                        hbtl_call_kwargs["isCustom"] = True
                        return functools.partial(hbtl.call, **hbtl_call_kwargs)

                    def export_mlir():
                        assert (
                            len(kwargs) == 0
                        ), "Keyword arguments are not supported for leap.custom to export."
                        infer_args_type = [
                            TensorType.from_mlir_value(ele)
                            if isinstance(ele, mlir.Value)
                            else ele
                            for ele in args
                        ]
                        hbtl_arg_types = [
                            hbtl_ops.get_tensor_type(
                                np.empty(ele.shape, dtype=ele.dtype.numpy)
                            )
                            if isinstance(ele, TensorType)
                            else ele
                            for ele in infer_args_type
                        ]
                        schema, hbtl_result_types = hbtl_ops.get_schema_and_infer_type(
                            module_name, op_name, hbtl_arg_types
                        )

                        result_types = [
                            TensorType.from_hbtl_tensor_type(hbtl_type).mlir_tensor_type
                            for hbtl_type in hbtl_result_types
                        ]
                        loc = _loc(module_name + "." + op_name)
                        inputs = [ele for ele in args if isinstance(ele, mlir.Value)]
                        params = [
                            ele for ele in args if not isinstance(ele, mlir.Value)
                        ]
                        with loc:
                            results = hbtl.call(
                                inputs,
                                str(schema),
                                isCustom=True,
                                parameters=params,
                                outputs_type=result_types,
                            )
                        if len(results) == 1:
                            return results[0]
                        else:
                            return results

                    # eager to execute the function
                    def eager_run():
                        assert (
                            len(kwargs) == 0
                        ), "Keyword arguments are not supported for leap.custom to export."
                        results = hbtl_ops.ops.__getattr__(module_name).__getattr__(
                            op_name
                        )(*args)
                        if len(results) == 1:
                            return results[0]
                        else:
                            return results

                    def has_mlir_function(args):
                        return any(isinstance(arg, mlir.Value) for arg in args)

                    global HBDK_EXPORT_FLAG
                    if has_torch_function(args):
                        return handle_torch_function(
                            cpp_custom_wrapper, args, *args, **kwargs
                        )
                    elif TORCH_EXPORT_FLAG in kwargs and kwargs[TORCH_EXPORT_FLAG]:
                        # pass horizon plugin tensor, translate the argument to plugin to export hbir
                        return export_for_torch()
                    elif has_mlir_function(args) or HBDK_EXPORT_FLAG:
                        # pass mlir.Values, need to construct graph
                        return export_mlir()
                    else:
                        # pass ndarray/tensor, need to run function
                        return eager_run()

                return cpp_custom_wrapper

        return OpFinder()


custom = ModuleFinder()
# leap_op = LeapOp.declare
leap_func = FunctionRegistry.declare
leap_export = FunctionRegistry.create

# Since we need to call hbtl.call for leap.custom, don't wrap it to hbir.leap
_wrap_imported_func(
    hbir, func_names_to_exclude=["custom", "constant", "numba", "triton_call"]
)

# Only expose "const_fake_quant" to users
_wrap_imported_func(qnt, func_names_to_include=["const_fake_quant"])


def load_library(name):
    """Load the C/C++ library

    Args:
        * name (str): The path to the .so file.
    """
    return Dispatcher.get().load(name)
