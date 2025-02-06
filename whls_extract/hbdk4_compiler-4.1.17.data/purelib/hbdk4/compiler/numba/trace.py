import ast
import inspect
import os
import sys

import numba
import re
import copy
import logging
import tempfile

import torch
import numpy as np
from numba import njit, typeof
from numba.core import types, extending

from hbdk4.compiler import ir as mlir, hbtl as hbtl_ops

from hbdk4.compiler.overlay import Module
from hbdk4.compiler.torch import export
from hbdk4.compiler.dialects.func import FuncOp, ReturnOp
from hbdk4.compiler.ops import hbir, hbtl
from hbdk4.compiler.dialects._ods_common import get_default_loc_context
from hbdk4.compiler.utils.types import TensorType, Dtype
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.ERROR)

DEFAULT_LOGGING_FORMATTER = " %(asctime)s - %(levelname)s : %(message)s"

formatter = logging.Formatter(DEFAULT_LOGGING_FORMATTER)

# file_handler = logging.FileHandler('hbdk4_compiler_translate.log')
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

mail_massage = "Please concat yinan.zhang@horizon.cc for more support requirements"


_regex_cpython = re.compile(r"define.*@*cpython")
_regex_cfunc = re.compile(r"define.*@*cfunc")


def _remove_cpython_and_cfunc(llvmir):
    def _extract_functions(module):
        cur = []
        for line in str(module).splitlines():
            if line.startswith("define"):
                # start of function
                assert not cur
                cur.append(line)
            elif line.startswith("}"):
                # end of function
                assert cur
                cur.append(line)
                yield True, cur
                cur = []
            elif cur:
                cur.append(line)
            else:
                yield False, [line]

    processed = []

    for is_func, lines in _extract_functions(llvmir):
        if is_func and _regex_cpython.match(lines[0]) is not None:
            continue
        if is_func and _regex_cfunc.match(lines[0]) is not None:
            continue
        processed += lines

    return "\n".join(processed)


def once_per_instance_method(method):
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "_{}_called".format(method.__name__), False):
            setattr(self, "_{}_called".format(method.__name__), True)
            return method(self, *args, **kwargs)
        else:
            raise Exception(
                "Method {} can only be called once per instance.".format(
                    method.__name__
                )
            )

    return wrapper


def _numba_signature_convert(signature):
    assert isinstance(signature, tuple)
    converted_signature = []
    for ty in signature:
        if isinstance(ty, types.npytypes.Array):
            converted_arr = ty.copy(layout="A")
            # TODO(yinan): Add warning or hint information when convert array layout from 'C' or 'F' to 'A'
            converted_signature.append(converted_arr)
        else:
            converted_signature.append(ty)
    return tuple(converted_signature)


class Translator:
    """ """

    def __init__(self, f, *inputs, **options):
        self.function = f
        self.arguments = inputs
        self.module = None
        self.llvm_ir: List = []
        self.library_name = (
            self.function.__name__
            if "library_name" not in options
            else options.get("library_name")
        )
        self.compile_options = options.get("compile_options")

    class FunctionVisitor(ast.NodeVisitor):
        def __init__(
            self, ctx, args, numba_lib_name, func_name, globals_dict, locals_dict=None
        ):

            self.env = dict()
            if globals_dict is not None:
                self.env = globals_dict
            if locals_dict is not None:
                self.env.update(locals_dict)
            self.variable_mlir_mapping = dict()
            assert isinstance(args, (list, tuple))
            self.args = args  # function arguments

            self.generated_numba_func: Dict = {}

            self.context = ctx
            self.module = None
            self.insertion_point = None

            self.llvm_ir: List = []
            self.numba_lib_name = numba_lib_name

            # func_name is needed when setting loc info
            self.func_name = func_name

        def generate_mlir_func_name(self, func_name, *args):
            name = func_name
            for i, arg in enumerate(args):
                if isinstance(arg, (np.ndarray, torch.Tensor)):
                    name += "." + str(
                        TensorType(list(arg.shape), Dtype(str(arg.dtype)))
                    )
                else:
                    name += "." + str(type(arg))
            return name

        def convert_to_mlir_type(self, variable):
            if isinstance(variable, tuple):
                return (self.convert_to_mlir_type(v) for v in variable)
            if isinstance(variable, list):
                return [self.convert_to_mlir_type(v) for v in variable]
            with self.context:
                assert isinstance(variable, torch.Tensor) or isinstance(
                    variable, np.ndarray
                ), "Support convert torch.Tensor or numpy.ndarray to mlir tensor, but the type is {}".format(
                    str(object=type(variable))
                )
                return TensorType(
                    variable.shape, Dtype(str(variable.dtype))
                ).mlir_tensor_type

        def generic_visit(self, node: ast.AST) -> Any:
            self.generic_visit_children(node)

        def generic_visit_children(self, node):
            for child in ast.iter_child_nodes(node):
                self.visit(child)

        @once_per_instance_method
        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            logger.debug("Enter visit ast.FunctionDef, visit %s ", {node.name})
            assert len(node.args.args) == len(self.args)
            # set loc info
            # TODO: the loc of arguments in FuncOp is still unknown, need to find a way to set it
            loc = mlir.Location.file(self.func_name, node.lineno, node.col_offset)
            with loc:
                with self.context:
                    self.module = mlir.Module.create()
                    func_op = FuncOp(
                        node.name,
                        ([self.convert_to_mlir_type(arg) for arg in self.args], []),
                    )
                    self.module.body.append(func_op)
                    entry_block = func_op.add_entry_block()
                    self.insertion_point = mlir.InsertionPoint(entry_block)
                for arg, mlir_arg in zip(node.args.args, entry_block.arguments):
                    self.variable_mlir_mapping[arg.arg] = mlir_arg
                for arg, v in zip(node.args.args, self.args):
                    self.env[arg.arg] = v
                self.generic_visit_children(node)

            logger.debug("Exit visit ast.FunctionDef")

        def visit_Attribute(self, node: ast.Attribute) -> Any:
            logger.debug("Enter visit ast.Attribute, visit %s ", {node.attr})

            # set loc info
            loc = mlir.Location.file(self.func_name, node.lineno, node.col_offset)
            with loc:
                if isinstance(node.value, ast.Name):
                    logger.debug(
                        "Exit visit ast.Attribute, the attribute is %s",
                        ".".join([node.value.id, node.attr]),
                    )
                    return [node.value.id, node.attr]
                elif isinstance(node.value, ast.Call):
                    _, attribute = self.visit_Call(node.value)
                    attribute.append(node.attr)
                    logger.debug(
                        "Exit visit ast.Attribute, the attribute is %s",
                        ".".join(attribute),
                    )
                    return attribute
                elif isinstance(node.value, ast.Subscript):
                    logger.debug(
                        "Exit visit ast.Attribute, the attribute is %s",
                        ".".join(
                            [
                                node.value.value.id
                                + "[{}]".format(node.value.slice.value.value),
                                node.attr,
                            ]
                        ),
                    )
                    return [
                        node.value.value.id,
                        str(node.value.slice.value.value),
                        node.attr,
                    ]
                elif isinstance(node.value, ast.Attribute):
                    attribute = self.visit_Attribute(node.value)
                    attribute.append(node.attr)
                    logger.debug(
                        "Exit visit ast.Attribute, the attribute is %s",
                        ".".join(attribute),
                    )
                    return attribute
                else:
                    raise ValueError(
                        "Attribute only support ast.Name, ast.Call and ast.Subscript"
                    )

        @staticmethod
        def _get_module_func_results_type_and_py_result(
            module: Module, module_func_name, *py_args
        ):
            func = module[module_func_name]
            obj_type = np.ndarray
            if len(py_args) > 0:
                obj_type = type(py_args[0])
            shape_list = [output.type.shape for output in func.outputs]
            dtype_list = (
                [output.type.np_dtype for output in func.outputs]
                if obj_type == np.ndarray
                else [output.type.torch_dtype for output in func.outputs]
            )
            results_type = [
                mlir.RankedTensorType.get(shape, dtype)
                for shape, dtype in zip(
                    shape_list,
                    [output.type.tensor.element_type for output in func.outputs],
                )
            ]
            py_result = []

            for shape, dtype in zip(shape_list, dtype_list):
                if obj_type == np.ndarray:
                    py_result.append(np.zeros(shape, dtype=dtype))
                else:
                    py_result.append(torch.zeros(shape, dtype=dtype))
            py_r = py_result[0] if len(py_result) == 1 else py_result
            return results_type, py_r

        def _handle_attribute_call(self, node):
            logger.debug("  Enter _handle_attribute_call")
            assert isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            attrs = self.visit_Attribute(node.func)
            attribute_str = ".".join(attrs)
            # TODO(yinan): Modify this to common Attribute, not just the two
            if attribute_str.endswith(".numpy") or attribute_str.endswith(
                "detach.numpy"
            ):
                assert len(attrs) == 2 or len(attrs) == 3, "Syntax error: "
                if isinstance(node.func.value, ast.Subscript):
                    name = attrs[0]
                    index = int(attrs[1])
                    assert name in self.variable_mlir_mapping, (
                        name + ": mlir value should be already traced"
                    )
                    assert name in self.env, (
                        name + ": pyobject should be already traced"
                    )
                    return (
                        self.env[name][index].detach().numpy(),
                        self.variable_mlir_mapping[name][index],
                    )
                name = attrs[0]
                return self.env[name].detach().numpy(), self.variable_mlir_mapping[name]
            elif attribute_str.startswith("torch.from_numpy"):
                assert (
                    len(node.args) == 1
                ), "torch.from_numpy only support for a argument"
                assert len(attrs) == 2, ""
                assert isinstance(node.args[0], ast.Name) or isinstance(
                    node.args[0], ast.Subscript
                )
                if isinstance(node.args[0], ast.Name):
                    arg_name = node.args[0].id
                    return (
                        torch.from_numpy(self.env[arg_name]),
                        self.variable_mlir_mapping[arg_name],
                    )
                else:
                    arg_name = node.args[0].value.id
                    arg_index = node.args[0].slice.value.value
                    assert arg_name in self.variable_mlir_mapping, (
                        arg_name + ": mlir value should be already traced"
                    )
                    assert arg_name in self.env, (
                        arg_name + ": pyobject should be already traced"
                    )
                    return (
                        torch.from_numpy(self.env[arg_name][arg_index]),
                        self.variable_mlir_mapping[arg_name][arg_index],
                    )
            else:
                return None, attrs

        def _numba_op_builder(self, mlir_args, results_type, jitted_func, *args):
            assert extending.is_jitted(
                jitted_func
            ), "The numba function should be jitted"
            logger.debug(
                "  Enter _numba_op_builder, building %s ", jitted_func.__name__
            )
            jitted_func_params = inspect.signature(jitted_func).parameters
            # if numba function's param is a varargs(*arg)
            if len(jitted_func_params.keys()) == 1 and any(
                param.kind == param.VAR_POSITIONAL
                for param in jitted_func_params.values()
            ):
                signature = tuple(map(typeof, list(args)))
                signature = _numba_signature_convert(signature)
                # need to wrap signatures to numba's Tuple type
                signature = tuple([types.containers.Tuple(signature)])
            else:
                assert len(inspect.signature(jitted_func).parameters.keys()) == len(
                    args
                )
                signature = tuple(map(typeof, list(args)))
                signature = _numba_signature_convert(signature)

            njit_func = jitted_func
            if signature not in jitted_func.overloads:
                njit_func = njit(signature)(jitted_func.py_func)
            fn_and_signature = (njit_func.__name__, signature)
            cres = njit_func.overloads[signature]
            fndesc = cres.fndesc

            if fn_and_signature not in self.generated_numba_func:

                with tempfile.NamedTemporaryFile(
                    mode="w+t", dir=".", prefix="temp", suffix=".ll", delete=False
                ) as llvm_ir_file:
                    self.generated_numba_func[fn_and_signature] = llvm_ir_file.name
                    ir = njit_func.inspect_llvm(signature)
                    processed_ir = _remove_cpython_and_cfunc(ir)
                    llvm_ir_file.write(processed_ir)
                    llvm_ir_file.flush()
                    llvm_ir_file.close()

                # self.llvm_ir.append(njit_func.inspect_llvm(signature))
            ir_file_name = self.generated_numba_func[fn_and_signature]

            with self.context, self.insertion_point:
                logger.debug("      Begin generating %s", jitted_func.__name__)
                numba_results = hbir.numba(
                    mlir_args,
                    os.path.dirname(ir_file_name),
                    os.path.basename(ir_file_name),
                    fndesc.mangled_name,
                    # "lib" + self.numba_lib_name + ".a",
                    outputs_type=results_type,
                )

                logger.debug("      Finish generating %s", jitted_func.__name__)
                logger.debug("  Exit _numba_op_builder")
                assert len(results_type) > 0, "numba should have at least one output"
                if len(results_type) == 1:
                    return numba_results[0]
                else:
                    return tuple(numba_results)

        def _torch_builder(self, name, mlir_args, result_types, py_func, *args):
            if inspect.isfunction(py_func):
                logger.debug("  Enter _torch_builder, building %s ", py_func.__name__)
            elif isinstance(py_func, torch.nn.Module):
                logger.debug(
                    "  Enter _torch_builder, building %s forward",
                    str(py_func.__class__),
                )
            assert isinstance(py_func, torch.nn.Module) or isinstance(
                py_func, torch.nn.functional
            )
            assert all(
                isinstance(arg, torch.Tensor) or isinstance(arg, np.ndarray)
                for arg in args
            )

            if isinstance(py_func, torch.nn.Module):
                traced = torch.jit.trace(py_func.eval(), args)
            else:
                traced = torch.jit.trace(py_func, args)

            with self.context:
                module = export(traced, args)
                operations = module.module.body.operations
                assert (
                    len(operations) == 1
                ), "Torch should be return a single function, not multiple functions"
                func = operations[0].detach_from_parent()
                arguments_mapping = {k: v for k, v in zip(func.arguments, mlir_args)}
                assert len(func.arguments) == len(mlir_args)

                for op in func.regions[0].blocks[0].operations:
                    op = op.detach_from_parent()
                    if not op.OPERATION_NAME == "func.return":
                        for i, operand in enumerate(op.operands):
                            if operand in func.arguments:
                                op.operands[i] = arguments_mapping[operand]
                        self.insertion_point.insert(op)
                    else:
                        logger.debug("  Exit _torch_builder")
                        assert (
                            len(op.operands) > 0
                        ), "Torch returnOp should have at least one operand"
                        if len(op.operands) == 1:
                            return op.operands[0]
                        else:
                            return tuple([operand for operand in op.operands])

            logger.debug("  Exit _torch_builder")
            return None

        def _module_builder(self, module, module_func_name, mlir_args, *args):
            logger.debug("  Enter _module_builder, building leap.call")
            assert isinstance(
                module, Module
            ), "The first arg of leap.call() must be Module object"
            assert isinstance(
                module_func_name, str
            ), "The second arg of leap.call() must be str object"
            assert all(
                isinstance(arg, torch.Tensor) or isinstance(arg, np.ndarray)
                for arg in args
            ), "Args type must be torch.Tensor or np.ndarray"
            module = module.clone()
            with self.context:
                func = module[module_func_name].opview
                arguments_mapping = {k: v for k, v in zip(func.arguments, mlir_args)}
                assert len(func.arguments) == len(mlir_args)

                for op in func.regions[0].blocks[0].operations:
                    op = op.detach_from_parent()
                    if not op.OPERATION_NAME == "func.return":
                        for i, operand in enumerate(op.operands):
                            if operand in func.arguments:
                                op.operands[i] = arguments_mapping[operand]
                        self.insertion_point.insert(op)
                    else:
                        logger.debug("  Exit _module_builder")
                        assert (
                            len(op.operands) > 0
                        ), "Module returnOp should have at least one operand"
                        if len(op.operands) == 1:
                            return op.operands[0]
                        else:
                            return tuple([operand for operand in op.operands])

            logger.debug("  Exit _module_builder")
            return None

        def _custom_op_builder(self, func_name, module_name, op_name, mlir_args):
            logger.debug("  Enter _custom_op_builder, building %s", func_name)
            infer_args = [
                TensorType.from_mlir_value(ele).hbtl_tensor_type
                if isinstance(ele, mlir.Value)
                else ele
                for ele in mlir_args
            ]
            schema, hbtl_types = hbtl_ops.get_schema_and_infer_type(
                module_name, op_name, infer_args
            )
            results_type = [
                TensorType.from_hbtl_tensor_type(hbtl_type).mlir_tensor_type
                for hbtl_type in hbtl_types
            ]
            with self.context, self.insertion_point:
                logger.debug("      Begin generating %s", func_name)
                inputs = [ele for ele in mlir_args if isinstance(ele, mlir.Value)]
                params = [ele for ele in mlir_args if not isinstance(ele, mlir.Value)]
                custom_op_results = hbtl.call(
                    inputs,
                    str(schema),
                    isCustom=True,
                    parameters=params,
                    outputs_type=results_type,
                )
                logger.debug("      Finish generating %s", func_name)
                logger.debug("  Exit _custom_op_builder")
                assert (
                    len(custom_op_results) > 0
                ), "custom op should have at least one output"
                if len(results_type) == 1:
                    return results_type, custom_op_results[0]
                else:
                    return results_type, tuple(custom_op_results)

        def visit_Constant(self, node: ast.Constant) -> Any:
            return node.value

        def visit_Index(self, node: ast.Index):
            # Deprecated since version 3.9
            assert sys.version_info < (3, 9), "ast.Index deprecated since version 3.9"
            return self.visit(node.value)

        def visit_Subscript(self, node: ast.Subscript) -> Any:
            arg_name = node.value.id
            assert isinstance(
                node.slice, (ast.Constant, ast.Index)
            ), "Only support the x[0], unsupported for x[0:2:1]"
            arg_index = self.visit(node.slice)
            assert arg_name in self.variable_mlir_mapping, (
                arg_name + ": mlir value should be already traced"
            )
            assert arg_name in self.env, (
                arg_name + ": pyobject should be already traced"
            )
            if sys.version_info >= (3, 11):
                pass
            py_arg = self.env[arg_name][arg_index]
            mlir_arg = self.variable_mlir_mapping[arg_name][arg_index]
            return py_arg, mlir_arg

        def visit_Call(self, node: ast.Call) -> Any:
            def get_node_trace(arg):
                if isinstance(arg, ast.Constant):
                    constant = self.visit(arg)
                    py_arg, mlir_arg = constant, constant
                elif isinstance(arg, ast.Name):
                    assert arg.id in self.variable_mlir_mapping, (
                        arg.id + ": mlir value should be already traced"
                    )
                    assert arg.id in self.env, (
                        arg.id + ": pyobject should be already traced"
                    )
                    py_arg = self.env[arg.id]
                    mlir_arg = self.variable_mlir_mapping[arg.id]
                elif isinstance(arg, ast.Call):
                    logger.debug("  arg is ast.Call")
                    py_arg, mlir_arg = self.visit_Call(arg)
                    assert (
                        py_arg is not None and mlir_arg is not None
                    ), "visit ast.Call {} should not return None.".format(arg.func.name)

                elif isinstance(arg, ast.Subscript):
                    logger.debug("  arg is ast.Subcript")
                    py_arg, mlir_arg = self.visit(arg)
                    # arg_name = arg.value.id
                    # arg_index = arg.slice.value.value
                elif isinstance(arg, ast.List) or isinstance(arg, ast.Tuple):
                    logger.debug("  arg is ast.List or ast.Tuple")
                    py_arg = []
                    mlir_arg = []
                    for x in arg.elts:
                        py_trace, mlir_trace = get_node_trace(x)
                        py_arg.append(py_trace)
                        mlir_arg.append(mlir_trace)
                else:
                    raise ValueError(
                        "The arguments of function should be ast.Constant, ast.Name, ast.Call, ast.Subscript, ast.List or ast.Tuple"
                    )
                return py_arg, mlir_arg

            def get_traced_py_mlir_args(node_args):
                py_args = []
                mlir_args = []
                for arg in node_args:
                    py_arg, mlir_arg = get_node_trace(arg)
                    py_args.append(py_arg)
                    mlir_args.append(mlir_arg)
                return py_args, mlir_args

            if isinstance(node.func, ast.Attribute):
                attribute = self.visit_Attribute(node.func)
                if (
                    len(attribute) == 4
                    and attribute[0] == "leap"
                    and attribute[1] == "custom"
                ):
                    func_name = ".".join(attribute)
                    logger.debug("Enter visit ast.Call, visit %s ", func_name)
                    py_args, mlir_args = get_traced_py_mlir_args(node.args)
                    module_name = attribute[2]
                    op_name = attribute[3]
                    py_result = hbtl_ops.ops.__getattr__(module_name).__getattr__(
                        op_name
                    )(*py_args)
                    loc = mlir.Location.file(
                        self.func_name, node.lineno, node.col_offset
                    )
                    with loc:
                        results_type, results = self._custom_op_builder(
                            func_name, module_name, op_name, mlir_args
                        )
                    if results_type is None:
                        logger.debug("Exit visit ast.Call, None is returned")
                        return py_result, None
                    elif len(results_type) == 1:
                        logger.debug("Exit visit ast.Call, a value is returned")
                        return py_result[0], results
                    else:
                        logger.debug("Exit visit ast.Call, a tuple is returned")
                        return py_result, results
                elif ".".join(attribute) == "leap.call":
                    func_name = "leap.call"
                    assert (
                        len(node.args) >= 2
                    ), "The arguments of leap.call() must be greater than 2"
                    module = node.args[0].id
                    module_func_name = node.args[1].value
                    logger.debug("Enter visit ast.Call, visit %s ", func_name)
                    py_args, mlir_args = get_traced_py_mlir_args(node.args[2:])
                    results = self._module_builder(
                        self.env[module], module_func_name, mlir_args, *py_args
                    )
                    (
                        results_type,
                        py_result,
                    ) = self._get_module_func_results_type_and_py_result(
                        self.env[module], module_func_name, *py_args
                    )
                    if results_type is None:
                        logger.debug("Exit visit ast.Call, None is returned")
                        return py_result, None
                    elif len(results_type) == 1:
                        logger.debug("Exit visit ast.Call, a value is returned")
                        return py_result, results
                    else:
                        logger.debug("Exit visit ast.Call, a tuple is returned")
                        return py_result, results
                else:
                    return self._handle_attribute_call(node)

            func_name = node.func.id
            logger.debug("Enter visit ast.Call, visit %s ", func_name)
            if func_name not in self.env:
                err_msg = func_name + " is not defined in globals"
                logger.error(err_msg)
                raise ValueError(err_msg)

            py_func = self.env[func_name]
            # Torch Module
            if inspect.isclass(py_func) and issubclass(py_func, torch.nn.Module):
                logger.debug(
                    "Exit visit ast.Call, the callee %s is a torch.nn.Module class",
                    func_name,
                )
                return py_func, None
            elif inspect.isclass(py_func):
                logger.error("Unsupported class")
                raise ValueError("Unsupported class")
            py_args, mlir_args = get_traced_py_mlir_args(node.args)
            # mlir_func_name = self.generate_mlir_func_name(func_name, *py_args)

            # set loc info
            loc = mlir.Location.file(self.func_name, node.lineno, node.col_offset)
            with loc:
                py_result = py_func(*py_args)
                if py_result is not None:
                    results_type = (
                        [self.convert_to_mlir_type(ret) for ret in py_result]
                        if (isinstance(py_result, tuple) or isinstance(py_result, list))
                        else [self.convert_to_mlir_type(py_result)]
                    )
                # Numba function
                if isinstance(py_func, numba.core.registry.CPUDispatcher):
                    assert callable(py_func) and not inspect.isclass(py_func)
                    results = self._numba_op_builder(
                        mlir_args, results_type, py_func, *py_args
                    )
                # Torch.nn.Module
                elif isinstance(py_func, torch.nn.Module):
                    mlir_func_name = self.generate_mlir_func_name(func_name, *py_args)
                    results = self._torch_builder(
                        mlir_func_name, mlir_args, results_type, py_func, *py_args
                    )
                if results_type is None:
                    logger.debug("Exit visit ast.Call, None is returned")
                    return py_result, None
                elif len(results_type) == 1:
                    logger.debug("Exit visit ast.Call, a value is returned")
                    return py_result, results
                else:
                    logger.debug("Exit visit ast.Call, a tuple is returned")
                    return py_result, results

        def _update_pyobject(self, node):
            logger.debug("  Enter _update_pyobject")
            module_node = ast.Module(body=[], type_ignores=[])
            module_node.body.append(node)
            node_code = compile(module_node, filename="<ast>", mode="exec")
            globals_dict = copy.copy(self.env)
            locals_dict = {}
            exec(node_code, globals_dict, locals_dict)
            self.env.update(locals_dict)
            return

        def visit_Assign(self, node: ast.Assign) -> Any:
            # Update the type
            logger.debug("Enter visit ast.Assign")

            def handle_single_assignment(lhs, rhs):
                if isinstance(lhs, ast.Tuple) and isinstance(rhs, ast.Tuple):
                    assert len(lhs.elts) == len(rhs.elts)
                    for lv, rv in zip(lhs.elts, rhs.elts):
                        handle_single_assignment(lv, rv)
                if isinstance(rhs, ast.Call):
                    with self.context:
                        pyobj, value = self.visit_Call(rhs)
                    # Update for python object
                    if value is None:  # ast.Attribute for pytorch class
                        self._update_pyobject(node)
                        return

                    if isinstance(lhs, ast.Tuple):
                        assert (
                            isinstance(pyobj, tuple)
                            and len(pyobj) == len(lhs.elts)
                            and len(pyobj) == len(value)
                        )
                        for i, elt in enumerate(lhs.elts):
                            assert isinstance(elt, ast.Name), (
                                "Invalid assignment statement, only support to assign "
                                "to a variable "
                            )
                            self.env[elt.id] = pyobj[i]
                            self.variable_mlir_mapping[elt.id] = value[i]

                    else:
                        assert isinstance(lhs, ast.Name), (
                            "Invalid assignment statement, only support to assign to a "
                            "variable "
                        )
                        self.env[lhs.id] = pyobj
                        self.variable_mlir_mapping[lhs.id] = value
                else:
                    raise ValueError(
                        "Unsupported assignment statement and invalid rvalue"
                    )

            for target in node.targets:
                handle_single_assignment(target, node.value)
            logger.debug("Exit visit ast.Assign")

        def visit_Return(self, node: ast.Return) -> Any:
            def get_traced_mlir_arg(return_arg):
                if isinstance(return_arg, ast.Name):
                    assert return_arg.id in self.variable_mlir_mapping, (
                        return_arg.id + ": mlir value should be already traced"
                    )
                    mlir_arg = self.variable_mlir_mapping[return_arg.id]
                elif isinstance(return_arg, ast.Call):
                    logger.debug("  return_arg is ast.Call")
                    py_arg, mlir_arg = self.visit_Call(return_arg)
                elif isinstance(return_arg, ast.Subscript):
                    _, mlir_arg = self.visit(return_arg)
                else:
                    raise ValueError(
                        "The arguments of function should be ast.Name or ast.Call"
                    )
                return mlir_arg

            logger.debug("Enter visit ast.Return")
            # set loc info
            loc = mlir.Location.file(self.func_name, node.lineno, node.col_offset)
            with loc:
                if (
                    isinstance(node.value, ast.Name)
                    or isinstance(node.value, ast.Call)
                    or isinstance(node.value, ast.Subscript)
                ):
                    mlir_var = get_traced_mlir_arg(node.value)
                    with self.context, self.insertion_point:
                        if isinstance(mlir_var, tuple) or isinstance(mlir_var, list):
                            ReturnOp(mlir_var)
                            self.func_op_return_types = [v.type for v in mlir_var]
                        else:
                            ReturnOp([mlir_var])
                            self.func_op_return_types = mlir_var.type
                elif isinstance(node.value, ast.List) or isinstance(
                    node.value, ast.Tuple
                ):
                    mlir_var = [get_traced_mlir_arg(elt) for elt in node.value.elts]
                    with self.context, self.insertion_point:
                        ReturnOp(mlir_var)
                        self.func_op_return_types = [v.type for v in mlir_var]
                else:
                    raise ValueError(
                        "Unsupported return value, only support ast.Name, ast.Call, ast.Subscript and corresponding Tuple/List"
                    )
            logger.debug("Exit visit ast.Return")

    def verify_module(self):
        assert all(op.verify() for op in self.module.body.operations)

    def generate_module(self):
        logger.debug("Generating module")
        sigs = inspect.signature(self.function)
        params = sigs.parameters.keys()
        assert len(params) == len(self.arguments)

        tree = ast.parse(inspect.getsource(self.function))
        with get_default_loc_context(None) as ctx:
            visitor = self.FunctionVisitor(
                ctx,
                self.arguments,
                self.library_name,
                self.function.__name__,
                self.function.__globals__,
            )
            visitor.visit(tree)
            # self.llvm_ir = visitor.llvm_ir

            self.module = visitor.module
            # add return type for FuncOp
            func_op = self.module.body.operations[0]
            return_types = visitor.func_op_return_types
            function_type = mlir.FunctionType.get(
                [arg.type for arg in func_op.arguments],
                return_types if isinstance(return_types, List) else [return_types],
            )
            func_op.attributes["function_type"] = mlir.TypeAttr.get(function_type)
            self.verify_module()
        logger.debug("Generating module done")

    def translate(self):
        self.generate_module()
