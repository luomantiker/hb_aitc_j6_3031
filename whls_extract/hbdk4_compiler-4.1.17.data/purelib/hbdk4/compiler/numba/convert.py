import inspect
import hbdk4.compiler
from hbdk4.compiler import ir as mlir
from hbdk4.compiler.overlay import Module, TensorType
from hbdk4.compiler.ops import hbir
from hbdk4.compiler.dialects._ods_common import get_default_loc_context
from numba import njit, typeof
from numba.core import extending
import numpy as np
import importlib
import os
from hbdk4.compiler.numba.trace import _remove_cpython_and_cfunc
import tempfile


def convert_custom_op_to_numba_op(module: Module):
    """For all functions in module, replace hbir.custom op with hbir.numba op"

    Args:
        module: Overlay.Module

    Returns:
        No returns
    """
    functions = module.functions
    ctx = get_default_loc_context()
    for function in functions:
        func = function.opview
        ops = func.regions[0].blocks[0].operations
        func_name = func.name.value
        for op in ops:
            if op.OPERATION_NAME == "hbir.custom":
                insertion_point = mlir.InsertionPoint(op)
                numba_results = _build_numba_op_from_custom_op(
                    op, ctx, insertion_point, func_name
                )
                # need to update users of hbir.custom op's return values
                for old_value, new_value in zip(op.results, numba_results):
                    old_value.replace_all_uses_with(new_value)
                # delete hbir.custom op
                op.operation.erase()
    return module


def _get_numba_signature(mlir_values):
    signatures = []
    for value in mlir_values:
        tensor_type = TensorType(value.type)
        py_var = np.zeros(tensor_type.shape, tensor_type.np_dtype)
        signatures.append(typeof(py_var).copy(layout="A"))
    return tuple(signatures)


def _get_function_from_file(file_path, func_name):
    spec = importlib.util.spec_from_file_location("module_name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, func_name, None)
    return function


def _check_custom_op(custom_op, jitted_func):
    py_func = jitted_func.py_func
    sigs = inspect.signature(py_func)
    func_name = py_func.__name__
    assert len(sigs.parameters) == len(
        custom_op.operands
    ), f"Invalid hbir.custom: The number of python numba function {func_name} inputs is inconsistent with the number of inputs in hbir.custom"

    py_args = []
    for value in custom_op.operands:
        tensor_type = TensorType(value.type)
        py_args.append(np.ones(tensor_type.shape, tensor_type.np_dtype))
    py_result = py_func(*py_args)
    assert len(py_result) == len(
        custom_op.results
    ), f"Invalid hbir.custom: The number of python numba function {func_name} outputs is inconsistent with the number of outputs in hbir.custom"


def _build_numba_op_from_custom_op(custom_op, ctx, insertion_point, func_name):
    signature = _get_numba_signature(custom_op.operands)
    file_path = custom_op.srcPath.value
    assert os.path.exists(file_path), "hbir.custom_op's srcPath '{}' not exist".format(
        file_path
    )
    jitted_func = _get_function_from_file(file_path, custom_op.entryFuncName.value)
    assert jitted_func, f"Not found {custom_op.entryFuncName.value} in {file_path}"
    assert extending.is_jitted(
        jitted_func
    ), f"{custom_op.entryFuncName.value} is not decorated by numba.njit"

    _check_custom_op(custom_op, jitted_func)

    if signature not in jitted_func.overloads:
        njit_func = njit([signature])(jitted_func.py_func)
    cres = njit_func.overloads[signature]
    fndesc = cres.fndesc

    # FIXME(yinan): Consider the case when numba python function is called repeatedly between preprocess or postprocess and in the middle of model
    llvm_ir_name = ""
    with tempfile.NamedTemporaryFile(
        mode="w+t", dir=".", prefix="temp", suffix=".ll", delete=False
    ) as llvm_ir_file:
        ir = njit_func.inspect_llvm(signature)
        processed_ir = _remove_cpython_and_cfunc(ir)
        llvm_ir_file.write(processed_ir)
        llvm_ir_file.flush()
        llvm_ir_file.close()
        llvm_ir_name = llvm_ir_file.name
    with ctx, insertion_point:
        numba_results = hbir.numba(
            [operands for operands in custom_op.operands],
            os.path.dirname(llvm_ir_name),
            os.path.basename(llvm_ir_name),
            fndesc.mangled_name,
            outputs_type=[mlir.ShapedType(result.type) for result in custom_op.results],
        )
        return numba_results
