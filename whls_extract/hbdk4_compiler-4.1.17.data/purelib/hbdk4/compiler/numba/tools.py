import os
import codecs
import subprocess
import sys
import random
import string

from hbdk4.compiler.numba.options import (
    Target,
    get_x86_compile_options,
    get_arm_compile_options,
)
from hbdk4.compiler.overlay import Module
from hbdk4.compiler._mlir_libs._hbdk import _compile_numba
from hbdk4.compiler.dialects._hbir_ops_gen import NumbaOp
from hbdk4.compiler.dialects._ods_common import _cext
from hbdk4.compiler.numba.convert import convert_custom_op_to_numba_op
import hbdk4.compiler
import tempfile
import re
from hbdk4.compiler import ir as mlir

from hbdk4.compiler.utils.cext import parse_enum

builder = _cext.ir.AttrBuilder


def _check_command_availability(command):
    try:
        # try to run
        subprocess.run(
            [command, "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False


def _run_command(command):
    proc = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)


def _generate_temp_library_name(
    func_name: str, prefix: str, suffix: str, target: Target
):
    output_temp_file = tempfile.NamedTemporaryFile(
        mode="w+t",
        dir=".",
        prefix=prefix + func_name + "_",
        suffix="_" + target.__str__() + suffix,
        delete=False,
    )
    return output_temp_file.name


def _generate_random_string(prefix, suffix, length=6):
    random_string = "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(length)
    )
    return f"{prefix}{random_string}{suffix}"


def walk(module, func):
    if isinstance(module, Module):
        module = module.module
    for f in module.body.operations:
        for region in f.regions:
            for block in region:
                for op in block:
                    func(op)


def has_numba_op(op):
    if isinstance(op, NumbaOp):
        return True
    for region in op.regions:
        for block in region.blocks:
            for op in block.operations:
                if has_numba_op(op):
                    return True
    return False


def compile_llvm_ir(module: Module, target: Target) -> Module:
    """

    :param module:
    :param emit_library:
    :return:
    """
    if not has_numba_op(module.module):
        return module

    cloned_module = module.clone()
    ir_file_set = set()
    walk(
        cloned_module,
        lambda op: ir_file_set.add(op.irFileName.value)
        if isinstance(op, NumbaOp)
        else None,
    )

    # NOTE-NUMBA-HAVENOT-LIBRARY
    # FIXME: hbir.numba should not have library path information, should as option to the pass
    tool_prefix = os.path.join(os.path.dirname(hbdk4.compiler.__file__), "_mlir_libs")
    llc = os.path.join(tool_prefix, "llc")
    clang = os.path.join(tool_prefix, "clang")
    ar = os.path.join(tool_prefix, "llvm-ar")
    for tool in [llc, clang, ar]:
        if not _check_command_availability(tool):
            raise EnvironmentError(
                f"{tool} not exist or failed to get version. Please check installation"
            )

    # static_library_name = _generate_random_string(prefix="libnumba_", suffix=".a")
    # NOTE-NUMBA-LIBRARY-NAME
    # FIXME: set the function name correctly
    length = 8
    random_string = "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(length)
    )
    static_library_name = (
        "libnumba_"
        + module[0].name
        + "_"
        + random_string
        + "_"
        + target.__str__()
        + ".a"
    )

    if target == Target.X86:
        cmd = [clang]
        cmd += [file_name for file_name in ir_file_set]
        cmd += ["-B" + tool_prefix]
        cmd += ["-fPIC"]
        cmd += ["--emit-static-lib"]
        cmd += ["-o", static_library_name]
        _run_command(cmd)
    elif target == Target.ARM:
        # Different with x86, arm should use llc generate object files, then use ar to pack
        object_files = []
        for ir_file in ir_file_set:
            object_file_name = _generate_temp_library_name(
                module[0].name, "temp_", ".o", Target.ARM
            )
            # object_file = tempfile.NamedTemporaryFile(
            #     mode="w+t", dir=".", prefix="temp_", suffix="_arm.o", delete=True
            # )
            cmd = [llc]
            cmd += [ir_file]
            cmd += ["-filetype=obj"]
            cmd += ["--mtriple=aarch64"]
            cmd += ["--relocation-model=pic"]
            cmd += ["-o"]
            cmd += [object_file_name]
            object_files.append(object_file_name)
            _run_command(cmd)
        cmd = [ar]
        cmd += ["rc"]
        cmd += [static_library_name]
        cmd += object_files
        _run_command(cmd)
    else:
        raise NotImplementedError(
            "Numba " + target + " compilation is not implemented yet"
        )

    ctx = cloned_module.module.context

    def update_numba_op(op):
        if isinstance(op, NumbaOp):
            op.attributes["libraryName"] = builder.get("StrAttr")(
                static_library_name, ctx
            )
            op.attributes["libraryMode"] = parse_enum("STATIC")

    walk(cloned_module, update_numba_op)
    return cloned_module


def _remove_extra_space(string):
    return re.sub(r"\s+", " ", string).strip()


def _replace_space_with_at(string):
    return re.sub(r" ", r"@", string.strip())


def compile_custom(module: Module) -> Module:
    """Compile the custom op to hbtl.call"""
    if isinstance(module, mlir.Module):
        module = Module(module)
    else:
        assert isinstance(
            module, Module
        ), "Only support hbdk4.compiler.Module or hbdk4.compiler.ir.Module to be compiled"
    m = convert_custom_op_to_numba_op(module)
    m = compile_numba(m)
    return m


def compile_numba(module: Module) -> Module:
    """Call the numba pass: gen-numba-wrapper and convert-numba-to-hbtl

    :param module:
    :param numba_compile_options
    :return: Module
    """
    if not has_numba_op(module.module):
        return module

    numba_compile_options = []
    arm_compile_options = get_arm_compile_options()
    if arm_compile_options:
        numba_compile_options.append(arm_compile_options)
    x86_compile_options = get_x86_compile_options()
    if x86_compile_options:
        numba_compile_options.append(x86_compile_options)

    if len(numba_compile_options) == 0:
        return module

    """NOTE-NUBMA-INTERPRETER
    TODO(yinan): the hbtl call should not have the library name, user load library by load_library
    If there is an option for x86 target, the module for x86 should be returned to run in interpreter
    """

    for compile_options in numba_compile_options:
        target = compile_options.target
        module = compile_llvm_ir(module, target)

        cc = compile_options.cc
        assert cc and cc == _remove_extra_space(
            cc
        ), "Invalid compiler for numba, there should be no space in compiler setting"

        output = compile_options.output_name

        # NOTE-NUMBA-LIBRARY-NAME
        # FIXME: set the function name correctly
        if not output:
            output = _generate_temp_library_name(module[0].name, "lib", ".so", target)
        assert output == _remove_extra_space(
            output
        ), "Invalid output name, there should be no space in output name"
        output = output if "/" in output or output.startswith("lib") else "lib" + output
        output = output if output.endswith(".so") else output + ".so"

        pass_options = []
        pass_options.append("output_path=" + os.path.basename(output))
        pass_options.append("cc=" + cc)

        c_flags = compile_options.c_flags
        cxx_flags = compile_options.cxx_flags
        ld_flags = compile_options.ld_flags

        for name, flag in zip(
            ["c_flags", "cxx_flags", "ld_flags"], [c_flags, cxx_flags, ld_flags]
        ):
            flag = _remove_extra_space(flag) if flag else flag
            flag = _replace_space_with_at(flag) if flag else flag
            pass_options.append(name + "=" + flag) if flag else None

        args = {"pass_options": pass_options}

        new_module = module.clone()
        _compile_numba(new_module.module, module.module.context, args)

    return new_module
