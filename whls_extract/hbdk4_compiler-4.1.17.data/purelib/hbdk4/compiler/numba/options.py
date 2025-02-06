from typing import Optional, List
from enum import Enum
import os
import warnings


class Target(Enum):
    ARM = 1
    RISCV = 2
    X86 = 3

    def __str__(self):
        if self == Target.ARM:
            return "arm"
        elif self == Target.RISCV:
            return "riscv"
        elif self == Target.X86:
            return "x86"


class CompileOptions:
    def __init__(
        self,
        target: Target,
        cc: str,
        output_name: Optional[str] = None,
        c_flags: Optional[str] = None,
        cxx_flags: Optional[str] = None,
        ld_flags: Optional[str] = None,
    ) -> None:
        assert (
            target
        ), "Please specify a target. Currently, supported targets include Target.ARM and Target.X86"
        # TODO: Prompts the user to give the head file directory and dynamic directory
        self.target = target
        self.output_name = output_name
        self.cc = cc
        self.c_flags = c_flags
        self.cxx_flags = cxx_flags
        self.ld_flags = ld_flags


class NumbaOptions:
    def __init__(
        self,
        compile_options_list: List[CompileOptions],
    ):
        assert (
            len(compile_options_list) > 0
        ), "Please give at least one set of compilation options as CompileOptions, but the compile_options is empty"
        self.compile_options_list = compile_options_list


def get_x86_compile_options():
    var_prefix_name = "HBDK_TARGET_X86_64_UNKNOWN_LINUX_GNU_"
    cxx = os.getenv(var_prefix_name + "CXX")
    cxx_flags = os.getenv(var_prefix_name + "CXXFLAGS")
    ld_flags = os.getenv(var_prefix_name + "LDFLAGS")
    if cxx is None:
        warnings.warn(
            "If the X86 compilation toolchain is not set up correctly, there will be problems at compile and interpret time if numba is used.\n"
        )
        warnings.warn("The dynamic library of x86 will not be generated.\n")
        return None
    return CompileOptions(
        target=Target.X86, cc=cxx, cxx_flags=cxx_flags, ld_flags=ld_flags
    )


def get_arm_compile_options():
    var_prefix_name = "HBDK_TARGET_AARCH64_UNKNOWN_LINUX_GNU_"
    cxx = os.getenv(var_prefix_name + "CXX")
    cxx_flags = os.getenv(var_prefix_name + "CXXFLAGS")
    ld_flags = os.getenv(var_prefix_name + "LDFLAGS")
    if cxx is None:
        warnings.warn(
            "If the ARM compilation toolchain is not set up correctly, there will be problems at compile and run time if numba is used\n"
        )
        warnings.warn("The dynamic library of arm will not be generated.\n")
        return None
    return CompileOptions(
        target=Target.ARM, cc=cxx, cxx_flags=cxx_flags, ld_flags=ld_flags
    )
