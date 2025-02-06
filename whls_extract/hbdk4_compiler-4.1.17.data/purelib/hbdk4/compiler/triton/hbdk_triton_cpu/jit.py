import functools
import os
import inspect
import textwrap
import tempfile
import subprocess
import shutil
from threading import stack_size
import torch
import numpy as np
import shlex
from dataclasses import dataclass
from typing import TypeVar, Optional
import logging
import hashlib
import ast
import hbdk4.compiler
from hbdk4.compiler import leap
from hbdk4.compiler import hbtl
from hbdk4.compiler.leap import TORCH_EXPORT_FLAG
from hbdk4.compiler.triton.compiler import ASTSource
from hbdk4.compiler.triton.runtime.jit import KernelParam, KernelArg, JITFunction
from hbdk4.compiler.triton.compiler import AttrsDescriptor
from hbdk4.compiler.triton.compiler.code_generator import kernel_suffix
from hbdk4.compiler.ops import hbir

from hbdk4.compiler._mlir_libs.libtriton import ir
from typing import Tuple

from hbdk4.compiler._mlir_libs.libtriton_vpu.analysis import (
    PointerType,
    analyze_inout_indices,
)

T = TypeVar("T")


@dataclass(frozen=True)
class HBDKTritonCPUOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    cluster_dims: tuple = (1, 1, 1)
    ptx_version: int = None
    enable_warp_specialization: bool = False
    enable_persistent: bool = False
    optimize_epilogue: bool = False
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    compile_only: bool = False


class CompiledCPUKernel:
    def __init__(self, so_path, launcher_name, arg_values):
        self.so_path = so_path
        self.launcher_name = launcher_name
        self.arg_values = arg_values

        # Register the shared library
        leap.load_library(so_path)
        # Set metadata
        self.metadata = {}
        # Set hbtl signature metadata
        # Now hbtl only supports from torch or numpy tensor to create hbtl tensor
        hbtl_args = [hbtl.get_tensor_type(arg) for arg in arg_values]
        hbtl_signature = hbtl.get_matched_schema(
            "hbdk_triton_cpu", self.launcher_name, hbtl_args
        )
        self.metadata["hbtl_signature"] = hbtl_signature
        # Set so path metadata
        self.metadata["so_path"] = self.so_path

    def __call__(self, *arg_values):
        # TODO: Support calling this kernel in the form that is the same as the python function. E.g. kernel[grid](x, y, output, n_elements, BLOCKSIZE=8)
        x = eval(f"leap.custom.hbdk_triton_cpu.{self.launcher_name}(*arg_values)")
        return x


class HBDKJITFunction(JITFunction[T]):
    _func_dict = {}

    def __init__(
        self, fn, cxx=None, cxxflags=None, ldflags=None, debug_dir=None, debug=False
    ):
        if fn.__name__ in HBDKJITFunction._func_dict:
            raise NameError(f"HBDKJITFunction name '{fn.__name__}' already exists!!!")

        self.fn = fn
        self.cxx_compiler = cxx
        self.compiler_flags = cxxflags
        self.linker_flags = ldflags
        self.debug_dir = debug_dir
        self.cached_kernel = {}
        self.signature = inspect.signature(fn)
        self.params = [
            KernelParam(i, param, False)
            for i, param in enumerate(self.signature.parameters.values())
        ]
        # add for supporting calling triton cpu jit'd function in another jit'd function
        self.debug = debug
        self.noinline = None
        self.arg_names = [p.name for p in self.params]

        # Function source code (used when converting from AST -> TTIR)
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def") :]
        self.has_boundary_check = False

        # Re-use docs of wrapped function
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__

        self.initial_args: Tuple
        # The triton kernel(fn) input outut indices
        self.input_indices = {}
        self.output_indices = {}
        self.input_output_indices = {}
        # For tensor that neither be input nor output, treat it as input
        self.extend_input_indices = {}

        # record the func name and the referenced python object
        HBDKJITFunction._func_dict[fn.__name__] = self

    def __getitem__(self, grid) -> T:
        from torch.overrides import has_torch_function, handle_torch_function

        callable_kernel = super().__getitem__(grid)

        # NOTE: Do not modify the name of the function, if modify it should sync to plugin.
        def triton_export_wrapper(*args, **kwargs):
            if has_torch_function(args):
                # get the arguments needed by hbir
                return handle_torch_function(
                    triton_export_wrapper, args, *args, **kwargs
                )

            if TORCH_EXPORT_FLAG in kwargs and kwargs[TORCH_EXPORT_FLAG]:
                compile_kwargs = kwargs
                compile_kwargs.pop(TORCH_EXPORT_FLAG)
                compile_kwargs["compile_only"] = True
                compiled_kernel, ttir_str = self.run(
                    *args, grid=grid, warmup=False, **compile_kwargs
                )
                hbir_params = []
                hbir_kwargs = {}
                input_indices = []
                output_indices = []
                tensor_index = 0
                for i, arg in enumerate(args):
                    if not self._type_of(self._key_of(arg)).startswith("*"):
                        hbir_params.append(arg)
                    else:
                        if i in self.extend_input_indices:
                            input_indices.append(tensor_index)
                        elif i in self.output_indices:
                            output_indices.append(tensor_index)
                        else:
                            raise NotImplementedError("Unsupported inputoutput indices")
                        tensor_index += 1
                hbir_kwargs["signature"] = str(
                    compiled_kernel.metadata["hbtl_signature"]
                )
                hbir_kwargs["library"] = str(compiled_kernel.metadata["so_path"])
                hbir_kwargs["inputIndices"] = input_indices
                hbir_kwargs["outputIndices"] = output_indices
                hbir_kwargs["ttir"] = ttir_str

                if len(hbir_params) > 0:
                    hbir_kwargs["parameters"] = hbir_params
                return functools.partial(hbir.triton_call, **hbir_kwargs)
            else:
                return callable_kernel(*args, **kwargs)

        return triton_export_wrapper

    def get_input_output_indices(self, ttir_str):
        inout_indices = analyze_inout_indices(ttir_str)
        if PointerType.INPUT in inout_indices:
            self.input_indices = {
                value: i for i, value in enumerate(inout_indices[PointerType.INPUT])
            }
        if PointerType.OUTPUT in inout_indices:
            self.output_indices = {
                value: i for i, value in enumerate(inout_indices[PointerType.OUTPUT])
            }
        if PointerType.INOUT in inout_indices:
            self.input_output_indices = {
                value: i for i, value in enumerate(inout_indices[PointerType.INOUT])
            }

    def get_extend_input_indices(self, signature):
        self.extend_input_indices = {}
        index = 0
        for i, ty in signature.items():
            if (
                ty.startswith("*")
                and i not in self.output_indices
                and i not in self.input_output_indices
            ):
                self.extend_input_indices[i] = index
                index = index + 1

    @staticmethod
    def copy_tensor(dst_tensor, src_tensor):
        assert torch.is_tensor(dst_tensor) and torch.is_tensor(
            src_tensor
        ), "Now only support torch.Tensor"
        assert (
            dst_tensor.shape == src_tensor.shape
        ), "Destination tensor shape should be same with the source tensor"
        assert (
            dst_tensor.stride() == src_tensor.stride()
        ), "Now only support with the same stride of destination tensor and source tensor"
        dst_tensor.copy_(src_tensor)

    def run(self, *args, grid, warmup, **kwargs):
        assert not warmup, "warmup should always be False"
        kwargs["debug"] = self.debug
        options = HBDKJITFunction._parse_options(kwargs)
        kwargs = {k: v for k, v in kwargs.items() if (k not in options.__dict__)}
        # Bind keyword args and set defaults
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        assert len(bound_args.arguments) == len(self.params)
        self.initial_args = bound_args.args
        # Canonicalize grid
        assert grid is not None
        if callable(grid):
            # Arguments are passed as a dict to `grid`, by contract
            grid = grid(dict(bound_args.arguments))
        grid_size = len(grid)
        grid_x = grid[0]
        grid_y = grid[1] if grid_size > 1 else 1
        grid_z = grid[2] if grid_size > 2 else 1
        args: list[KernelArg] = [
            KernelArg(arg_value, param)
            for (_, arg_value), param in zip(bound_args.arguments.items(), self.params)
        ]

        # Build constant dict
        constants = {
            arg.param.num: arg.value
            for arg in args
            if arg.param.is_constexpr or (arg.value is None)
        }
        for i, arg in constants.items():
            if callable(arg):
                raise TypeError(f"Callable constexpr at index{i} is not supported")

        # Build kernel's signature -> this does not include constexpr arguments
        signature = {
            arg.param.num: HBDKJITFunction._type_of(HBDKJITFunction._key_of(arg.value))
            for arg in args
            if not arg.param.is_constexpr
        }

        # Build kernel's name
        kernel_name = "_".join(
            [self.fn.__name__, kernel_suffix(signature.values(), AttrsDescriptor())]
        )

        tree = ast.parse(self.src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and (node.func.attr == "load" or node.func.attr == "store")
                ) or (
                    isinstance(node.func, ast.Name)
                    and (node.func.id == "load" or node.func.id == "store")
                ):
                    keywords = [keyword.arg for keyword in node.keywords]
                    if "boundary_check" in keywords:
                        self.has_boundary_check = True

        # From Python AST to Triton IR
        src = ASTSource(self, signature, constants)
        context = ir.context()
        ir.load_dialects(context)
        module = src.make_ir(options, context)
        self.get_input_output_indices(module.str())
        self.get_extend_input_indices(signature)
        arg_values = [
            arg.value for i, arg in enumerate(args) if (i in self.extend_input_indices)
        ]
        arg_values.extend(
            [
                arg.value
                for i, arg in enumerate(args)
                if (
                    not arg.param.is_constexpr
                    and i not in self.extend_input_indices
                    and i not in self.output_indices
                )
            ]
        )
        cache_key = self._generate_cache_key(kernel_name, args, grid)
        if self.cached_kernel.get(cache_key) is not None:
            kernel = self.cached_kernel.get(cache_key)
        else:
            # All intermediate contents will be saved in a temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save Triton IR to file
                if self.debug_dir is not None:
                    tmpdir = self.debug_dir
                triton_ir_path = os.path.join(tmpdir, "kernel_ttir.mlir")
                with open(triton_ir_path, "w") as f:
                    print(module.str(), file=f)
                # Triton IR to Linalg IR
                inlined_triton_ir_path = os.path.join(
                    tmpdir, "kernel_inlined_ttir.mlir"
                )
                triton_opt_path = HBDKJITFunction._get_dep_tool_path("triton-opt")
                command = [
                    triton_opt_path,
                    triton_ir_path,
                    "--inline",
                    "-o",
                    inlined_triton_ir_path,
                ]
                if self.has_boundary_check:
                    # Call this pass to convert block ptr to common ptr when boundary check exists
                    command.insert(3, "--triton-rewrite-tensor-pointer")
                subprocess.check_call(command)

                linalg_ir_path = os.path.join(tmpdir, "kernel_linalgir.mlir")
                triton_shared_opt_path = HBDKJITFunction._get_dep_tool_path(
                    "triton-shared-opt"
                )
                subprocess.check_call(
                    [
                        triton_shared_opt_path,
                        inlined_triton_ir_path,
                        "--triton-to-linalg",
                        "-o",
                        linalg_ir_path,
                    ]
                )
                # Linalg IR to LLVM IR
                llvm_ir_path = os.path.join(tmpdir, "kernel_llvmir.mlir")
                mlir_opt_path = HBDKJITFunction._get_dep_tool_path("mlir-opt")
                subprocess.check_call(
                    [
                        mlir_opt_path,
                        linalg_ir_path,
                        "--convert-linalg-to-affine-loops",
                        # Remove to avoid bufferize error caused by "bufferize-function-boundaries" option added by this pass when deal with cf dialect
                        # "--eliminate-empty-tensors",
                        "--empty-tensor-to-alloc-tensor",
                        "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                        "--lower-affine",
                        "--convert-linalg-to-loops",
                        "--convert-vector-to-scf",
                        "--convert-scf-to-cf",
                        "--convert-arith-to-llvm",
                        "--convert-math-to-llvm",
                        "--convert-complex-to-llvm",
                        "--convert-vector-to-llvm",
                        "--convert-index-to-llvm",
                        "--memref-expand",
                        "--expand-strided-metadata",
                        "--finalize-memref-to-llvm",
                        "--convert-cf-to-llvm",
                        "--convert-func-to-llvm",
                        "--lower-affine",
                        "--convert-arith-to-llvm",
                        "--reconcile-unrealized-casts",
                        "-o",
                        llvm_ir_path,
                    ]
                )
                # LLVM IR in MLIR to the actual LLVM IR
                actual_llvm_ir_path = os.path.join(tmpdir, "kernel.ir")
                mlir_translate_path = HBDKJITFunction._get_dep_tool_path(
                    "mlir-translate"
                )
                subprocess.check_call(
                    [
                        mlir_translate_path,
                        llvm_ir_path,
                        "--mlir-to-llvmir",
                        "-o",
                        actual_llvm_ir_path,
                    ]
                )
                # Actual LLVM IR to general assembly
                assembly_path = os.path.join(tmpdir, "kernel.s")
                llc_path = HBDKJITFunction._get_dep_tool_path("llc")
                subprocess.check_call(
                    [
                        llc_path,
                        "-relocation-model=pic",
                        actual_llvm_ir_path,
                        "-o",
                        assembly_path,
                    ]
                )

                # Generate launcher source code
                launcher_name = f"{cache_key}_launcher"
                launcher_src = self._generate_launcher(
                    launcher_name, signature, kernel_name, grid_x, grid_y, grid_z
                )
                launcher_src_path = os.path.join(tmpdir, "launcher.cpp")
                # launcher_src_path = os.path.join(os.getcwd(), "launcher.cpp")
                with open(launcher_src_path, "w") as f:
                    print(launcher_src, file=f)

                # Compile the launcher and the kernel into a .so
                # Prepare dependent files
                parent_dir = os.path.dirname(os.path.abspath(__file__))
                c_runner_utils_cpp_src_path = os.path.join(
                    parent_dir, "CRunnerUtils.cpp"
                )
                c_runner_utils_cpp_src_path = os.path.join(
                    parent_dir, "CRunnerUtils.cpp"
                )
                c_runner_utils_h_src_path = os.path.join(parent_dir, "CRunnerUtils.h")
                msan_h_src_path = os.path.join(parent_dir, "Msan.h")
                shutil.copy(c_runner_utils_cpp_src_path, tmpdir)
                shutil.copy(c_runner_utils_h_src_path, tmpdir)
                shutil.copy(msan_h_src_path, tmpdir)
                so_path = os.path.join(tmpdir, "kernel.so")
                compile_cmd = [
                    self._get_cxx_compiler(),
                    f"-I{HBDKJITFunction._get_hbdk4_runtime_path('include')}",
                    "-fPIC",
                    "--std=c++17",
                    launcher_src_path,
                    assembly_path,
                    os.path.realpath(
                        f"{HBDKJITFunction._get_hbdk4_runtime_path('lib')}/libclang_rt.builtins.a"
                    ),
                    "--shared",
                    "-o",
                    so_path,
                ]
                compile_options = shlex.split(self._get_compiler_flags())
                link_options = shlex.split(self._get_linker_flags())
                compile_cmd.extend(compile_options)
                compile_cmd.extend(link_options)
                subprocess.check_call(compile_cmd)
                saved_so_path = os.path.join(
                    os.getcwd(),
                    f"{cache_key}_kernel.so",
                )
                shutil.copy(so_path, saved_so_path)
                kernel: CompiledCPUKernel = CompiledCPUKernel(
                    saved_so_path, launcher_name, arg_values
                )
                self.cached_kernel[cache_key] = kernel
        # Run it
        if not options.compile_only:
            result = kernel(*arg_values)
            if not isinstance(result, list) and not isinstance(result, tuple):
                result = [result]
            assert len(result) == len(self.output_indices)
            for v, i in zip(result, self.output_indices):
                HBDKJITFunction.copy_tensor(self.initial_args[i], v)

        return kernel, module.str()

    def _get_cxx_compiler(self) -> str:
        if self.cxx_compiler is not None:
            return self.cxx_compiler
        compiler = os.getenv("HBDK_CXX", "")
        if compiler == "":
            logging.warning("HBDK_CXX is not set! Using default cxx compiler 'g++'")
            return "g++"
        return compiler

    def _get_compiler_flags(self) -> str:
        if self.compiler_flags is not None:
            return self.compiler_flags
        flags = os.getenv("HBDK_CXXFLAGS", "")
        if flags == "":
            logging.warning("HBDK_CXXFLAGS is not set!")
        return flags

    def _get_linker_flags(self) -> str:
        if self.linker_flags is not None:
            return self.linker_flags
        flags = os.getenv("HBDK_LDFLAGS", "")
        if flags == "":
            logging.warning("HBDK_LDFLAGS is not set!")
        return flags

    @staticmethod
    def _get_dep_tool_path(tool_name: str) -> str:
        tool_prefix = os.path.join(
            os.path.dirname(hbdk4.compiler.__file__), "_mlir_libs"
        )
        path: str = os.path.join(tool_prefix, tool_name)
        assert os.path.exists(path), f"Cannot find {tool_name} in path '{path}'"
        return path

    @staticmethod
    def _get_hbdk4_runtime_path(include_or_lib: str) -> str:
        try:
            import hbdk4.runtime
        except ImportError:
            logging.error(
                "Triton cpu compile relies on hbdk runtime whl package, which is not installed. So far, only 'hbdk4-runtime-x86-64-unknown-linux-gnu-nash' is supported"
            )
        else:
            path = os.path.join(
                hbdk4.runtime.__path__._path[0],
                f"x86_64_unknown_linux_gnu/nash/{include_or_lib}",
            )
            assert os.path.exists(
                path
            ), f"{path} not exists, check your hbdk runtime package's version"
            return path

    @staticmethod
    def _mlir_type_to_cpp_type(ty):
        if ty[0] == "*":
            return "void*"
        return {
            "i1": "bool",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u8": "uint8_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "hbtl::half_t",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]

    @staticmethod
    def _mlir_type_to_hbtl_type(ty):
        if ty[0] == "*":
            return "void*"
        # TODO: the python binding of hbtl only supports int64_t for integer arguments, which is far from optimal
        return {
            "i1": "bool",
            "i8": "int64_t",
            "i16": "int64_t",
            "i32": "int64_t",
            "i64": "int64_t",
            "u32": "int64_t",
            "u64": "int64_t",
            "fp16": "hbtl::half_t",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }[ty]

    @staticmethod
    def _parse_options(opts):
        args = {
            k: opts[k]
            for k in HBDKTritonCPUOptions.__dataclass_fields__.keys()
            if k in opts
        }
        return HBDKTritonCPUOptions(**args)

    def _generate_cache_key(self, kernel_name, args, grid):
        sig_key = tuple(
            arg.signature_key() for arg in args if not arg.param.is_constexpr
        )
        spec_key = tuple(
            arg.specialization_key() for arg in args if not arg.param.do_not_specialize
        )
        constexpr_key = tuple(arg.value for arg in args if arg.param.is_constexpr)
        output_shapes = []
        for i in self.output_indices:
            output_shapes.append(args[i].value.shape)
        key = str(
            [self.fn.__name__, sig_key, constexpr_key, spec_key, grid, output_shapes]
        )
        hash_code = hashlib.md5(key.encode("utf-8")).hexdigest()
        cache_key = f"{kernel_name}_{hash_code}"
        return cache_key

    def _generate_launcher(
        self, launcher_name, signature, kernel_name, grid_x, grid_y, grid_z
    ):
        new_line = "\n"
        indent = " " * 4
        hbtl_params = []
        hbtl_infer_params = []
        for i, _ in enumerate(self.output_indices):
            hbtl_params.append("ude::TensorRef &outs" + str(i))
            hbtl_infer_params.append("ude::TensorRef &outs" + str(i))

        # TODO: Support inputoutput indices
        for i, _ in enumerate(self.extend_input_indices):
            hbtl_params.append("const ude::TensorRef &ins" + str(i))
            hbtl_infer_params.append("const ude::TensorRef &ins" + str(i))

        for i, ty in signature.items():
            if not ty.startswith("*"):
                param: str = (
                    HBDKJITFunction._mlir_type_to_hbtl_type(ty) + " param" + str(i)
                )
                hbtl_params.append(param)
                hbtl_infer_params.append(param)

        generated_code = []
        # generate for output
        for i, type in signature.items():
            if i in self.output_indices:
                index = self.output_indices[i]
                assert type.startswith("*")
                cpp_type = HBDKJITFunction._mlir_type_to_cpp_type(
                    type[type.find("*") + 1 :]
                )
                generated_code.append(
                    indent
                    + "void *outs"
                    + str(index)
                    + "Data = outs"
                    + str(index)
                    + ".ptr;"
                )
                generated_code.append(
                    indent
                    + "StridedMemRefType<char, 0> outs"
                    + str(index)
                    + "Memref = {static_cast<char *>(outs"
                    + str(index)
                    + "Data), static_cast<char *>(outs"
                    + str(index)
                    + "Data), 0};"
                )
            elif i in self.extend_input_indices:
                index = self.extend_input_indices[i]
                assert type.startswith("*")
                cpp_type = HBDKJITFunction._mlir_type_to_cpp_type(
                    type[type.find("*") + 1 :]
                )
                generated_code.append(
                    indent
                    + "const void *ins"
                    + str(index)
                    + "Data = ins"
                    + str(index)
                    + ".ptr;"
                )
                generated_code.append(
                    indent
                    + "StridedMemRefType<const char, 0> ins"
                    + str(index)
                    + "Memref = {static_cast<const char *>(ins"
                    + str(index)
                    + "Data), static_cast<const char *>(ins"
                    + str(index)
                    + "Data), 0};"
                )
            # TODO(yinan): Support inputoutput indices

        generated_args_code = []
        generated_params_code = []
        for i, type in signature.items():
            if type.startswith("*"):
                if i in self.extend_input_indices:
                    index = self.extend_input_indices[i]
                    generated_args_code.append("0, &ins" + str(index) + "Memref")
                    generated_params_code.append("int64_t, void*")
                elif i in self.output_indices:
                    index = self.output_indices[i]
                    generated_args_code.append("0, &outs" + str(index) + "Memref")
                    generated_params_code.append("int64_t, void *")
                else:
                    # TODO(yinan): Support inputoutput indices
                    raise NotImplementedError("Unsupported input-output")
            else:
                cpp_type = HBDKJITFunction._mlir_type_to_cpp_type(type)
                generated_args_code.append(
                    "static_cast<" + cpp_type + ">(param" + str(i) + ")"
                )
                generated_params_code.append(cpp_type)

        generated_args_code_str = ", ".join(generated_args_code)
        generated_params_code_str = ", ".join(generated_params_code)
        if generated_args_code:
            generated_args_code_str += ", "
        if generated_params_code:
            generated_params_code_str += ", "

        infer_code = []
        for initial_index, i in self.output_indices.items():
            ty = signature[initial_index][1:]
            element_ty = {
                "i1": "bool8",
                "fp16": "f16",
                "fp32": "f32",
                "fp64": "f64",
                "i8": "si8",
                "i16": "si16",
                "i32": "si32",
                "i64": "si64",
                "u8": "ui8",
                "u16": "ui16",
                "u32": "ui32",
                "u64": "ui64",
            }
            if ty not in element_ty:
                raise NotImplementedError(
                    "Unsupported type in hbtl element type: " + ty
                )
            shape = self.initial_args[initial_index].shape
            shape_str = [str(s) for s in shape]
            infer_code.append(f"outs{i}.shape = {{{', '.join(shape_str)}}};")
            infer_code.append(f"outs{i}.dtype = ude::Type::{element_ty[ty]};")

        return f"""
#include "ude/public/Protocols.h"
#include "ude/public/Common.h"
#include "ude/public/Library.h"
#include "ude/public/Status.h"

#include "CRunnerUtils.h"
#include "CRunnerUtils.cpp"

extern "C" {{
  void {kernel_name}({generated_params_code_str}int, int, int, int, int, int);
}}

namespace hbdk_triton_cpu {{
  ude::Status launcherInfer({', '.join(hbtl_infer_params)}) {{
    {(new_line + indent).join(infer_code)}
    return ude::Status::success();
  }}

  ude::Status launcherImpl({', '.join(hbtl_params)}) {{

    constexpr int32_t GRID_X_SIZE = {grid_x};
    constexpr int32_t GRID_Y_SIZE = {grid_y};
    constexpr int32_t GRID_Z_SIZE = {grid_z};

    /// Get raw data pointer and construct StrideMemRefType
    {new_line.join(generated_code)}

    /// Launch several programs
    for (int32_t blockIdxX = 0; blockIdxX < GRID_X_SIZE; ++blockIdxX) {{
      for (int32_t blockIdxY = 0; blockIdxY < GRID_Y_SIZE; ++blockIdxY) {{
        for (int32_t blockIdxZ = 0; blockIdxZ < GRID_Z_SIZE; ++blockIdxZ) {{
          {kernel_name}({generated_args_code_str}GRID_X_SIZE, GRID_Y_SIZE, GRID_Z_SIZE, blockIdxX, blockIdxY, blockIdxZ);
        }}
      }}
    }}

    return ude::Status::success();
  }}
}}

// NOLINTNEXTLINE
UDE_LIBRARY(hbdk_triton_cpu, CUSTOM) {{
 m.def<{len(self.output_indices)}>("hbdk_triton_cpu::{launcher_name}", hbdk_triton_cpu::launcherInfer, hbdk_triton_cpu::launcherImpl);
}}

"""


# -------------
# Decorators
# -------------


def triton_cpu_jit(
    fn: Optional[T] = None,
    *,
    cxx: Optional[str] = None,
    cxxflags: Optional[str] = None,
    ldflags: Optional[str] = None,
    debug_dir: Optional[str] = None,
    debug: Optional[bool] = False,
) -> HBDKJITFunction[T]:
    def decorator(fn: T) -> HBDKJITFunction[T]:
        assert callable(fn)
        return HBDKJITFunction(fn, cxx, cxxflags, ldflags, debug_dir, debug)

    if fn is not None:
        return decorator(fn)
    else:
        return decorator
