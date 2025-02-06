import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Tuple


# import triton._C.libtriton.triton as _triton
import triton._C.libvpu_backend_for_triton.triton_vpu as _triton_vpu
from triton.common import _build
from triton.common.backend import BaseBackend, register_backend
from triton.compiler.make_launcher import get_cache_manager, version_key, make_so_cache_key
from triton.compiler.utils import generate_cu_signature
from triton.runtime import jit
from triton.runtime.driver import HIPDriver



def ty_to_cpp(ty):
    if ty[0] == '*':
        return "hipDeviceptr_t"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def generate_launcher_hip(constants, signature, ids):
    # print("generate_launcher_hip")
    start_desc = len(signature)
    # signature = generate_cu_signature(constants, signature, ids)
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    def _extracted_type(ty):
        if ty[0] == '*':
            return "PyObject*"
        return {
            'i1': 'int32_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            'fp16': 'float',
            'bf16': 'float',
            'fp32': 'float',
            'f32': 'float',
            'fp64': 'double',
        }[ty]

    def format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "uint32_t": "I",
            "int32_t": "i",
            "uint64_t": "K",
            "int64_t": "L",
        }[ty]

    format = "iiiiiiiiiKKOOO" + ''.join([format_of(_extracted_type(ty)) for ty in signature.values()])

    # generate glue code
    folded_without_constexprs = [c for c in ids['ids_of_folded_args'] if c not in ids['ids_of_const_exprs']]
    params = [i for i in signature.keys() if i >= start_desc or (i not in constants and i not in folded_without_constexprs)]
    src = f"""
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <Python.h>
#include <stdbool.h>
#include <dlfcn.h>

static inline void gpuAssert(hipError_t code, const char *file, int line)
{{
   if (code != HIP_SUCCESS)
   {{
      const char* prefix = "Triton Error [HIP]: ";
       const char* str = hipGetErrorString(code);
      char err[1024] = {{0}};
      snprintf(err, 1024, "%s Code: %d, Messsage: %s", prefix, code, str );
      PyErr_SetString(PyExc_RuntimeError, err);
   }}
}}

#define HIP_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, hipStream_t stream, hipFunction_t function{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  // printf("_launch hip kernel\\n");
  void *params[] = {{ {', '.join(f"&arg{i}" for i in params)} }};
  if (gridX*gridY*gridZ > 0) {{
      HIP_CHECK(hipModuleLaunchKernel(function, gridX, gridY, gridZ, 64*num_warps, 1, 1, shared_memory, stream, params, 0));
    }}
  }}

typedef struct _DevicePtrInfo {{
    hipDeviceptr_t dev_ptr;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(ret);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    uint64_t dev_ptr;
    hipError_t status = hipPointerGetAttribute(&dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
    if (status == hipErrorInvalidValue) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    ptr_info.dev_ptr = (hipDeviceptr_t)dev_ptr;
    Py_DECREF(ret);
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
   // printf("launch\\n");
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int num_warps;
  int num_ctas;
  int clusterDimX;
  int clusterDimY;
  int clusterDimZ;
  int shared_memory;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *compiled_kernel = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &num_warps, &num_ctas, &clusterDimX, &clusterDimY, &clusterDimZ, &shared_memory, &_stream, &_function, &launch_enter_hook, &launch_exit_hook, &compiled_kernel{', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''})) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None) {{
    PyObject_CallObject(launch_enter_hook, args);
  }}


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (hipStream_t)_stream, (hipFunction_t)_function{', ' + ', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items()) if len(signature) > 0 else ''});

  if (launch_exit_hook != Py_None) {{
    PyObject_CallObject(launch_exit_hook, args);
  }}

  if(PyErr_Occurred()) {{
    return NULL;
  }}
  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


def make_stub(name, signature, constants, ids, **kwargs):
    # name of files that are cached
    so_cache_key = make_so_cache_key(version_key(), signature, constants, ids, **kwargs)
    so_cache_manager = get_cache_manager(so_cache_key)
    so_name = f"{name}.so"
    # retrieve stub from cache if it exists
    cache_path = so_cache_manager.get_file(so_name)
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = generate_launcher_hip(constants, signature, ids)
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            so = _build(name, src_path, tmpdir)
            with open(so, "rb") as f:
                return so_cache_manager.put(f.read(), so_name, binary=True)
    else:
        return cache_path

# AMDGCN translation


def get_amdgcn_bitcode_paths(arch):
    # print("get_amdgcn_bitcode_paths")
    gpu_arch_agnostic_bitcode_libraries = ["opencl.bc",
                                           "ocml.bc",
                                           "ockl.bc",
                                           "oclc_finite_only_off.bc",
                                           "oclc_daz_opt_off.bc",
                                           "oclc_correctly_rounded_sqrt_on.bc",
                                           "oclc_unsafe_math_off.bc",
                                           "oclc_wavefrontsize64_on.bc",
                                           "oclc_abi_version_400.bc",]

    gfx_arch = arch[1]
    gfx_arch_id = re.search('gfx(\\w+)', gfx_arch).group(1).strip()

    gpu_arch_specific_bitcode_library = 'oclc_isa_version_' + gfx_arch_id + ".bc"
    current_dir = Path(__file__)
    bitcode_path_dir = os.path.join(current_dir.parent.resolve(), "lib/bitcode/")

    amdgcn_bitcode_paths = {}
    i = 0
    for bc_lib in gpu_arch_agnostic_bitcode_libraries:
        bc_path = bitcode_path_dir + bc_lib
        if os.path.exists(bc_path):
            amdgcn_bitcode_paths['library_' + str(i)] = bc_path
            i += 1
    bc_gfx_path = bitcode_path_dir + gpu_arch_specific_bitcode_library
    if os.path.exists(bc_gfx_path):
        amdgcn_bitcode_paths['library_' + str(i)] = bc_gfx_path

    # print(f"amdgcn_bitcode_paths: {amdgcn_bitcode_paths}")
    return amdgcn_bitcode_paths


def get_amdgpu_arch_fulldetails():
    # print("get_amdgpu_arch_fulldetails")
    """
    get the amdgpu fulll ISA details for compiling:
    i.e., arch_triple: amdgcn-amd-amdhsa; arch_name: gfx906; arch_features: sramecc+:xnack-
    """
    try:
        # TODO: package vpu.cc with Triton
        vpu_path_dir = os.getenv("VPU_PATH", default="/opt/vpu")
        vpuinfo = subprocess.check_output(vpu_path_dir + '/bin/vpuinfo').decode()
        gfx_arch_details = re.search('amd.*', vpuinfo).group(0).strip().split('--')
        arch_triple = gfx_arch_details[0]
        arch_name_features = gfx_arch_details[1].split(':')
        arch_name = arch_name_features[0]
        arch_features = ""

        if (len(arch_name_features) == 3):
            arch_features = "+" + re.search('\\w+', arch_name_features[1]).group(0) + ","\
                            "-" + re.search('\\w+', arch_name_features[2]).group(0)
        return [arch_triple, arch_name, arch_features]
    except BaseException:
        return None


def get_kernel_name(src: str, pattern: str) -> str:
    # print("get_kernel_name")
    '''
    Get kernel name from PTX code.
    This Kernel name is required when launching the kernel.
    '''
    # There is a name mangling in PTX codegen, so the original kernel names in Triton IR are not available in PTX/cubin.
    assert src
    for line in src.split('\n'):
        line = line.strip()
        if line.startswith(pattern):
            return line.split()[-1]


def get_arch_details(arch: list):
    # get arch info
    gfx_arch = os.environ.get('MI_GPU_ARCH', arch[1])
    gfx_triple = arch[0]
    gfx_features = arch[2]
    if gfx_arch is None:
        raise RuntimeError('gfx_arch is None (not specified)')

    return gfx_arch, gfx_triple, gfx_features


def update_extern_libs(extern_libs: dict, arch: str):
    # append extern_libs
    extern_libs.update(get_amdgcn_bitcode_paths(arch))
    for key in list(extern_libs):
        if extern_libs[key] == '' or extern_libs[key] is None:
            extern_libs.pop(key)

    # check extern libs
    if extern_libs:
        for name, path in extern_libs.items():
            if len(name) == 0 or len(path) == 0:
                raise RuntimeWarning(f"extern_lib has empty value, {name}: {path}")

    names = list(extern_libs.keys())
    paths = list(extern_libs.values())
    return names, paths


# passes
def ttir_to_ttgir_vpu(module: str, compute_capability:int, num_warps:int, num_stages:int):
    return _triton_vpu.translate_ttir_to_ttgir_vpu(module, compute_capability, num_warps, num_stages)


def optimize_ttgir_vpu():
    pass


def ttgir_to_llir_vpu(module: str, extern_libs: dict, arch:str):
    names, paths = update_extern_libs(extern_libs, arch)
    llvmIR = _triton_vpu.translate_ttgir_to_llvmir(module, names, paths)
    return llvmIR


def llir_to_amdgcn_and_hsaco(module: str, arch: str):
    '''
    Translate TritonGPU module to HSACO code based on full details of gpu architecture.
    :param mod: a TritonGPU dialect module
    :return:
        - AMDGCN code
        - Path to HSACO object
    '''
    gfx_arch, gfx_triple, gfx_features = get_arch_details(arch)
    return _triton_vpu.translate_llvmir_to_hsaco(module, gfx_arch, gfx_triple, gfx_features)

# fused pass


def ttir_to_amdgcn_and_hsaco(module, context, arch, num_warps, num_stages, extern_libs) -> Tuple[str, str]:
    gfx_arch, gfx_triple, gfx_features = get_arch_details(arch)
    names, paths = update_extern_libs(extern_libs, arch)
    return _triton_vpu.translate_triton_ir_to_amdgcn_and_hsaco(str(module), gfx_arch, gfx_triple, gfx_features, num_warps, num_stages, names, paths)


class HIPBackend(BaseBackend):
    def __init__(self, device_type: str) -> None:
        # print("HIPBackend.__init__")
        super(HIPBackend, self).__init__(device_type)
        self.driver = HIPDriver()
        self.stub_so_path = ""

    def add_stages(self, arch: str, extern_libs: dict, stages:dict, num_warps=4, num_stages=3):
        # print(f"HIPBackend.add_stages{arch, extern_libs, stages, num_warps, num_stages}")

        # add stages
        stages["ttgir"] = (lambda path: Path(path).read_text(),
                            lambda src: ttir_to_ttgir_vpu(str(src), 0, num_warps, num_stages))
        stages["llir"] = (lambda path: Path(path).read_text(),
                            lambda src: ttgir_to_llir_vpu(src, extern_libs, arch))
        stages["amdgcn"] = (lambda path: Path(path).read_text(),
                            lambda src: llir_to_amdgcn_and_hsaco(src, arch))

    def add_meta_info(self, ir, module, next_module, metadata, asm):
        pass

        # if ir == "amdgcn":
        #     # print("HIPBackend.add_meta_info")
        #     asm[ir] = str(next_module[0])

        #     metadata["name"] = get_kernel_name(next_module[0], pattern='.globl')

        #     if "shared" not in metadata.keys():
        #         metadata["shared"] = _triton_vpu.get_shared_memory_size(module)
        #         # metadata["shared"] = 32
        #         # metadata["shared"] = 0
        #     asm["hsaco_path"] = next_module[1]

    def get_driver(self):
        # print("HIPBackend.get_driver")
        return self.driver

    def get_stream(self, idx=None):
        # print("HIPBackend.get_stream")
        return jit.get_cuda_stream()

    def get_device_properties(self, device):
        # print("HIPBackend.get_device_properties")
        return self.driver.utils.get_device_properties(device)

    def get_current_device(self):
        # print("HIPBackend.get_current_device")
        return jit.get_current_device()

    def set_current_device(self, device):
        # print("HIPBackend.set_current_device")
        return jit.set_current_device(device)

    def get_load_binary_fn(self):
        # print("HIPBackend.get_load_binary_fn")
        return self.driver.utils.load_binary

    def get_kernel_bin(self):
        # print("HIPBackend.get_kernel_bin")
        return "hsaco_path"

    def get_architecture_descriptor(self, **kwargs):
        # print("HIPBackend.get_architecture_descriptor")
        return get_amdgpu_arch_fulldetails()

    def make_launcher_stub(self, name, signature, constants, ids):
        # print("HIPBackend.make_launcher_stub")
        self.stub_so_path = make_stub(name, signature, constants, ids)
        return self.stub_so_path

    def get_shared_memory_size(self, module):
        # print("HIPBackend.get_shared_memory_size")
        return _triton_vpu.get_shared_memory_size(module)

    def get_num_warps(self, module):
        # print("HIPBackend.get_num_warps")
        return _triton_vpu.get_num_warps(module)

register_backend("hip", HIPBackend)
