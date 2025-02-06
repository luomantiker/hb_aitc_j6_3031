# This Directory contains the HBTL Cuda Kernel sample

The HBTL Cuda Kernel can utilize all the foundational
infrastructure from HBTL, achieved by specifying the
location of HBTL's header files and dynamically linking
HBTL. It supports two compilation modes: the first is
to build together with HBTL, and the second is to
build standalone.

## Build with HBTL or HBDK

The first building method will utilize the build tools
configured within HBTL, and there is no need to specify
the location of the hbtl dynamic library.

## Build standalone

The second building method requires specifying the
following variables (Some optional markings using
'optional'):

  1. CMAKE_C_COMPILER: Used to specify the version
   of the C compiler. [optional]
  2. CMAKE_CXX_COMPILER: Used to specify the
   version of the CXX compiler. [optional]
  3. CMAKE_MAKE_PROGRAM: Used to specify the
   build tool. If not specified, the
   makefile will be used. [optional]
  4. LINKER_PATH: Used to specify the
   location of the linker that the user wants
   to use. If not specified, on systems with older
   linkers, there might be difficulties in
   completing the CUDA configuration.  [optional]
  5. HBTL_LIB_PATH: Used to specify the location
   of the hbtl dynamic library. [required]
  6. CUDAToolkit_ROOT: Used to specify the
   location of the CUDA Toolkit. option. [optional]

If not specified, optional variables might fail to
initialize CUDA correctly.

## Code Structure

- sample
  - include
  - CudaKernel

1. The include directory contains declarations of
  functions with host-device interaction, same as
  CPP header files.

2. The CudaKernel directory includes code that
   runs on both the host and device sides.
   Kernels running on the device side are
   implemented in .cu files. Unlike C++,
   explicit instantiation of templates is
   required within them.

3. Tensor can handle memory allocation on the
   GPU and synchronization on the host effectively,
   but this is limited to the use of tensors
   with contiguous storage.

4. HBTL provides CudaContext for controlling
   certain GPU operations.
