cmake_minimum_required(VERSION 3.20)

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0056 NEW)

project(hbtl_sample LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(HBTL_WHOLE_BUILD)
  message(STATUS "Building sample as a child of hbtl")
else(HBTL_WHOLE_BUILD)
  message(STATUS "Building sample standalone")
endif(HBTL_WHOLE_BUILD)

if(NOT HBTL_WHOLE_BUILD)

  find_package(CUDAToolkit REQUIRED)

  message(STATUS "CUDA verision: ${CUDAToolkit_VERSION}")
  message(STATUS "CUDA headers: ${CUDAToolkit_INCLUDE_DIRS}")
  message(STATUS "CUDA nvcc: ${CUDAToolkit_NVCC_EXECUTABLE}")

  set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})

  message(STATUS "cuda compiler id is ${CMAKE_CUDA_COMPILER_ID}")

  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_C_COMPILER})
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
  endif()

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -forward-unknown-to-host-compiler")

  # When CMake initializing CUDA, it uses system default /usr/bin/ld
  # But CentOS default ld is too old to work
  # Set CMAKE_CUDA_FLAGS seem the only way to set linker flag to choose the correct linker here
  # We reset CMAKE_CUDA_FLAGS after enable_language to not pollute the flags
  set(CMAKE_CUDA_FLAGS_OLD ${CMAKE_CUDA_FLAGS})

  # expect to static link libstdc++, exclude libgcc and libstdc++
  set(linker_flags
      "-static-libstdc++ -Wl,--exclude-libs=libgcc.a -Wl,-rpath,${LINKLIB_PATH} -Wno-fuse-ld-path -pthread"
  )
  set(CMAKE_MODULE_LINKER_FLAGS
      "${CMAKE_MODULE_LINKER_FLAGS} ${linker_flags} -Wl,--exclude-libs=libstdc++.a"
  )
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${linker_flags}")
  set(CMAKE_SHARED_LINKER_FLAGS
      "${CMAKE_SHARED_LINKER_FLAGS} ${linker_flags} -Wl,--exclude-libs=libstdc++.a"
  )

  message(STATUS "CMake shared Linker Flags: ${CMAKE_SHARED_LINKER_FLAGS}")

  if(CMAKE_SHARED_LINKER_FLAGS)
    string(REPLACE " " ";" splitted_option ${CMAKE_SHARED_LINKER_FLAGS})
    foreach(flag ${splitted_option})

      string(FIND ${flag} "-stdlib=libc++" contain_stdlibc)

      if("${contain_stdlibc}" EQUAL "-1")
        string(FIND ${flag} "-Wl," contain_wl)

        set(normalized_flag)
        if(${contain_wl} EQUAL "-1")
          set(normalized_flag "-Xcompiler ${flag}")
        else()
          string(REPLACE "-Wl," "-Xlinker " normalized_flag ${flag})
        endif()

        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${normalized_flag}")
      endif()

    endforeach(flag ${splitted_option})
  endif(CMAKE_SHARED_LINKER_FLAGS)

  message(STATUS "CUDA FLAGS ${CMAKE_CUDA_FLAGS}")

  enable_language(CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_OLD}")

  # sample need to include hbtl include path
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/spdlog
                   ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/spdlog EXCLUDE_FROM_ALL)
endif(NOT HBTL_WHOLE_BUILD)

if(NOT (TARGET hbtl))
  if(NOT HBTL_LIB_PATH)
    message(
      FATAL_ERROR
        "Must specify the absolute path of libhbtl.so in HBTL_LIB_PATH if TARGET \"hbtl\" not exist"
    )
  endif(NOT HBTL_LIB_PATH)

  message(STATUS "Import libhbtl.so from {HBTL_LIB_PATH}")

  add_library(hbtl SHARED IMPORTED GLOBAL)
  set_target_properties(hbtl PROPERTIES IMPORTED_LOCATION "${HBTL_LIB_PATH}")
endif(NOT (TARGET hbtl))

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(CudaKernel)
