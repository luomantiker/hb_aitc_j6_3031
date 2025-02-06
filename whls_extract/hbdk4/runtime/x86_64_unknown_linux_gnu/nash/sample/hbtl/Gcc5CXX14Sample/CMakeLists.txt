cmake_minimum_required(VERSION 3.20)

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0056 NEW)

project(hbtl_gcc5_sample LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

message(STATUS "Testing build hbtl using gcc5.5")

add_definitions(-DFMT_HEADER_ONLY)

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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/spdlog
                 ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/spdlog EXCLUDE_FROM_ALL)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/unittest
                 ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/unittest EXCLUDE_FROM_ALL)

if(NOT HBTL_LIB_PATH)
  message(
    FATAL_ERROR
      "Must specify the absolute path of libhbtl.so in HBTL_LIB_PATH if TARGET \"hbtl\" not exist"
  )
endif(NOT HBTL_LIB_PATH)

message(STATUS "Import libhbtl.so from ${HBTL_LIB_PATH}")

add_library(hbtl SHARED IMPORTED GLOBAL)
set_target_properties(hbtl PROPERTIES IMPORTED_LOCATION "${HBTL_LIB_PATH}")

add_subdirectory(gcc5CXX14)
