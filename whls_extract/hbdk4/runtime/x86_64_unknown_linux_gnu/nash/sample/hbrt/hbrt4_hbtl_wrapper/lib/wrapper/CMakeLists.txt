set(sources wrapper.cpp)
include_directories(${HBRT4_HBTL_WRAPPER_INCLUDE_DIR})
add_hbrt_library(hbrt4-hbtl-interface STATIC ${sources})

add_dependencies(hbrt4-hbtl-interface hbtl_header)

add_dependencies(hbrt-rust-link-common-depend-libs hbrt4-hbtl-interface)

target_link_libraries(hbrt4-hbtl-interface
                      PUBLIC spdlog_header_only hbtl_header hbrt4_legacy_header)
target_link_libraries(hbrt4-hbtl-interface PRIVATE hbtl)

add_dependencies(hbrt-rust-link-common-depend-libs hbrt4-hbtl-interface)

# This share library be used by run model of hbdk3
add_library(hbtl-interface SHARED ${sources})

add_dependencies(hbrt-rust-link-common-depend-libs hbtl-interface)

add_dependencies(hbtl-interface hbtl_header)
target_link_libraries(hbtl-interface PUBLIC spdlog_header_only hbtl_header
                                            hbrt4_legacy_header)
target_link_libraries(hbtl-interface PRIVATE hbtl)

add_definitions(-DLIB="${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
add_dependencies(hbrt4-hbtl-interface silverc)

if(USE_OMP AND FOR_DEV_USE)
  add_definitions(-DONEDNN)
  add_dependencies(hbrt4-hbtl-interface external)
endif()
