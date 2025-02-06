set(HBRT4_HBTL_WRAPPER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(HBRT4_HBTL_WRAPPER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(lib)

add_hbrt_library(hbrt4_hbtl_wrapper_header INTERFACE)
target_include_directories(hbrt4_hbtl_wrapper_header
                           INTERFACE ${HBRT4_HBTL_WRAPPER_INCLUDE_DIR})

hbut_install("rt;sim" DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} RT_SIM_DEST
             sample/hbrt/)
