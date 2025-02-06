add_hbrt4_public_library(HBRT4PublicExampleCpuBpuMixedModel
                         cpu_bpu_mixed_model.cpp)
target_include_directories(
  HBRT4PublicExampleCpuBpuMixedModel
  PUBLIC ${HBRT_SOURCE_DIR}/hbrt4_hbtl_wrapper/include
         ${HBTOP_SOURCE_DIR}/hbtl/include)

set(example_link_libs hbrt4 hbtl hbtl-interface)

if((NOT CMAKE_CROSSCOMPILING) AND HBDK_BUILD_SIMULATOR)
  list(APPEND example_link_libs hbdk_sim_x86)
endif()
if(HBDK_BUILD_HAS_B25_SERIES)
  list(APPEND example_link_libs cnn_intf)
  add_compile_definitions(EXAMPLE_B25)
  target_include_directories(HBRT4PublicExampleCpuBpuMixedModel
                             PUBLIC ${HBRT_SOURCE_DIR}/cnn_intf/include)
elseif(HBDK_BUILD_HAS_B30_SERIES)
  list(APPEND example_link_libs hb3rd_libbpu_b30)
  add_compile_definitions(EXAMPLE_B30)
endif()

target_link_libraries(HBRT4PublicExampleCpuBpuMixedModel
                      PRIVATE ${example_link_libs})

add_dependencies(hbrt4-public-all HBRT4PublicExampleCpuBpuMixedModel)

if(HBDK_WHOLE_BUILD)
  hbut_install("rt;sim" DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} RT_SIM_DEST
               sample/hbrt/)
endif()
add_hbrt4_public_executable(HBRT4PublicExample cpu_bpu_mixed_model_example.cpp)
target_link_libraries(HBRT4PublicExample
                      PUBLIC HBRT4PublicExampleCpuBpuMixedModel hbrt4 hbtl)
add_dependencies(hbrt4-public-all HBRT4PublicExample)
