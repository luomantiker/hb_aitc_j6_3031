add_library(udetest SHARED TestReg.cpp)
target_link_libraries(udetest PRIVATE hbtl)

if(NOT HBDK_WHOLE_BUILD)
  set_target_properties(udetest PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                           "${CMAKE_BINARY_DIR}/lib")
endif(NOT HBDK_WHOLE_BUILD)
