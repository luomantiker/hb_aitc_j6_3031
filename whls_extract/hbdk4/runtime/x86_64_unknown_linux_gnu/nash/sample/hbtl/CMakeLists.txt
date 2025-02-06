# add_subdirectory(GpuSample)
add_subdirectory(Ude)

if(HBDK_WHOLE_BUILD)
  hbut_install("rt;sim" DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ RT_SIM_DEST
               sample/hbtl/)
endif()
