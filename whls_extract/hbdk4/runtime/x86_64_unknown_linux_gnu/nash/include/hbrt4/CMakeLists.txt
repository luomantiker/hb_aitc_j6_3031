add_subdirectory(ModelInfo)

hbut_install("rt;sim" DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} RT_SIM_DEST
             include/)
