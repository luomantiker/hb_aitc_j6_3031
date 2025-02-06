add_executable(hbtl_gcc5_test TestRunbyValue.cpp)

target_link_libraries(hbtl_gcc5_test PRIVATE hbtl_gtest_main hbtl
                                             spdlog::spdlog_header_only)
