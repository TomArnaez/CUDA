cmake_minimum_required(VERSION 3.17)

include_directories(${TEST_SOURCE_DIR}/src)
add_executable (Test test.cpp)
target_link_libraries(Test, NvOF)
add_test(NAME MyTest COMMAND Test)