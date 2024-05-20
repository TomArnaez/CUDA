cmake_minimum_required(VERSION 3.17)

project(NvOF VERSION 1.0 LANGUAGES CXX CUDA)

###############################################################################
# C++ compiler setup
###############################################################################

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

###############################################################################
# Dependencies
###############################################################################

find_package(OpenCV REQUIRED)
find_package(tl-expected REQUIRED)
find_package(CUDAToolkit REQUIRED)

###############################################################################
# Program
###############################################################################

file(GLOB_RECURSE HEADER_FILES "src/*.hpp")
file(GLOB_RECURSE SRC_FILES "src/*.cpp")

set_source_files_properties(${HEADER_FILES} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${SRC_FILES} PROPERTIES LANGUAGE CUDA)

add_library(NvOF SHARED ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(NvOF PUBLIC tl::expected)
set_target_properties(NvOF PROPERTIES CUDA_ARCHITECTURES OFF) # Not sure abut this setting

add_executable (demo src/main.cpp)
target_link_libraries (demo PUBLIC NvOF ${OpenCV_LIBS})