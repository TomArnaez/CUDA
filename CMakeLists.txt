cmake_minimum_required(VERSION 3.17)

project(GPUViewer VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")

###############################################################################
# Dependencies
###############################################################################

##########
#  CUDA  #
##########

find_package(CUDAToolkit REQUIRED)

###############################################################################
# Program
###############################################################################

include(FindCUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src)
file(GLOB SRC_FILES "src/*.cpp")

add_executable(MyExecutable ${SRC_FILES})
target_compile_options(MyExecutable PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
set_target_properties(MyExecutable PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES OFF) # Not sure abut this setting