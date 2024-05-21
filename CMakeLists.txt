cmake_minimum_required(VERSION 3.17)

project(GPUViewer VERSION 1.0 LANGUAGES CXX CUDA)

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

list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")

###############################################################################
# Dependencies
###############################################################################

find_package(VulkanHeaders CONFIG)
find_package(OpenCV REQUIRED)
find_package(tl-expected REQUIRED)
find_package(CUDAToolkit REQUIRED)

###############################################################################
# Program
###############################################################################

include(FindCUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src)

set (SRC_FILES
    src/correctioncore.cpp
    src/corrections.cpp
    src/vulkaninterop.cpp
)

set (HEADER_FILES
    src/correctioncore.hpp
    src/corrections.hpp
    src/vulkaninterop.hpp
)

set_source_files_properties(${HEADER_FILES} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${SRC_FILES} PROPERTIES LANGUAGE CUDA)

add_library(CudaCorrection SHARED ${SRC_FILES} ${HEADER_FILES})

target_compile_options(CudaCorrection PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

target_link_libraries(CudaCorrection PUBLIC tl::expected)
set_target_properties(CudaCorrection PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES OFF) # Not sure abut this setting

add_executable (demo src/main.cpp)
target_link_libraries(demo PRIVATE CudaCorrection tl::expected ${OpenCV_LIBS} Vulkan::Headers)