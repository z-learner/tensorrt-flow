cmake_minimum_required(VERSION 3.10)

project(test)

set(CMAKE_CXX_STANDARD 17)

find_package(benchmark REQUIRED)
find_package(tensorrt_flow REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${tensorrt_flow_INCLUDE_DIRS})
include_directories(${benchmark_INCLUDE_DIRS})

##### memory test #####
add_executable(memory_test test_memory.cc)
target_link_libraries(memory_test benchmark::benchmark tensorrt_flow::hetero_memory)
##### memory test #####


