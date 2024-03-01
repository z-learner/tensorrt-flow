cmake_minimum_required(VERSION 3.10)

project(yolo)

project(yolo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --shared -Xcompiler -fPIC --expt-relaxed-constexpr")

find_package(tensorrt_flow REQUIRED)
find_package(CUDA REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(/home/friden/code/TensorRT-8.5.2.2/include)
include_directories(/home/friden/code/TensorRT-8.5.2.2/samples/common)
set(TENSORRT_LIB_PATH "//home/friden/code/TensorRT-8.5.2.2/lib")
file(GLOB TENSORRT_LIBS "${TENSORRT_LIB_PATH}/*.so")
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${tensorrt_flow_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

file(GLOB CUDA_SROUCE  src/*.cu)
file(GLOB CC_SROUCE  src/*.cc)


cuda_add_library(affine_kernel SHARED ${CUDA_SROUCE})
target_link_libraries(affine_kernel ${CUDA_LIBRARIES})
set_property(TARGET affine_kernel PROPERTY CUDA_ARCHITECTURES 75)

add_executable(yolo  main.cc ${CC_SROUCE})
target_link_libraries(yolo affine_kernel fmt::fmt ${TENSORRT_LIBS} ${OpenCV_LIBS} ${CUDA_LIBRARIES} tensorrt_flow::tensorrt_flow tensorrt_flow::image_process )
