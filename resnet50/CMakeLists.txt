cmake_minimum_required(VERSION 3.10)

project(resnet50)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")


find_package(tensorrt_flow REQUIRED)
find_package(CUDA REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)
find_library(CUDNN_LIBRARY cudnn HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

include_directories(/home/friden/code/TensorRT-8.5.2.2/include)
include_directories(/home/friden/code/TensorRT-8.5.2.2/samples/common)
set(TENSORRT_LIB_PATH "//home/friden/code/TensorRT-8.5.2.2/lib")
file(GLOB TENSORRT_LIBS "${TENSORRT_LIB_PATH}/*.so")

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${tensorrt_flow_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(resnet50_infer main.cc trt_resnet50.cc calibrator.cc)
target_link_libraries(resnet50_infer fmt::fmt ${TENSORRT_LIBS} ${OpenCV_LIBS} ${CUDA_LIBRARIES} tensorrt_flow::cuda_algo_kernel tensorrt_flow::tensorrt_flow tensorrt_flow::image_process )
