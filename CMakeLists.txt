cmake_minimum_required(VERSION 3.10)

project(tensorrt_flow)
project(tensorrt_flow LANGUAGES CXX CUDA)



set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --shared -Xcompiler -fPIC --expt-relaxed-constexpr")

include_directories(include)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB MEMORY_CC_SOURCES  src/memory/*.cc)
file(GLOB TENSORRT_CC_SOURCES  src/tensorrt/*.cc)
file(GLOB CUDA_ALGO_KERNEL_CU_SOURCES  src/cuda/*.cu)
file(GLOB IMAGE_PEOCESS_CC_SOURCES  src/image_process/*.cc)


include_directories(/home/friden/code/TensorRT-8.5.2.2/include)
include_directories(/home/friden/code/TensorRT-8.5.2.2/samples/common)
set(TENSORRT_LIB_PATH "/home/friden/code/TensorRT-8.5.2.2/lib")
file(GLOB TENSORRT_LIBS "${TENSORRT_LIB_PATH}/*.so")

add_library(hetero_memory SHARED ${MEMORY_CC_SOURCES})
target_link_libraries(hetero_memory ${CUDA_LIBRARIES})

cuda_add_library(cuda_algo_kernel SHARED ${CUDA_ALGO_KERNEL_CU_SOURCES})
target_link_libraries(cuda_algo_kernel ${CUDA_LIBRARIES})
# just for 2080TI
set_property(TARGET cuda_algo_kernel PROPERTY CUDA_ARCHITECTURES 75)

add_library(image_process SHARED ${IMAGE_PEOCESS_CC_SOURCES})
target_link_libraries(image_process PRIVATE ${CUDA_LIBRARIES} ${OpenCV_LIBS} cuda_algo_kernel)

add_library(tensorrt_flow SHARED ${TENSORRT_CC_SOURCES})
target_link_libraries(tensorrt_flow PRIVATE fmt::fmt ${TENSORRT_LIBS} ${CUDA_LIBRARIES})


target_include_directories(hetero_memory
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_include_directories(cuda_algo_kernel
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_include_directories(image_process
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)


target_include_directories(tensorrt_flow
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

install(TARGETS hetero_memory tensorrt_flow cuda_algo_kernel image_process
    EXPORT tensorrt_flowtargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(EXPORT
      tensorrt_flowtargets
      NAMESPACE  "tensorrt_flow::"
      DESTINATION  lib/cmake/tensorrt_flow
)

install(DIRECTORY include/tensorrt_flow DESTINATION include)

include(CMakePackageConfigHelpers)

configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/tensorrt_flowConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/tensorrt_flowConfig.cmake
        INSTALL_DESTINATION lib/cmake
        )

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/tensorrt_flowConfig.cmake DESTINATION lib/cmake/tensorrt_flow)
