cmake_minimum_required(VERSION 3.10)

project(tensorrt_flow)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

SET(CMAKE_BUILD_TYPE "Debug")

include_directories(include)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
find_package(fmt REQUIRED)

# file(GLOB CUDA_SOURCES  src/cuda/*.cu src/cuda/*.cc)
file(GLOB MEMORY_CC_SOURCES  src/memory/*.cc)
file(GLOB TENSORRT_CC_SOURCES  src/tensorrt/*.cc)


include_directories(/root/Code/TensorRT-8.5.2.2/include)
include_directories(/root/Code/TensorRT-8.5.2.2/samples/common)
set(TENSORRT_LIB_PATH "/root/Code/TensorRT-8.5.2.2/lib")
file(GLOB TENSORRT_LIBS "${TENSORRT_LIB_PATH}/*.so")

add_library(hetero_memory SHARED ${MEMORY_CC_SOURCES})
target_link_libraries(hetero_memory ${CUDA_LIBRARIES})


add_library(tensorrt_flow SHARED ${TENSORRT_CC_SOURCES})
target_link_libraries(tensorrt_flow PRIVATE fmt::fmt ${TENSORRT_LIBS} ${CUDA_LIBRARIES})

target_include_directories(hetero_memory
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
target_include_directories(tensorrt_flow
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

install(TARGETS hetero_memory tensorrt_flow
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
