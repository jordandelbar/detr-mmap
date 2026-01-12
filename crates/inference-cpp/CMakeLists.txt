cmake_minimum_required(VERSION 3.20)
project(inference-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(OpenCV REQUIRED)

# FlatBuffers (manual paths)
set(FLATBUFFERS_INCLUDE_DIR "/usr/include" CACHE PATH "FlatBuffers include directory")

# CUDA paths (manual, no find_package to avoid compiler check issues)
set(CUDA_INCLUDE_DIRS "/usr/local/cuda-13.1/include" CACHE PATH "CUDA include directory")
set(CUDA_LIBRARIES cudart CACHE STRING "CUDA runtime library")

# Generate FlatBuffers C++ headers from schemas
set(SCHEMA_DIR "${CMAKE_SOURCE_DIR}/../schema")
set(GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${GENERATED_DIR})

# Generate frame schema
add_custom_command(
    OUTPUT ${GENERATED_DIR}/frame_generated.h
    COMMAND flatc --cpp -o ${GENERATED_DIR} ${SCHEMA_DIR}/frame.fbs
    DEPENDS ${SCHEMA_DIR}/frame.fbs
    COMMENT "Generating frame FlatBuffers C++ header"
)

# Generate detection schema
add_custom_command(
    OUTPUT ${GENERATED_DIR}/detection_generated.h
    COMMAND flatc --cpp -o ${GENERATED_DIR} ${SCHEMA_DIR}/detection.fbs
    DEPENDS ${SCHEMA_DIR}/detection.fbs
    COMMENT "Generating detection FlatBuffers C++ header"
)

# TensorRT paths
set(TENSORRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT installation root")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include" CACHE PATH "TensorRT include directory")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib" CACHE PATH "TensorRT library directory")

# If system TensorRT is available, use it
if(NOT EXISTS "${TENSORRT_INCLUDE_DIR}")
    set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
    set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
endif()

# Library with all core functionality
add_library(inference-core
    src/semaphore.cpp
    src/frame_reader.cpp
    src/detection_writer.cpp
    src/preprocessing.cpp
    src/tensorrt_backend.cpp
    src/postprocessing.cpp
    ${GENERATED_DIR}/frame_generated.h
    ${GENERATED_DIR}/detection_generated.h
)

target_include_directories(inference-core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${GENERATED_DIR}
    ${FLATBUFFERS_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(inference-core PUBLIC
    rt          # POSIX message queues
    pthread     # Threading
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    nvinfer      # TensorRT
    nvonnxparser # ONNX parser for TensorRT
    cudart       # CUDA runtime
)

target_link_directories(inference-core PUBLIC
    ${TENSORRT_LIB_DIR}
    /usr/local/cuda-13.1/lib64
)

# Main executable
add_executable(inference-cpp
    src/main.cpp
)

target_link_libraries(inference-cpp PRIVATE
    inference-core
)

# Enable warnings
target_compile_options(inference-core PRIVATE
    -Wall -Wextra -Wpedantic
)
target_compile_options(inference-cpp PRIVATE
    -Wall -Wextra -Wpedantic
)
