cmake_minimum_required(VERSION 3.20)
project(inference-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find FlatBuffers
find_package(Flatbuffers REQUIRED)

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

# Library with all core functionality
add_library(inference-core
    src/semaphore.cpp
    src/frame_reader.cpp
    src/detection_writer.cpp
    ${GENERATED_DIR}/frame_generated.h
    ${GENERATED_DIR}/detection_generated.h
)

target_include_directories(inference-core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${GENERATED_DIR}
    ${FLATBUFFERS_INCLUDE_DIR}
)

target_link_libraries(inference-core PUBLIC
    rt          # POSIX message queues
    pthread     # Threading
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
