cmake_minimum_required(VERSION 3.14)
project(moshi_tools CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find GGML from llama.cpp
# Set LLAMA_CPP_DIR to your llama.cpp checkout, e.g.:
#   cmake .. -DLLAMA_CPP_DIR=/path/to/llama.cpp
set(LLAMA_CPP_DIR "" CACHE PATH "Path to llama.cpp")
if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR "LLAMA_CPP_DIR must be set. Use: cmake .. -DLLAMA_CPP_DIR=/path/to/llama.cpp")
endif()
set(GGML_INCLUDE_DIR "${LLAMA_CPP_DIR}/ggml/include")
set(LLAMA_BUILD_DIR "${LLAMA_CPP_DIR}/build" CACHE PATH "Path to llama.cpp build dir")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${GGML_INCLUDE_DIR}
)

# Link directories
link_directories(${LLAMA_BUILD_DIR}/bin)

# Moshi STT library
add_library(moshi_stt_lib STATIC src/moshi_stt.cpp)
target_link_libraries(moshi_stt_lib ggml-base)

# Moshi STT CLI tool
add_executable(moshi_stt tools/moshi_stt.cpp)
target_link_libraries(moshi_stt moshi_stt_lib ggml-base)
set_target_properties(moshi_stt PROPERTIES
    BUILD_RPATH "${LLAMA_BUILD_DIR}/bin"
    INSTALL_RPATH "${LLAMA_BUILD_DIR}/bin"
)

# Mimi decode tool
add_executable(mimi_decode tools/mimi_decode.cpp)
target_link_libraries(mimi_decode ggml-base)
set_target_properties(mimi_decode PROPERTIES
    BUILD_RPATH "${LLAMA_BUILD_DIR}/bin"
    INSTALL_RPATH "${LLAMA_BUILD_DIR}/bin"
)

# Mimi encoder library
add_library(mimi_lib STATIC src/mimi.cpp)
target_link_libraries(mimi_lib ggml-base)

# Mimi encode CLI tool
add_executable(mimi_encode tools/mimi_encode.cpp)
target_link_libraries(mimi_encode mimi_lib ggml-base)
set_target_properties(mimi_encode PROPERTIES
    BUILD_RPATH "${LLAMA_BUILD_DIR}/bin"
    INSTALL_RPATH "${LLAMA_BUILD_DIR}/bin"
)

# Moshi Audio Server (with Mimi encoder + STT)
add_executable(moshi_server tools/moshi_server.cpp)
target_compile_definitions(moshi_server PRIVATE CPPHTTPLIB_NO_EXCEPTIONS)
target_link_libraries(moshi_server mimi_lib moshi_stt_lib ggml-base)
set_target_properties(moshi_server PROPERTIES
    BUILD_RPATH "${LLAMA_BUILD_DIR}/bin"
    INSTALL_RPATH "${LLAMA_BUILD_DIR}/bin"
)

# Tests
enable_testing()

add_test(NAME test_stt_basic
    COMMAND ${CMAKE_SOURCE_DIR}/tests/test_stt.sh ${CMAKE_BINARY_DIR}/moshi_stt
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
