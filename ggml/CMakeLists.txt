cmake_minimum_required(VERSION 3.16)
project(moshi-ggml VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(MOSHI_GGML_BUILD_TESTS "Build tests" ON)
option(MOSHI_GGML_METAL "Enable Metal backend (Apple Silicon)" ON)
option(MOSHI_GGML_ACCELERATE "Enable Accelerate framework (macOS)" ON)

# Find ggml - requires llama.cpp to be available
# Users should set LLAMA_CPP_DIR to point to their llama.cpp installation
if(NOT DEFINED LLAMA_CPP_DIR)
    set(LLAMA_CPP_DIR "$ENV{HOME}/github/llama.cpp" CACHE PATH "Path to llama.cpp")
endif()

if(EXISTS "${LLAMA_CPP_DIR}/ggml/include/ggml.h")
    message(STATUS "Found llama.cpp at ${LLAMA_CPP_DIR}")
else()
    message(FATAL_ERROR "Could not find llama.cpp at ${LLAMA_CPP_DIR}. "
                        "Please set LLAMA_CPP_DIR to the llama.cpp directory.")
endif()

# Include ggml headers
include_directories(
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link directories
link_directories(${LLAMA_CPP_DIR}/build/bin)

# Platform-specific settings
if(APPLE)
    if(MOSHI_GGML_METAL)
        add_definitions(-DGGML_USE_METAL)
        find_library(METAL_LIBRARY Metal REQUIRED)
        find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
        find_library(METALKIT_LIBRARY MetalKit REQUIRED)
    endif()

    if(MOSHI_GGML_ACCELERATE)
        add_definitions(-DGGML_USE_ACCELERATE)
        find_library(ACCELERATE_LIBRARY Accelerate REQUIRED)
    endif()
endif()

# Moshi GGML library sources
set(MOSHI_GGML_SOURCES
    src/moshi_lm.cpp
    src/moshi_lm.c
)

# Create static library
add_library(moshi-ggml STATIC ${MOSHI_GGML_SOURCES})

target_include_directories(moshi-ggml PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/include
)

# Link against ggml libraries
target_link_libraries(moshi-ggml
    ggml
    ggml-base
    ggml-cpu
)

if(APPLE)
    if(MOSHI_GGML_METAL)
        target_link_libraries(moshi-ggml
            ggml-metal
            ${METAL_LIBRARY}
            ${FOUNDATION_LIBRARY}
            ${METALKIT_LIBRARY}
        )
    endif()

    if(MOSHI_GGML_ACCELERATE)
        target_link_libraries(moshi-ggml ${ACCELERATE_LIBRARY})
    endif()
endif()

# Tests
if(MOSHI_GGML_BUILD_TESTS)
    enable_testing()

    add_executable(test-rmsnorm tests/test_rmsnorm.cpp)
    target_link_libraries(test-rmsnorm moshi-ggml)
    add_test(NAME test-rmsnorm COMMAND test-rmsnorm)
endif()

# Install
install(TARGETS moshi-ggml
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)
