cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ML-MCTS CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------
# Neural network support — auto-enabled if LibTorch is found
# Provide path via: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
# or set LIBTORCH_PATH environment variable
# ---------------------------------------------------------
find_package(Torch QUIET HINTS $ENV{LIBTORCH_PATH})

if(Torch_FOUND)
    message(STATUS "LibTorch found — neural network support: ON")
else()
    message(STATUS "LibTorch not found — building in random rollout mode")
    message(STATUS "  To enable neural support, provide LibTorch:")
    message(STATUS "    cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..")
endif()

# ---------------------------------------------------------
# Collect all source files under src/ (non-recursive)
# ---------------------------------------------------------
file(GLOB SRC_FILES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

# ---------------------------------------------------------
# Collect all header files under include/ (non-recursive)
# ---------------------------------------------------------
file(GLOB INCLUDE_FILES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)

# ---------------------------------------------------------
# Add Games/ subdirectory (this builds the Games library)
# ---------------------------------------------------------
add_subdirectory(Games)

# ---------------------------------------------------------
# Define the executable
# ---------------------------------------------------------
add_executable(ML-MCTS
    ${SRC_FILES}
    ${INCLUDE_FILES}
)


# ---------------------------------------------------------
# Define code definitions
# ---------------------------------------------------------
set(INFERENCE_MAX_DELAY     5)
set(MAX_BATCH_SIZE          64)
set(VIRTUAL_LOSS_VALUE      1.0)
set(MCTS_PUCT_CONSTANT      1.2)
set(SIMULATIONS_PER_MOVE    1000)
set(PREALLOCATED_NODES      1000000)
set(MCTS_THREAD_WORKERS     6)
set(MAX_PENDING_REQUESTS    64)

set(GAME_TYPE UltimateXO)
set(GAME_INCLUDE "UltimateXO/UltimateXO.h")


configure_file(
    ${CMAKE_SOURCE_DIR}/include/config.h.in
    ${CMAKE_BINARY_DIR}/generated/config.h
)

target_include_directories(ML-MCTS PRIVATE
    ${CMAKE_BINARY_DIR}/generated
)

# ---------------------------------------------------------
# Include directories for ML-MCTS
# ---------------------------------------------------------
target_include_directories(ML-MCTS PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/moodycamel"
)

# ---------------------------------------------------------
# Link libraries
# ---------------------------------------------------------
target_link_libraries(ML-MCTS PRIVATE Games)


if(Torch_FOUND)
    target_compile_definitions(ML-MCTS PRIVATE NEURAL_ENABLED)
    target_include_directories(ML-MCTS PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(ML-MCTS PRIVATE ${TORCH_LIBRARIES})
endif()

# Debug build settings
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(ML-MCTS PRIVATE -g -O0 
        -fsanitize=address,undefined 
        -fno-omit-frame-pointer
    )
    target_link_options(ML-MCTS PRIVATE 
        -fsanitize=address,undefined
    )
endif()

# Profile build settings
if(CMAKE_BUILD_TYPE STREQUAL "Profile")
    target_compile_options(ML-MCTS PRIVATE 
        -O3                        # optimised like release
        -g                         # debug symbols for perf/flamegraph
        -fno-omit-frame-pointer    # proper stack unwinding
    )
endif()