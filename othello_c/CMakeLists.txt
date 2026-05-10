cmake_minimum_required(VERSION 3.18)
project(othello_c C)

# ---------------------------------------------------------------------------
# Compile options
# ---------------------------------------------------------------------------
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Common warning flags. -O3/-march=native for Release; sanitizers for Debug.
set(COMMON_C_FLAGS -Wall -Wextra -Wno-unused-parameter)
add_compile_options(${COMMON_C_FLAGS})
add_compile_options($<$<CONFIG:Release>:-O3>)
add_compile_options($<$<CONFIG:Release>:-DNDEBUG>)
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    add_compile_options($<$<CONFIG:Release>:-march=native>)
endif()
add_compile_options($<$<CONFIG:Debug>:-O0>)
add_compile_options($<$<CONFIG:Debug>:-g>)
add_compile_options($<$<CONFIG:Debug>:-fsanitize=address,undefined>)
add_link_options($<$<CONFIG:Debug>:-fsanitize=address,undefined>)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

# ONNX Runtime (C API). Search order:
#   1. -DONNXRUNTIME_ROOT_DIR=...          (CMake cache var)
#   2. ONNXRUNTIME_ROOT_DIR env var
#   3. brew --prefix onnxruntime           (macOS)
#   4. extracted onnxruntime-*/ dir under the project root or the parent repo
#      (e.g. when you ran `tar xz` next to CMakeLists.txt)
#   5. system locations: /usr/local, /opt/onnxruntime
set(_ORT_HINTS)
if(ONNXRUNTIME_ROOT_DIR)
    list(APPEND _ORT_HINTS "${ONNXRUNTIME_ROOT_DIR}")
endif()
if(DEFINED ENV{ONNXRUNTIME_ROOT_DIR})
    list(APPEND _ORT_HINTS "$ENV{ONNXRUNTIME_ROOT_DIR}")
endif()
if(APPLE)
    execute_process(COMMAND brew --prefix onnxruntime
                    OUTPUT_VARIABLE _BREW_ORT
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(_BREW_ORT)
        list(APPEND _ORT_HINTS "${_BREW_ORT}")
    endif()
endif()
# Extracted release tarball lying next to CMakeLists.txt (or in parent repo)
file(GLOB _ORT_LOCAL_EXTRACTED
     "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-*"
     "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-*"
     "/content/onnxruntime-*"
     "/usr/local/onnxruntime-*")
foreach(_p ${_ORT_LOCAL_EXTRACTED})
    if(IS_DIRECTORY "${_p}")
        list(APPEND _ORT_HINTS "${_p}")
    endif()
endforeach()
list(APPEND _ORT_HINTS /usr/local /opt/onnxruntime /usr)

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_c_api.h
    HINTS ${_ORT_HINTS}
    PATH_SUFFIXES include include/onnxruntime onnxruntime
)
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS ${_ORT_HINTS}
    PATH_SUFFIXES lib lib64
)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found.\n"
        "  searched hints: ${_ORT_HINTS}\n"
        "Fix one of:\n"
        "  - cmake -DONNXRUNTIME_ROOT_DIR=/path/to/extracted/onnxruntime ...\n"
        "  - export ONNXRUNTIME_ROOT_DIR=/path/to/extracted/onnxruntime\n"
        "  - on macOS: brew install onnxruntime\n"
        "On Colab, after `tar xz` the GPU tarball, point ONNXRUNTIME_ROOT_DIR\n"
        "at the extracted onnxruntime-linux-x64-gpu-X.Y.Z/ directory.")
endif()
message(STATUS "ONNX Runtime header: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib   : ${ONNXRUNTIME_LIB}")

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(othello_core STATIC
    src/othello.c
    src/zobrist.c
    src/tree.c
    src/nn.c
    src/mcts.c
    src/coach.c
    src/arena.c
)
target_include_directories(othello_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(othello_core PUBLIC
    ${ONNXRUNTIME_LIB}
    Threads::Threads
    m
)

# ---------------------------------------------------------------------------
# Executable + benchmark
# ---------------------------------------------------------------------------
add_executable(othello_c src/main.c)
target_link_libraries(othello_c PRIVATE othello_core)

add_executable(bench_mcts benchmarks/bench_mcts.c)
target_link_libraries(bench_mcts PRIVATE othello_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

add_executable(test_othello tests/test_othello.c)
target_link_libraries(test_othello PRIVATE othello_core)
add_test(NAME othello_rules
    COMMAND test_othello ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixture_othello.bin)

add_executable(test_tree tests/test_tree.c)
target_link_libraries(test_tree PRIVATE othello_core)
add_test(NAME tree_sanity COMMAND test_tree)

add_executable(test_nn tests/test_nn.c)
target_link_libraries(test_nn PRIVATE othello_core)
add_test(NAME nn_parity
    COMMAND test_nn
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/random_model.onnx
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/nn_fixture.bin)

add_executable(test_mcts tests/test_mcts.c)
target_link_libraries(test_mcts PRIVATE othello_core)
add_test(NAME mcts_serial_parity
    COMMAND test_mcts
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/random_model.onnx
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/mcts_fixture.bin)

add_executable(test_mcts_threaded tests/test_mcts_threaded.c)
target_link_libraries(test_mcts_threaded PRIVATE othello_core)
add_test(NAME mcts_threaded
    COMMAND test_mcts_threaded
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/random_model.onnx)
