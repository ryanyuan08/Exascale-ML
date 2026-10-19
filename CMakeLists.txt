cmake_minimum_required(VERSION 3.14)

project(
    BurgersEquationSolver
    VERSION 1.0
    DESCRIPTION "Basic 1d Burgers Equation Solver"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Default to an optimized build. Without this, a plain `cmake ..` produces an
# unoptimized binary (no -O flags), which makes the numerical kernels several
# times slower. Single-config generators only; multi-config generators (e.g.
# Visual Studio) pick the configuration at build time instead.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo
                                                 MinSizeRel)
    message(STATUS "No build type specified, defaulting to ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_definitions(
    MODEL_PATH="${CMAKE_SOURCE_DIR}/neuralnet/output/model.pt")

include(FetchContent)

set(_local_torch_dir "${PROJECT_SOURCE_DIR}/lib/cpp/python/torch")
set(_torch_config "${_local_torch_dir}/share/cmake/Torch/TorchConfig.cmake")

if(NOT EXISTS "${_torch_config}")
    message(
        FATAL_ERROR
            "torch not found at ${_local_torch_dir}. "
            "Install it first with:\n" "    python3 scripts/setup_deps.py")
endif()

set(Torch_DIR "${_local_torch_dir}/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP true)
FetchContent_MakeAvailable(json)

add_library(
    burgers_core STATIC
    src/burgers.cpp
    src/initial_condition_generator.cpp
    src/math_utils.cpp
    src/metadata_writer.cpp
    src/schemes/ftcs.cpp
    src/schemes/lax_wendroff.cpp
    src/schemes/godunov.cpp
    src/schemes/ftcs_conservative.cpp
    include/burger_scheme.h
    include/burgers.h
    include/initial_condition_generator.h
    include/math_utils.h
    include/metadata_writer.h
    include/nn.h)

target_include_directories(burgers_core PUBLIC "${PROJECT_SOURCE_DIR}/include"
                                               "${PROJECT_BINARY_DIR}/include")

# Enable M_PI.
target_compile_definitions(burgers_core PUBLIC _USE_MATH_DEFINES)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

target_link_libraries(burgers_core PUBLIC torch nlohmann_json::nlohmann_json)

add_executable(burgers src/main.cpp)
target_link_libraries(burgers PRIVATE burgers_core)

add_executable(training_data_generator src/training_data_generator.cpp)
target_link_libraries(training_data_generator PRIVATE burgers_core)
