cmake_minimum_required (VERSION 3.11)
# Try to get version from Git tags
execute_process(
    COMMAND git describe --tags --abbrev=0
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Fallback if git fails
if(NOT GIT_TAG)
    set(GIT_TAG "0.0.0")
endif()

project(MeanSquareDisplacement VERSION ${GIT_TAG} LANGUAGES CXX)
message(STATUS "MeanSquareDisplacement version: ${GIT_TAG}")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
option(USE_BOOST "Use Boost-qi for reading, highly improves reading performance" OFF)
option(USE_CUDA "Compile in hybrid CPU/GPU mode, requires nvcc" ON)
option(BUILD_TESTS "Build the tests" ON)
option(BUILD_EXECUTABLE "Build the executable" ON)
if(NOT CMAKE_CUDA_ARHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "all-major")
endif()
if(USE_CUDA)
  enable_language(CUDA)
  set(USE_CUDA ON)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(mean_square_displacement)

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
