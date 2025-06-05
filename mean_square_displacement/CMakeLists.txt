cmake_minimum_required(VERSION 3.11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(FFTW REQUIRED COMPONENTS FLOAT_LIB DOUBLE_LIB)
add_library(msd SHARED src/msd.cpp)
target_include_directories(msd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(msd PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(msd::msd ALIAS msd) # For FetchContent compatibility

add_library(autocorr STATIC src/autocorr.cpp)

add_library(autocorr_cpu STATIC src/autocorr_cpu.cpp)
target_link_libraries(autocorr_cpu PUBLIC FFTW::Float FFTW::Double)
target_link_libraries(autocorr PUBLIC autocorr_cpu)

if(USE_CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED YES)
  add_library(autocorr_cuda STATIC src/autocorr_cuda.cu)
  set_target_properties(autocorr_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
  target_link_libraries(autocorr_cuda PUBLIC cufft)
  target_link_libraries(autocorr PUBLIC autocorr_cuda)
endif()

target_link_libraries(msd PUBLIC autocorr)

if(BUILD_EXECUTABLE)
  add_executable(MeanSquareDisplacement src/MeanSquareDisplacement.cpp)
  target_link_libraries(MeanSquareDisplacement msd)
  set_target_properties(MeanSquareDisplacement PROPERTIES OUTPUT_NAME "msd")
  install(TARGETS MeanSquareDisplacement RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/defines.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/defines.h)

find_package(Git)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(COMMAND bash  ${PROJECT_SOURCE_DIR}/cmake/gitversion.sh OUTPUT_VARIABLE GITCOMMIT)
else()
  SET(GITCOMMIT "unknown")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/gitversion.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/gitversion.h)


if(BUILD_PYTHON)
  find_package(Python 3
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)
  find_package(nanobind QUIET)
  if (NOT nanobind_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      nanobind
      GIT_REPOSITORY https://github.com/wjakob/nanobind/
      GIT_TAG        v2.6.0
    )
    FetchContent_MakeAvailable(nanobind)
  endif()
  nanobind_add_module(wrapper STABLE_ABI python/mean_square_displacement/wrapper.cpp)
  target_link_libraries(wrapper PRIVATE msd)
  install(TARGETS wrapper LIBRARY DESTINATION ${Python_SITEARCH}/mean_square_displacement)
endif()

if(INSTALL_HEADER_FILES)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION include
)
endif()
install(TARGETS msd LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(
  EXPORT msdTargets
  FILE msdTargets.cmake
  NAMESPACE msd::
  DESTINATION lib/cmake/msd
)
