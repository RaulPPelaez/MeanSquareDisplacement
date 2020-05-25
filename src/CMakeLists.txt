cmake_minimum_required (VERSION 3.8)
add_executable(MeanSquareDisplacement MeanSquareDisplacement.cpp)
if(NOT DONT_USE_CUDA)
  SET_SOURCE_FILES_PROPERTIES(MeanSquareDisplacement.cpp PROPERTIES LANGUAGE CUDA)
  target_link_libraries(MeanSquareDisplacement cufft)
endif()

set_target_properties(MeanSquareDisplacement PROPERTIES OUTPUT_NAME "msd")

if(USE_CPU)

  find_package(FFTW QUIET COMPONENTS SIMPLE)

  if(NOT FFTW_FOUND)
    if(USE_MKL)
      find_package(FFTW QUIET REQUIRED COMPONENTS MKL SIMPLE)
      include_directories(MeanSquareDisplacement ${FFTW_fftw3.h_DIRS})
      set(USE_MKL ON)
    else()
      set(USE_CPU OFF)
    endif()
  endif()
  if(USE_CPU)
    include_directories(MeanSquareDisplacement ${FFTW_INCLUDE_DIRS})
    link_directories(MeanSquareDisplacement ${FFTW_LIBRARY_DIRS})

    target_link_libraries(MeanSquareDisplacement  ${FFTW_LIBRARIES})
  endif()
endif()

install(TARGETS MeanSquareDisplacement RUNTIME DESTINATION bin)

configure_file ( "defines.h.in" ${CMAKE_CURRENT_SOURCE_DIR}/defines.h  )


find_package(Git)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(COMMAND bash  ${PROJECT_SOURCE_DIR}/cmake/gitversion.sh OUTPUT_VARIABLE GITCOMMIT)
else()
  SET(GITCOMMIT "unknown")
endif()




configure_file("gitversion.h.in" "${CMAKE_CURRENT_SOURCE_DIR}/gitversion.h")
