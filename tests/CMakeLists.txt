include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.16.0
)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(CTest)
include(GoogleTest)
add_executable(test_msd test_msd.cpp)
target_include_directories(test_msd PUBLIC ${PROJECT_SOURCE_DIR}/mean_square_displacement/include)
target_link_libraries(
  test_msd
  PRIVATE
  GTest::gtest_main
  msd
)
gtest_discover_tests(test_msd)
