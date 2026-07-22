cmake_minimum_required(VERSION 3.14)
project(TextNormalizer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# build the Normalizer Library
add_library(normalizer normalizer.cpp)
target_include_directories(normalizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Fetch and Load Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0 
)

# Fetch pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# Build the Python extension module
pybind11_add_module(normalizer_python bindings.cpp)
target_link_libraries(normalizer_python PRIVATE normalizer)

# downloads and makes GTest targets available
FetchContent_MakeAvailable(googletest)

# Build and Link Tests
enable_testing()

add_executable(normalizer_tests test_normalizer.cpp)

target_link_libraries(normalizer_tests
    normalizer
    gtest_main 
)

include(GoogleTest)
gtest_discover_tests(normalizer_tests)