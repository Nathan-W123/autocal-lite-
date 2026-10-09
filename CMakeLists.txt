cmake_minimum_required(VERSION 3.16)
project(autocal_lite
        VERSION 0.1.0
        DESCRIPTION "Targetless stereo/IMU extrinsic calibration with drift tracking, built on the Aether6 core"
        LANGUAGES CXX)

# Build system reused from Aether6: same dependency strategy (system package first, fetched
# otherwise), same warning set, same layout (library + apps + Catch2 tests). yaml-cpp is
# not needed here, so the only dependencies are Eigen and, for the tests, Catch2.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(AETHER_BUILD_TESTS "Build the Catch2 unit-test suite" ON)
option(AETHER_ENABLE_WARNINGS "Enable a strict warning set" ON)
option(AETHER_ENABLE_WERROR "Treat warnings as errors" OFF)

include(FetchContent)

find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found on the system - fetching it")
  FetchContent_Declare(Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(Eigen3)
  # Treat the fetched headers like a system package so -Werror only applies to our code
  # (GCC reports false-positive -Wmaybe-uninitialized inside Eigen 3.4 at -O2).
  get_target_property(_eigen_includes eigen INTERFACE_INCLUDE_DIRECTORIES)
  set_target_properties(eigen PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_eigen_includes}")
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------------------
# Core library: the reused Aether6 core (core/math/util/sensors/estimation/mc) plus the
# new calibration module (calib/).
# ---------------------------------------------------------------------------------------
file(GLOB_RECURSE AETHER_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(aether ${AETHER_SOURCES})
target_include_directories(aether PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aether PUBLIC Eigen3::Eigen Threads::Threads)

function(aether_set_warnings target)
  if(AETHER_ENABLE_WARNINGS)
    if(MSVC)
      target_compile_options(${target} PRIVATE /W4)
    else()
      target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow
                                               -Wnon-virtual-dtor -Wcast-align -Wunused
                                               -Woverloaded-virtual -Wdouble-promotion)
    endif()
  endif()
  if(AETHER_ENABLE_WERROR AND NOT MSVC)
    target_compile_options(${target} PRIVATE -Werror)
  endif()
endfunction()

aether_set_warnings(aether)

# ---------------------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------------------
set(AUTOCAL_APPS autocal_run autocal_mc)
foreach(app IN LISTS AUTOCAL_APPS)
  add_executable(${app} apps/${app}.cpp)
  target_link_libraries(${app} PRIVATE aether)
  aether_set_warnings(${app})
  set_target_properties(${app} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endforeach()

# ---------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------
if(AETHER_BUILD_TESTS)
  enable_testing()
  find_package(Catch2 3 QUIET)
  if(NOT Catch2_FOUND)
    message(STATUS "Catch2 v3 not found on the system - fetching it")
    FetchContent_Declare(Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.5.2
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
  else()
    if(EXISTS "${Catch2_DIR}/../../../share/Catch2/extras")
      list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}/../../../share/Catch2/extras")
    endif()
  endif()

  file(GLOB AETHER_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
  add_executable(autocal_tests ${AETHER_TEST_SOURCES})
  target_link_libraries(autocal_tests PRIVATE aether Catch2::Catch2WithMain)
  aether_set_warnings(autocal_tests)
  set_target_properties(autocal_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  include(CTest)
  include(Catch OPTIONAL RESULT_VARIABLE AETHER_CATCH_MODULE)
  if(AETHER_CATCH_MODULE)
    catch_discover_tests(autocal_tests)
  else()
    add_test(NAME autocal_tests COMMAND autocal_tests)
  endif()
endif()

message(STATUS "autocal-lite ${PROJECT_VERSION}: build type = ${CMAKE_BUILD_TYPE}, tests = ${AETHER_BUILD_TESTS}")
