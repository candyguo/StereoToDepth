cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sgm_study)

SET(CMAKE_BUILD_TYPE Debug)
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

add_executable(sgm_study main.cc)
target_link_libraries(sgm_study sgm ${OpenCV_LIBS})
