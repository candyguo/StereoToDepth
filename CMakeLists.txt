cmake_minimum_required(VERSION 3.5)
project(StereoToDepth)

set( CMAKE_BUILD_TYPE Release )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

set(OpenCV_DIR /usr/local/opencv249/share/OpenCV)
find_package( OpenCV 2.4.9 REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )


include_directories( "/usr/include/eigen3" )

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib )

include_directories(
        ${PROJECT_SOURCE_DIR}/
        ${PROJECT_SOURCE_DIR}/include/
        ${EIGEN3_INCLUDE_DIR}
)
add_subdirectory( ${PROJECT_SOURCE_DIR}/src/ )
set(SOURCE_FILES main.cpp)
add_executable(StereoToDepth ${SOURCE_FILES})
target_link_libraries(StereoToDepth stereo_lib)