add_library(stereo_lib stereoframe.cpp)
target_link_libraries(stereo_lib
        ${OpenCV_LIBS}
        )