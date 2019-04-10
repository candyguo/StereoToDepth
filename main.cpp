#include <iostream>
#include "stereoframe.h"
#include <sys/stat.h>

using namespace stereo_mapping;

int main() {
    Config config;
    int windows_width = config.getConfigPara().at("window_width").get<int>();
    int window_height = config.getConfigPara().at("window_height").get<int>();
    int search_radius = config.getConfigPara().at("search_radius").get<int>();
    int gradient_threshold = config.getConfigPara().at("gradient_threshold").get<int>();

    //构造内参矩阵
    cv::Mat left_intrinsic_mat = (cv::Mat_<double>(3,3) <<
            458.654,0.0,367.215,
            0.0,457.296,248.375,
            0.0,0.0,1.0);
    cv::Mat right_intrinsic_mat = (cv::Mat_<double>(3,3) <<
            457.587,0.0,379.999,
            0.0,456.134,255.238,
            0.0,0.0,1.0);

    cv::Mat left_discoff = (cv::Mat_<double>(4,1) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);
    cv::Mat right_discoff = (cv::Mat_<double>(4,1) << -0.28368365,  0.07451284, -0.00010473, -3.55590700e-05);

    cv::Mat fundmental_matrix = util::CalculateFundamentalMatrix(left_intrinsic_mat,right_intrinsic_mat);

    string path = config.getConfigPara().at("euroc_stereo_data_source").get<string>() + "/depth/";
    mkdir(path.c_str(),S_IRWXU);

    StereoFrameReader fr(config,StereoDataType::EurocStereo);
    while(stereoframe::Ptr frame = fr.next()) {
        cv::Mat left_depth = util::getDepthMapFromStereo(frame->left, frame->right,
                                                         left_intrinsic_mat,
                                                         left_discoff, right_intrinsic_mat,
                                                         right_discoff, fundmental_matrix,
                                                         frame->id,window_height,windows_width,
                                                         search_radius,gradient_threshold);

        cv::imshow("left_depth", left_depth);
        string filename = fr.image_files[frame->id];
        filename.erase(filename.length() - 4, 4);
        cv::imwrite(path + filename + ".png", left_depth);
        cv::waitKey(10);
    }
    return 0;
}