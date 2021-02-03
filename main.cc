#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "sgm.h"

int main(int argc, char** argv) {
    std::string dir = "/media/guochengcheng/perception_g/tda4_day/";
    for(int i = 12186; i < 13000; i++) {

      const std::string left_image_name = "/home/NULLMAX/guochengcheng/0913/left/17424.bmp";
      const std::string right_image_name = "/home/NULLMAX/guochengcheng/0913/right/17424.bmp";

      cv::Mat left_img = cv::imread(left_image_name, cv::IMREAD_GRAYSCALE);
      cv::Mat right_img = cv::imread(right_image_name, cv::IMREAD_GRAYSCALE);

//      cv::Mat left_img = cv::imread(dir + "left/" + std::to_string(i) + ".bmp", cv::IMREAD_GRAYSCALE);
//      cv::Mat right_img = cv::imread(dir + "right/" + std::to_string(i) + ".bmp", cv::IMREAD_GRAYSCALE);

      cv::imshow("origin left", left_img);
      cv::imshow("origin right", right_img);

      SGMOption option;
      option.min_disparity = 0;
      option.max_disparity = 32;
      option.num_path = 8;
      option.P1 = 10;
      option.P2 = 150;
      option.is_lr_check = true;
      option.is_check_unique = true;
      option.LR_check_thres = 1.0f;
      option.uniqueness_ratio = 0.9;
      option.is_remove_sparkles = true;
      option.min_sparkle_area = 30;
      option.is_fill_holes = true;

      const int img_width = left_img.cols;
      const int img_height = left_img.rows;

      SemiGlobalMatching sgm;
      sgm.Initialize(img_width, img_height, option);

      float* disp_left = new float[img_width * img_height];
      sgm.Match(left_img.data, right_img.data, disp_left);

      SGMUtils::ShowDispImage(disp_left, img_width, img_height, std::to_string(i) + ".png");

      cv::waitKey(0);

      delete []disp_left;
      disp_left = nullptr;
    }


    std::cout<< "sgm from zero to one " << std::endl;
}