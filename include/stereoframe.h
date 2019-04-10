//
// Created by ccfy on 19-4-10.
//

#ifndef STEREOTODEPTH_STEREOFRAME_H
#define STEREOTODEPTH_STEREOFRAME_H

#include "common.h"

namespace stereo_mapping {

    class stereoframe {
    public:
        typedef shared_ptr<stereoframe> Ptr;

    public:
        stereoframe(){}

    public:
        int id = -1;
        cv::Mat left,right;
        //当前双目相机中间到世界坐标系的变换
        Eigen::Isometry3d T_f_w = Eigen::Isometry3d::Identity();
        CAMERA_INTRINSIC_PARAMETERS left_camera;
        CAMERA_INTRINSIC_PARAMETERS right_camera;
        //当前双目的视差影像
        cv::Mat disparity;
        //当前双目的深度影像
        cv::Mat depth;
    };

    enum StereoDataType {
        UnrealStereo = 0,
        EurocStereo = 1,
        TumStereo = 2
    };

    class StereoFrameReader
    {
    public:
        StereoFrameReader(Config config, StereoDataType dataType = StereoDataType::UnrealStereo) {
            this->config = config;
            DataType = dataType;
            if(DataType == StereoDataType::UnrealStereo)
                init_unreal();
            else if(DataType == StereoDataType::EurocStereo)
                init_euroc();
            else
                std::cout<<"current datatype is not supported"<<std::endl;
        }

        stereoframe::Ptr next();// 获取currentIndex的影像

        void reset() {
            currentIndex = startIndedx;
        }

        stereoframe::Ptr get(const int& index) {
            if(index < 0 || index > image_files.size()) {
                std::cout<<"index is invalid"<<std::endl;
                return nullptr;
            }
            currentIndex = index;
            return next();
        }

        int currentIndex = 0;
        vector<string> image_files;

    protected:
        void init_unreal(); //对读取类的成员变量进行初始化
        void init_euroc();
    private:
        // image read index
        int startIndedx = 0;
        Config config;
        string dataset_dir;
        StereoDataType DataType = StereoDataType::UnrealStereo;
        CAMERA_INTRINSIC_PARAMETERS left_camera;
        CAMERA_INTRINSIC_PARAMETERS right_camera;
    };
}

#endif //STEREOTODEPTH_STEREOFRAME_H
