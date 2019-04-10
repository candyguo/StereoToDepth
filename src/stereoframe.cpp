//
// Created by ccfy on 19-4-10.
//
#include "stereoframe.h"

using namespace stereo_mapping;

void StereoFrameReader::init_unreal()
{
    dataset_dir = config.getConfigPara().at("unreal_stereo_data_source").get<string>();
    string refpose_file = dataset_dir + "/ref_pose.txt";
    ifstream fin(refpose_file);
    if(!fin) {
        std::cerr <<"ref_pose.txt was not found,which is necessary for pointcloud generation" << std::endl;
        return;
    }
    while(!fin.eof()) {
        int id;
        double x,y,z,rotation_x,rotation_y,rotation_z;
        fin >> id >> x >> y >> z >> rotation_x >> rotation_y >> rotation_z;
        std::stringstream img_name;
        img_name << std::setfill('0') << std::setw(10) << id << ".tiff";
        if(!fin.good()) {
            break;
        }
        image_files.push_back(img_name.str());

    }
    std:cout<<"stereo image size is "<<image_files.size()<<std::endl;
    startIndedx = config.getConfigPara().at("start_index").get<int>();
    currentIndex = startIndedx;
}

void StereoFrameReader::init_euroc()
{
    dataset_dir = config.getConfigPara().at("euroc_stereo_data_source").get<string>();
    string cam0_file = dataset_dir + "/cam0/data.csv";
    ifstream fin(cam0_file);
    if(!fin) {
        std::cerr <<"cam0 was not found,which is necessary for pointcloud generation" << std::endl;
        return;
    }
    string tmp;
    getline(fin,tmp);
    double time;
    string filename;
    while(!fin.eof()) {
        fin >> time >>filename;
        if(!fin.good()) {
            break;
        }
        filename.erase(0,1);
        image_files.push_back(filename);
    }
    std:cout<<"stereo image size is "<<image_files.size()<<std::endl;
    startIndedx = config.getConfigPara().at("start_index").get<int>();
    currentIndex = startIndedx;
}

//通过reader获得当前帧指针
stereoframe::Ptr StereoFrameReader::next()
{
    if(currentIndex < 0 || currentIndex > image_files.size()) {
        std::cout<<"current index is invalid"<<std::endl;
        return nullptr;
    }
    stereoframe::Ptr frame(new stereoframe);
    if(DataType == StereoDataType::UnrealStereo)
    {
        frame->id = currentIndex;
        frame->left = cv::imread(dataset_dir + "dump_images/" + "left/" + image_files[currentIndex]);
        frame->right = cv::imread(dataset_dir + "dump_images/" + "right/" + image_files[currentIndex]);
        if(frame->left.data == nullptr || frame->right.data == nullptr)
            return nullptr;
    }
    else if(DataType == StereoDataType::EurocStereo)
    {
        frame->id = currentIndex;
        frame->left = cv::imread(dataset_dir + "/cam0/data/" + image_files[currentIndex],cv::IMREAD_GRAYSCALE);
        frame->right = cv::imread(dataset_dir + "/cam1/data/" + image_files[currentIndex],cv::IMREAD_GRAYSCALE);
        if(frame->left.data == nullptr || frame->right.data == nullptr)
        {
            std::cout<<"frame image data read failed"<<std::endl;
            return nullptr;
        }
    }
    currentIndex++;
    return frame;
}




