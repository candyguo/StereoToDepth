// c++ std library
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
using namespace std;

//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

//OpenCV
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/cvaux.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/legacy/legacy.hpp>

//boost
#include <boost/format.hpp>
#include <boost/timer.hpp>
#include <boost/lexical_cast.hpp>

#include "json.hpp"

using json = nlohmann::json;

namespace stereo_mapping
{
    struct CAMERA_INTRINSIC_PARAMETERS
    {
        double cx = 0;
        double cy = 0;
        double fx = 0;
        double fy = 0;
        double d0 = 0;
        double d1 = 0;
        double d2 = 0;
        double d3 = 0;
        //stereo baseline
        double base_line = 0;
    };

    class Config{
    public:
        Config() {
            setParameterFile();
        }

    public:
        json getConfigPara()
        {
            return config_para;
        }

    protected:
        void setParameterFile()
        {
            std::string filename = "../config/StereoDepth.json";
            std::ifstream config_file(filename);
            config_file >> config_para;
            config_file.close();
        }

    private:
        json config_para;
    };

    class util {
    public:
        static cv::Mat CalculateFundamentalMatrix(cv::Mat left_intrinsic_mat,cv::Mat right_intrinsic_mat)
        {
            Eigen::Matrix4d T_B_C0;
            T_B_C0 << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                    0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                    -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                    0.0, 0.0, 0.0, 1.0;

            Eigen::Matrix4d T_B_C1;
            T_B_C1 << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                    0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                    -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                    0.0, 0.0, 0.0, 1.0;
            auto T_C0_C1 = T_B_C0.inverse() * T_B_C1;
            auto T_C1_C0 = T_B_C1.inverse() * T_B_C0;
            cv::Mat R_C0_C1 = (cv::Mat_<double>(3,3) <<
                                                     T_C0_C1(0,0),T_C0_C1(0,1),T_C0_C1(0,2),
                    T_C0_C1(1,0),T_C0_C1(1,1),T_C0_C1(1,2),
                    T_C0_C1(2,0),T_C0_C1(2,1),T_C0_C1(2,2));
            //求解基础矩阵
            cv::Mat transposed_right;
            cv::Mat tcross = (cv::Mat_<double>(3,3)<<
                                                   0,T_C0_C1(2,3),-T_C0_C1(1,3),
                    -T_C0_C1(2,3),0,T_C0_C1(0,3),
                    T_C0_C1(1,3),-T_C0_C1(0,3),0);

            cv::transpose(right_intrinsic_mat.inv(),transposed_right);
            cv::Mat fundmental_matrix = transposed_right * (tcross*R_C0_C1.inv()) * (left_intrinsic_mat.inv());
            return fundmental_matrix;
        }

        static cv::Mat getDepthMapFromStereo(cv::Mat left, cv::Mat right,cv::Mat left_intrinsic_mat,cv::Mat left_discoff,
                                             cv::Mat right_intrinsic_mat,cv::Mat right_discoff,cv::Mat fundmental_matrix,
                                             int id, int window_height,int windows_width,
                                             int search_radius,int gradient_threshold)
        {
            Eigen::Matrix4d T_B_C0, T_B_C1;
            T_B_C0 << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                    0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                    -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                    0.0, 0.0, 0.0, 1.0;
            T_B_C1 << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                    0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                    -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                    0.0, 0.0, 0.0, 1.0;
            auto T_C0_C1 = T_B_C0.inverse() * T_B_C1;
            auto T_C1_C0 = T_B_C1.inverse() * T_B_C0;

            cv::Mat T1 = (cv::Mat_<float>(3,4)<<
                                              1,0,0,0,
                    0,1,0,0,
                    0,0,1,0);
            cv::Mat T2 = (cv::Mat_<float>(3,4)<<
                                              T_C1_C0(0,0),T_C1_C0(0,1),T_C1_C0(0,2),T_C1_C0(0,3),
                    T_C1_C0(1,0),T_C1_C0(1,1),T_C1_C0(1,2),T_C1_C0(1,3),
                    T_C1_C0(2,0),T_C1_C0(2,1),T_C1_C0(2,2),T_C1_C0(2,3));

            cv::Mat left_depth = cv::Mat::zeros(left.rows,left.cols,CV_16UC1);
            vector<int> valid_depth = {};
            cv::Mat left_undistort,right_undistort;
            cv::undistort(left,left_undistort,left_intrinsic_mat,left_discoff);
            cv::undistort(right,right_undistort,right_intrinsic_mat,right_discoff);

            vector<cv::Point2f> pts_1;
            vector<cv::Point2f> pts_2;
            int idx = 0;
            for(int v=window_height/2 + 1;v<left_undistort.rows - (window_height/2 + 1);v++)
            {
                std::cout<<"current frame id: "<<id<< "current row: "<<v<<std::endl;
                for(int u=windows_width/2 + 1;u<left_undistort.cols-(windows_width/2 + 1);u++)
                {
                    //图像梯度计算
                    idx = v*left_undistort.cols + u;
                    if(v == 0 || v == (left_undistort.rows-1) || u == 0 || u == (left_undistort.cols-1))
                        continue;
                    float dx = 0.5f*(left_undistort.data[idx+1] - left_undistort.data[idx-1]) ;
                    float dy = 0.5f*(left_undistort.data[idx+left_undistort.cols] - left_undistort.data[idx-left_undistort.cols]);
                    if(!std::isfinite(dx))
                        dx=0;
                    if(!std::isfinite(dy))
                        dy=0;
                    if((dx*dx+dy*dy) < gradient_threshold)
                        continue;

                    valid_depth.push_back(idx);
                    Eigen::Vector2d target_point = GetMatchedPointOnPolorline(
                            left_undistort,right_undistort,fundmental_matrix,Eigen::Vector2d(u,v),windows_width
                            ,window_height,search_radius);
                    pts_1.push_back(pixel2cam(cv::Point2f(u,v),left_intrinsic_mat));
                    pts_2.push_back(pixel2cam(cv::Point2f(target_point(0,0),target_point(1,0)),
                                              right_intrinsic_mat));
                }
            }
            cv::Mat Pts_4d;
            cv::triangulatePoints(T1, T2 ,pts_1,pts_2,Pts_4d);
            ushort* depthdata = (ushort*)left_depth.data;
            for(int i = 0;i<Pts_4d.cols;i++) {
                cv::Mat x = Pts_4d.col(i);
                Eigen::Vector3d p(x.at<float>(0,0) / x.at<float>(3,0),
                                  x.at<float>(1,0) / x.at<float>(3,0),
                                  std::abs(x.at<float>(2,0) / x.at<float>(3,0)));
                depthdata[valid_depth[i]] = ushort(std::abs(float(p(2) * 1000.0)));
            }
            return left_depth;

        }

        // 双线性灰度插值
        static double getBilinearInterpolatedValue( const cv::Mat& img, const Eigen::Vector2d& pt ) {
            uchar* d = & img.data[ int(pt(1,0))*img.step+int(pt(0,0)) ];
            double xx = pt(0,0) - floor(pt(0,0));
            double yy = pt(1,0) - floor(pt(1,0));
            return  (( 1-xx ) * ( 1-yy ) * double(d[0]) +
                     xx* ( 1-yy ) * double(d[1]) +
                     ( 1-xx ) *yy* double(d[img.step]) +
                     xx*yy*double(d[img.step+1]))/255.0;
        }


        /*
         * 计算两幅图像对应两个点的NCC相关值
         * ncc_window_width : 窗口半宽度\
         * ncc_window_height: 窗口半高度
         */
        static double NCC (const cv::Mat& ref, const cv::Mat& curr, const Eigen::Vector2d& pt_ref,
                           const Eigen::Vector2d& pt_curr,const int ncc_window_width,
                           const int ncc_window_height)
        {
            // 零均值-归一化互相关
            // 先算均值
            double mean_ref = 0, mean_curr = 0;
            int ncc_area = (2*ncc_window_width+1) * (2*ncc_window_height+1);
            vector<double> values_ref, values_curr; // 参考帧和当前帧的均值
            for ( int x=-ncc_window_width; x<=ncc_window_width; x++ )
                for ( int y=-ncc_window_height; y<=ncc_window_height; y++ )
                {
                    //std::cout<<"start" << y+pt_ref(1,0) << " "<< x+pt_ref(0,0)<<std::endl;
                    double value_ref = double(ref.ptr<uchar>( int(y+pt_ref(1,0)) )[ int(x+pt_ref(0,0)) ])/255.0;
                    mean_ref += value_ref;

                    double value_curr = getBilinearInterpolatedValue( curr, pt_curr+Eigen::Vector2d(x,y) );
                    mean_curr += value_curr;

                    values_ref.push_back(value_ref);
                    values_curr.push_back(value_curr);
                }
            mean_ref /= ncc_area;
            mean_curr /= ncc_area;

            // 计算 Zero mean NCC
            double numerator = 0, demoniator1 = 0, demoniator2 = 0;
            for ( int i=0; i<values_ref.size(); i++ )
            {
                double n = (values_ref[i]-mean_ref) * (values_curr[i]-mean_curr);
                numerator += n;
                demoniator1 += (values_ref[i]-mean_ref)*(values_ref[i]-mean_ref);
                demoniator2 += (values_curr[i]-mean_curr)*(values_curr[i]-mean_curr);
            }
            return numerator / sqrt( demoniator1*demoniator2+1e-10 );   // 防止分母出现零
        }

        /*
         * 基于NCC规则,根据左右相的变换矩阵在极线上搜索同名点
         * window_size_width :窗口宽度
         * window_size_height:窗口高度
         * search_radius: 目标图像上以source_point为原点的搜索半径
         */
        static Eigen::Vector2d GetMatchedPointOnPolorline(const cv::Mat source_image,const cv::Mat target_image,
                                                          const cv::Mat fundamental, const Eigen::Vector2d source_point,
                                                          const int window_size_width, const int window_size_height,
                                                          const double search_radius)
        {
            //在极线上进行ncc匹配
            double max_score = -1.0;
            double tmp_score = 0;
            int max_x = 0;
            int max_y = 0;
            int start = std::max((window_size_width-1)/2,static_cast<int>(source_point(0,0) - search_radius*3));
            int end = std::min(target_image.cols - (window_size_width-1)/2,static_cast<int>(source_point(0,0) + search_radius));

            for(int x = start;x <= end; x++)
            {
                //通过极线计算目标点的Y坐标
                vector<cv::Vec3f> lines;
                vector<cv::Point2f> ps{cv::Point2f(source_point(0,0),source_point(1,0))};
                cv::computeCorrespondEpilines(ps, 1, fundamental, lines);
                auto target_line = lines[0];
                auto target_y = -(target_line[2] + target_line[0] * target_image.cols) / target_line[1];

                tmp_score = NCC(source_image, target_image, source_point, Eigen::Vector2d(x,target_y),
                                (window_size_width-1)/2,(window_size_height-1)/2);
                if(tmp_score > max_score)
                {
                    max_x = x;
                    max_y = target_y;
                    max_score = tmp_score;
                }
            }

            return Eigen::Vector2d(max_x,max_y);
        }

        static cv::Point2f pixel2cam(const cv::Point2d& p, const cv::Mat& K) {
            return cv::Point2f((p.x - K.at<double>(0,2)) / K.at<double>(0,0),
                               (p.y - K.at<double>(1,2)) / K.at<double>(1,1));
        }
    };

}