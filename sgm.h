#pragma once
#include <vector>
#include <string>

struct SGMOption {
    int num_path;
    int min_disparity;
    int max_disparity;

    int P1;
    int P2;

    float LR_check_thres;
    float uniqueness_ratio;
    float min_sparkle_area;

    bool is_lr_check = false;
    bool is_check_unique = false;
    bool is_remove_sparkles = false;
    bool is_fill_holes = false;

    SGMOption() : num_path(8), min_disparity(0), max_disparity(128),P1(10), P2(150) {}
};

class SGMUtils {
public:
    static void census_transform(unsigned char* source, unsigned int* census,
            const int img_width, const int img_height);
    static int hamming_dist(const unsigned int census_x, const unsigned int census_y);

    static void cost_aggregate_left_right(const unsigned char* img_data, const int image_width,
            const int image_height, const int min_disparity, const int max_disparity,
            const int p1, const int p2_init, const unsigned char* cost_init, unsigned char* cost_aggr);

    static void cost_aggregate_right_left(const unsigned char* img_data, const int image_width,
            const int image_height, const int min_disparity, const int max_disparity,
            const int p1, const int p2_init, const unsigned char* cost_init, unsigned char* cost_aggr);

    static void cost_aggregate_top_bottom(const unsigned char* img_data, const int image_width,
            const int image_height, const int min_disparity, const int max_disparity,
            const int p1, const int p2_init, const unsigned char* cost_init, unsigned char* cost_aggr);

    static void cost_aggregate_botton_top(const unsigned char* img_data, const int image_width,
            const int image_height, const int min_disparity, const int max_disparity,
            const int p1, const int p2_init, const unsigned char* cost_init, unsigned char* cost_aggr);

    static void MedianFilter(float* in, float* out, const int img_width, const int img_height,
            const int wnd_size);

    static void RemoveSparkles(float* disparity, const int img_width, const int img_height, const int diff_insame,
            const int min_sparkle_area);

    static void ShowDispImage(float* disp, const int& image_width, const int& image_height,
            const std::string& win_name);
};


class SemiGlobalMatching {
public:
    SemiGlobalMatching();
    ~SemiGlobalMatching();

public:
    bool Initialize(const int img_width, const int img_height, const SGMOption& option);
    bool Match(unsigned char* left_img, unsigned char* right_img, float* left_disp);
    //when image changed, reset needs to be called to re-initialize
    bool Reset(const int img_width, const int img_height, const SGMOption& option);

private:
    void CensusTransform() const;
    void ComputeCost() const;
    void CostAggregation() const;
    void ComputeDisparity() const;
    void ComputeDisparityRight() const;
    void LRCheck();

    void Release();

    void FillHolesInDispMap();

private:
    SGMOption option_;
    int img_width_;
    int img_height_;

    unsigned char* left_img_;
    unsigned char* right_img_;

    unsigned int* census_left_;
    unsigned int* census_right_;

    unsigned char* cost_init_;
    unsigned short* cost_aggr_;

    float* disp_left_;
    float* disp_right_;

    float* middle_filter_disp_left_;

    bool is_initialize_;

private:
    /*
     *  from left to right(1) from right to left(2)
     *  from top to bottom(3) from bottom to top(4)
     *  from tl to br(5)      from br to tl(6)
     *  from tr to bl(7)      from bl to tr(8)
     */
    unsigned char* cost_aggr_1_;
    unsigned char* cost_aggr_2_;
    unsigned char* cost_aggr_3_;
    unsigned char* cost_aggr_4_;
    unsigned char* cost_aggr_5_;
    unsigned char* cost_aggr_6_;
    unsigned char* cost_aggr_7_;
    unsigned char* cost_aggr_8_;

    //remove outlier and wrong pixel by lr check
    std::vector<std::pair<int, int>> occlusions_;
    std::vector<std::pair<int, int>> mismatches_;
};