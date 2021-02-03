#include "sgm.h"
#include "memory"
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*
 *     [1, 4, 6]
 *     [2, 5, 8]
 *     [1, 9, 3]
 *     census value of pixel 5 is (110100101)
 */
void SGMUtils::census_transform(unsigned char* source, unsigned int* census,
        const int img_width, const int img_height) {
    // 5 * 5 census transform
    if(source == nullptr || census == nullptr) {
        return;
    }

    for(int j = 2; j < img_height - 2; j++) {
         for(int i = 2; i < img_width - 2; i++) {
            unsigned char center_pixel = source[j * img_width + i];
            int census_val = 0;
            //census value is col ordered from left to right
            for(int k = -2; k <= 2; k++) {
                for(int l = -2; l <= 2; l++) {
                    census_val <<= 1;
                    unsigned char gray = source[(j + k) * img_width + i + l];
                    if(gray < center_pixel) {
                        census_val += 1;
                    }
                }
            }
            census[j * img_width + i] = census_val;
        }
    }
}

int SGMUtils::hamming_dist(const unsigned int census_x, const unsigned int census_y) {
    int dist = 0;
    int val = census_x ^ census_y;
    while(val) {
        dist++;
        val &= (val - 1);
    }
    return dist;
}

void SGMUtils::cost_aggregate_left_right(const unsigned char* img_data, const int image_width,
        const int image_height, const int min_disparity, const int max_disparity,
        const int p1, const int p2_init, const unsigned char* cost_init, unsigned char* cost_aggr) {
    const int disp_range = max_disparity - min_disparity;

    for(int j = 0; j < image_height; j++) {
        auto cost_init_row = cost_init + j * image_width * disp_range;
        auto cost_aggr_row = cost_aggr + j * image_width * disp_range;
        auto image_row = img_data + j * image_width;

        unsigned char gray = *image_row;
        unsigned char gray_last = gray;

        std::vector<unsigned char> cost_last_path(disp_range + 2, UINT8_MAX);

        // initalize first element of every row
        memcpy(cost_aggr_row, cost_init_row, sizeof(unsigned char) * disp_range);
        memcpy(&cost_last_path[1], cost_aggr_row, sizeof(unsigned char) * disp_range);

        // move pointer
        cost_init_row += disp_range;
        cost_aggr_row += disp_range;
        image_row += 1;

        short mincost_last_path = UINT8_MAX;
        for(auto cost : cost_last_path) {
            if(cost < mincost_last_path) {
                mincost_last_path = cost;
            }
        }

        for(int i = 1; i < image_width; i++) {
            short min_cost = UINT8_MAX; // for recording mincost of last path
            gray = *image_row;
            for(int d = 0; d < disp_range; d++) {
                unsigned char cost = cost_init_row[d];
                unsigned short l1 = cost_last_path[d+1];
                unsigned short l2 = cost_last_path[d] + p1;
                unsigned short l3 = cost_last_path[d+2] + p1;
                unsigned short l4 = mincost_last_path + std::max(p1, p2_init / (abs(gray - gray_last)+1));

                const short cost_s = cost + static_cast<unsigned char>(std::min(std::min(l1, l2), std::min(l3,l4)) - mincost_last_path);
                cost_aggr_row[d] = cost_s;
                min_cost = std::min(min_cost, cost_s);
            }
            //reset cost_last_path and min_cost
            mincost_last_path = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_row, sizeof(unsigned char) * disp_range);

            image_row += 1;
            cost_init_row += disp_range;
            cost_aggr_row += disp_range;
            gray_last = gray;
        }
    }
}

void SGMUtils::cost_aggregate_right_left(const unsigned char* img_data, const int image_width,
        const int image_height, const int min_disparity, const int max_disparity,
        const int p1, const int p2_init, const unsigned char* cost_init, unsigned char* cost_aggr) {
    int disp_range = max_disparity - min_disparity;
    for(int j = 0; j < image_height; j++) {
        auto cost_init_row = cost_init + j * image_width * disp_range + (image_width - 1) * disp_range;
        auto cost_aggr_row = cost_aggr + j * image_width * disp_range + (image_width - 1) * disp_range;
        auto image_row = img_data + j * image_width + image_width - 1;

        unsigned char gray = *image_row;
        unsigned char gray_last = gray;

        std::vector<unsigned char> cost_last_path(disp_range + 2, UINT8_MAX);

        // first pixel of every row
        memcpy(cost_aggr_row, cost_init_row, sizeof(unsigned char) * disp_range);
        memcpy(&cost_last_path[1], cost_aggr_row, sizeof(unsigned char) * disp_range);

        cost_init_row -= disp_range;
        cost_aggr_row -= disp_range;
        image_row -= 1;

        short mincost_last_path = UINT8_MAX;
        for(auto cost : cost_last_path) {
            if(cost < mincost_last_path) {
                mincost_last_path = cost;
            }
        }
        for(int i = 1; i < image_width; i++) {
            gray = *image_row;
            unsigned char min_cost = UINT8_MAX;
            for(int d = 0; d < disp_range; d++) {
                unsigned char cost = cost_init_row[d];
                unsigned short l1 = cost_last_path[d+1];
                unsigned short l2 = cost_last_path[d] + p1;
                unsigned short l3 = cost_last_path[d+2] + p1;
                unsigned short l4 = mincost_last_path + std::max(p1, p2_init / (abs(gray_last - gray) + 1));
                unsigned char cost_s = cost + static_cast<unsigned char>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_path);
                // record min_cost after aggregate
                min_cost = std::min(min_cost, cost_s);
                cost_aggr_row[d] = cost_s;
            }
            mincost_last_path = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_row, sizeof(unsigned char) * disp_range);

            image_row -= 1;
            cost_init_row -= disp_range;
            cost_aggr_row -= disp_range;
            gray_last = gray;
        }
    }
}

void SGMUtils::cost_aggregate_top_bottom(const unsigned char* img_data, const int image_width,
                                         const int image_height, const int min_disparity, const int max_disparity,
                                         const int p1, const int p2_init, const unsigned char* cost_init, unsigned char* cost_aggr) {
    const int disp_range = max_disparity - min_disparity;
    for(int i = 0; i < image_width; i++) {
        auto cost_init_col = cost_init + i * disp_range;
        auto cost_aggr_col = cost_aggr + i * disp_range;
        auto image_col = img_data + i;

        unsigned char gray = *image_col;
        unsigned char gray_last = *image_col;

        std::vector<unsigned char> cost_last_path(disp_range + 2, UINT8_MAX);

        //construct first pixel of current col 's cost aggr and future cost+last_path
        memcpy(cost_aggr_col, cost_init_col, sizeof(unsigned char) * disp_range);
        memcpy(&cost_last_path[1], cost_aggr_col, sizeof(unsigned char) * disp_range);

        cost_init_col += image_width * disp_range;
        cost_aggr_col += image_width * disp_range;
        image_col += image_width;

        unsigned char mincost_last_path = UINT8_MAX;
        for(auto cost : cost_last_path) {
            mincost_last_path = std::min(mincost_last_path, cost);
        }

        for(int j = 1; j < image_height;j++) {
            gray = *image_col;
            unsigned char min_cost = UINT8_MAX;
            for(int d = 0; d < disp_range; d++) {
                unsigned char cost = cost_init_col[d];
                unsigned short l1 = cost_last_path[d+1];
                unsigned short l2 = cost_last_path[d] + p1;
                unsigned short l3 = cost_last_path[d+2] + p1;
                unsigned short l4 = mincost_last_path + std::max(p1, p2_init / (abs(gray_last - gray) + 1));
                unsigned char cost_s = cost + static_cast<unsigned char>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_path);
                // record min_cost after aggregate
                min_cost = std::min(min_cost, cost_s);
                cost_aggr_col[d] = cost_s;
            }
            mincost_last_path = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_col, sizeof(unsigned char) * disp_range);

            image_col += image_width;
            cost_init_col += disp_range * image_width;
            cost_aggr_col += disp_range * image_width;
            gray_last = gray;
        }
    }
}

// aggregate path
void SGMUtils::cost_aggregate_botton_top(const unsigned char* img_data, const int image_width,
                                         const int image_height, const int min_disparity, const int max_disparity,
                                         const int p1, const int p2_init, const unsigned char* cost_init, unsigned char* cost_aggr) {
    const int disp_range = max_disparity - min_disparity;
    for(int i = 0; i < image_width; i++) {
        auto cost_init_col = cost_init + i * disp_range + (image_height - 1) * image_width * disp_range;
        auto cost_aggr_col = cost_aggr + i * disp_range + (image_height - 1) * image_width * disp_range;
        auto image_col = img_data + i + (image_height - 1) * image_width;

        unsigned char gray = *image_col;
        unsigned char gray_last = *image_col;

        std::vector<unsigned char> cost_last_path(disp_range + 2, UINT8_MAX);

        //construct first pixel of current col 's cost aggr and future cost+last_path
        memcpy(cost_aggr_col, cost_init_col, sizeof(unsigned char) * disp_range);
        memcpy(&cost_last_path[1], cost_aggr_col, sizeof(unsigned char) * disp_range);

        cost_init_col -= image_width * disp_range;
        cost_aggr_col -= image_width * disp_range;
        image_col -= image_width;

        unsigned char mincost_last_path = UINT8_MAX;
        for(auto cost : cost_last_path) {
            mincost_last_path = std::min(mincost_last_path, cost);
        }

        for(int j = 1; j < image_height;j++) {
            gray = *image_col;
            unsigned char min_cost = UINT8_MAX;
            for(int d = 0; d < disp_range; d++) {
                unsigned char cost = cost_init_col[d];
                unsigned short l1 = cost_last_path[d+1];
                unsigned short l2 = cost_last_path[d] + p1;
                unsigned short l3 = cost_last_path[d+2] + p1;
                unsigned short l4 = mincost_last_path + std::max(p1, p2_init / (abs(gray_last - gray) + 1));
                unsigned char cost_s = cost + static_cast<unsigned char>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_path);
                // record min_cost after aggregate
                min_cost = std::min(min_cost, cost_s);
                cost_aggr_col[d] = cost_s;
            }
            //all disparity cost and min disparity cost of last path
            mincost_last_path = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_col, sizeof(unsigned char) * disp_range);

            image_col -= image_width;
            cost_init_col -= disp_range * image_width;
            cost_aggr_col -= disp_range * image_width;
            gray_last = gray;
        }
    }
}

void SGMUtils::MedianFilter(float* in, float* out, const int img_width, const int img_height,
                  const int wnd_size) {
    for(int j = 0; j < img_height; j++) {
        for(int i = 0; i < img_width; i++) {
            int radius = wnd_size / 2;
            int size = wnd_size * wnd_size;
            std::vector<float> window;
            window.reserve(size);
            for(int r = -radius; r <= radius; r++) {
                for(int l = -radius; l <= radius; l++) {
                    int row = j + r;
                    int col = i + l;
                    if(row < 0 || row >= img_height || col < 0 || col >= img_width)
                        continue;
                    window.emplace_back(in[row * img_width + col]);
                }
            }
            std::sort(window.begin(), window.end());
            out[j * img_width + i] = window[window.size() / 2];
        }
    }
}

void SGMUtils::RemoveSparkles(float* disparity, const int img_width, const int img_height, const int diff_insame,
                    const int min_sparkle_area) {
    std::vector<bool> visited(img_width * img_height, false);
    for(int j = 0; j < img_height; j++) {
        for(int i = 0; i < img_width; i++) {
            if(visited[j * img_width + i] || disparity[j * img_width + i] <= 0)
                continue;
            visited[j * img_width + i] = true;
            std::vector<std::pair<int,int>> pixel_vector;
            std::queue<std::pair<int, int>> pixel_queue;
            pixel_vector.push_back(std::make_pair(j,i));
            pixel_queue.push(std::make_pair(j, i));
            while(!pixel_queue.empty()) {
                auto cur_pixel = pixel_queue.front();
                pixel_queue.pop();
                int row = cur_pixel.first;
                int col = cur_pixel.second;
                float disp_base = disparity[row * img_width + col];
                // 8 neighbour add
                for(int r = -1; r <= 1; r++) {
                    for(int c = -1; c <= 1; c++) {
                        if(r == 0 && c == 0)
                            continue;
                        int neigh_row = row + r;
                        int neigh_col = col + c;
                        if(neigh_row < 0 || neigh_row >= img_height || neigh_col < 0 || neigh_col >= img_width)
                            continue;
                        float neigh_disparity = disparity[neigh_row * img_width + neigh_col];
                        if(neigh_disparity < 0) {
                            visited[neigh_row * img_width + neigh_col] = true;
                            continue;
                        }
                        if(!visited[neigh_row * img_width + neigh_col] && fabs(neigh_disparity - disp_base) < diff_insame) {
                            pixel_vector.push_back(std::make_pair(neigh_row, neigh_col));
                            pixel_queue.push(std::make_pair(neigh_row, neigh_col));
                            visited[neigh_row * img_width + neigh_col] = true;
                        }
                    }
                }
            }

            if(pixel_vector.size() < min_sparkle_area) {
                for(auto& pix : pixel_vector) {
                    disparity[pix.first * img_width + pix.second] = -1.0;
                }
            }
        }
    }
}

void SGMUtils::ShowDispImage(float* disp_image, const int& image_width, const int& image_height,
                             const std::string& win_name) {
    cv::Mat disp_mat = cv::Mat(image_height,image_width, CV_8UC1);
    for(int j = 0; j < image_height; j++) {
        for(int i = 0; i < image_width; i++) {
            float disp = disp_image[j * image_width + i];
            if(disp < 0.0)
                disp_mat.data[j * image_width + i] = 0.0;
            else
                disp_mat.data[j * image_width + i] = static_cast<char>(disp);
        }
    }
    cv::normalize(disp_mat, disp_mat, 0, 255,cv::NORM_MINMAX);

    cv::Mat tmp, color;
    //disparity.convertTo(tmp, CV_8UC1,1.0 / 128.0 * 255.0);
    cv::applyColorMap(disp_mat, color, cv::COLORMAP_JET);
    cv::imshow("color disparity",color);

    //cv::imshow(win_name, disp_mat);
}

SemiGlobalMatching::SemiGlobalMatching() {

}

SemiGlobalMatching::~SemiGlobalMatching() {
    Release();
}

bool SemiGlobalMatching::Initialize(const int img_width, const int img_height, const SGMOption &option) {
    img_width_ = img_width;
    img_height_ = img_height;
    option_ = option;

    census_left_ = new unsigned int[img_width_ * img_height_];
    census_right_ = new unsigned int[img_width_ * img_height_];

    const int disp_range = option.max_disparity - option.min_disparity;
    if(disp_range <= 0)
        return false;

    // build cost volume
    cost_init_ = new unsigned char[img_width_ * img_height_ * disp_range];
    cost_aggr_ = new unsigned short[img_width_ * img_height_ * disp_range];

    cost_aggr_1_ = new unsigned char[img_width_ * img_height_ * disp_range];
    cost_aggr_2_ = new unsigned char[img_width_ * img_height_ * disp_range];
    cost_aggr_3_ = new unsigned char[img_width_ * img_height_ * disp_range];
    cost_aggr_4_ = new unsigned char[img_width_ * img_height_ * disp_range];
    cost_aggr_5_ = new unsigned char[img_width_ * img_height_ * disp_range];
    cost_aggr_6_ = new unsigned char[img_width_ * img_height_ * disp_range];
    cost_aggr_7_ = new unsigned char[img_width_ * img_height_ * disp_range];
    cost_aggr_8_ = new unsigned char[img_width_ * img_height_ * disp_range];

    disp_left_ = new float[img_width_ * img_height_];
    disp_right_ = new float[img_width_ * img_height_];

    middle_filter_disp_left_ = new float[img_width * img_height];

    is_initialize_ = census_left_ && census_right_ &&
            cost_init_ && cost_aggr_ && disp_left_;
}

void SemiGlobalMatching::Release() {
    if(census_left_) {
        delete []census_left_;
        census_left_ = nullptr;
    }
    if(census_right_) {
        delete []census_right_;
        census_right_ = nullptr;
    }
    if(cost_init_) {
        delete []cost_init_;
        cost_init_ = nullptr;
    }
    if(cost_aggr_) {
        delete []cost_aggr_;
        cost_aggr_ = nullptr;
    }
    if(cost_aggr_1_) {
        delete []cost_aggr_1_;
        cost_aggr_1_ = nullptr;
    }
    if(cost_aggr_2_) {
        delete []cost_aggr_2_;
        cost_aggr_2_ = nullptr;
    }
    if(cost_aggr_3_) {
        delete []cost_aggr_3_;
        cost_aggr_3_ = nullptr;
    }
    if(cost_aggr_4_) {
        delete []cost_aggr_4_;
        cost_aggr_4_ = nullptr;
    }
    if(cost_aggr_5_) {
        delete []cost_aggr_5_;
        cost_aggr_5_ = nullptr;
    }
    if(cost_aggr_6_) {
        delete []cost_aggr_6_;
        cost_aggr_6_ = nullptr;
    }
    if(cost_aggr_7_) {
        delete []cost_aggr_7_;
        cost_aggr_7_ = nullptr;
    }
    if(cost_aggr_8_) {
        delete []cost_aggr_8_;
        cost_aggr_8_ = nullptr;
    }
    if(disp_left_) {
        delete []disp_left_;
        disp_left_ = nullptr;
    }
    if(disp_right_) {
        delete []disp_right_;
        disp_right_ = nullptr;
    }
    if(middle_filter_disp_left_) {
        delete []middle_filter_disp_left_;
        middle_filter_disp_left_ = nullptr;
    }
    is_initialize_ = false;
}

bool SemiGlobalMatching::Reset(const int img_width, const int img_height, const SGMOption &option) {
    Release();
    Initialize(img_width,img_height,option);
}

bool SemiGlobalMatching::Match(unsigned char *left_img, unsigned char *right_img, float *left_disp) {
    if(!is_initialize_)
        return false;
    if(left_img == nullptr || right_img == nullptr)
        return false;

    left_img_ = left_img;
    right_img_ = right_img;

    CensusTransform();

    ComputeCost();

    CostAggregation();

    ComputeDisparity();

    if(option_.is_lr_check) {
        ComputeDisparityRight();
        //LRCheck();
    }

    if(option_.is_remove_sparkles) {
        //SGMUtils::RemoveSparkles(disp_left_,img_width_, img_height_, 2.0f, option_.min_sparkle_area);
    }

    if(option_.is_fill_holes) {
        //FillHolesInDispMap();
    }

    //SGMUtils::MedianFilter(disp_left_, middle_filter_disp_left_, img_width_, img_height_, 3);

    //memcpy(left_disp, middle_filter_disp_left_, sizeof(float) * img_width_ * img_height_);
    memcpy(left_disp, disp_left_, sizeof(float) * img_width_ * img_height_);
    return true;
}

void SemiGlobalMatching::CensusTransform() const {
    // transform original image to census image
    SGMUtils::census_transform(left_img_, census_left_, img_width_, img_height_);
    SGMUtils::census_transform(right_img_, census_right_, img_width_, img_height_);
}

// cost volume calculation based on disparity major order
// build a cost volume which size is width * height * disp_range
void SemiGlobalMatching::ComputeCost() const {
    const int min_disparity = option_.min_disparity;
    const int max_disparity = option_.max_disparity;

    const int disp_range = max_disparity - min_disparity;

    for(int j = 0; j < img_height_; j++) {
        for(int i = 0; i < img_width_; i++) {
            int left_census_value = census_left_[j * img_width_ + i];
            for(int d = min_disparity; d < max_disparity; d++) {
                auto& cost = cost_init_[j * img_width_ * disp_range + i * disp_range + (d - min_disparity)];
                if((i - d) < 0) {
                    cost = UINT8_MAX;
                    continue;
                }
                int right_census_value = census_right_[j * img_width_ + i - d];
                cost = SGMUtils::hamming_dist(left_census_value, right_census_value);
            }
        }
    }
}

// winner take all (WTA), find min cost disparity from disparity cost volume
void SemiGlobalMatching::ComputeDisparity() const {
    const int min_disparity = option_.min_disparity;
    const int max_disparity = option_.max_disparity;
    const int disp_range = max_disparity - min_disparity;

    auto cost_ptr = cost_aggr_;
    auto disparity = disp_left_;

    for(int j = 0; j < img_height_; j++) {
        for(int i = 0; i < img_width_; i++) {
            int min_cost = UINT16_MAX;
            int best_disparity = 0;
            int sec_min_cost = UINT16_MAX;
            for(int d = min_disparity; d < max_disparity; d++) {
                // it's ok to build a local cost array to store cost of every disparity
                int cost = cost_ptr[j * img_width_ * disp_range + i * disp_range + (d - min_disparity)];
                if(cost < min_cost) {
                    min_cost = cost;
                    best_disparity = d;
                }
            }

            // uniqueness check
            if(option_.is_check_unique) {
                for(int d = min_disparity; d < max_disparity; d++) {
                    if(d == best_disparity)
                        continue;
                    int cost = cost_ptr[j * img_width_ * disp_range + i * disp_range + (d - min_disparity)];
                    sec_min_cost = std::min(sec_min_cost, cost);
                }
                std::cout<< std::endl;
                if((sec_min_cost - min_cost) <= static_cast<int>(min_cost * (1 - option_.uniqueness_ratio))) {
                    disparity[j * img_width_ + i] = -1.0;
                    continue;
                }
            }

            // subpixel para fit
            if(best_disparity == min_disparity || best_disparity == (max_disparity - 1)) {
                disparity[j * img_width_ + i] = -1.0;
                continue;
            }
            // 最优视差前一个视差的代价值cost_1，后一个视差的代价值cost_2
            unsigned short cost1 = cost_ptr[j * img_width_ * disp_range + i * disp_range + (best_disparity - 1 - min_disparity)];
            unsigned short cost2 = cost_ptr[j * img_width_ * disp_range + i * disp_range + (best_disparity + 1 - min_disparity)];
            // 解一元二次曲线极值
            const unsigned short denom = std::max(1, cost1 + cost2 - 2 * min_cost);
            disparity[j * img_width_ + i] = static_cast<float >(best_disparity) +
                    static_cast<float>(cost1 - cost2) / (denom * 2.0f);
        }
    }
}

void SemiGlobalMatching::CostAggregation() const {
  const int max_disparity = option_.max_disparity;
  const int min_disparity = option_.min_disparity;
  SGMUtils::cost_aggregate_left_right(left_img_, img_width_,img_height_,min_disparity,max_disparity,
          option_.P1, option_.P2, cost_init_, cost_aggr_1_);
  SGMUtils::cost_aggregate_right_left(left_img_,img_width_,img_height_, min_disparity,max_disparity,
          option_.P1, option_.P2, cost_init_, cost_aggr_2_);
  SGMUtils::cost_aggregate_top_bottom(left_img_, img_width_, img_height_, min_disparity,max_disparity,
          option_.P1, option_.P2, cost_init_, cost_aggr_3_);
  SGMUtils::cost_aggregate_botton_top(left_img_, img_width_, img_height_, min_disparity, max_disparity,
          option_.P1, option_.P2, cost_init_, cost_aggr_4_);

  int size = img_height_ * img_width_ * (max_disparity - min_disparity);
  for(int i = 0; i < size; i++) {
      // 4 path aggregate cost
      cost_aggr_[i] = cost_aggr_1_[i] + cost_aggr_2_[i] + cost_aggr_3_[i] + cost_aggr_4_[i];
  }
}

// generate right image disparity based on left image cost_aggr array
void SemiGlobalMatching::ComputeDisparityRight() const {
    const int max_disparity = option_.max_disparity;
    const int min_disparity = option_.min_disparity;
    const int disp_range = max_disparity - min_disparity;

    auto cost_ptr = cost_aggr_;
    auto disparity = disp_right_;

    for (int j = 0; j < img_height_; j++) {
        for (int i = 0; i < img_width_; i++) { // for every pixel in right image
            unsigned short min_cost = UINT16_MAX;
            unsigned short sec_min_cost = UINT16_MAX;
            int best_disparity = min_disparity;
            for (int d = min_disparity; d < max_disparity; d++) {
                int col_left = i + d;
                if (col_left < 0 || col_left >= img_width_)
                    continue;
                unsigned short cost = cost_ptr[j * img_width_ * disp_range + col_left * disp_range + (d - min_disparity)];
                if (cost < min_cost) {
                    min_cost = cost;
                    best_disparity = d;
                }
            }

            // uniqueness check
            if(option_.is_check_unique) {
                for(int d = min_disparity; d < max_disparity; d++) {
                    if(d == best_disparity)
                        continue;
                    unsigned short cost = cost_ptr[j * img_width_ * disp_range + (i + d) * disp_range + (d - min_disparity)];
                    sec_min_cost = std::min(sec_min_cost, cost);
                }
                if((sec_min_cost - min_cost) <= static_cast<int>(min_cost * (1 - option_.uniqueness_ratio))) {
                    disparity[j * img_width_ + i] = -1.0;
                    continue;
                }
            }

            if (best_disparity == min_disparity || best_disparity == (max_disparity - 1)) {
                disparity[j * img_width_ + i] = -1.0;
                continue;
            }
            // subpixel para fit
            // 最优视差前一个视差的代价值cost_1，后一个视差的代价值cost_2
            unsigned short cost1 = cost_ptr[j * img_width_ * disp_range + i * disp_range +
                                            (best_disparity - 1 - min_disparity)];
            unsigned short cost2 = cost_ptr[j * img_width_ * disp_range + i * disp_range +
                                            (best_disparity + 1 - min_disparity)];
            // 解一元二次曲线极值
            const unsigned short denom = std::max(1, cost1 + cost2 - 2 * min_cost);
            disparity[j * img_width_ + i] = static_cast<float>(best_disparity) +
                                            static_cast<float>(cost1 - cost2) / (denom * 2.0f);
        }
    }
}

void SemiGlobalMatching::LRCheck() {
    occlusions_.clear();
    mismatches_.clear();
    for(int j = 0; j < img_height_; j++) {
        for(int i = 0; i < img_width_; i++) {
            float& disp = disp_left_[j * img_width_ + i];
            if(disp < 0.0) {
                mismatches_.emplace_back(std::make_pair(j, i));
                continue;
            }
            int col_right = static_cast<int>(j - disp + 0.5);
            if(col_right >= 0 && col_right < img_width_) {
                float disp_r = disp_right_[j * img_width_ + col_right];
                if(fabs(disp - disp_r) > option_.LR_check_thres) {
                    // match point disparity exceed threshold
                    int col_l1 = static_cast<int>(col_right + disp_r + 0.5);
                    if(col_l1 > 0 && col_l1 < img_width_) {
                        float disp_l = disp_left_[j * img_width_ + col_l1];
                        if(disp_l > disp) {
                            occlusions_.emplace_back(std::make_pair(j, i));
                        } else {
                            mismatches_.emplace_back(std::make_pair(j ,i));
                        }
                    } else {
                        mismatches_.emplace_back(std::make_pair(j, i));
                    }
                    disp = -1.0;
                }
            } else {
                mismatches_.emplace_back(std::make_pair(j, i));
                disp = -1.0;
            }
        }
    }
}

void SemiGlobalMatching::FillHolesInDispMap() {
    std::vector<float> disp_collect;
    // eight direction line search
    double angles[8] = {M_PI, 3 * M_PI / 4, 2 * M_PI / 4, M_PI / 4, 0, 7 * M_PI / 4, 6 * M_PI / 4, 5 * M_PI / 4};
    float* disp_ptr = disp_left_;
    // first loop, process occlusion; second loop, process mismatch; third loop, process rest invalid pixel
    for(int i = 0; i < 3; i++) {
        auto& trig_pixels = i == 0? occlusions_ : mismatches_;
        std::vector<std::pair<int, int>> inv_pixels;
        if(i == 2) {
            for(int j = 0; j < img_height_; j++) {
                for(int i = 0; i < img_width_; i++) {
                    auto pixel = disp_ptr[j * img_width_ + i];
                    if(pixel < 0) {
                        inv_pixels.emplace_back(std::make_pair(j,i));
                    }
                }
            }
            trig_pixels = inv_pixels;
        }

        for(auto& pixel : trig_pixels) {
            int y = pixel.first;
            int x = pixel.second;
            std::vector<float> disp_collect;
            double unit_dist = 1;
            for(int n = 1; n <= 8; n++) {
                double angle = angles[n - 1];
                if((n % 2) == 0)
                    unit_dist = sqrt(2);
                for(int unit = 1; ; unit++) {
                    int cur_y = round(y + cos(angle) * unit * unit_dist);
                    int cur_x = round(x + sin(angle) * unit * unit_dist);
                    // border check
                    if(cur_x < 0 || cur_x >= img_width_ || cur_y < 0 || cur_y >= img_height_)
                        break;
                    float disp = disp_ptr[cur_y * img_width_ + cur_x];
                    if(disp >= 0) {
                        disp_collect.emplace_back(disp);
                        break; // current direction found
                    }
                }
            }
            if(disp_collect.empty())
                continue;

            std::sort(disp_collect.begin(), disp_collect.end());
            if(i == 0) {
                if(disp_collect.size() > 1) {
                    disp_ptr[y * img_width_ + i] = disp_collect[1]; // second minimum disp as background disparity
                } else {
                    disp_ptr[y * img_width_ + i] = disp_collect[0];
                }
            } else {
                // select disparity middle value
                disp_ptr[y * img_width_ + i] = disp_collect[disp_collect.size() / 2];
            }
        }
    }
}

