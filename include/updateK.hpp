#ifndef updateK_hpp
#define updatek_hpp

#include <vector>
#include <opencv2/core.hpp>

#include "Point.hpp"

std::vector<skelx::Point> updateK(cv::Mat &img, std::vector<skelx::Point> &pointset, int t, double threshold = 0.95, double a = 1.0, int c = 12);

#endif  