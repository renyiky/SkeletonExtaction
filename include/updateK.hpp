#ifndef updateK_hpp
#define updatek_hpp

#include <vector>
#include <opencv2/core.hpp>

#include "Point.hpp"

void updateK(cv::Mat &img, std::vector<skelx::Point> &pointset, int t);

#endif  