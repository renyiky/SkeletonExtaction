#ifndef updateK_hpp
#define updatek_hpp

#include <vector>
#include <opencv2/core.hpp>

#include "Point.hpp"

// update k in each iteration
// when k is larger than 10, we set it as the @param upperLimit
void updateK(cv::Mat &img, std::vector<skelx::Point> &pointset, int upperLimit);

#endif  