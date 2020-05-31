#ifndef getNeighborsOfK_hpp
#define getNeighborsOfK_hpp

#include <opencv2/core.hpp>

#include "Point.hpp"

int setNeighborsOfK(cv::Mat &img, skelx::Point &point, const int k);

#endif