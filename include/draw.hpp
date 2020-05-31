#ifndef DRAWWWWW_HPP
#define DRAWWWWW_HPP

#include <opencv2/core.hpp>
#include <vector>

#include "Point.hpp"

cv::Mat draw(cv::Mat src, std::vector<struct skelx::Point> pointset);

#endif