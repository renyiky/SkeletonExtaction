#ifndef VISUALIZE_HPP
#define VISUALIZE_HPP

#include <opencv2/core.hpp>
#include <vector>

#include "Point.hpp"

void visualize(const cv::Mat &img, const std::vector<skelx::Point> pointset, int iter);

#endif