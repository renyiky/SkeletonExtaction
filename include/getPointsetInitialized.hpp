#ifndef GETPOINTSETINITIALIZED_HPP
#define GETPOINTSETINITIALIZED_HPP

#include <vector>
#include <opencv2/core.hpp>

#include "Point.hpp"

std::vector<struct skelx::Point> getPointsetInitialized(cv::Mat &img);

#endif