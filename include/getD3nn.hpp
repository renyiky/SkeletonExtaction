#ifndef GETD3NN_HPP
#define GETD3NN_HPP

#include <opencv2/core.hpp>
#include "Point.hpp"

double getD3nn(cv::Mat &img, const struct skelx::Point &point);

#endif