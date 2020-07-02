#ifndef GETD4NN_HPP
#define GETD4NN_HPP

#include <opencv2/core.hpp>
#include "Point.hpp"

double getD4nn(cv::Mat &img, const struct skelx::Point &point);

#endif