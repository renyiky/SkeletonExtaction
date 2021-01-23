#pragma once

#include <string>
#include <opencv2/core.hpp>

cv::Mat contract(cv::Mat img, std::string filename, const double detailFactor, const bool perturbationFlag);