#ifndef ZSALG_HPP
#define ZSALG_HPP

#include <opencv2/core.hpp>

// Implementation of Zhang, Suen algorithm.
// won't change the source img.
// frontground: white
// background: black
cv::Mat ZSalg(cv::Mat img);

#endif