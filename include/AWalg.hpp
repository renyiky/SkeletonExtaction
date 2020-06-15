#ifndef AWALG_HPP
#define AWALG_HPP

#include <opencv2/core.hpp>

// Implementation of AW algrithm.
// Here are three conditions concerned:
// (a): Does the pixel belong to one of the following two cases:...
// (b): Does the pixel belong to an extremity of a zigzag diagonal line
// (c): if condition a and b are both NOT satisfied, we apply the 20 rules
cv::Mat AWalg(cv::Mat img);

#endif