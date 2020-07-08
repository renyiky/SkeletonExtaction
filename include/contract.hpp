#ifndef contracttt_hpp
#define contracttt_hpp

#include <string>
#include <opencv2/core.hpp>

// Extract skeleton.
// The parameter detailFactor is used in PCA, and controls the degree of details the skeleton would have
cv::Mat contract(cv::Mat img, std::string filename, const double detailFactor = 10.0);

#endif