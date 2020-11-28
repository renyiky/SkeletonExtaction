#pragma once

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "Point.hpp"

namespace skelx{
    void visualize(const cv::Mat &img, const std::vector<skelx::Point> pointset, int iter);
    void updateK(cv::Mat &img, std::vector<skelx::Point> &pointset, int upperLimit);
    double getD3nn(cv::Mat &img, const struct skelx::Point &point);
    std::vector<struct skelx::Point> getPointsetInitialized(cv::Mat &img);
    int setNeighborsOfK(cv::Mat &img, skelx::Point &point, const int k);
    cv::Mat draw(const cv::Mat &src, std::vector<struct skelx::Point> &pointset);
    void refreshPointset(cv::Mat &img, std::vector<skelx::Point> &pointset);
    void movePoint(std::vector<skelx::Point> &pointset, double threshold);
    void computeUi(cv::Mat &img, std::vector<skelx::Point> &pointset, double threshold);
    void PCA(cv::Mat &img, std::vector<skelx::Point> &pointset, double threshold, double detailFactor);
    int setUpperLimitOfK(cv::Mat &img);
    cv::Mat postProcess(cv::Mat &img, const double detailFactor, const double thinningFactor, std::vector<skelx::Point> pointSet);
}