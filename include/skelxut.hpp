#pragma once

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "Point.hpp"

namespace skelx{
    int computeK(cv::Mat &img);
    void movePoint(std::vector<skelx::Point> &pointset);
    std::vector<struct skelx::Point> getPointsetInitialized(cv::Mat &img);
    cv::Mat draw(const cv::Mat &src, std::vector<struct skelx::Point> &pointset);
    void computeUi(cv::Mat &img, std::vector<skelx::Point> &pointset, const int k, const bool perturbationFlag);
    void PCA(cv::Mat &img, std::vector<skelx::Point> &pointset, double detailFactor);
    void visualize(const cv::Mat &img, const std::vector<skelx::Point> pointset, const int iter);
    cv::Mat postProcess(cv::Mat &img, const double detailFactor, const int k, const bool perturbationFlag);
    int gaussCircleCount(const int r);
    int computeSearchRadius(const cv::Mat &img);
    bool setRadiusNeighbors(cv::Mat &img, skelx::Point &point, const int radius, bool perturbationFlag);
}