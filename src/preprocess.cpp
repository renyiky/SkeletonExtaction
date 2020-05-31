#include <opencv2/core.hpp>

#include "preprocess.hpp"

cv::Mat preprocess(cv::Mat img){
/*
process raw image
*/
    typedef unsigned int uInt;
    cv::Mat background,ret;
    background = cv::Mat::zeros(img.rows + 2, img.cols + 2, CV_8U);
    ret = cv::Mat::zeros(img.rows, img.cols, CV_8U);
    for(uInt i = 0; i < img.rows; ++i){
        for(uInt j = 0; j < img.cols; ++j){
            background.at<uchar>(i + 1, j + 1) = img.at<uchar>(i, j);
        }
    }
    for(uInt i = 0; i < img.rows; ++i){
        for(uInt j = 0; j < img.cols; ++j){
            if(img.at<uchar>(i, j) == 0){
                uInt flag = 0;
                for(uInt m = 0; m < 3 && flag == 0; ++m){
                    for(uInt n = 0; n < 3 && flag == 0; ++n){
                        if(background.at<uchar>(i + m,j + n) != 0){
                            ret.at<uchar>(i, j) = 255;
                            flag = 1;
                            break;
                        }
                    }
                }
            }
        }
    }
    return ret;
}
