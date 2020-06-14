#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

// invert img
Mat invert(Mat img){
    int rows = img.rows, cols = img.cols;
    uchar *p;
    for(int i = 0; i < rows; ++i){
        p = img.ptr<uchar>(i);
        for(int j = 0; j < cols; ++j){
            p[j] = p[j] > 180 ? 0 : 255;
        }
    }
    return img;
}

// fullfill the black holes by dilation and erosion
Mat fullfill(Mat img){
    Mat ret = img.clone();
    // dilation part
    // form of structure: 3x3
    for(int x = 1; x < img.rows - 1; ++x){
        for(int y =1 ; y < img.cols -1 ; ++y){
            for(int i = -1; i < 2; ++i){
                for(int j = -1; j < 2; ++j){
                    if(img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
                        ret.at<uchar>(x, y) = 255;
                    }
                }
            }
        }
    }

    img = ret.clone();
    // erosion part
    // form of structure: 3x3
    for(int x = 1; x < img.rows - 1; ++x){
        for(int y =1 ; y < img.cols -1 ; ++y){
            for(int i = -1; i < 2; ++i){
                for(int j = -1; j < 2; ++j){
                    if(img.at<uchar>(x + i, y + j) == 0 && !(i == 0 && j == 0)){
                        ret.at<uchar>(x, y) = 0;
                    }
                }
            }
        }
    }
    return ret;
}
