#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

Mat preProcess(Mat &img){\
    int rows = img.rows, cols = img.cols;
    Mat ret = Mat::zeros(rows, cols, CV_8U);

    return ret;
}

void removeWhiteSkeleton(Mat &img){
    for(int x = 0; x< img.rows; ++x){
        for(int y = 0; y < img.cols; ++y){
            if(img.at<uchar>(x, y) != 0){
                int flag = 0;
                for(int i = -1; (i < 2) && (flag == 0); ++i){
                    for(int j = -1; (j < 2) && (flag == 0); ++j){
                        if(x + i >= 0 && x + i < img.rows && y + j >= 0 && y + j < img.cols && img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
                            flag = 1;
                            break;
                        }
                    }
                }
                if(flag == 0){
                    img.at<uchar>(x, y) = 0;
                }
            }
        }
    }
}