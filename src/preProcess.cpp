#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

void invert(Mat &img);

Mat preProcess(Mat img){
    invert(img);
    return img;
}

void invert(Mat &img){
    int rows = img.rows, cols = img.cols;
    uchar *p;
    for(int i = 0; i < rows; ++i){
        p = img.ptr<uchar>(i);
        for(int j = 0; j < cols; ++j){
            p[j] = p[j] > 180 ? 0 : 255;
        }
    }
}