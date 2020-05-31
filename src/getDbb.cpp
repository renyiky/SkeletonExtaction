#include <cmath>
#include <opencv2/core.hpp>

#include "getDbb.hpp"

using namespace std;
using namespace cv;

double getDbb(Mat &img){
    double left = img.cols + 1,
            right = -1,
            up = img.rows + 1,
            down = -1;
    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.cols; ++j){
            if(img.at<uchar>(i, j) != 0){
                left = left < j ? left : j;
                right = right > j ? right : j;
                up = up < i ? up : i;
                down = down > i ? down : i;
            }
        }
    }
    return sqrt((right - left) * (right - left) + (down - up) * (down - up));
}
