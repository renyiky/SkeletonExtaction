#include <opencv2/core.hpp>

#include "draw.hpp"
#include "Point.hpp"

using namespace std;
using namespace cv;

// draw the pointset, and return the new img
Mat draw(Mat src, vector<struct skelx::Point> pointset){
   
    int rows = src.rows, cols = src.cols;
    Mat ret = Mat::zeros(rows, cols, CV_8U);
    for(skelx::Point p : pointset){
        ret.at<uchar>(p.pos[0],p.pos[1]) = 255;
    }
    return ret;
}