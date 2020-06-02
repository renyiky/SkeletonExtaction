#include <opencv2/core.hpp>
#include <vector>

#include "Point.hpp"

using namespace std;
using namespace cv;

void visualize(const Mat &img, const vector<skelx::Point> pointset){
    // note that BGR ! (29, 147, 248)
    
    Mat KNNVisual(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));

    // KNN visualization
    for(skelx::Point p : pointset){
        KNNVisual.at<uchar>(p.pos[0], p.pos[1], 0) = 255;
        KNNVisual.at<uchar>(p.pos[0], p.pos[1], 1) = 255;
        KNNVisual.at<uchar>(p.pos[0], p.pos[1], 2) = 255;
    }
    Mat PCAVisual = KNNVisual.clone();

    
}