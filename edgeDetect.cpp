#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

using namespace cv;
using namespace std;

int main(){
    Mat img1 = imread("results/1_preProcessed.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("results/3_bone.png", IMREAD_GRAYSCALE);

    for(int i = 0; i<img1.rows; ++i){
        for(int j = 0; j<img1.cols; ++j){
            img1.at<uchar>(i, j) = img1.at<uchar>(i, j) - img2.at<uchar>(i, j);
        }
    }
    imwrite("source.png", img1);
    return 0;
}