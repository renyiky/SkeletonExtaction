#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){
    string filename = argv[1];
    Mat img = imread(filename + ".png", IMREAD_COLOR);

    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.cols; ++j){
            if(img.at<uchar>(i, j, 0) == 47 && img.at<uchar>(i, j, 1) == 47 && img.at<uchar>(i, j, 2) == 47){
                img.at<uchar>(i, j, 0) = 255;
                img.at<uchar>(i, j, 1) = 255;
                img.at<uchar>(i, j, 2) = 255;
            }else if(img.at<uchar>(i, j, 0) == 0 && img.at<uchar>(i, j, 1) == 0 && img.at<uchar>(i, j, 0) == 0){
                img.at<uchar>(i, j, 0) = 255;
                img.at<uchar>(i, j, 1) = 255;
                img.at<uchar>(i, j, 2) = 255;
            }
        }
    }
    imwrite("results/" + filename + ".png", img);
}