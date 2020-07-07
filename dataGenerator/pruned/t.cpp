#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>

using namespace std;
using namespace cv;

int main(){
    string filename ="seasnake06";
    Mat img = imread(filename + ".png", IMREAD_GRAYSCALE);

    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.cols; ++j){
            if(img.at<uchar>(i, j) != 255){
                img.at<uchar>(i, j) = 0;
            }
        }
    }
    imwrite("results/" + filename + ".png", img);
}