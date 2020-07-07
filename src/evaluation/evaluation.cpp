#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){
    string filename = argv[1], 
            srcPath = "experimentsMaterial/results/",
            gtPath = "experimentsMaterial/groundtruth/";
    Mat src = imread(srcPath + filename + ".png", IMREAD_GRAYSCALE);
    

    return 0;
}