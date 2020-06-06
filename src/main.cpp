#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>

#include "contract.hpp"

using namespace std;
using namespace cv;

void output(Mat &img, string name);

int main(int argc, char *argv[]){
    string filename = argv[1], inputPath = "dataset/";
    Mat img = imread(inputPath+filename+".png", IMREAD_GRAYSCALE);
    output(img, "raw");
    img = contract(img, filename);
    output(img, "final");
    return 0;
}

void output(Mat &img, string name){
    string outputPath = "results/";
    imwrite(outputPath + "0_" + name + ".png", img);
}