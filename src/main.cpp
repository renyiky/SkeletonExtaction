#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>

#include "preprocess.hpp" 

using namespace std;
using namespace cv;

void output(Mat &img, string name);

int count = 0;

int main(){
    string filename = "man", inputPath = "dataset/";
    Mat img = imread(inputPath+filename+".png", IMREAD_GRAYSCALE);
    output(img, "raw");
    img=preprocess(img);
    output(img, "preProcessed");


    return 0;
}

void output(Mat &img, string name){
    string outputPath = "results/";
    imwrite(outputPath + to_string(::count) + "_" + name + ".png", img);
    ++::count;
}