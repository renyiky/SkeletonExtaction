#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <unistd.h>

#include "contract.hpp"
#include "preProcess.hpp"
#include "ZSalg.hpp"
#include "GHalg.hpp"

using namespace std;
using namespace cv;

void output(Mat &img, string name);
void superpose(Mat img, string filename, string inputPath);

int main(int argc, char *argv[]){
    string filename = argv[1], inputPath = "dataset/";
    Mat img = imread(inputPath+filename+".png", IMREAD_GRAYSCALE);
    output(img, "raw");

    img = invert(img);
    output(img, "invert");

    // img = fullfill(img);
    // output(img, "fullfill_" + filename);

    // img = ZSalg(img);

    img = GHalg(img);

    // img = contract(img, filename);
    output(img, "final_" + filename);

    // superpose(img, filename, "results/");
    return 0;
}

void output(Mat &img, string name){
    string outputPath = "results/";
    imwrite(outputPath + "0_" + name + ".png", img);
}