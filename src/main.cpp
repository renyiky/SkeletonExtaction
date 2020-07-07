#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <unistd.h>

#include "contract.hpp"
#include "preProcess.hpp"
#include "ZSalg.hpp"
#include "GHalg.hpp"
#include "AWalg.hpp"
#include "HybridAlg.hpp"

using namespace std;
using namespace cv;

void output(Mat &img, string name);
void superpose(Mat img, string filename, string inputPath);

int main(int argc, char *argv[]){
    string filename = argv[1], inputPath = "experimentsMaterial/resources/";
    Mat img = imread(inputPath+filename+".png", IMREAD_GRAYSCALE);
    output(img, "raw_" + filename);

    // img = invert(img);
    // output(img, "invert");

    // img = fullfill(img);
    // output(img, "fullfill_" + filename);

    Mat imgZS = ZSalg(img);
    output(imgZS, "final_ZS_" + filename);

    Mat imgGH = GHalg(img);
    output(imgGH, "final_GH_" + filename);

    Mat imgAW = AWalg(img);
    output(imgAW, "final_AW_" + filename);

    Mat imgHybrid = HybridAlg(img);
    output(imgHybrid, "final_Hybrid_" + filename);

    img = contract(img, filename, 20);
    output(img, "extracted_" + filename);
    img = AWalg(img);
    output(img, "final_" + filename);
    // superpose(img, filename, "results/");
    return 0;
}

void output(Mat &img, string name){
    string outputPath = "experimentsMaterial/results/";
    imwrite(outputPath + "0_" + name + ".png", img);
}