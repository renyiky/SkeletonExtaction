#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <iostream>

#include "contract.hpp"
#include "preProcess.hpp"
#include "previousAlgs.hpp"

using namespace std;
using namespace cv;

void output(Mat &img, string name);
void superpose(Mat img, string filename, string inputPath);

string inputPath = "experimentsMaterial/resources/",
        outputPath = "results/";

int main(int argc, char *argv[]){
    double thinningFactor = 0.8;
    double detailFactor = stod(argv[2]);
    if(argc == 4) thinningFactor = stod(argv[3]);
    string filename = argv[1];
    Mat img = imread(inputPath+filename+".png", IMREAD_GRAYSCALE);
    output(img, "raw_" + filename);

    // img = invert(img);
    // output(img, "invert");

    // img = fullfill(img);
    // output(img, "fullfill_" + filename);

    // Mat imgZS = ZSalg(img);
    // output(imgZS, "final_ZS_" + filename);

    // Mat imgGH = GHalg(img);
    // output(imgGH, "final_GH_" + filename);

    // Mat imgAW = AWalg(img);
    // output(imgAW, "final_AW_" + filename);

    // Mat imgHybrid = HybridAlg(img);
    // output(imgHybrid, "final_Hybrid_" + filename);

    img = contract(img, filename, detailFactor, thinningFactor);
    imwrite(outputPath + "4_final_" + filename + "_" + to_string(static_cast<int>(detailFactor)) + "_" + to_string(thinningFactor) + ".png", img);

    // postprocess
    // img = AWalg(img);
    // imwrite(outputPath + "0_final_" + filename + "_" + to_string(static_cast<int>(detailFactor)) + ".png", img);
    
    cout << filename + "'s current detail factor = " << detailFactor << "\n" 
            "thinningFactor = " << thinningFactor << endl;
    return 0;
}

void output(Mat &img, string name){
    // string outputPath = "experimentsMaterial/results/";
    imwrite(outputPath + "0_" + name + ".png", img);
}