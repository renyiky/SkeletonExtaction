#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <iostream>
#include <functional>

#include "contract.hpp"
#include "preProcess.hpp"
#include "previousAlgs.hpp"

using namespace std;
using namespace cv;

void output(const Mat &img, string name);

string inputPath = "experimentsMaterial/resources/",
        outputPath = "results/";

int main(int argc, char *argv[]){
    if(argc <= 2){cout<<"parameter insufficient."<<endl; return -1;}
    double detailFactor = stod(argv[2]);
    string filename = argv[1];
    Mat img = imread(inputPath+filename+".png", IMREAD_GRAYSCALE);
    output(img, "raw_" + filename);

    // vector<function<Mat(Mat)> > previousAlgs{ZSalg, GHalg, AWalg, HybridAlg};
    // vector<string> prefixes {"final_ZS_",
    //                         "final_GH_",
    //                         "final_AW_",
    //                         "final_Hybrid_"};

    // for(int i = 0; i < previousAlgs.size(); ++i)
    //     output(previousAlgs[i](img), prefixes[i] + filename);

    img = contract(img, filename, detailFactor);
    imwrite(outputPath + "0_final_" + filename + "_" + to_string(static_cast<int>(detailFactor)) + ".png", img);

    cout << filename + "'s current detail factor = " << detailFactor << endl;
    return 0;
}

void output(const Mat &img, string name){
    imwrite(outputPath + "0_" + name + ".png", img);
}