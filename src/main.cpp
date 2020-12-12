#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <iostream>
#include <functional>
#include <ctime>

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

    clock_t start_t;
    double total_t;
    vector<function<Mat(Mat)> > previousAlgs{ZSalg, AWalg, GHalg, HybridAlg};
    vector<string> prefixes {"ZS",
                            "AW",
                            "GH",
                            "Hybrid"};

    for(int i = 0; i < previousAlgs.size(); ++i){
        start_t = clock();
        output(previousAlgs[i](img), "0_final_" + prefixes[i] + "_" + filename);
        total_t = static_cast<double>(clock() - start_t) / CLOCKS_PER_SEC;
        cout << prefixes[i] << " time consumed: " << total_t << endl;
    }

    start_t = clock();
    img = contract(img, filename, detailFactor);
    total_t = static_cast<double>(clock() - start_t) / CLOCKS_PER_SEC;
    cout << "Time consumed: " << total_t << endl;
    imwrite(outputPath + "0_final_" + filename + "_" + to_string(static_cast<int>(detailFactor)) + ".png", img);

    cout << filename + "'s current detail factor = " << detailFactor << endl;
    return 0;
}

void output(const Mat &img, string name){
    imwrite(outputPath + name + ".png", img);
}