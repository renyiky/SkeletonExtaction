#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

string resPath("experimentsMaterial/results/");
vector<string> methodName = {"ZS", "AW", "GH", "Hybrid", "DM", "Ours"};

Mat invert(Mat img){
    int rows = img.rows, cols = img.cols;
    uchar *p;
    for(int i = 0; i < rows; ++i){
        p = img.ptr<uchar>(i);
        for(int j = 0; j < cols; ++j){
            p[j] = p[j] > 180 ? 0 : 255;
        }
    }
    return img;
}

int main(int argc, char** argv){
    string df(argv[2]);
    resPath += (string(argv[1]) + "_" + df + "/");
    Mat background = imread(resPath + "0_raw_" + string(argv[1]) + ".png", IMREAD_GRAYSCALE);
    if(background.data == NULL) cout << "raw image open failed." << endl;
    vector<string> prefixes = {"0_final_ZS_",
                                "0_final_AW_",
                                "0_final_GH_",
                                "0_final_Hybrid_",
                                "DM_",
                                "0_final_"};
    vector<Mat> images;
    for(int i = 0; i < 5; ++i){
        Mat im(imread(resPath + prefixes[i] + string(argv[1]) + ".png", IMREAD_GRAYSCALE));
        if(im.data == NULL) cout<<"image loading failed."<<endl;
        images.push_back(im);
    }
    images.push_back((imread(resPath + prefixes.back() + string(argv[1]) + +"_" + df + ".png", IMREAD_GRAYSCALE)));

    Mat inverted_bg = invert(background);
    for(int k = 0; k < images.size(); ++k){
        Mat ret = inverted_bg.clone();
        for(int i = 0; i < images[k].rows; ++i){
            for(int j = 0; j < images[k].cols; ++j){
                if(images[k].at<uchar>(i, j) == 255){
                    ret.at<uchar>(i, j) = 255;
                }
            }
        }
        imwrite(resPath + "synthesized_" + prefixes[k] + string(argv[1]) + ".png", ret);
    }
    return 0;
}