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

vector<string> prefixes = {"0_final_ZS_",
                            "0_final_AW_",
                            "0_final_GH_",
                            "0_final_Hybrid_",
                            "DM_",
                            "0_final_"};

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

Mat gray(Mat img){
    Mat ret = invert(img);
    int rows = ret.rows, cols = ret.cols;
    uchar *p;
    for(int i = 0; i < rows; ++i){
        p = ret.ptr<uchar>(i);
        for(int j = 0; j < cols; ++j){
            p[j] = p[j] == 0 ? 127 : 255;
        }
    }
    return ret;
}

void blackAndWhite(Mat &background, vector<Mat> &images, string name){
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
        imwrite(resPath + "synthesized_" + prefixes[k] + name + ".png", ret);
    }
}

void grayAndBlack(Mat &background, vector<Mat> &images, string name){
    
    Mat ret = background.clone();
    uchar *p;
    for(int i = 0; i < ret.rows; ++i){
        p = ret.ptr<uchar>(i);
        for(int j = 0; j < ret.cols; ++j){
            p[j] = p[j] == 0 ? 255 : 127;
        }
    }

    for(int k = 0; k < images.size(); ++k){
        Mat new_ret = ret.clone();
        for(int i = 0; i < images[k].rows; ++i){
            for(int j = 0; j < images[k].cols; ++j){
                if(images[k].at<uchar>(i, j) == 255){
                    new_ret.at<uchar>(i, j) = 0;
                }
            }
        }
        imwrite(resPath + "synthesized_gray_black_" + prefixes[k] + name + ".png", new_ret);
    }
}

int main(int argc, char** argv){
    if(argc <= 2){cout<<"parameter insufficient."<<endl; return -1;}
    string df(argv[2]);
    resPath += (string(argv[1]) + "_" + df + "/");
    Mat background = imread(resPath + "0_raw_" + string(argv[1]) + ".png", IMREAD_GRAYSCALE);
    if(background.data == NULL) {cout << "raw image open failed." << endl; return -1;}

    vector<Mat> images;
    for(int i = 0; i < 5; ++i){
        Mat im(imread(resPath + prefixes[i] + string(argv[1]) + ".png", IMREAD_GRAYSCALE));
        if(im.data == NULL) {cout<<"image loading failed."<<endl; return -1;}
        images.push_back(im);
    }
    images.push_back((imread(resPath + prefixes.back() + string(argv[1]) + +"_" + df + ".png", IMREAD_GRAYSCALE)));

    // grayAndBlack(background, images, argv[1]);
    blackAndWhite(background, images, argv[1]);
    return 0;
}