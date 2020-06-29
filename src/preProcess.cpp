#include <opencv2/core.hpp>
#include <vector>

using namespace std;
using namespace cv;

Mat smoothing(Mat img);

// invert img
Mat invert(Mat img){
    int rows = img.rows, cols = img.cols;
    uchar *p;
    for(int i = 0; i < rows; ++i){
        p = img.ptr<uchar>(i);
        for(int j = 0; j < cols; ++j){
            p[j] = p[j] > 180 ? 0 : 255;
        }
    }
    img = smoothing(img);
    return img;
}

// fullfill the black holes by dilation and erosion
Mat fullfill(Mat img){
    Mat ret = img.clone();
    // dilation part
    // form of structure: 3x3
    for(int x = 1; x < img.rows - 1; ++x){
        for(int y =1 ; y < img.cols -1 ; ++y){
            for(int i = -1; i < 2; ++i){
                for(int j = -1; j < 2; ++j){
                    if(img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
                        ret.at<uchar>(x, y) = 255;
                    }
                }
            }
        }
    }

    img = ret.clone();
    // erosion part
    // form of structure: 3x3
    for(int x = 1; x < img.rows - 1; ++x){
        for(int y =1 ; y < img.cols -1 ; ++y){
            for(int i = -1; i < 2; ++i){
                for(int j = -1; j < 2; ++j){
                    if(img.at<uchar>(x + i, y + j) == 0 && !(i == 0 && j == 0)){
                        ret.at<uchar>(x, y) = 0;
                    }
                }
            }
        }
    }
    return ret;
}

// remove noise
Mat smoothing(Mat img){
    Mat ret = img.clone();
    int count = 1;
    while(count != 0){
        count = 0;
        for(int i = 0; i < img.rows; ++i){
            for(int j = 0; j < img.rows; ++j){
                if(img.at<uchar>(i, j) != 0){
                    int x = i,
                        y = j;
                    vector<int> p2 = {x - 1, y},
                                p3 = {x - 1, y + 1},
                                p4 = {x, y + 1},
                                p5 = {x + 1, y + 1},
                                p6 = {x + 1, y},
                                p7 = {x + 1, y - 1},
                                p8 = {x, y - 1},
                                p9 = {x - 1, y - 1};
                    vector<vector<int> > neighbors = {p2, p3, p4, p5, p6, p7, p8, p9};
                    vector<int> neighborsValue = {};
                    for(vector<int> k : neighbors){
                        if(k[0] >= 0 && k[0] < img.rows && k[1] >= 0 && k[1] < img.cols){
                            neighborsValue.push_back(img.at<uchar>(k[0], k[1]) == 0 ? 0 : 1);
                        }else{
                            neighborsValue.push_back(0);
                        }
                    }

                    int sum = 0;
                    for(int i : neighborsValue){
                        sum += i;
                    }
                    if(sum == 1){
                        ret.at<uchar>(i, j) = 0;
                        ++count;
                    }
                }
            }
        }
        img = ret.clone();
    }
    return ret;
}