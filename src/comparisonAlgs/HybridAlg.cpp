#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
#include <opencv2/highgui.hpp>

#include "previousAlgs.hpp"

using namespace std;
using namespace cv;

vector<int> createNeighborsValueHybridVer(Mat &img, const vector<vector<int> > &neighbors);
int checkBpOfPstar(Mat &img, vector<int> pos);
int subIterationOne(Mat &img, vector<int> pos);
int subIterationTwo(Mat &img, vector<int> pos);

Mat HybridAlg(Mat img){
    int count = 1,
        c = 0;
    Mat ret = img.clone();

    while(count != 0){
        count = 0;
        for(int i = 0; i < img.rows; ++i){
            for(int j = 0; j < img.cols; ++j){
                if(img.at<uchar>(i, j) != 0){
                    // when (i + j) % 2 == 0
                    if((i + j) % 2 == 0 && subIterationOne(img, {i, j})){
                        ret.at<uchar>(i, j) = 0;
                        ++count;
                    }
                }
            }
        }
        img = ret.clone();
        for(int i = 0; i < img.rows; ++i){
            for(int j = 0; j < img.cols; ++j){
                if(img.at<uchar>(i, j) != 0){
                    // when (i + j) % 2 != 0
                    if((i + j) % 2 != 0 && subIterationTwo(img, {i, j})){
                        ret.at<uchar>(i, j) = 0;
                        ++count;
                    }
                }
            }
        }
        img = ret.clone();
        // ++c;
        // imwrite("results/" + to_string(c) + ".png", img);
    }
    return ret;
}

// used for (i + j) % 2 == 0 
int subIterationOne(Mat &img, vector<int> pos){
    int x = pos[0],
        y = pos[1];

    vector<int> p2 = {x - 1, y},
                p3 = {x - 1, y + 1},
                p4 = {x, y + 1},
                p5 = {x + 1, y + 1},
                p6 = {x + 1, y},
                p7 = {x + 1, y - 1},
                p8 = {x, y - 1},
                p9 = {x - 1, y - 1};
    vector<vector<int> > neighbors = {p2, p3, p4, p5, p6, p7, p8, p9};
    vector<int> neighborsValue = createNeighborsValueHybridVer(img, neighbors);

    int Cp = static_cast<int>(!neighborsValue[0] && (neighborsValue[1] || neighborsValue[2])) +
                static_cast<int>(!neighborsValue[2] && (neighborsValue[3] || neighborsValue[4])) +
                static_cast<int>(!neighborsValue[4] && (neighborsValue[5] || neighborsValue[6])) +
                static_cast<int>(!neighborsValue[6] && (neighborsValue[7] || neighborsValue[0])),
        Bp = 0;
    for(int i : neighborsValue){
        Bp += i;
    }

    if(Cp == 1 &&
        Bp >= 2 && Bp <= 7 &&
        neighborsValue[0] * neighborsValue[2] * neighborsValue[4] == 0 &&
        neighborsValue[2] * neighborsValue[4] * neighborsValue[6] == 0){
            return 1;
        }else{
            return 0;
        }
}

// used for (i + j) % 2 != 0
int subIterationTwo(Mat &img, vector<int> pos){
    int x = pos[0],
        y = pos[1];

    vector<int> p2 = {x - 1, y},
                p3 = {x - 1, y + 1},
                p4 = {x, y + 1},
                p5 = {x + 1, y + 1},
                p6 = {x + 1, y},
                p7 = {x + 1, y - 1},
                p8 = {x, y - 1},
                p9 = {x - 1, y - 1};
    vector<vector<int> > neighbors = {p2, p3, p4, p5, p6, p7, p8, p9};
    vector<int> neighborsValue = createNeighborsValueHybridVer(img, neighbors);

    int Cp = static_cast<int>(!neighborsValue[0] && (neighborsValue[1] || neighborsValue[2])) +
                static_cast<int>(!neighborsValue[2] && (neighborsValue[3] || neighborsValue[4])) +
                static_cast<int>(!neighborsValue[4] && (neighborsValue[5] || neighborsValue[6])) +
                static_cast<int>(!neighborsValue[6] && (neighborsValue[7] || neighborsValue[0])),
        Bp = 0;

    for(int i : neighborsValue){
        Bp += i;
    }

    if(Bp == 1 && 
        ((neighborsValue[1] == 1 && checkBpOfPstar(img, {x - 1, y + 1})) || 
        (neighborsValue[3] == 1 && checkBpOfPstar(img, {x + 1, y + 1})) ||
        (neighborsValue[5] == 1 && checkBpOfPstar(img, {x + 1, y - 1})) ||
        (neighborsValue[7] == 1 && checkBpOfPstar(img, {x - 1, y - 1})))){
            return 0;
    }

    if(Cp == 1 &&
        Bp >= 1 && Bp <= 7 &&
        neighborsValue[0] * neighborsValue[2] * neighborsValue[6] == 0 &&
        neighborsValue[0] * neighborsValue[4] * neighborsValue[6] == 0){
            return 1;
        }else{
            return 0;
        }
}

vector<int> createNeighborsValueHybridVer(Mat &img, const vector<vector<int> > &neighbors){
    vector<int> neighborsValue = {};
    for(vector<int> i : neighbors){
        if(i[0] >= 0 && i[0] < img.rows && i[1] >= 0 && i[1] < img.cols){
            neighborsValue.push_back(img.at<uchar>(i[0], i[1]) == 0 ? 0 : 1);
        }else{
            neighborsValue.push_back(0);
        }
    }
    return neighborsValue;
}

// return 1: shall retain the pixel
int checkBpOfPstar(Mat &img, vector<int> pos){
    int x = pos[0],
        y = pos[1];

    vector<int> p2 = {x - 1, y},
                p3 = {x - 1, y + 1},
                p4 = {x, y + 1},
                p5 = {x + 1, y + 1},
                p6 = {x + 1, y},
                p7 = {x + 1, y - 1},
                p8 = {x, y - 1},
                p9 = {x - 1, y - 1};
    vector<vector<int> > neighbors = {p2, p3, p4, p5, p6, p7, p8, p9};
    vector<int> neighborsValue = createNeighborsValueHybridVer(img, neighbors);
    int Bp = 0;
    for(int i : neighborsValue){
        Bp += i;
    }
    if(Bp <= 2){
        return 1;
    }else{
        return 0;
    }
}