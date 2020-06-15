#include <opencv2/core.hpp>
#include <vector>

#include "ZSalg.hpp"

using namespace std;
using namespace cv;

int isSatisfiedIterOne(Mat &img, const vector<int> &pos);
int isSatisfiedIterTwo(Mat &img, const vector<int> &pos);
Mat subIterationOne(Mat &img);
Mat subIterationTwo(Mat &img);

int C1 = 0;
int C2 = 0;

// Implementation of Zhang, Suen algorithm.
// won't change the source img.
// frontground: white
// background: black
Mat ZSalg(Mat img){
    do{
        C1 = 0;
        C2 = 0;
        img = subIterationOne(img);
        img = subIterationTwo(img);
    }while(C1 != 0 || C2 != 0);
    return img;
}

Mat subIterationOne(Mat &img){
    Mat ret = img.clone();
    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.cols; ++j){
            if(img.at<uchar>(i, j) != 0 && isSatisfiedIterOne(img, {i, j})){
                ret.at<uchar>(i, j) = 0;
                ++C1;
            }
        }
    }
    return ret;
}

Mat subIterationTwo(Mat &img){
    Mat ret = img.clone();
    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.cols; ++j){
            if(img.at<uchar>(i, j) != 0 && isSatisfiedIterTwo(img, {i, j})){
                ret.at<uchar>(i, j) = 0;
                ++C2;
            }
        }
    }
    return ret;
}

// Condition (a): B(p) is the number of nonzero neighbors of P
// Condition (b): A(p) is the number of 01 patterns in the ordered set P2, P3..., P9
// Condition (c): P2 * P4 * P6 = 0
// Condition (d): P4 * P6 * P8 = 0
int isSatisfiedIterOne(Mat &img, const vector<int> &pos){
    int x = pos[0], 
        y = pos[1],
        countA = 0,
        countB = 0;
    vector<int> p2 = {x - 1, y},
                p3 = {x - 1, y + 1},
                p4 = {x, y + 1},
                p5 = {x + 1, y + 1},
                p6 = {x + 1, y},
                p7 = {x + 1, y - 1},
                p8 = {x, y - 1},
                p9 = {x - 1, y -1};
    vector<vector<int> > neighbors = {p2, p3, p4, p5, p6, p7, p8, p9};
    
    // condition a
    for(vector<int> i : neighbors){
        if(i[0] >= 0 && i[0] < img.rows && i[1] >= 0 && i[1] < img.cols && img.at<uchar>(i[0], i[1]) != 0){
            ++countA;
        }
    }

    // condition b
    vector<int> neighborsValue = {};
    for(vector<int> i : neighbors){
        if(i[0] >= 0 && i[0] < img.rows && i[1] >= 0 && i[1] < img.cols){
            neighborsValue.push_back(img.at<uchar>(i[0], i[1]) == 0 ? 0 : 1);
        }else{
            neighborsValue.push_back(0);
        }
    }
    neighborsValue.push_back(neighborsValue[0]);    // from p9 back to p2

    for(int i = 0; i < neighborsValue.size(); ++i){
        if(i + 1 < neighborsValue.size() && neighborsValue[i] == 0 && neighborsValue[i + 1] == 1){
            ++countB;
        }
    }

    return (countA >= 2 && countA <= 6 && // condition a
            countB == 1 &&  // condition b
            neighborsValue[0] * neighborsValue[2] * neighborsValue[4] == 0 &&   //condition c
            neighborsValue[2] * neighborsValue[4] * neighborsValue[6] == 0); // condition d
}

// Condition (a): B(p) is the number of nonzero neighbors of P
// Condition (b): A(p) is the number of 01 patterns in the ordered set P2, P3..., P9
// Condition (c): P2 * P4 * P8 = 0
// Condition (d): P2 * P6 * P8 = 0
int isSatisfiedIterTwo(Mat &img, const vector<int> &pos){
    int x = pos[0], 
        y = pos[1],
        countA = 0,
        countB = 0;
    vector<int> p2 = {x - 1, y},
                p3 = {x - 1, y + 1},
                p4 = {x, y + 1},
                p5 = {x + 1, y + 1},
                p6 = {x + 1, y},
                p7 = {x + 1, y - 1},
                p8 = {x, y - 1},
                p9 = {x - 1, y -1};
    vector<vector<int> > neighbors = {p2, p3, p4, p5, p6, p7, p8, p9};
    
    // condition a
    for(vector<int> i : neighbors){
        if(i[0] >= 0 && i[0] < img.rows && i[1] >= 0 && i[1] < img.cols && img.at<uchar>(i[0], i[1]) != 0){
            ++countA;
        }
    }

    // condition b
    vector<int> neighborsValue = {};
    for(vector<int> i : neighbors){
        if(i[0] >= 0 && i[0] < img.rows && i[1] >= 0 && i[1] < img.cols){
            neighborsValue.push_back(img.at<uchar>(i[0], i[1]) == 0 ? 0 : 1);
        }else{
            neighborsValue.push_back(0);
        }
    }
    neighborsValue.push_back(neighborsValue[0]);    // from p9 back to p2

    for(int i = 0; i < neighborsValue.size(); ++i){
        if(i + 1 < neighborsValue.size() && neighborsValue[i] == 0 && neighborsValue[i + 1] == 1){
            ++countB;
        }
    }

    return (countA >= 2 && countA <= 6 && // condition a
            countB == 1 &&  // condition b
            neighborsValue[0] * neighborsValue[2] * neighborsValue[6] == 0 &&    // condition c
            neighborsValue[0] * neighborsValue[4] * neighborsValue[6] == 0);   // condition d
}