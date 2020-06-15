#include <opencv2/core.hpp>
#include <vector>
#include <iostream>

#include "GHalg.hpp"

using namespace std;
using namespace cv;

int isA1Satisfied(Mat &img, vector<int> pos, int iter);
int isA2Satisfied(Mat &img, vector<int> pos);

Mat GHalg(Mat img){
    int count = 1,
        iter = 0;
    
    Mat ret = img.clone();
    while(count != 0){
        // ++iter;
        count = 0;

        // subfield one
        for(int i = 0; i < img.rows; ++i){
            if(i % 2 == 0){
                for(int j = 0; j < img.cols; j += 2){
                    if(img.at<uchar>(i, j) != 0 && isA2Satisfied(img, {i, j})){
                        ret.at<uchar>(i, j) = 0;
                        ++count;
                    }
                }
            }else{
                for(int j = 1; j < img.cols; j += 2){
                    if(img.at<uchar>(i, j) != 0 && isA2Satisfied(img, {i, j})){
                        ret.at<uchar>(i, j) = 0;
                        ++count;
                    }
                }
            }
        }
        img = ret.clone();
        // subfield two
        for(int i = 0; i < img.rows; ++i){
            if(i % 2 == 0){
                for(int j = 1; j < img.cols; j += 2){
                    if(img.at<uchar>(i, j) != 0 && isA2Satisfied(img, {i, j})){
                        ret.at<uchar>(i, j) = 0;
                        ++count;
                    }
                }
            }else{
                for(int j = 0; j < img.cols; j += 2){
                    if(img.at<uchar>(i, j) != 0 && isA2Satisfied(img, {i, j})){
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

// check if the conditions of algorithm A1 are satisfied
// condition (a): C(p) = 1;
// condition (b): 2 <= N(p) <= 3;
// condition (c): one of the following:
//                  1. (p2 || p3 || !p5) || p4 = 0 for odd iters;
//                  2. (p6 || p7 || !p1) || p8 = 0 for even iters;
int isA1Satisfied(Mat &img, vector<int> pos, int iter){
    int x = pos[0],
        y = pos[1];
    vector<int> p1 = {x - 1, y -1},
                p2 = {x - 1, y},
                p3 = {x - 1, y + 1},
                p4 = {x, y + 1},
                p5 = {x + 1, y + 1},
                p6 = {x + 1, y},
                p7 = {x + 1, y - 1},
                p8 = {x, y -1};
    vector<vector<int> > neighbors = {p1, p2, p3, p4, p5, p6, p7, p8};
    vector<int> neighborsValue = {};
    
    for(vector<int> i : neighbors){
        if(i[0] >= 0 && i[0] < img.rows && i[1] >= 0 && i[1] < img.cols){
            neighborsValue.push_back(img.at<uchar>(i[0], i[1]) == 0 ? 0 : 1);
        }else{
            neighborsValue.push_back(0);
        }
    }

    // condition a
    int Cp = static_cast<int>(!neighborsValue[1] && (neighborsValue[2] || neighborsValue[3])) +
                static_cast<int>(!neighborsValue[3] && (neighborsValue[4] || neighborsValue[5])) +
                static_cast<int>(!neighborsValue[5] && (neighborsValue[6] || neighborsValue[7])) +
                static_cast<int>(!neighborsValue[7] && (neighborsValue[0] || neighborsValue[1]));
    
    // condition b
    int N1 = static_cast<int>(neighborsValue[0] || neighborsValue[1]) +
                static_cast<int>(neighborsValue[2] || neighborsValue[3]) +
                static_cast<int>(neighborsValue[4] || neighborsValue[5]) +
                static_cast<int>(neighborsValue[6] || neighborsValue[7]),
    
        N2 = static_cast<int>(neighborsValue[1] || neighborsValue[2]) +
                static_cast<int>(neighborsValue[3] || neighborsValue[4]) +
                static_cast<int>(neighborsValue[5] || neighborsValue[6]) +
                static_cast<int>(neighborsValue[7] || neighborsValue[0]);
        
    int Np = N1 < N2 ? N1 : N2;
    
    // condition c
    int res;
    if(iter % 2 == 0){
        res = neighborsValue[1] || neighborsValue[2] || !neighborsValue[4];
    }else{
        res = neighborsValue[5] || neighborsValue[6] || !neighborsValue[0];
    }

    return Cp == 1 &&   // condition a
            Np >= 2 && Np <= 3 &&   // condition b
            res == 0;   // condition c
}

// check if the conditions of algorithm A2 are satisfied
// condition (a): C(p) = 1;
// condition (b): p is 4-connected to S(hat);
// condition (c): B(p) > 1
int isA2Satisfied(Mat &img, vector<int> pos){
    int x = pos[0],
        y = pos[1];
    vector<int> p1 = {x - 1, y -1},
            p2 = {x - 1, y},
            p3 = {x - 1, y + 1},
            p4 = {x, y + 1},
            p5 = {x + 1, y + 1},
            p6 = {x + 1, y},
            p7 = {x + 1, y - 1},
            p8 = {x, y -1};
    vector<vector<int> > neighbors = {p1, p2, p3, p4, p5, p6, p7, p8};
    vector<int> neighborsValue = {};
    
    for(vector<int> i : neighbors){
        if(i[0] >= 0 && i[0] < img.rows && i[1] >= 0 && i[1] < img.cols){
            neighborsValue.push_back(img.at<uchar>(i[0], i[1]) == 0 ? 0 : 1);
        }else{
            neighborsValue.push_back(0);
        }
    }

    // condition a
    int Cp = static_cast<int>(!neighborsValue[1] && (neighborsValue[2] || neighborsValue[3])) +
                static_cast<int>(!neighborsValue[3] && (neighborsValue[4] || neighborsValue[5])) +
                static_cast<int>(!neighborsValue[5] && (neighborsValue[6] || neighborsValue[7])) +
                static_cast<int>(!neighborsValue[7] && (neighborsValue[0] || neighborsValue[1]));

    // condition b (p2 = 0) or (p4 = 0) or (p6 = 0) or (p8 = 0)
    
    // condition c
    int count = 0;
    for(int i : neighborsValue){
        count += i;
    }

    return Cp == 1 &&
            (neighborsValue[1] == 0 || neighborsValue[3] == 0 || neighborsValue[5] == 0 || neighborsValue[7] == 0) &&
            count > 1;
}
