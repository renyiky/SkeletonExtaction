#include <vector>
#include <opencv2/core.hpp>
#include <cmath>
#include <iostream>

#include "Point.hpp"
#include "getDbb.hpp"
#include "getD4nn.hpp"

using namespace std;
using namespace cv;

// initialize the pointset
vector<struct skelx::Point> getPointsetInitialized(Mat &img){
    double dbb = getDbb(img);  // diagonal length of bounding box

    vector<struct skelx::Point> pointset;
    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.cols; ++j){
            if(img.at<uchar>(i, j) != 0){
                skelx::Point p;
                p.pos[0] = i;
                p.pos[1] = j;
                pointset.push_back(p);
            }
        }
    }
    // set k0
    int num = pointset.size();
    for(struct skelx::Point &p : pointset){
        p.d4nn = getD4nn(img, p);
        double dnn = 3 * p.d4nn;
        int x = p.pos[0],
            y = p.pos[1];
        vector<vector<double> > neighborsCount = {};
        for(int i = -dnn; i < dnn + 1; ++i){
            for(int j = -dnn; j < dnn + 1; ++j){
                if(pow((i * i + j * j), 0.5) <= dnn && x + i >= 0 && x + i < img.rows && y + j >= 0 && y + j < img.cols && img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
                    neighborsCount.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
                }
            }
        }

        p.k = p.k0 = neighborsCount.size(); // static_cast<int>(dbb / (pow(num, 1/3) * p.d4nn));
    }
    return pointset;
}
