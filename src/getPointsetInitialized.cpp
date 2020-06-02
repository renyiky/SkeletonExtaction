#include <vector>
#include <opencv2/core.hpp>
#include <cmath>
#include <iostream>

#include "Point.hpp"
#include "getDbb.hpp"
#include "getD3nn.hpp"

using namespace std;
using namespace cv;

// initialize the pointset
vector<struct skelx::Point> getPointsetInitialized(Mat &img){
    double dbb = getDbb(img);  // diagonal length of bounding box
    cout<<dbb<<endl;
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
    for(struct skelx::Point &i : pointset){
        i.d3nn = getD3nn(img, i);
        i.k = i.k0 = 50; // static_cast<int>(dbb / (pow(num, 1/3) * i.d3nn));
    }
    return pointset;
}
