#include <opencv2/core.hpp>
#include <vector>

#include "getD3nn.hpp"

using namespace std;
using namespace cv;

// get d3nn of param point
double getD3nn(Mat &img, const struct skelx::Point &point){
    int x = point.pos[0], y = point.pos[1];
    int radius = 0,
        rows = img.rows,
        cols = img.cols;
    vector<int> neighbors = {};

    while(neighbors.size() < 3){
        ++radius;
        for(int i = -radius; i < radius + 1; ++i){
            if(i == -radius || i == radius){
                for(int j = -radius; j < radius + 1; ++j){
                    if(x + i >= 0 && x + i < rows && y + j >= 0 
                    && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
                    && !(i == 0 && j == 0)){
                        neighbors.push_back(radius);
                    }
                }
            }else{
                int j = -radius;
                if(x + i >= 0 && x + i < rows && y + j >= 0 
                && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
                && !(i == 0 && j == 0)){
                    neighbors.push_back(radius);
                }

                j = radius;
                if(x + i >= 0 && x + i < rows && y + j >= 0 
                && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
                && !(i == 0 && j == 0)){
                    neighbors.push_back(radius);
                }
            }
        }
    }
    return (neighbors[0] + neighbors[1] + neighbors[2]) / 3;
}