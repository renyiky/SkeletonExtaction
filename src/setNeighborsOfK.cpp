#include <vector>
#include <opencv2/core.hpp>
#include <cmath>

#include "Point.hpp"
#include "setNeighborsOfK.hpp"

using namespace std;
using namespace cv;

int setNeighborsOfK(Mat &img, skelx::Point &point, const int k){
    int radius = 0,
        rows = img.rows,
        cols = img.cols,
        x = point.pos[0],
        y = point.pos[1];
    vector<vector<double> > neighbors{};

    while(neighbors.size() < k){
        ++radius;
        neighbors = {};

        // rectangle search
        // for(int i = -radius; i < radius + 1; ++i){
        //     if(i == -radius || i == radius){
        //         for(int j = -radius; j < radius + 1; ++j){
        //             if(/*pow(i * i + j * j, 0.5) <= radius && */x + i >= 0 && x + i < rows && y + j >= 0 
        //             && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
        //             && !(i == 0 && j == 0)){
        //                 neighbors.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
        //             }
        //         }
        //     }else{
        //         int j = -radius;
        //         if(/*pow(i * i + j * j, 0.5) <= radius &&*/ x + i >= 0 && x + i < rows && y + j >= 0 
        //         && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
        //         && !(i == 0 && j == 0)){
        //             neighbors.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
        //         }

        //         j = radius;
        //         if(/*pow(i * i + j * j, 0.5) <= radius &&*/ x + i >= 0 && x + i < rows && y + j >= 0 
        //         && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
        //         && !(i == 0 && j == 0)){
        //             neighbors.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
        //         }
        //     }
        // }


        // circle search
        for(int i = -radius; i < radius + 1; ++i){
            for(int j = -radius; j < radius + 1; ++j){
                if(pow((i * i + j * j), 0.5) <= radius && x + i >= 0 && x + i < img.rows && y + j >= 0 && y + j < img.cols && img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
                    neighbors.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
                }
            }
        }

    }
    if(neighbors.size() != 0){
        point.neighbors = neighbors;
        return 1;
    }
    else{
        return 0;
    }
}

int regularize(skelx::Point centerPoint){
    double xi = centerPoint.pos[0], yi = centerPoint.pos[1];
    vector<vector<double> > connectedDomain = {{xi, yi, 0}}; // neighborset inlucdes center point, and set the connected domain flag of center point as 0;
    vector<vector<double> > neighbors = centerPoint.neighbors;

    unsigned int count = 0;
    while (count < neighbors.size())
    {
        double xj = neighbors[count][0], yj = neighbors[count][1];
        
        
        // for(int i = -1; i < 2; ++i){
        //     for(int j = -1; j < 2; ++j){
        //         if(!(i == 0 && j == 0) && xi + i == xj && yi + j == yj){

        //         }
        //     }
        // }
    }
    

                // if(!(i == 0 && j == 0) && xi + i == xj && yi + j == yj){
                //     return 1;
                // }

    return 0;
}