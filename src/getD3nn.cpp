#include <opencv2/core.hpp>
#include <vector>

#include "getD3nn.hpp"

using namespace std;
using namespace cv;

// get d3nn of param point
// d3nn is the average length of edges in the 3-nearest-neighbors graph
double getD3nn(Mat &img, const struct skelx::Point &point){
    int x = point.pos[0], 
        y = point.pos[1],
        rows = img.rows,
        cols = img.cols;
    int radius = 0;
    vector<double> neighborsRadius = {};

    // rectangle search
    while(neighborsRadius.size() < 3){
        ++radius;
        for(int i = -radius; i < radius + 1; ++i){
            if(i == -radius || i == radius){
                for(int j = -radius; j < radius + 1; ++j){
                    if(x + i >= 0 && x + i < rows && y + j >= 0 
                        && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
                        && !(i == 0 && j == 0)){
                        neighborsRadius.push_back(static_cast<double>(radius));
                    }
                }
            }else{
                int j = -radius;
                if(x + i >= 0 && x + i < rows && y + j >= 0 
                    && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
                    && !(i == 0 && j == 0)){
                    neighborsRadius.push_back(static_cast<double>(radius));
                }

                j = radius;
                if(x + i >= 0 && x + i < rows && y + j >= 0 
                    && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
                    && !(i == 0 && j == 0)){
                    neighborsRadius.push_back(static_cast<double>(radius));
                }
            }
        }
    }

    // circle search
    // while(neighborsRadius.size() < 4){
    //     ++radius;
    //     neighborsRadius = {};
    //     for(double i = -radius; i < radius + 1; ++i){
    //         for(double j = -radius; j < radius + 1; ++j){
    //             if(pow((i * i + j * j), 0.5) <= radius && x + i >= 0 && x + i < img.rows && y + j >= 0 && y + j < img.cols && img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
    //                 neighborsRadius.push_back(pow((i * i + j * j), 0.5));
    //             }
    //         }
    //     }
    // }
    // sort(neighborsRadius.begin(), neighborsRadius.end());
    return (neighborsRadius[0] + neighborsRadius[1] + neighborsRadius[2]) / 3;   // /3
}