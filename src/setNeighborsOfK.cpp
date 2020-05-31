#include <vector>
#include <opencv2/core.hpp>

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
    vector<vector<int> > neighbors{};

    while(neighbors.size() < k){
        ++radius;
        for(int i = -radius; i < radius + 1; ++i){
            if(i == -radius || i == radius){
                for(int j = -radius; j < radius + 1; ++j){
                    if(x + i >= 0 && x + i < rows && y + j >= 0 
                    && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
                    && !(i == 0 && j == 0)){
                        neighbors.push_back({i, j});
                    }
                }
            }else{
                int j = -radius;
                if(x + i >= 0 && x + i < rows && y + j >= 0 
                && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
                && !(i == 0 && j == 0)){
                    neighbors.push_back({i, j});
                }

                j = radius;
                if(x + i >= 0 && x + i < rows && y + j >= 0 
                && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
                && !(i == 0 && j == 0)){
                    neighbors.push_back({i, j});
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