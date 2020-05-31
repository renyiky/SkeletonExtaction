#include <opencv2/core.hpp>

#include <vector>
#include <iostream>

#include "getPointsetInitialized.hpp"
#include "Point.hpp"
#include "setNeighborsOfK.hpp"

using namespace std;
using namespace cv;

void computeUi(Mat &img, vector<skelx::Point> &pointset);

int contract(Mat &img, string filename){
    vector<skelx::Point> pointset = getPointsetInitialized(img);    // set coordinates, k0, d3nn
    


}

void computeUi(Mat &img, vector<skelx::Point> &pointset){
    for(skelx::Point &p : pointset){
        // get k nearest neighbors
        vector<double> ui{0.0, 0.0};

        if(!setNeighborsOfK(img, p, p.k)){  // set ui neighbors
            cout<<"neighbors insufficient!"<<endl;
        }
        for(vector<int> nei: p.neighbors){
            ui[0] += nei[0];
            ui[1] += nei[1];
        }
        ui[0] = ui[0]/static_cast<double>(p.neighbors.size());
        ui[1] = ui[1]/static_cast<double>(p.neighbors.size());
        p.ui = {ui[0], ui[1]};
    }
}

void PCA(Mat &img, vector<skelx::Point> &pointset){
    for(skelx::Point &p: pointset){
        if(!setNeighborsOfK(img, p, 3 * p.d3nn)){   // set PCA neighbors
            cout<<"PCA neighbors insufficient!"<<endl;
        }
    }



}