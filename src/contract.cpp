#include <iostream>

#include "Point.hpp"
#include "skelxut.hpp"

using namespace std;
using namespace cv;

Mat contract(Mat img, string filename, const double detailFactor, const double thinningFactor){
    double sigmaHat = 0.0,
            preSigmaHat = sigmaHat;
    int count = 0,  // count if sigmaHat remains unchanged
        t = 0,  // count for iterations
        upperLimit = skelx::setUpperLimitOfK(img);  // set the upper limit of k, it would be used when update k during each iteration
    vector<skelx::Point> pointset = skelx::getPointsetInitialized(img);    // set coordinates, k0, d3nn

    while(true){    // I remove the threshold 0.95
        skelx::computeUi(img, pointset, 0.95);
        skelx::PCA(img, pointset, 0.95, detailFactor);
        skelx::movePoint(pointset, 0.95);
        skelx::refreshPointset(img, pointset);
        for(skelx::Point &p : pointset){
            sigmaHat += p.sigma;
        }
        sigmaHat /= pointset.size();
        updateK(img, pointset, upperLimit);

        std::cout<<"iter:"<<t + 1<<"   sigmaHat = "<<sigmaHat<<endl;
        ++t;

        // check if sigmaHat remains unchanged
        // if it doesn't change for 3 times, stop extracting
        if(sigmaHat == preSigmaHat){
            if(count == 2) return skelx::postProcess(img, detailFactor, thinningFactor, pointset);
            else ++count;
        }else{
            preSigmaHat = sigmaHat;
            count = 0;
        }
    }
}