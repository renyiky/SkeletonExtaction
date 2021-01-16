#include <iostream>

#include "Point.hpp"
#include "skelxut.hpp"

using namespace std;
using namespace cv;

static double sigmaHat = 0.0;
static double preSigmaHat = sigmaHat;

static int countTimes = 0;   // count if sigmaHat remains unchanged
static int t = 0;   // count for iterations
static int k = 0;   // k nearest neighbors
static int r = 0;

Mat contract(Mat img, string filename, const double detailFactor, const bool perturbationFlag){
    // k = skelx::computeK(img);  // compute k
    r = skelx::computeSearchRadius(img);
    r = 5;
        // cout<<r<<endl;
        // return img;
    while(true){
        vector<skelx::Point> pointset = skelx::getPointsetInitialized(img);
        skelx::computeUi(img, pointset, r, perturbationFlag);
        // skelx::visualize(img, pointset, t);
        skelx::PCA(img, pointset, detailFactor);
        skelx::movePoint(pointset);
        img = skelx::draw(img, pointset);


        for(skelx::Point &p : pointset) sigmaHat += p.sigma;
        sigmaHat /= pointset.size();

        imwrite("results/" + to_string(t) + ".png", img);
        ++t;
        std::cout<<"iter:"<<t<<"   sigmaHat = "<<sigmaHat<<endl;

        // check if sigmaHat remains unchanged
        // if it doesn't change for 3 times, go to postprocessing
        if(sigmaHat == preSigmaHat){
            if(countTimes == 2) return skelx::postProcess(img, detailFactor, r, perturbationFlag);
            else ++countTimes;
        }else{
            preSigmaHat = sigmaHat;
            countTimes = 0;
        }
    }
}