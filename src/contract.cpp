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

Mat contract(Mat img, string filename, const double detailFactor, const double thinningFactor){
    k = skelx::computeK(img);  // compute k

    while(true){
        vector<skelx::Point> pointset = skelx::getPointsetInitialized(img);    // set coordinates, k
        skelx::computeUi(img, pointset, k);
        skelx::PCA(img, pointset, detailFactor);
        skelx::movePoint(pointset);
        img = skelx::draw(img, pointset);

        // if(t % 10 == 0) skelx::visualize(img, pointset, t);

        for(skelx::Point &p : pointset) sigmaHat += p.sigma;
        sigmaHat /= pointset.size();

        std::cout<<"iter:"<<t + 1<<"   sigmaHat = "<<sigmaHat<<endl;

        ++t;
        // check if sigmaHat remains unchanged
        // if it doesn't change for 3 times, stop extracting
        if(sigmaHat == preSigmaHat){
            if(countTimes == 2) {
                imwrite("results/0_raw_skel.png", draw(img, pointset));
                return skelx::postProcess(img, detailFactor, thinningFactor, k);}
            // if(count == 2) return img;
            else ++countTimes;
        }else{
            preSigmaHat = sigmaHat;
            countTimes = 0;
        }
    }
}