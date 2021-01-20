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
static int minRadius = 0;

Mat contract(Mat img, string filename, const double detailFactor, const bool perturbationFlag){
    k = skelx::computeK(img);  // compute k
    minRadius = skelx::computeMinimumSearchRadius(k);
    // k = 75;
    // k = 69;  // k for deer01 tail
    // k = 17; // for 901  , checked and passed: 17, 18, 19
    // k = 24; // for 902, k = 24
    // k = 36; // for 903, k = 36

    cout << "k = " << k << endl;

    while(true){
        vector<skelx::Point> pointset = skelx::getPointsetInitialized(img);
        skelx::computeUi(img, pointset, k, minRadius, perturbationFlag);
        skelx::PCA(img, pointset, detailFactor);
        // skelx::visualize(img, pointset, t);
        skelx::movePoint(pointset);
        img = skelx::draw(img, pointset);
        skelx::cleanImage(img);

        for(skelx::Point &p : pointset) sigmaHat += p.sigma;
        sigmaHat /= pointset.size();

        // imwrite("results/" + to_string(t) + ".png", img);
        ++t;
        std::cout << "iter:" << t << "   sigmaHat = " << sigmaHat << endl;

        // check if sigmaHat remains unchanged
        // if it doesn't change for 3 times, go to postprocessing
        if(sigmaHat == preSigmaHat){
            if(countTimes == 2) {
                vector<skelx::Point> pointset = skelx::getPointsetInitialized(img);
                skelx::computeUi(img, pointset, k, minRadius, perturbationFlag);
                skelx::PCA(img, pointset, detailFactor);
                return skelx::postProcess(img, pointset);
            }
            else ++countTimes;
        }else{
            preSigmaHat = sigmaHat;
            countTimes = 0;
        }
    }
}