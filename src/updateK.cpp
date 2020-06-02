#include <opencv2/core.hpp>
#include <vector>
#include <cmath>

#include "Point.hpp"

using namespace std;
using namespace cv;

// update k every @param c iterations, and only update point whose sigma is less than the @param threshold
// @param threshold: only update the point's k which sigma is less than threshold
// @param t: current number of iterations
// @param a: growth rate, set 1.0 as default
// @param c: update k every c iteration, set 12 as default
void updateK(vector<skelx::Point> &pointset, double threshold, int t, double a = 1.0, int c = 12){

    // find sigma min
    double sigmaMin = 999999;
    for(skelx::Point &p : pointset){
        sigmaMin = sigmaMin > p.sigma ? p.sigma : sigmaMin;
    }
    
    for(skelx::Point &p : pointset){
        p.deltaK = p.k0 * (a * a * (2 * (static_cast<double>(t) / static_cast<double>(c)) - 1) + 2 * a) * exp((-(p.sigma - sigmaMin) * (p.sigma - sigmaMin)) / ((0.95 - sigmaMin) * (0.95 - sigmaMin)));

        
    }


}