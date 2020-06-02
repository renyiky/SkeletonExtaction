#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include <iostream>

#include "Point.hpp"

using namespace std;
using namespace cv;

double thetaFunc(double t, double r0);

// update k every @param c iterations, and only update point whose sigma is less than the @param threshold
// @param threshold: only update the point's k which sigma is less than threshold, set 0.95 as default
// @param t: current number of iterations
// @param a: growth rate, set 1.0 as default
// @param c: update k every c iteration, set 12 as default
vector<skelx::Point> updateK(Mat &img, vector<skelx::Point> &pointset, int t, double threshold, double a, int c){
    vector<skelx::Point> ret={};
    // find sigma min
    double sigmaMin = 999999;
    for(skelx::Point &p : pointset){
        sigmaMin = sigmaMin > p.sigma ? p.sigma : sigmaMin;
    }
    
    // set deltaK for each point
    for(skelx::Point &p : pointset){
        p.deltaK = p.k0 * (a * a * (2 * (static_cast<double>(t) / static_cast<double>(c)) - 1) + 2 * a) * exp((-(p.sigma - sigmaMin) * (p.sigma - sigmaMin)) / ((0.95 - sigmaMin) * (0.95 - sigmaMin)));
    }

    // Gaussian weighting and PCA weighting
    for(skelx::Point p : pointset){
        double numerator = 0.0, denominator = 0.0;
        for(vector<double> nei : p.PCAneighbors){
            for(skelx::Point temp : pointset){
                if(temp.pos[0] == nei[0] && temp.pos[1] == nei[1]){
                    numerator += ((temp.k + temp.deltaK) * thetaFunc(temp.k + temp.deltaK, temp.k0));
                    denominator += thetaFunc(temp.k + temp.deltaK, temp.k0);
                    cout<<"point:"<<nei[0]<<" "<<nei[1]<<endl;
                    break;
                }
            }
        }
        p.k = numerator / denominator;
        ret.push_back(p);
    }
    return ret;
}

// we set k0 as r0
double thetaFunc(double t, double r0){
    return exp(- t * t / ((r0 / 2) * (r0 / 2)));
}