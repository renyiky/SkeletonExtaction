#include <opencv2/core.hpp>
#include <vector>
#include <cmath>
#include <iostream>

#include "Point.hpp"
#include "getD3nn.hpp"
#include "updateK.hpp"

using namespace std;
using namespace cv;

// double thetaFunc(double t, double r0);
// vector<vector<vector<double> > > drawKandDeltakMap(const Mat &img, const vector<skelx::Point> &pointset);

// update k every @param c iterations, and only update point whose sigma is less than the @param threshold
// @param threshold: only update the point's k which sigma is less than threshold, set 0.95 as default
// @param t: current number of iterations
// @param a: growth rate, set 1.0 as default
// @param c: update k every c iteration, set 12 as default
// vector<skelx::Point> updateK(Mat &img, vector<skelx::Point> &pointset, int t, double threshold, double a, int c){
//     vector<skelx::Point> ret={};
//     // find sigma min
//     double sigmaMin = 999999;
//     for(skelx::Point &p : pointset){
//         sigmaMin = sigmaMin > p.sigma ? p.sigma : sigmaMin;
//     }
    
//     // set deltaK for each point
//     for(skelx::Point &p : pointset){
//         p.deltaK = p.k0 * (a * a * (2 * (static_cast<double>(t) / static_cast<double>(c)) - 1) + 2 * a) * exp((-(p.sigma - sigmaMin) * (p.sigma - sigmaMin)) / ((0.95 - sigmaMin) * (0.95 - sigmaMin)));
//     }

//     // refresh PCAneighbors
//     for(skelx::Point &p : pointset){
//         double dnn = 3 * p.d3nn, 
//                     x = p.pos[0],
//                     y = p.pos[1];
//         p.PCAneighbors = {};

//         for(int i = -dnn; i < dnn + 1; ++i){
//             for(int j = -dnn; j < dnn + 1; ++j){
//                 if(x + i >= 0 && x + i < img.rows && y + j >= 0 && y + j < img.cols && img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
//                     p.PCAneighbors.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
//                 }
//             }
//         }
//     }

//     vector<vector<vector<double> > > kMap = drawKandDeltakMap(img, pointset);

//     // Gaussian weighting and PCA weighting
//     for(skelx::Point p : pointset){
//         if(p.sigma < threshold){
//             double numerator = 0.0, denominator = 0.0;
//             for(vector<double> nei : p.PCAneighbors){
//                 numerator += ((kMap[nei[0]][nei[1]][3] + kMap[nei[0]][nei[1]][4]) * thetaFunc(kMap[nei[0]][nei[1]][3] + kMap[nei[0]][nei[1]][4], kMap[nei[0]][nei[1]][2]));
//                 denominator += thetaFunc(kMap[nei[0]][nei[1]][3] + kMap[nei[0]][nei[1]][4], kMap[nei[0]][nei[1]][2]);
//                 // cout<<"point:"<<nei[0]<<" "<<nei[1]<<endl;
//             }
//             cout << "before: "<<p.k<<endl;
//             p.k = numerator / denominator;
//             cout<<"after: "<<p.deltaK<<"  k:"<<p.k<<"  numerator: "<<numerator<<" deno: "<<denominator<<"\n"<<endl;
//         }
//         ret.push_back(p);
//     }
//     return ret;
// }

// // we set k0 as r0
// double thetaFunc(double t, double r0){
//     return exp(- t * t / ((r0 / 2) * (r0 / 2)));
// }

// vector<vector<vector<double> > > drawKandDeltakMap(const Mat &img, const vector<skelx::Point> &pointset){
//     vector<vector<vector<double> > > ret(img.rows, vector<vector<double> >(img.cols, vector<double>(5, 0.0)));

//     for(skelx::Point p : pointset){
//         ret[p.pos[0]][p.pos[1]][0] = p.pos[0];
//         ret[p.pos[0]][p.pos[1]][1] = p.pos[1];
//         ret[p.pos[0]][p.pos[1]][2] = static_cast<double>(p.k0) * p.d3nn;
//         ret[p.pos[0]][p.pos[1]][3] = static_cast<double>(p.k);
//         ret[p.pos[0]][p.pos[1]][4] = p.deltaK;

//         if(p.pos[0] == 208 && p.pos[1] == 51){
//             for(double i : ret[208][51]){
//                 cout<<"ho "<<i<<" "<<flush;
//             }
//         } 
//     }


//     return ret;
// }

// update k in each iteration
// when k is larger than 10, we set it as the @param upperLimit
void updateK(Mat &img, vector<skelx::Point> &pointset, int upperLimit){
    for(struct skelx::Point &p : pointset){
        double dnn = 3 * p.d3nn;
        int x = p.pos[0],
            y = p.pos[1];
        vector<vector<double> > neighborsCount = {};
        for(int i = -dnn; i < dnn + 1; ++i){
            for(int j = -dnn; j < dnn + 1; ++j){
                if(pow((i * i + j * j), 0.5) <= dnn && x + i >= 0 && x + i < img.rows && y + j >= 0 && y + j < img.cols && img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
                    neighborsCount.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
                }
            }
        }
        // set the upper limit of K = 20
        p.k = 10 < neighborsCount.size() ? upperLimit : neighborsCount.size();
        // if(neighborsCount.size() >= 20){
        //     cout<<p.k<<"  "<< neighborsCount.size()<<endl;
        // }
    }
}