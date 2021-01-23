#include <iostream>

#include "Point.hpp"
#include "skelxut.hpp"

using namespace std;
using namespace cv;

static double sigmaHat = 0.0;
static double preSigmaHat = sigmaHat;

static int countTimes = 0;   // count if sigmaHat remains unchanged
static int t = 0;   // count for iterations
static int k = 22;   // k nearest neighbors

Mat contract(Mat img, string filename, const double detailFactor, const bool perturbationFlag){
    // k = skelx::computeK(img);  // compute k
    // k = 22;
    // minRadius = skelx::computeMinimumSearchRadius(k);
    // cout << "k = " << k << endl;

    // int x = 130, y = 130;
    // vector<skelx::Point> _pointset;
    // _pointset.push_back(skelx::Point(x, y));
    // vector<vector<double>> neighbors;

    // double radius = 0.1;
    // while(radius < 23.){
    //     // cout<<radius<<endl;
    //     vector<vector<double>> new_neighbors;
    //     for(int i = -radius; i < radius + 1; i++){
    //         for(int j = -radius; j < radius + 1; j++){
    //             if(pow((i * i + j * j), 0.5) <= radius && x + i >= 0 && x + i < img.rows && y + j >= 0 && 
    //                 y + j < img.cols && img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
    //                     if(find(neighbors.begin(), neighbors.end(), vector<double>{static_cast<double>(i), static_cast<double>(j)}) == neighbors.end()){
    //                         new_neighbors.push_back({static_cast<double>(i), static_cast<double>(j)});
    //                     }
    //             }
    //         }
    //     }
    //     if(!new_neighbors.empty()){
    //         cout<<"{";
    //         // cout<<"new neighbors:\n";
    //         for(int i = 0; i < new_neighbors.size(); ++i){
    //             cout<<"{"<<static_cast<int>(new_neighbors[i][0])<<","<<static_cast<int>(new_neighbors[i][1])<<"}";
    //             if(i != new_neighbors.size() - 1) cout<<",";
    //             neighbors.push_back(new_neighbors[i]);
    //         }
    //         cout<<"},   // r = "<<radius<<endl;
    //         _pointset[0].neighbors = neighbors;
    //     }
    //     radius += 0.1;
    // }
    // return img;




    while(true){
        vector<skelx::Point> pointset = skelx::getPointsetInitialized(img);
        skelx::computeUi(img, pointset, k, perturbationFlag);
        skelx::PCA(img, pointset, detailFactor);
        // skelx::visualize(img, pointset, t);
        skelx::movePoint(pointset);
        img = skelx::draw(img, pointset);
        // skelx::cleanImage(img);

        for(skelx::Point &p : pointset) sigmaHat += p.sigma;
        sigmaHat /= pointset.size();

        ++t;
        std::cout << "iter:" << t << "   sigmaHat = " << sigmaHat << endl;

        // check if sigmaHat remains unchanged
        // if it doesn't change for 3 times, go to postprocessing
        if(sigmaHat == preSigmaHat){
            if(countTimes == 2) {
                // imwrite("results/0_extracted_" + filename + "_" + to_string(static_cast<int>(detailFactor)) + ".png", img);
                vector<skelx::Point> pointset = skelx::getPointsetInitialized(img);
                skelx::computeUi(img, pointset, k, perturbationFlag);
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