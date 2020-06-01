#include <opencv2/core.hpp>

#include <vector>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <cmath>

#include "getPointsetInitialized.hpp"
#include "Point.hpp"
#include "setNeighborsOfK.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

int contract(Mat &img, string filename){
    vector<skelx::Point> pointset = getPointsetInitialized(img);    // set coordinates, k0, d3nn
    
    skelx::computeUi(img, pointset);
    skelx::PCA(img, pointset);
    skelx::movePoint(pointset); 
}

namespace skelx{

    

    vector<skelx::Point> movePoint(vector<skelx::Point> pointset){  // note: no referenced
        for(skelx::Point p : pointset){
            p.pos[0] += p.deltaX[0];
            p.pos[1] += p.deltaX[1];
        }
        return pointset;
    }

    // set ui for each xi
    void computeUi(Mat &img, vector<skelx::Point> &pointset){
        for(skelx::Point &p : pointset){
            // get k nearest neighbors
            vector<double> ui{0.0, 0.0};

            if(!setNeighborsOfK(img, p, p.k)){  // set ui neighbors
                cout<<"neighbors insufficient!"<<endl;
            }
            for(vector<double> nei: p.neighbors){
                ui[0] += (nei[0] - p.pos[0]);
                ui[1] += (nei[1] - p.pos[1]);
            }
            ui[0] = ui[0]/static_cast<double>(p.neighbors.size());
            ui[1] = ui[1]/static_cast<double>(p.neighbors.size());
            p.ui = {ui[0], ui[1]};
        }
    }

    // pca process, after this, the sigma, principalVec, 
    // covMat and deltaX of xi would be set
    void PCA(Mat &img, vector<skelx::Point> &pointset){

        // get and set covMat for each xi
        for(skelx::Point &xi: pointset){
            if(!setNeighborsOfK(img, xi, 3 * xi.d3nn)){   // set PCA neighbors
                cout<<"PCA neighbors insufficient!"<<endl;
            }

            // calculate center point, namely xi
            vector<double> centerPoint{0.0, 0.0};
            for(vector<double> &xj: xi.neighbors){
                centerPoint[0] += xj[0];
                centerPoint[1] += xj[1];
            }
            centerPoint[0] /= static_cast<double>(xi.neighbors.size());
            centerPoint[1] /= static_cast<double>(xi.neighbors.size());

            vector<vector<double> > covMat(2, vector<double>(2,0.0));  // create cov Matrix
            for(vector<double> &xj: xi.neighbors){
                vector<double> xixj = {xj[0] - centerPoint[0], xj[1] - centerPoint[1]};

                covMat[0][0] += xixj[0] * xixj[0];
                covMat[0][1] += xixj[0] * xixj[1];
                covMat[1][0] += xixj[1] * xixj[0];
                covMat[1][1] += xixj[1] * xixj[1];
            }

            covMat[0][0] /= xi.neighbors.size();
            covMat[0][1] /= xi.neighbors.size();
            covMat[1][0] /= xi.neighbors.size();
            covMat[1][1] /= xi.neighbors.size();
            xi.covMat = covMat;
        }
    
        // get eigen value and eigen vectors, and set sigma, principalVec for each xi
        for(skelx::Point &xi: pointset){
            MatrixXd covMatEigen(2, 2);
            covMatEigen(0, 0) = xi.covMat[0][0];
            covMatEigen(0, 1) = xi.covMat[0][1];
            covMatEigen(1, 0) = xi.covMat[1][0];
            covMatEigen(1, 1) = xi.covMat[1][1];

            EigenSolver<MatrixXd> es(covMatEigen);
            int maxIndex;
            double lamda1 = es.eigenvalues().col(0)(0).real(), 
                    lamda2=es.eigenvalues().col(0)(1).real();

            lamda1 > lamda2 ? maxIndex = 0 : maxIndex =1;
            double sigma = es.eigenvalues().col(0)(maxIndex).real() / (lamda1 + lamda2);
            vector<double> maxEigenVec{es.eigenvectors().col(maxIndex)(0).real(), es.eigenvectors().col(maxIndex)(1).real()};
            
            xi.sigma = sigma;
            xi.principalVec = maxEigenVec;
        }
    
        // set deltaX for each xi
        for(skelx::Point &xi: pointset){
            vector<double> deltaX{0.0, 0.0};
            double cosTheta;
            // compute cos<pV, ui>, namely cosTheta
            cosTheta = xi.ui[0] * xi.principalVec[0] + xi.ui[1] * xi.principalVec[1];   // numerator
            cosTheta /= (pow(pow(xi.principalVec[0], 2) + pow(xi.principalVec[1], 2), 0.5) + pow(pow(xi.ui[0], 2) + pow(xi.ui[1], 2), 0.5));

            if(cosTheta < 0){
                xi.principalVec[0] = -xi.principalVec[0];
                xi.principalVec[1] = -xi.principalVec[1];
                cosTheta = -cosTheta;
            }

            double uiMod = pow(pow(xi.ui[0], 2) + pow(xi.ui[1], 2), 0.5);
            deltaX[0] = uiMod * cosTheta * (1 - xi.sigma) * xi.principalVec[0] + xi.ui[0] - uiMod * cosTheta * xi.principalVec[0];
            deltaX[1] = uiMod * cosTheta * (1 - xi.sigma) * xi.principalVec[1] + xi.ui[1] - uiMod * cosTheta * xi.principalVec[1];

            xi.deltaX = deltaX;
        }
    }

}

