#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>

#include "getPointsetInitialized.hpp"
#include "Point.hpp"
#include "setNeighborsOfK.hpp"
#include "getD3nn.hpp"
#include "updateK.hpp"
#include "visualize.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace skelx{

    Mat draw(Mat src, vector<struct skelx::Point> pointset){
        int rows = src.rows, cols = src.cols;
        Mat ret = Mat::zeros(rows, cols, CV_8U);
        for(skelx::Point p : pointset){
            if(p.pos[0] >= 0 && p.pos[0] < rows && p.pos[1] >= 0 && p.pos[1] < cols){
                ret.at<uchar>(p.pos[0],p.pos[1]) = 255;
            }
        }
        return ret;
    }

    // remove duplicates which result from moving point, and refresh d3nn for each point
    void refreshPointset(Mat &img, vector<skelx::Point> &pointset){
        unsigned int count = 0;

        while(count < pointset.size()){
            skelx::Point temp = pointset[count];
            unsigned int i = count + 1;
            while (i < pointset.size())
            {
                if(pointset[i].pos[0] == temp.pos[0] && pointset[i].pos[1] == temp.pos[1]){
                    pointset.erase(pointset.begin() + i);
                }else{
                    ++i;
                }
            }
            ++count;
        }

        img = skelx::draw(img, pointset);
        for(skelx::Point &p : pointset){
            p.d3nn = getD3nn(img, p);
        }
    }

    // move points toward deltaX
    // only move the point whose sigma is less than the threshold
    void movePoint(vector<skelx::Point> &pointset, double threshold){
        for(skelx::Point &p : pointset){
            if(p.sigma < threshold){
                p.pos[0] += static_cast<int>(p.deltaX[0]);
                p.pos[1] += static_cast<int>(p.deltaX[1]);
            }
        }
    }

    // set ui for each xi based on p.k,
    // neighbors and ui of xi would be set
    void computeUi(Mat &img, vector<skelx::Point> &pointset, double threshold){
        for(skelx::Point &p : pointset){
            if(p.sigma > threshold){
                continue;
            }
            // get k nearest neighbors
            vector<double> ui{0.0, 0.0};

            if(!setNeighborsOfK(img, p, p.k)){  // set ui neighbors
                std::cout<<"neighbors insufficient!"<<endl;
            }
            for(vector<double> nei: p.neighbors){
                // if it's connected in the same domain with center point
                // if(nei[2] == 0){
                ui[0] += (nei[0] - p.pos[0]);
                ui[1] += (nei[1] - p.pos[1]);
                // }else{
                //     // if not, decrease its influence to center point
                //     double distance = pow((nei[0] - p.pos[0]) * (nei[0] - p.pos[0]) + (nei[1] - p.pos[1]) * (nei[1] - p.pos[1]), 0.5),
                //             influenceRatio = exp(-(distance - 0.5) * (distance - 0.5) / 2);
                //     ui[0] += ((nei[0] - p.pos[0]) * influenceRatio);
                //     ui[1] += ((nei[1] - p.pos[1]) * influenceRatio);
                // }
            }
            ui[0] = ui[0] / static_cast<double>(p.neighbors.size());
            ui[1] = ui[1] / static_cast<double>(p.neighbors.size());
            p.ui = {ui[0], ui[1]};

            // compute the angle of current cluster of pixels, namely alpha, represented by its cosAlpha
            double minCosAlpha = 2.0;
            for(vector<double> neighbor : p.neighbors){
                vector<double> tempVec = {neighbor[0] - p.pos[0], neighbor[1] - p.pos[1]};
                // calculate the angle by the method of arccosTheta,
                // so calculate the inner product firstly
                double innerProduct = tempVec[0] * p.ui[0] + tempVec[1] * p.ui[1],
                        cosAlpha = innerProduct / (pow(tempVec[0] * tempVec[0] + tempVec[1] * tempVec[1], 0.5) * pow(p.ui[0] * p.ui[0] + p.ui[1] * p.ui[1], 0.5));
                minCosAlpha = minCosAlpha < cosAlpha ? minCosAlpha : cosAlpha;
            }
            p.cosAlpha = minCosAlpha;
            // cout<<p.pos[0]<<"  "<<p.pos[1]<<"  "<<p.cosAlpha<<endl;
            // if(isnan(p.cosAlpha)){
            //     cout<<"ui: "<<p.ui[0]<<"  "<<p.ui[1]<<endl;
            // }
        }
    }

    // pca process, after this, the sigma, principalVec, 
    // PCAneighbors, covMat and deltaX of xi would be set
    // only do PCA for the point whose sigma is less than the @param threshold
    void PCA(Mat &img, vector<skelx::Point> &pointset, double threshold){

        // get and set covMat for each xi
        for(skelx::Point &xi: pointset){
            if(xi.sigma > threshold){
                continue;
            }
            // get PCA neighbors which is in 3 * d3nn distance
            double dnn = 2 * 4 * xi.d3nn, // 9 times of d3nn
                    x = xi.pos[0],
                    y = xi.pos[1];
            xi.PCAneighbors = {};

            for(int i = -dnn; i < dnn + 1; ++i){
                for(int j = -dnn; j < dnn + 1; ++j){
                    if(pow((i * i + j * j), 0.5) <= dnn && x + i >= 0 && x + i < img.rows && y + j >= 0 && y + j < img.cols && img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
                        xi.PCAneighbors.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
                    }
                }
            }

            // calculate center point, namely xi
            vector<double> centerPoint{0.0, 0.0};
            for(vector<double> &xj: xi.PCAneighbors){
                centerPoint[0] += xj[0];
                centerPoint[1] += xj[1];
            }
            centerPoint[0] /= static_cast<double>(xi.PCAneighbors.size());
            centerPoint[1] /= static_cast<double>(xi.PCAneighbors.size());

            vector<vector<double> > covMat(2, vector<double>(2,0.0));  // create cov Matrix
            for(vector<double> &xj: xi.PCAneighbors){
                vector<double> xixj = {xj[0] - centerPoint[0], xj[1] - centerPoint[1]};

                covMat[0][0] += xixj[0] * xixj[0];
                covMat[0][1] += xixj[0] * xixj[1];
                covMat[1][0] += xixj[1] * xixj[0];
                covMat[1][1] += xixj[1] * xixj[1];
            }

            covMat[0][0] /= xi.PCAneighbors.size();
            covMat[0][1] /= xi.PCAneighbors.size();
            covMat[1][0] /= xi.PCAneighbors.size();
            covMat[1][1] /= xi.PCAneighbors.size();

            if(isnan(covMat[0][0])){
                std::cout<<"NaN covMat occured."<<endl;
            }
            xi.covMat = covMat;
        }
        
        // get eigen value and eigen vectors, and set sigma, principalVec for each xi
        for(skelx::Point &xi: pointset){
            if(xi.sigma > threshold){
                continue;
            }
            MatrixXd covMatEigen(2, 2);
            covMatEigen(0, 0) = xi.covMat[0][0];
            covMatEigen(0, 1) = xi.covMat[0][1];
            covMatEigen(1, 0) = xi.covMat[1][0];
            covMatEigen(1, 1) = xi.covMat[1][1];

            EigenSolver<MatrixXd> es(covMatEigen);
            int maxIndex;
            double lamda1 = es.eigenvalues().col(0)(0).real(), 
                    lamda2 = es.eigenvalues().col(0)(1).real();

            lamda1 > lamda2 ? maxIndex = 0 : maxIndex = 1;
            double sigma = es.eigenvalues().col(0)(maxIndex).real() / (lamda1 + lamda2);
            vector<double> maxEigenVec{es.eigenvectors().col(maxIndex)(0).real(), es.eigenvectors().col(maxIndex)(1).real()};
            
            if(isnan(sigma)){
                xi.sigma = 0.5;
                std::cout<<"NaN sigma occured."<<endl;
            }else{
                xi.sigma = sigma;
            }
            xi.principalVec = maxEigenVec;
        }
    
        // set deltaX for each xi
        for(skelx::Point &xi: pointset){
            if(xi.sigma > threshold){
                continue;
            }
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
            // if(!isnan(xi.cosAlpha)){
            deltaX[0] = xi.ui[0] - uiMod * cosTheta * xi.sigma * xi.principalVec[0];    // deltaX[0] = max(xi.ui[0] - uiMod * cosTheta * xi.sigma * xi.principalVec[0], xi.ui[0] - uiMod * cosTheta * xi.sigma * xi.principalVec[0] * 2 * std::exp(-(xi.cosAlpha - 1) * (xi.cosAlpha - 1)));
            deltaX[1] = xi.ui[1] - uiMod * cosTheta * xi.sigma * xi.principalVec[1];    // deltaX[1] = max(xi.ui[1] - uiMod * cosTheta * xi.sigma * xi.principalVec[1], xi.ui[1] - uiMod * cosTheta * xi.sigma * xi.principalVec[1] * 2 * std::exp(-(xi.cosAlpha - 1) * (xi.cosAlpha - 1)));
            // }else{
            //     deltaX[0] = 0;
            //     deltaX[1] = 0;
            // }
            // cout<<xi.neighbors.size()<<"   "<<xi.PCAneighbors.size()<<endl;
            xi.deltaX = deltaX;
        }
    }

    // remove isolate point, namely noise
    void postProcess(Mat &img){
        for(int x = 0; x< img.rows; ++x){
            for(int y = 0; y < img.cols; ++y){
                if(img.at<uchar>(x, y) != 0){
                    int flag = 0;
                    for(int i = -1; (i < 2) && (flag == 0); ++i){
                        for(int j = -1; (j < 2) && (flag == 0); ++j){
                            if(x + i >= 0 && x + i < img.rows && y + j >= 0 && y + j < img.cols && img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
                                flag = 1;
                                break;
                            }
                        }
                    }
                    if(flag == 0){
                        img.at<uchar>(x, y) = 0;
                    }
                }
            }
        }
    }
}

Mat contract(Mat img, string filename){
    double sigmaHat = 0.0, preSigmaHat = sigmaHat;
    int count = 0,  // count if sigmaHat remains unchanged
        t = 0;  // times of iterations
    vector<skelx::Point> pointset = getPointsetInitialized(img);    // set coordinates, k0, d3nn

    while(sigmaHat < 0.95){
        skelx::computeUi(img, pointset, 0.95);
        skelx::PCA(img, pointset, 0.95);

        // if(t % 10 == 0 && t != 0){
        //     visualize(img, pointset, t);
        // }

        skelx::movePoint(pointset, 0.95);
        skelx::refreshPointset(img, pointset);
        for(skelx::Point &p : pointset){
            sigmaHat += p.sigma;
        }
        sigmaHat /= pointset.size();

        updateK(img, pointset, t + 1, 1);

        imwrite("results/" + to_string(t + 1) + "_" + filename + ".png", img);
        std::cout<<"iter:"<<t + 1<<"   sigmaHat = "<<sigmaHat<<endl;
        ++t;

        // check if sigmaHat remains unchanged
        // if it doesn't change for 3 times, stop extracting
        if(sigmaHat == preSigmaHat){
            if(count == 2){
                skelx::postProcess(img);
                return img;
            }else{
                ++count;
            }
        }else{
            preSigmaHat = sigmaHat;
            count = 0;
        }
    }
    skelx::postProcess(img);
    return img;
}