#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <map>

#include "skelxut.hpp"
#include "Point.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace skelx{

    void visualize(const Mat &img, const vector<skelx::Point> pointset, int iter){
        // note that BGR ! center:(29, 147, 248), KNNneighbor:(124, 191, 160), PCAneighbor:(130, 232, 255)
        int count = 0;
        Mat origin(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));

        Mat_<Vec3b> _origin = origin;
        // set origin image
        for(skelx::Point p : pointset){
            _origin(p.pos[0], p.pos[1])[0] = 255;
            _origin(p.pos[0], p.pos[1])[1] = 255;
            _origin(p.pos[0], p.pos[1])[2] = 255;
        }
        origin = _origin;

        for(skelx::Point p : pointset){
            cout << "Point" << count<< ": deltaX:[ " << p.deltaX[0] << " , " << p.deltaX[1] << " ]" << " cosTheta:[" << p.cosTheta << "] "
                    << "ui:[ " << p.ui[0] << " , " << p.ui[1] << " ] sigma:[" <<p.sigma << "]"<< endl;
            Mat KNNvisual = origin.clone();
            Mat_<Vec3b> _KNNvisual = KNNvisual;

            // draw center point, red
            _KNNvisual(p.pos[0], p.pos[1])[0] = 0;
            _KNNvisual(p.pos[0], p.pos[1])[1] = 0;
            _KNNvisual(p.pos[0], p.pos[1])[2] = 255;
            KNNvisual = _KNNvisual;

            // draw KNN neighbors, green
            for(vector<double> nei : p.neighbors){
                _KNNvisual(nei[0], nei[1])[0] = 0;
                _KNNvisual(nei[0], nei[1])[1] = 255;
                _KNNvisual(nei[0], nei[1])[2] = 0;
            }
            KNNvisual = _KNNvisual;
            imwrite("results/visualization/iter" + to_string(iter) + "_" + to_string(count) + "_KNN.png", KNNvisual);
            ++count;
        }
    }

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
            // set the upper limit of K 
            p.k = 10 < neighborsCount.size() ? upperLimit : neighborsCount.size();
        }
    }

    // get d3nn of param point
    // d3nn is the average length of edges in the 3-nearest-neighbors graph
    double getD3nn(Mat &img, const struct skelx::Point &point){
        int x = point.pos[0], 
            y = point.pos[1],
            rows = img.rows,
            cols = img.cols;
        int radius = 0;
        vector<double> neighborsRadius = {};

        // circular search
        while(neighborsRadius.size() < 3){
            ++radius;
            neighborsRadius = {};
            for(double i = -radius; i < radius + 1; ++i){
                for(double j = -radius; j < radius + 1; ++j){
                    if(pow((i * i + j * j), 0.5) <= radius && x + i >= 0 && x + i < img.rows && y + j >= 0 && y + j < img.cols && img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
                        neighborsRadius.push_back(pow((i * i + j * j), 0.5));
                    }
                }
            }
        }
        sort(neighborsRadius.begin(), neighborsRadius.end());
        return (neighborsRadius[0] + neighborsRadius[1] + neighborsRadius[2]) / 3;   // /3
    }
    
    // initialize the pointset
    vector<struct skelx::Point> getPointsetInitialized(Mat &img){
        vector<struct skelx::Point> pointset;
        for(int i = 0; i < img.rows; ++i){
            for(int j = 0; j < img.cols; ++j){
                if(img.at<uchar>(i, j) != 0){
                    skelx::Point p;
                    p.pos[0] = i;
                    p.pos[1] = j;
                    pointset.push_back(p);
                }
            }
        }
        // set k0
        int num = pointset.size();
        for(struct skelx::Point &p : pointset){
            p.d3nn = getD3nn(img, p);
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
            p.k = neighborsCount.size();
        }
        return pointset;
    }

    int setNeighborsOfK(Mat &img, skelx::Point &point, const int k){
        int radius = 0,
            rows = img.rows,
            cols = img.cols,
            x = point.pos[0],
            y = point.pos[1];
        vector<vector<double> > neighbors{};

        while(neighbors.size() < k){
            ++radius;
            neighbors = {};
            // circular search
            for(int i = -radius; i < radius + 1; ++i){
                for(int j = -radius; j < radius + 1; ++j){
                    if(pow((i * i + j * j), 0.5) <= radius && x + i >= 0 && x + i < img.rows && y + j >= 0 && y + j < img.cols && img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
                        neighbors.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
                    }
                }
            }
        }

        if(neighbors.size() != 0){
            point.neighbors = neighbors;
            return 1;
        }
        else{
            return 0;
        }
    }

    Mat draw(const Mat &src, vector<struct skelx::Point> &pointset){
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
        map<vector<double>, skelx::Point> dataset;
        for(skelx::Point p : pointset){
            dataset[p.pos] = p;
        }
        pointset.clear();
        for(map<vector<double>, skelx::Point>::iterator iter = dataset.begin(); iter != dataset.end(); ++iter){
            pointset.push_back(iter->second);
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
                ui[0] += (nei[0] - p.pos[0]);
                ui[1] += (nei[1] - p.pos[1]);
            }
            ui[0] = ui[0] / static_cast<double>(p.neighbors.size());
            ui[1] = ui[1] / static_cast<double>(p.neighbors.size());
            p.ui = {ui[0], ui[1]};
        }
    }

    // pca process, after this, the sigma, principalVec, 
    // PCAneighbors, covMat, cosTheta and deltaX of xi would be set.
    // only do PCA for the point whose sigma is less than the parameter threshold.
    // the parameter detailFactor which is set to 10.0 as default can control the degree of detail the algorithm would produce,
    // the larger the detailFactor, the more details the skeleton would have.
    void PCA(Mat &img, vector<skelx::Point> &pointset, double threshold, double detailFactor){

        for(skelx::Point &xi: pointset){
            if(xi.sigma > threshold){
                continue;
            }else if(xi.ui[0] == 0 && xi.ui[1] == 0){
                xi.deltaX = {0, 0};
                xi.cosTheta = 0.0;
                continue;
            }
            // calculate center point, namely xi
            vector<double> centerPoint{0.0, 0.0};
            for(vector<double> &xj: xi.neighbors){
                centerPoint[0] += xj[0];
                centerPoint[1] += xj[1];
            }
            centerPoint[0] /= static_cast<double>(xi.neighbors.size());
            centerPoint[1] /= static_cast<double>(xi.neighbors.size());

            vector<vector<double> > covMat(2, vector<double>(2, 0.0));  // create cov Matrix
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

            if(isnan(covMat[0][0])){
                std::cout<<"NaN covMat occured."<<endl;
            }
            xi.covMat = covMat;

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

            vector<double> deltaX{0.0, 0.0};
            double cosTheta;
            // compute cos<pV, ui>, namely cosTheta
            cosTheta = xi.ui[0] * xi.principalVec[0] + xi.ui[1] * xi.principalVec[1];   // numerator
            cosTheta /= ((pow(pow(xi.principalVec[0], 2) + pow(xi.principalVec[1], 2), 0.5)) * (pow(pow(xi.ui[0], 2) + pow(xi.ui[1], 2), 0.5)));

            if(cosTheta < 0){
                xi.principalVec[0] = -xi.principalVec[0];
                xi.principalVec[1] = -xi.principalVec[1];
                cosTheta = -cosTheta;
            }
            xi.cosTheta = cosTheta;
            double uiMod = pow(pow(xi.ui[0], 2) + pow(xi.ui[1], 2), 0.5),
                    jumpFunction = 2.0 / (1 + exp((xi.sigma - 0.7723) * (xi.sigma - 0.7723) * 1500.0)) + 1; // the 0.7723 comes from the mean of (0.755906 + 0.7875 + 0.773625) which are referred to 3 diffenrent rectangle conditions

            deltaX[0] = xi.ui[0] * std::exp(- (cosTheta * cosTheta) * detailFactor) * jumpFunction;
            deltaX[1] = xi.ui[1] * std::exp(- (cosTheta * cosTheta) * detailFactor) * jumpFunction;
            xi.deltaX = deltaX;
        }
    }

    // set the upper limit of p.k
    int setUpperLimitOfK(Mat &img){
        double left = img.cols + 1,
            right = -1,
            up = img.rows + 1,
            down = -1;
        for(int i = 0; i < img.rows; ++i){
            for(int j = 0; j < img.cols; ++j){
                if(img.at<uchar>(i, j) != 0){
                    left = left < j ? left : j;
                    right = right > j ? right : j;
                    up = up < i ? up : i;
                    down = down > i ? down : i;
                }
            }
        }
        return static_cast<int>(sqrt((right - left) * (down - up)) / 10);
    }

    bool isKeyPos(vector<vector<int> > &keyPointPos, vector<int> pos){
        for(auto &i : keyPointPos){
            if(i[0] == pos[0] && i[1] == pos[1]) return true;
        }
        return false;
    }

    bool isRemovable(Mat &img, vector<int> pos){
        Mat bw, ret, labelImage;
        ret = img.clone();
        // try to remove it
        ret.at<uchar>(pos[0], pos[1]) = 0;
        cv::threshold(ret, bw, 0, 255, THRESH_BINARY);
        if(connectedComponents(bw, labelImage) != 2) return false;
        else return true;
    }

    // thin the raw skeleton
    // remove points whose cosTheta is less than the paramether threshold
    // return the new final image
    Mat thin(Mat &img, vector<skelx::Point> &pointset, float threshold){
        vector<vector<int> > keyPointPos;  // store the positions of key points which shall not be removed
        vector<skelx::Point> keyPointSet;
        for(auto &i : pointset){
            if(i.cosTheta >= threshold && (abs(i.pos[0]) > 10e-3 || abs(i.pos[1]) > 10e-3)){
                keyPointSet.push_back(i);
                keyPointPos.push_back({static_cast<int>(i.pos[0]), static_cast<int>(i.pos[1])});
            
            }
        }

        Mat ret = img.clone();
        bool flag = true; // flag to show if there is no point can be removed
        while(flag){
            flag = false;


            // from right to left
            for(int x = 0; x < ret.rows; ++x){
                for(int y = ret.cols - 1; y >= 0; --y){
                    if(ret.at<uchar>(x, y) != 0 && !isKeyPos(keyPointPos, {x, y}) && isRemovable(ret, {x, y})){
                        ret.at<uchar>(x, y) = 0;
                        flag = true;
                        break;
                    }
                }
            }

            // from left to right
            for(int x = 0; x < ret.rows; ++x){
                for(int y = 0; y < ret.cols; ++y){
                    if(ret.at<uchar>(x, y) != 0 && !isKeyPos(keyPointPos, {x, y}) && isRemovable(ret, {x, y})){
                        ret.at<uchar>(x, y) = 0;
                        flag = true;
                        break;
                    }
                }
            }
            // from bottom to top
            for(int y = 0; y < ret.cols; ++y){
                for(int x = ret.rows - 1; x >= 0; --x){
                    if(ret.at<uchar>(x, y) != 0 && !isKeyPos(keyPointPos, {x, y}) && isRemovable(ret, {x, y})){
                        ret.at<uchar>(x, y) = 0;
                        flag = true;
                        break;
                    }
                }
            }
            // from top to bottom
            for(int y = 0; y < ret.cols; ++y){
                for(int x = 0; x < ret.rows; ++x){
                    if(ret.at<uchar>(x, y) != 0 && !isKeyPos(keyPointPos, {x, y}) && isRemovable(ret, {x, y})){
                        ret.at<uchar>(x, y) = 0;
                        flag = true;
                        break;
                    }
                }
            }


            

        }
        return ret;
        // visualize(img, keyPointSet, 0);
        // return draw(img, keyPointSet);
    }

    // remove isolate point, namely noise,
    // and fill one-pixel holes
    Mat postProcess(Mat &img, const double detailFactor, const double thinningFactor, vector<skelx::Point> pointSet){
        // remove isolate point

        

        for(int x = 0; x < img.rows; ++x){
            for(int y = 0; y < img.cols; ++y){
                if(img.at<uchar>(x, y) != 0){
                    int flag = 0;
                    for(int i = -1; i < 2 && flag == 0; ++i){
                        for(int j = -1; j < 2 && flag == 0; ++j){
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

        // fill one-pixel holes
        for(int x = 0; x < img.rows; ++x){
            for(int y = 0; y < img.cols; ++y){
                if(img.at<uchar>(x, y) == 0){
                    // 4-neighbors considered
                    vector<int> up = {x, y - 1},
                                down = {x , y + 1},
                                right = {x + 1, y},
                                left = {x - 1, y};
                    vector<vector<int> > fourNeighbors = {up, down, right, left};
                    int sum = 0;
                    for(vector<int> &p : fourNeighbors){
                        if(p[0] >= 0 && p[0] < img.rows && p[1] >= 0 && p[1] < img.cols && img.at<uchar>(p[0], p[1]) == 255){
                            sum += 1;
                        }
                    }
                    if(sum == 4){
                        img.at<uchar>(x, y) = 255;
                    }
                }
            }
        }
        
        // vector<skelx::Point> pointset = getPointsetInitialized(img);
        // computeUi(img, pointset, 1.0);
        // PCA(img, pointset, 1.0, detailFactor);

        // visualize(img, pointset, 0);
        return thin(img, pointSet, thinningFactor);
        // return img;
    }
}