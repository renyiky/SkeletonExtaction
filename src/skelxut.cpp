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
#include <string>
#include <omp.h>

#include "skelxut.hpp"
#include "Point.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace skelx{
    int gaussCircleCount(const int r){
        int Nr = 0; // Number of pixels in Radius r
        int i = 0;
        int r2 = r * r;
        while(4 * i + 1 <= r2){
            Nr += r2 / (4 * i + 1) - r2 / (4 * i + 3);
            ++i;
        }
        return Nr * 4 + 1;
    }

    void visualize(const Mat &img, const vector<skelx::Point> pointset, const int iter){
        // note that BGR ! center:(255, 0, 0), KNNneighbor:(0, 255, 0)
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
            // cout << "Point" << count<< ": deltaX:[ " << p.deltaX[0] << " , " << p.deltaX[1] << " ]" << " cosTheta:[" << p.cosTheta << "] "
            //         << "ui:[ " << p.ui[0] << " , " << p.ui[1] << " ] sigma:[" <<p.sigma << "]"<< endl;

            // draw center point, red
            Mat KNNvisual = origin.clone();
            Mat_<Vec3b> _KNNvisual = KNNvisual;
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

    // initialize the pointset
    vector<struct skelx::Point> getPointsetInitialized(Mat &img){
        vector<struct skelx::Point> pointset;
        for(int i = 0; i < img.rows; ++i){
            for(int j = 0; j < img.cols; ++j){
                if(img.at<uchar>(i, j) != 0){
                    // skelx::Point p(i, j);
                    // p.pos[0] = i;
                    // p.pos[1] = j;
                    pointset.push_back(skelx::Point(i, j));
                }
            }
        }
        return pointset;
    }

    Mat drawNeighborGraph(const Mat &img, const vector<vector<double> > &neighbors, const Point &samplePixel){
        Mat ret = Mat::zeros(img.rows, img.cols, CV_8U);
        for(const vector<double> &p : neighbors){
            ret.at<uchar>(p[0], p[1]) = 255;
        }
        ret.at<uchar>(samplePixel.pos[0], samplePixel.pos[1]) = 255;
        return ret;
    }

    void BFS(const Mat &img, vector<vector<double>> &searchedNeighbors, const vector<double> &pos){
        double x = pos[0];
        double y = pos[1];
        vector<vector<double>> new_neighbors;
        vector<double> p1 = {x - 1, y - 1},
                p2 = {x - 1, y},
                p3 = {x - 1, y + 1},
                p4 = {x, y + 1},
                p5 = {x + 1, y + 1},
                p6 = {x + 1, y},
                p7 = {x + 1, y - 1},
                p8 = {x, y - 1};
        vector<vector<double> > neighbors = {p1, p2, p3, p4, p5, p6, p7, p8};

        for(vector<double> i : neighbors){
            if(i[0] >= 0 && i[0] < img.rows && i[1] >= 0 && i[1] < img.cols 
                && img.at<uchar>(i[0], i[1]) != 0 
                && find(searchedNeighbors.begin(), searchedNeighbors.end(), vector<double>{i[0], i[1]}) == searchedNeighbors.end()){
                searchedNeighbors.push_back({static_cast<double>(i[0]), static_cast<double>(i[1])});
                new_neighbors.push_back({static_cast<double>(i[0]), static_cast<double>(i[1])});
            }
        }
        
        for(auto v : new_neighbors) BFS(img, searchedNeighbors, v);
    }

    vector<vector<double>> repositionNeighbors(const Mat &img, const vector<vector<double>> &neighbors, const skelx::Point &center){
        vector<vector<double>> ret_neighbors{{center.pos[0], center.pos[1]}};
        BFS(img, ret_neighbors, {center.pos[0], center.pos[1]});
        ret_neighbors.erase(ret_neighbors.begin());
        return ret_neighbors;
    }

    // search k nearest neighbors
    // and do perturbation test
    bool setNeighborsOfK(Mat &img, skelx::Point &point, const int k, bool perturbationFlag){
        int radius = 0,
            rows = img.rows,
            cols = img.cols,
            x = point.pos[0],
            y = point.pos[1];
        vector<vector<double> > neighbors{};

        // compute the minumum value of radius containing k neighbors
        radius = sqrt(1. + 2. * k) / 2 - 0.5;

        while(neighbors.size() < k){
            neighbors = {};
            // circular search
            for(int i = -radius; i < radius + 1; ++i){
                for(int j = -radius; j < radius + 1; ++j){
                    if(pow((i * i + j * j), 0.5) <= radius && x + i >= 0 && x + i < img.rows && y + j >= 0 && y + j < img.cols && img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
                        neighbors.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
                    }
                }
            }
            ++radius;
        }

        if(perturbationFlag){
            double sumX = 0;
            double sumY = 0;
            for(auto i : neighbors){
                sumX += i[0] - x;
                sumY += i[1] - y;
            }
            if(sumX != 0 || sumY != 0){
                Mat neighborGraph = drawNeighborGraph(img, neighbors, point);
                Mat binImg, labels, stats, centroids;
                cv::threshold(neighborGraph, binImg, 0, 255, cv::THRESH_OTSU);
                if(cv::connectedComponentsWithStats (binImg, labels, stats, centroids) != 2){
                    neighbors = repositionNeighbors(neighborGraph, neighbors, point);
                }
            }
        }

        if(neighbors.size() != 0){
            point.neighbors = neighbors;
            return true;
        }
        else return false;
    }

    // draw points on a new img
    Mat draw(const Mat &src, vector<struct skelx::Point> &pointset){
        Mat ret = Mat::zeros(src.rows, src.cols, CV_8U);
        for(skelx::Point &p : pointset){
            if(p.pos[0] >= 0 && p.pos[0] < src.rows && p.pos[1] >= 0 && p.pos[1] < src.cols) 
                ret.at<uchar>(p.pos[0], p.pos[1]) = 255;
        }
        return ret;
    }

    // move points toward deltaX
    void movePoint(vector<skelx::Point> &pointset){
        for(skelx::Point &p : pointset){
            p.pos[0] += static_cast<int>(p.deltaX[0]);
            p.pos[1] += static_cast<int>(p.deltaX[1]);
        }
    }

    // set ui for each xi based on k nearest neighbors.
    // neighbors and ui of xi would be set
    void computeUi(Mat &img, vector<skelx::Point> &pointset, const int k, const bool perturbationFlag){
        for(skelx::Point &p : pointset){
            if(!setNeighborsOfK(img, p, k, perturbationFlag)) std::cout<<"neighbors insufficient!"<<endl;

            vector<double> ui{0.0, 0.0};
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
    // covMat, cosTheta and deltaX of xi would be set.
    // the parameter detailFactor which is set to 1.0 as default can control the degree of detail the algorithm would produce,
    // the larger the detailFactor, the more detail is included in the final result.
    void PCA(Mat &img, vector<skelx::Point> &pointset, double detailFactor){
        for(skelx::Point &xi: pointset){
        // #pragma omp parallel for
        // for(int i = 0; i < pointset.size(); ++i){
        //     skelx::Point &xi = pointset[i];
            if(xi.ui[0] == 0 && xi.ui[1] == 0){
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

            MatrixXd covMatEigen(2, 2);
            covMatEigen(0, 0) = covMat[0][0];
            covMatEigen(0, 1) = covMat[0][1];
            covMatEigen(1, 0) = covMat[1][0];
            covMatEigen(1, 1) = covMat[1][1];

            EigenSolver<MatrixXd> es(covMatEigen);
            int maxIndex;
            double lamda1 = es.eigenvalues().col(0)(0).real(), 
                    lamda2 = es.eigenvalues().col(0)(1).real();

            lamda1 > lamda2 ? maxIndex = 0 : maxIndex = 1;
            double sigma = es.eigenvalues().col(0)(maxIndex).real() / (lamda1 + lamda2);
            vector<double> maxEigenVec{es.eigenvectors().col(maxIndex)(0).real(), es.eigenvectors().col(maxIndex)(1).real()};

            // if(isnan(sigma)){
            //     xi.sigma = 0.5;
            //     std::cout<<"NaN sigma occured."<<endl;
            // }else{
            //     xi.sigma = sigma;
            // }
            xi.sigma = sigma;
            xi.principalVec = maxEigenVec;

            // compute cos<pV, ui>, namely cosTheta
            double cosTheta;
            cosTheta = xi.ui[0] * xi.principalVec[0] + xi.ui[1] * xi.principalVec[1];   // numerator
            cosTheta /= ((pow(pow(xi.principalVec[0], 2) + pow(xi.principalVec[1], 2), 0.5)) * (pow(pow(xi.ui[0], 2) + pow(xi.ui[1], 2), 0.5)));

            if(cosTheta < 0){
                xi.principalVec[0] = -xi.principalVec[0];
                xi.principalVec[1] = -xi.principalVec[1];
                cosTheta = -cosTheta;
            }
            xi.cosTheta = cosTheta;
            double uiMod = pow(pow(xi.ui[0], 2) + pow(xi.ui[1], 2), 0.5),
                    scale = 10.0,
                    jumpFunction = 2.0 / (1 + exp((xi.sigma - 0.7723) * (xi.sigma - 0.7723) * 1500.0)) + 1; // the 0.7723 comes from the mean of (0.755906 + 0.7875 + 0.773625) which are referred to 3 diffenrent rectangle conditions

            xi.deltaX[0] = xi.ui[0] * std::exp(- pow(cosTheta, 2.0) * detailFactor * scale) * jumpFunction;
            xi.deltaX[1] = xi.ui[1] * std::exp(- pow(cosTheta, 2.0) * detailFactor * scale) * jumpFunction;
        }
    }

    // compute k
    int computeK(Mat &img){
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

    // check if the parameter pos is included in keypoints
    bool isKeyPos(vector<skelx::Point > &keyPointset, vector<int> pos){
        for(auto &i : keyPointset){
            if(i.pos[0] == pos[0] && i.pos[1] == pos[1]) return true;
        }
        return false;
    }

    // Check that if we delete this point, the connectivity of the graph will change.
    bool isRemovable(Mat &img, vector<int> pos){
        Mat bw, ret, labelImage;
        ret = img.clone();
        // try to remove it
        ret.at<uchar>(pos[0], pos[1]) = 0;
        cv::threshold(ret, bw, 0, 255, THRESH_BINARY);
        if(connectedComponents(bw, labelImage) != 2) return false;
        else return true;
    }

    // for further thinning
    Mat postProcess(Mat &img, const double detailFactor, const int k, const bool perturbationFlag){
        // remove isolate points
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

        vector<skelx::Point> pointset = skelx::getPointsetInitialized(img);
        skelx::computeUi(img, pointset, k, perturbationFlag);
        skelx::PCA(img, pointset, detailFactor);

        vector<skelx::Point> keypointset;   // store keypoints which shall not be removed
        for(auto &i : pointset){
            if(i.cosTheta >= 0.8 && (abs(i.ui[0]) >= 1.0 || abs(i.ui[1]) >= 1.0)) keypointset.push_back(i);
        }

        // find centroids of the clusters of keypoints
        Mat keyMap = draw(img, keypointset);
        Mat binImg, labels, stats, centroids;
        cv::threshold(keyMap, binImg, 0, 255, cv::THRESH_OTSU);
        cv::connectedComponentsWithStats (binImg, labels, stats, centroids);
        keypointset = {};   // clear keypointset for accpetance for centroids
        for(int i = 1; i < centroids.rows; ++i){    // exclude the background label
                keypointset.push_back(Point(static_cast<int>(centroids.at<double>(i, 1)), static_cast<int>(centroids.at<double>(i, 0))));   // note that the generated points need to swap positions
        }

        // ui based thinning algorithm
        sort(pointset.begin(), pointset.end(), 
            [](Point &p1, Point &p2)->bool{
                return sqrt(p1.ui[0] * p1.ui[0] + p1.ui[1] * p1.ui[1]) > sqrt(p2.ui[0] * p2.ui[0] + p2.ui[1] * p2.ui[1]);
            });

        for(Point &p : pointset){
            if(!isKeyPos(keypointset, {static_cast<int>(p.pos[0]), static_cast<int>(p.pos[1])}) && isRemovable(img, {static_cast<int>(p.pos[0]), static_cast<int>(p.pos[1])})){
                img.at<uchar>(p.pos[0], p.pos[1]) = 0;
            }
        }
        return img;
    }
}

// Mat postProcess(Mat &img, const double detailFactor, const int k, const bool perturbationFlag){
//         // remove isolate points
//         // for(int x = 0; x < img.rows; ++x){
//         //     for(int y = 0; y < img.cols; ++y){
//         //         if(img.at<uchar>(x, y) != 0){
//         //             int flag = 0;
//         //             for(int i = -1; i < 2 && flag == 0; ++i){
//         //                 for(int j = -1; j < 2 && flag == 0; ++j){
//         //                     if(x + i >= 0 && x + i < img.rows && y + j >= 0 && y + j < img.cols && img.at<uchar>(x + i, y + j) != 0 && !(i == 0 && j == 0)){
//         //                         flag = 1;
//         //                         break;
//         //                     }
//         //                 }
//         //             }
//         //             if(flag == 0){
//         //                 img.at<uchar>(x, y) = 0;
//         //             }
//         //         }
//         //     }
//         // }



//         bool flag = true;
//         int t = 0;
//         while(flag){
            
//             vector<skelx::Point> pointset = skelx::getPointsetInitialized(img);
//             skelx::computeUi(img, pointset, k, perturbationFlag);
//             skelx::PCA(img, pointset, detailFactor);
//             vector<skelx::Point> keypointset;   // store keypoints which shall not be removed
//             for(auto &i : pointset){
//                 if(i.cosTheta >= 0.8 && (abs(i.ui[0]) >= 1.0 || abs(i.ui[1]) >= 1.0)) keypointset.push_back(i);
//             }

//             // find centroids of the clusters of keypoints
//             Mat keyMap = draw(img, keypointset);
//             Mat binImg, labels, stats, centroids;
//             cv::threshold(keyMap, binImg, 0, 255, cv::THRESH_OTSU);
//             cv::connectedComponentsWithStats (binImg, labels, stats, centroids);
//             keypointset = {};   // clear keypointset for accpetance for centroids
//             for(int i = 1; i < centroids.rows; ++i){    // exclude the background label
//                     keypointset.push_back(Point(static_cast<int>(centroids.at<double>(i, 1)), static_cast<int>(centroids.at<double>(i, 0))));   // note that the generated points need to swap positions
//             }


//             flag = false;
//             sort(pointset.begin(), pointset.end(), 
//                 [](Point &p1, Point &p2)->bool{
//                     return sqrt(p1.ui[0] * p1.ui[0] + p1.ui[1] * p1.ui[1]) > sqrt(p2.ui[0] * p2.ui[0] + p2.ui[1] * p2.ui[1]);
//                 });

//             for(Point &p : pointset){
//                 if(sqrt(p.ui[0] * p.ui[0] + p.ui[1] * p.ui[1]) == 0) break;
//                 if(!isKeyPos(keypointset, {static_cast<int>(p.pos[0]), static_cast<int>(p.pos[1])}) && isRemovable(img, {static_cast<int>(p.pos[0]), static_cast<int>(p.pos[1])})){
//                     img.at<uchar>(p.pos[0], p.pos[1]) = 0;
//                     flag = true;
//                 }
//             }

//             imwrite("results/" + to_string(t++) + ".png", img);
//             // pointset = skelx::getPointsetInitialized(img);
            
//             // skelx::computeUi(img, pointset, k, perturbationFlag);
//             // skelx::PCA(img, pointset, detailFactor);
//         }
//         // ui based thinning algorithm
//         return img;
//     }
// }