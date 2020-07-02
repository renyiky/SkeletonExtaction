#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

#include "Point.hpp"
#include "visualize.hpp"

using namespace std;
using namespace cv;

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
        // if(count % 10 != 0){
        if(p.pos[0] > 100){
            ++count;
            continue;
        }
        Mat KNNvisual = origin.clone();
        Mat_<Vec3b> _KNNvisual = KNNvisual;
        // draw center point, red
        _KNNvisual(p.pos[0], p.pos[1])[0] = 0;
        _KNNvisual(p.pos[0], p.pos[1])[1] = 0;
        _KNNvisual(p.pos[0], p.pos[1])[2] = 255;
        KNNvisual = _KNNvisual;
        Mat PCAvisual = KNNvisual.clone();
        Mat_<Vec3b> _PCAvisual = PCAvisual;
        // draw KNN neighbors, green
        for(vector<double> nei : p.neighbors){
            _KNNvisual(nei[0], nei[1])[0] = 0;
            _KNNvisual(nei[0], nei[1])[1] = 255;
            _KNNvisual(nei[0], nei[1])[2] = 0;
        }
        KNNvisual = _KNNvisual;
        // draw PCA neighbors, blue
        for(vector<double> nei : p.PCAneighbors){
            _PCAvisual(nei[0], nei[1])[0] = 202;
            _PCAvisual(nei[0], nei[1])[1] = 188;
            _PCAvisual(nei[0], nei[1])[2] = 62;
        }
        PCAvisual = _PCAvisual;
        imwrite("results/visualization/iter" + to_string(iter) + "_" + to_string(count) + "_KNN.png", KNNvisual);
        imwrite("results/visualization/iter" + to_string(iter) + "_" + to_string(count) + "_PCA.png", PCAvisual);
        ++count;
    }
}