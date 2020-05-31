#include <opencv2/core.hpp>

#include <vector>

#include "getPointsetInitialized.hpp"
#include "Point.hpp"


using namespace std;
using namespace cv;

unsigned int contract(Mat &img, string filename){
    vector<skelx::Point> pointset = getPointsetInitialized(img);    // set coordinates, k0, d3nn


}




void computeUi(Mat &img, vector<skelx::Point> &pointset){
    for(skelx::Point i : pointset){




    }
}