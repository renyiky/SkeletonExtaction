#include <opencv2/core.hpp>
#include <vector>

using namespace std;
using namespace cv;

Mat AWalg(Mat img){

}

int isSatisfied(Mat &img, vector<int> pos){
    int x = pos[0],
        y = pos[1];
    vector<int> p1 = {x - 1, y -1},
            p2 = {x - 1, y},
            p3 = {x - 1, y + 1},
            p4 = {x, y + 1},
            p5 = {x + 1, y + 1},
            p6 = {x + 1, y},
            p7 = {x + 1, y - 1},
            p8 = {x, y - 1};
    vector<vector<int> > neighbors = {p1, p2, p3, p4, p5, p6, p7, p8};
    vector<int> neighborsValue = {};
    
    for(vector<int> i : neighbors){
        if(i[0] >= 0 && i[0] < img.rows && i[1] >= 0 && i[1] < img.cols){
            neighborsValue.push_back(img.at<uchar>(i[0], i[1]) == 0 ? 0 : 1);
        }else{
            neighborsValue.push_back(0);
        }
    }

    // here below are templates
    vector<int> rule1 = {1, 2, 0, 0, 2, 1, 1, 1},
                rule2 = 


}