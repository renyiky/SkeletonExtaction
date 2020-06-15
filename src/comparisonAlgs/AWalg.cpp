#include <opencv2/core.hpp>
#include <vector>

using namespace std;
using namespace cv;

int isRulesSatisfied(Mat &img, vector<int> pos);

// here below are templates, namely rules
vector<int> rule1 = {1, 2, 0, 0, 2, 1, 1, 1},
            rule2 = {1, 1, 2, 0, 0, 2, 1, 1},
            rule3 = {1, 1, 1, 1, 2, 0, 0, 2},
            rule4 = {1, 1, 1, 2, 0, 0, 2, 1},
            rule5 = {1, 2, 0, 0, 0, 0, 2, 1},
            rule6 = {1, 1, 2, 0, 0, 0, 0, 2},
            rule7 = {1, 1, 1, 0, 1, 1, 1, 1},
            rule8 = {1, 1, 1, 1, 1, 0, 1, 1},
            rule9 = {2, 0, 0, 0, 0, 2, 1, 1},
            rule10 = {0, 0, 0, 0, 2, 1, 1, 2},
            rule11 = {2, 1, 1, 2, 0, 0, 0, 0},
            rule12 = {0, 2, 1, 1, 2, 0, 0, 0},
            rule13 = {0, 0, 0, 2, 1, 1, 2, 0},
            rule14 = {0, 0, 2, 1, 1, 2, 0, 0},
            rule15 = {1, 1, 1, 1, 1, 1, 1, 0},
            rule16 = {1, 0, 1, 1, 1, 1, 1, 1},
            rule17 = {0, 2, 1, 1, 1, 1, 2, 0},
            rule18 = {2, 1, 1, 1, 1, 2, 0, 0},
            rule19 = {0, 0, 2, 1, 1, 1, 1, 2},
            rule20 = {2, 0, 0, 2, 1, 1, 1, 1};
vector<vector<int> > rules = {rule1, rule2, rule3, rule4, rule5,
                                rule6, rule7, rule8, rule9, rule10,
                                rule11, rule12, rule13, rule14, rule15,
                                rule16, rule17, rule18, rule19, rule20};

// Implementation of AW algrithm.
// Here are three conditions concerned:
// (a): Does the pixel belong to one of the following two cases:...
// (b): Does the pixel belong to an extremity of a zigzag diagonal line
// (c): if condition a and b are both NOT satisfied, we apply the 20 rules
Mat AWalg(Mat img){
    int count = 1;
    Mat ret = img.clone();

    while(count != 0){
        count = 0;
        for(int i = 0; i < img.rows; ++i){
            for(int j = 0; j < img.cols; ++j){
                if(img.at<uchar>(i, j) != 0 && isRulesSatisfied(img, {i, j})){
                    ret.at<uchar>(i, j) = 0;
                    ++count;
                }
            }
        }
        img = ret.clone();
    }
    return ret;
}

int isRulesSatisfied(Mat &img, vector<int> pos){
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

    // check if neighborsValue meets one of the rules
    for(vector<int> rule : rules){
        int count = 0;
        for(int i = 0; i < rules.size(); ++i){
            if(rule[i] == 2){
                ++count;
            }else if(rule[i] == neighborsValue[i]){
                ++count;
            }else{
                break;
            }
        }
        if(count == rule.size()){
            return 1;
        }
    }
    // doesn't meet any rule, ret 0
    return 0;
}