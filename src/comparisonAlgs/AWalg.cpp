#include <opencv2/core.hpp>
#include <vector>

using namespace std;
using namespace cv;

int isRuleSatisfied(Mat &img, vector<int> pos);
vector<int> createNeighborsValue(Mat &img, const vector<vector<int> > &neighbors);

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
                if(img.at<uchar>(i, j) != 0 && isRuleSatisfied(img, {i, j})){
                    ret.at<uchar>(i, j) = 0;
                    ++count;
                }
            }
        }
        img = ret.clone();
    }
    return ret;
}

int isRuleSatisfied(Mat &img, vector<int> pos){
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
    vector<int> neighborsValue = createNeighborsValue(img, neighbors);

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

int isConditionOneSatisfied(Mat &img, vector<int> pos){
    int x = pos[0],
        y = pos[1];
    
    // we always consider the point y as center
    
    // vertical case
    vector<int> pz = {x - 1, y},
                p1 = {x - 2, y - 1},
                p2 = {x - 2, y},
                p3 = {x - 2, y + 1},
                p4 = {x - 1, y + 1},
                p5 = {x, y + 1},
                p6 = {x + 1, y + 1},
                p7 = {x + 1, y},
                p8 = {x + 1, y - 1},
                p9 = {x, y - 1},
                p10 = {x - 1, y - 1};
    vector<vector<int> > neighbors = {pz, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10};
    vector<int> neighborsValue = createNeighborsValue(img, neighbors);

    if(neighborsValue[0] == 1 && neighborsValue[2] == 0 && 
        neighborsValue[4] == 1 && neighborsValue[5] == 1 &&
        neighborsValue[7] == 0 && neighborsValue[9] == 1 &&
        neighborsValue[10] == 1){
            return 1;
        }

    // horizontal case
    vector<int> pz = {x, y - 1},
                p1 = {x - 1, y - 2},
                p2 = {x - 1, y - 1},
                p3 = {x - 1, y},
                p4 = {x - 1, y + 1},
                p5 = {x, y + 1},
                p6 = {x + 1, y + 1},
                p7 = {x + 1, y},
                p8 = {x + 1, y - 1},
                p9 = {x + 1, y - 2},
                p10 = {x, y - 2};
    vector<vector<int> > neighbors = {pz, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10};
    vector<int> neighborsValue = createNeighborsValue(img, neighbors);

    if(neighborsValue[0] == 1 && neighborsValue[2] == 1 &&
        neighborsValue[3] == 1 && neighborsValue[5] == 0 &&
        neighborsValue[7] == 1 && neighborsValue[8] == 1 &&
        neighborsValue[10] == 0){
            return 1;
        }

    return 0;
}

int stepOne(Mat &img, vector<int> pos){
    int x = pos[0],
        y = pos[1];
    
    // vertical case
    vector<int> p1 = {x - 1, y},
                pz = {x + 1, y},    // when w is above
                p2 = {x + 2, y};
    vector<vector<int> > neighbors = {p1, pz, p2};
    vector<int> neighborsValue = createNeighborsValue(img, neighbors);

    if(neighborsValue[0] == 0 && neighborsValue[1] == 1 && neighborsValue [2] == 0){
        return 1;   // means vertical and w is above
    }

    vector<int> p1 = {x - 2, y},
                pz = {x - 1, y},    // when w is below
                p2 = {x + 1, y};
    vector<vector<int> > neighbors = {p1, pz, p2};
    vector<int> neighborsValue = createNeighborsValue(img, neighbors);
    
    if(neighborsValue[0] == 0 && neighborsValue[1] == 1 && neighborsValue [2] == 0){
        return 2;   // means vertical and w is below
    }

    // horizontal case
    vector<int> p1 = {x, y - 1},
                pz = {x, y + 1},    // when w is on left side
                p2 = {x, y + 2};
    vector<vector<int> > neighbors = {p1, pz, p2};
    vector<int> neighborsValue = createNeighborsValue(img, neighbors);
    
    if(neighborsValue[0] == 0 && neighborsValue[1] == 1 && neighborsValue[2] == 0){
        return 3;   // means horizontal and w is on left side
    }

    vector<int> p1 = {x, y - 2},
                pz = {x, y - 1},    // when w is on right side
                p2 = {x, y + 1};
    vector<vector<int> > neighbors = {p1, pz, p2};
    vector<int> neighborsValue = createNeighborsValue(img, neighbors);

    if(neighborsValue[0] == 0 && neighborsValue[1] == 1 && neighborsValue[2] == 0){
        return 4;   // means horizontal and w is on right side
    }

    return 0;   // means this pixel doesn't belong to two pixels wide in the vertical or horizontal direction
}

vector<int> createNeighborsValue(Mat &img, const vector<vector<int> > &neighbors){
    vector<int> neighborsValue = {};
    for(vector<int> i : neighbors){
        if(i[0] >= 0 && i[0] < img.rows && i[1] >= 0 && i[1] < img.cols){
            neighborsValue.push_back(img.at<uchar>(i[0], i[1]) == 0 ? 0 : 1);
        }else{
            neighborsValue.push_back(0);
        }
    }
    return neighborsValue;
}

// actual step 2 of the whole iteration
int stepTwo(Mat &img, vector<int> pos){
    int x = pos[0],
        y = pos[1];
    
    vector<int> pz = {x + 1, y},
                p1 = {x - 1, y - 1},
                p2 = {x - 1, y},
                p3 = {x - 1, y + 1},
                p4 = {x, y + 1},
                p5 = {x + 1, y + 1},
                p6 = {x + 2, y + 1},
                p7 = {x + 2, y},
                p8 = {x + 2, y - 1},
                p9 = {x + 1, y - 1},
                p10 = {x, y - 1};
    vector<vector<int> > neighbors = {pz, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10};
    vector<int> neighborsValue = createNeighborsValue(img, neighbors);

    if(neighborsValue[0] == 1 && neighborsValue[2] == 0 &&
        neighborsValue[4] == 1 && neighborsValue[5] == 1 &&
        neighborsValue[7] == 0 && neighborsValue[9] == 1 &&
        neighborsValue[10] == 1){
            return 0;   // stop calculations for this pixel
        }
    
    vector<int> pz = {x - 1, y},
                p1 = {x - 2, y - 1},
                p2 = {x - 2, y},
                p3 = {x - 2, y + 1},
                p4 = {x - 1, y + 1},
                p5 = {x, y + 1},
                p6 = {x + 1, y + 1},
                p7 = {x + 1, y},
                p8 = {x + 1, y - 1},
                p9 = {x, y - 1},
                p10 = {x - 1, y - 1};
    vector<vector<int> > neighbors = {pz, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10};
    vector<int> neighborsValue = createNeighborsValue(img, neighbors);

    if(neighborsValue[0] == 1 && neighborsValue[2] == 0 && 
        neighborsValue[4] == 1 && neighborsValue[5] == 1 &&
        neighborsValue[7] == 0 && neighborsValue[9] == 1 &&
        neighborsValue[10] == 1){
            return 1;   // delete this pixel and stop calculations for it
        }else{
            return 2;   // go to 4, to check if it belongs to an extremity of a zigzag diagonal line
        }
}

// actual step 3 of the whole iteration
int stepThree(Mat &img, vector<int> pos){
    int x = pos[0],
        y = pos[1];
    
    vector<int> pz = {x, y + 1},
                p1 = {x - 1, y - 1},
                p2 = {x - 1, y},
                p3 = {x - 1, y + 1},
                p4 = {x - 1, y + 2},
                p5 = {x, y + 2},
                p6 = {x + 1, y + 2},
                p7 = {x + 1, y + 1},
                p8 = {x + 1, y},
                p9 = {x + 1, y - 1},
                p10 = {x, y - 1};
    vector<vector<int> > neighbors = {pz, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10};
    vector<int> neighborsValue = createNeighborsValue(img, neighbors);

    if(neighborsValue[0] == 1 && neighborsValue[2] == 1 &&
        neighborsValue[3] == 1 && neighborsValue[5] == 0 &&
        neighborsValue[7] == 1 && neighborsValue[8] == 1 &&
        neighborsValue[10] == 0){
            return 0;   // stop calculations for this pixel
        }

    vector<int> pz = {x, y - 1},
                p1 = {x - 1, y - 2},
                p2 = {x - 1, y - 1},
                p3 = {x - 1, y},
                p4 = {x - 1, y + 1},
                p5 = {x, y + 1},
                p6 = {x + 1, y + 1},
                p7 = {x + 1, y},
                p8 = {x + 1, y - 1},
                p9 = {x + 1, y - 2},
                p10 = {x, y - 2};
    vector<vector<int> > neighbors = {pz, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10};
    vector<int> neighborsValue = createNeighborsValue(img, neighbors);

    if(neighborsValue[0] == 1 && neighborsValue[2] == 1 &&
        neighborsValue[3] == 1 && neighborsValue[5] == 0 &&
        neighborsValue[7] == 1 && neighborsValue[8] == 1 &&
        neighborsValue[10] == 0){
            return 1;   // delete this pixel and stop calculations for it
        }else{
            return 2;   // go to step 5, check if extremity or not
        }

}

int stepFour(Mat &img, vector<int> pos){
    int x = pos[0],
        y = pos[1];
    
    vector<int> pz = {x + 1, y},
                p1 = {x - 1, y - 1},
                p2 = {x - 1, y},
                p3 = {x - 1, y + 1},
                p4 = {x , y + 1},
                p5 = {x + 1, y + 1},
                p6 = {x + 2, y + 1},
                p7 = {x + 2, y},
                p8 = {x + 2, y - 1},
                p9 = {x + 1, y - 1},
                p10 = {x, y - 1};
     vector<vector<int> > neighbors = {pz, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10};
     vector<int> neighborsValue = createNeighborsValue(img, neighbors);
     if(neighborsValue[0] == 1 && neighborsValue[1] == 0 &&
     neighborsValue[2] == 0 && neighborsValue[3] == 0 &&
     neighborsValue[4] == 0 && neighborsValue[5] == 0 &&
     neighborsValue[6] == 0 && neighborsValue[7] == 0 &&
     neighborsValue[8] == 1 && neighborsValue[9] == 1 &&
     neighborsValue[10] == 0 &&){
         return 0;  // stop calculations for this pixel
    }

    vector<int> pz = {x + 1, y},
                p1 = {x - 1, y - 1},
                p2 = {x - 1, y},
                p3 = {x - 1, y + 1},
                p4 = {x , y + 1},
                p5 = {x + 1, y + 1},
                p6 = {x + 2, y + 1},
                p7 = {x + 2, y},
                p8 = {x + 2, y - 1},
                p9 = {x + 1, y - 1},
                p10 = {x, y - 1};
     vector<vector<int> > neighbors = {pz, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10};
     vector<int> neighborsValue = createNeighborsValue(img, neighbors);


}