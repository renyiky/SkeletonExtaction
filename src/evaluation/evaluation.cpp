#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iomanip>

using namespace std;
using namespace cv;

double f1ScoreExam(const Mat &groundTruth, const Mat &src);
double thinningRateExam(const Mat &img);
double connectivityMeasureExam(const Mat &img);
double sensitivityMeasureExam(const Mat &img);
vector<int> searchNeighborsValue(const Mat &img, vector<vector<int> > neighbors);
int countTriangle(const Mat &img, const vector<int> &pos);

int main(int argc, char *argv[]){
    string filename = argv[1], 
            resultsPath = "experimentsMaterial/results/",
            groundTruthPath = "experimentsMaterial/groundtruth/",
            gtFilePath = groundTruthPath + filename + ".png",
            srcFilePath = resultsPath + filename + "_" + argv[2] + "/" + "0_final_" + filename +"_" + argv[2] + ".png",
            ZSFilePath = resultsPath + filename + "_" + argv[2] + "/0_final_ZS_" + filename + ".png",
            AWFilePath = resultsPath + filename + "_" + argv[2] + "/0_final_AW_" + filename + ".png",
            GHFilePath = resultsPath + filename + "_" + argv[2] + "/0_final_GH_" + filename + ".png",
            HybridFilePath = resultsPath + filename + "_" + argv[2] + "/0_final_Hybrid_" + filename + ".png",
            rawFilePath = resultsPath + filename + "_" + argv[2] + "/0_raw_" + filename + ".png";
    
    Mat src = imread(srcFilePath, IMREAD_GRAYSCALE),
        ZS = imread(ZSFilePath, IMREAD_GRAYSCALE),
        AW = imread(AWFilePath, IMREAD_GRAYSCALE),
        GH = imread(GHFilePath, IMREAD_GRAYSCALE),
        Hybrid = imread(HybridFilePath, IMREAD_GRAYSCALE),
        groundTruth = imread(gtFilePath, IMREAD_GRAYSCALE),
        raw = imread(rawFilePath, IMREAD_GRAYSCALE);
    vector<Mat> resources = {src, ZS, AW, GH, Hybrid};   
    vector<string> names = {"Ours", "ZS", "AW", "GH", "Hybrid"};
    
    // examinations are below    
    vector<double> f1Scores = {},
                    TR = {},
                    CM = {},
                    SM = {};
    for(Mat &i : resources){
        // F1 Score
        f1Scores.push_back(f1ScoreExam(groundTruth, i));

        // Thinning Rate
        TR.push_back(thinningRateExam(i));

        // Connectivity Measure
        CM.push_back(connectivityMeasureExam(i));

        // Sensitivity Measure
        SM.push_back(sensitivityMeasureExam(i));
    }

    // print all results
    for(int i = 0; i < names.size(); ++i){
        cout<<setiosflags(ios::fixed)<<setprecision(7)<<flush;
        cout<<setw(7)<<names[i]<<"  |  F1 = "<<f1Scores[i]<<"  |  TR = "<<TR[i]<<flush;
        cout.unsetf(ios_base::fixed);
        cout<<setw(5)<<"  |  CM = "<<setw(2)<<CM[i]<<setw(5)<<"  |  SM = "<<setw(2)<<SM[i]<<endl;
    }

    return 0;
}

// F1 score examination
double f1ScoreExam(const Mat &groundTruth, const Mat &src){
    int truePositive = 0, trueNegative = 0,
        falsePositive = 0, falseNegative = 0;
    for(int i = 0; i < groundTruth.rows; ++i){
        for(int j = 0; j < groundTruth.cols; ++j){
            // when it should be predicted as negtive
            if(groundTruth.at<uchar>(i, j) == 0 && src.at<uchar>(i, j) == 0){
                ++trueNegative;
            }else if(groundTruth.at<uchar>(i, j) == 0 && src.at<uchar>(i, j) != 0){
                ++falsePositive;
            // when it should be predicted as positive
            }else if(groundTruth.at<uchar>(i, j) != 0 && src.at<uchar>(i, j) == 0){
                ++falseNegative;
            }else{
                ++truePositive;
            }
        }
    }
    double precision = static_cast<double>(truePositive) / (static_cast<double>(truePositive) + static_cast<double>(falsePositive)),
            recall = static_cast<double>(truePositive) / (static_cast<double>(truePositive) + static_cast<double>(falseNegative));
    
    return (2 * precision * recall) / (precision + recall);
}

// Thinning Rate(TR) examination,
// when it equals to 1, means that the image is perfectly thinned.
double thinningRateExam(const Mat &img){
    int TM1 = 0,
        temp = max(img.rows, img.cols),
        TM2 = 4 * (temp - 1) * (temp - 1);
    
    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.cols; ++j){
            if(img.at<uchar>(i, j) != 0){
                TM1 += countTriangle(img, {i, j});
            }
        }
    }
    return 1.0 - static_cast<double>(TM1) / static_cast<double>(TM2);
}

// p9 p2 p3
// p8 p1 p4
// p7 p6 p5
vector<int> searchNeighborsValue(const Mat &img, vector<vector<int> > neighbors){
    vector<int> neighborsValue = {};
    for(vector<int> &p : neighbors){
        if(p[0] >= 0 && p[0] < img.rows && p[1] >= 0 && p[1] < img.cols){
            if(img.at<uchar>(p[0], p[1]) != 0){
                neighborsValue.push_back(1);
            }else{
                neighborsValue.push_back(0);
            }
        }else{
            neighborsValue.push_back(0);
        }
    }
    return neighborsValue;
}

int countTriangle(const Mat &img, const vector<int> &pos){
    int x = pos[0],
        y = pos[1];
    vector<int> p1 = {x, y},
                p2 = {x - 1, y},
                p3 = {x - 1, y + 1},
                p4 = {x, y + 1},
                p5 = {x + 1, y + 1},
                p6 = {x + 1, y},
                p7 = {x + 1, y - 1},
                p8 = {x, y - 1},
                p9 = {x - 1, y - 1};
    vector<vector<int> > neighbors = {p1, p2, p3, p4, p5, p6, p7, p8, p9};
    vector<int> neighborsValue = searchNeighborsValue(img, neighbors);
    int count = neighborsValue[0] * 
                (neighborsValue[7] * neighborsValue[8] + 
                neighborsValue[8] * neighborsValue[1] + 
                neighborsValue[1] * neighborsValue[2] + 
                neighborsValue[2] * neighborsValue[3]);

    return count;
}

// Connectivity Measure(CM) examination
// The closer the CM of skeleton to that of original image, the better connectivity the skeleton has.
double connectivityMeasureExam(const Mat &img){
    int CM = 0;
    for(int x = 0; x < img.rows; ++x){
        for(int y = 0; y < img.cols; ++y){
            if(img.at<uchar>(x, y) != 0){
                vector<int> p2 = {x - 1, y},
                            p3 = {x - 1, y + 1},
                            p4 = {x, y + 1},
                            p5 = {x + 1, y + 1},
                            p6 = {x + 1, y},
                            p7 = {x + 1, y - 1},
                            p8 = {x, y - 1},
                            p9 = {x - 1, y - 1};
                vector<vector<int> > neighbors = {p2, p3, p4, p5, p6, p7, p8, p9};
                vector<int> neighborsValue = searchNeighborsValue(img, neighbors);
                int sum = 0;
                for(int n : neighborsValue){
                    sum += n;
                }
                if(sum < 2){
                    CM += 1;
                }
            }
        }
    }
    return CM;  // note that CM is implicitly cast into double here
}

// Sensitivity Measure(SM) examination.
// Actually, to count the cross-points in skeleton.
// The fewer the cross-points, the higher the immunity of the algorithm to noise.
double sensitivityMeasureExam(const Mat &img){
    int SM = 0;
    for(int x = 0; x < img.rows; ++x){
        for(int y = 0; y < img.cols; ++y){
            if(img.at<uchar>(x, y) != 0){
                vector<int> p2 = {x - 1, y},
                            p3 = {x - 1, y + 1},
                            p4 = {x, y + 1},
                            p5 = {x + 1, y + 1},
                            p6 = {x + 1, y},
                            p7 = {x + 1, y - 1},
                            p8 = {x, y - 1},
                            p9 = {x - 1, y - 1};
                vector<vector<int> > neighbors = {p2, p3, p4, p5, p6, p7, p8, p9, p2};
                vector<int> neighborsValue = searchNeighborsValue(img, neighbors);
                int sum = 0;
                for(int i = 0; i < neighborsValue.size() - 1; ++i){
                    if(neighborsValue[i] == 0 && neighborsValue[i + 1] == 1){
                        sum += 1;
                    }
                }
                if(sum > 2){
                    SM += 1;
                }
            }
        }
    }
    return SM;  // note that SM is implicitly cast into double here.
}

// Execution Time.
// The real time taken by the algorithm.
// Calculated as the mean of 100 different experiments.


// Thinning Speed.
// used to evaluate the number of pixels thinned per time unit(second).

