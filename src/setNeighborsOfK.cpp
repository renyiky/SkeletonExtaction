#include <vector>
#include <opencv2/core.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "Point.hpp"
#include "setNeighborsOfK.hpp"

using namespace std;
using namespace cv;

int isNextTo(vector<double> nei, vector<double> point);
vector<vector<vector<double> > > connectFunc(vector<vector<double> > curPointset,  vector<vector<double> > centerDomain,  vector<vector<double> > outerDomain);
void regularize(skelx::Point &centerPoint);

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

        // rectangle search
        // for(int i = -radius; i < radius + 1; ++i){
        //     if(i == -radius || i == radius){
        //         for(int j = -radius; j < radius + 1; ++j){
        //             if(/*pow(i * i + j * j, 0.5) <= radius && */x + i >= 0 && x + i < rows && y + j >= 0 
        //             && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
        //             && !(i == 0 && j == 0)){
        //                 neighbors.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
        //             }
        //         }
        //     }else{
        //         int j = -radius;
        //         if(/*pow(i * i + j * j, 0.5) <= radius &&*/ x + i >= 0 && x + i < rows && y + j >= 0 
        //         && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
        //         && !(i == 0 && j == 0)){
        //             neighbors.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
        //         }

        //         j = radius;
        //         if(/*pow(i * i + j * j, 0.5) <= radius &&*/ x + i >= 0 && x + i < rows && y + j >= 0 
        //         && y + j < cols && img.at<uchar>(x + i, y + j) != 0 
        //         && !(i == 0 && j == 0)){
        //             neighbors.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
        //         }
        //     }
        // }


        // circle search
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
        // regularize(point);  // set connected domain in each neighbor[2]
        return 1;
    }
    else{
        return 0;
    }
}

// void regularize(skelx::Point &centerPoint){
//     vector<vector<double> > centerDomain = {{centerPoint.pos[0], centerPoint.pos[1]}},
//                             outerDomain = {};
            
//     vector<vector<vector<double> > > domains;
//     vector<vector<double> > neighbors = centerPoint.neighbors;

//     int flag = 0;
            
//     for(vector<double> nei : neighbors){
//         for(vector<double> point : centerDomain){
//             if(isNextTo(nei, point)){
//                 flag = 1;
//                 domains = connectFunc({nei}, centerDomain, outerDomain);
//                 centerDomain = domains[0];
//                 outerDomain = domains[1];
//                 break;
//             }
//         }
//         if(flag == 0){
//             outerDomain.push_back(nei);
//         }else{
//             flag = 0;
//         }
//     }

//     vector<vector<double> > regularizedNeighbors = {};
    
//     centerDomain.erase(centerDomain.begin());   // erase the first point which is the center point, not the neighbor
//     for(vector<double> i : centerDomain){
//         regularizedNeighbors.push_back({i[0], i[1], 0});
//     }
//     for(vector<double> i : outerDomain){
//         regularizedNeighbors.push_back({i[0], i[1], 1});
//     }

//     centerPoint.neighbors = regularizedNeighbors;
// }

// vector<vector<vector<double> > > 
// connectFunc(vector<vector<double> > curPointset,  vector<vector<double> > centerDomain,  vector<vector<double> > outerDomain){
//     for(vector<double> i : curPointset){
//         centerDomain.push_back(i);
        
//         vector<vector<double> >::iterator iter = find(outerDomain.begin(), outerDomain.end(), i);
//         if(iter != outerDomain.end()){
//             outerDomain.erase(iter);
//         }
//     }
//     if(!outerDomain.size()){
//         return {centerDomain, outerDomain};
//     }
//     // cout<<outerDomain[0][0]<<"  "<<outerDomain[0][1]<<endl;
//     vector<vector<double> > newSet = {};
//     for(vector<double> i : curPointset){
//         for(vector<double> j : outerDomain){
//             if(isNextTo(i , j) && (find(newSet.begin(), newSet.end(), j) == newSet.end())){
//                 newSet.push_back(j);
//             }
//         }
//     }
//     if(!newSet.size()){
//         return {centerDomain, outerDomain};
//     }
//     return connectFunc(newSet, centerDomain, outerDomain);
// }

// int isNextTo(vector<double> nei, vector<double> point){
//     if((nei[0] + 1 == point[0] && nei[1] == point[1]) ||
//         (nei[0] - 1 == point[0] && nei[1] == point[1]) ||
//         (nei[0] == point[0] && nei[1] + 1 == point[1]) ||
//         (nei[0] == point[0] && nei[1] - 1 == point[1]) ||
//         (nei[0] + 1 == point[0] && nei[1] + 1 == point[1]) ||
//         (nei[0] + 1 == point[0] && nei[1] - 1 == point[1]) ||
//         (nei[0] - 1 == point[0] && nei[1] + 1 == point[1]) ||
//         (nei[0] - 1 == point[0] && nei[1] - 1 == point[1])){
//             return 1;
//     }else{
//         return 0;
//     }
// }

// int isInteger(double num){
//     return static_cast<double>(static_cast<int>(num)) == num;
// }