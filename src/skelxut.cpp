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

// comment out following line to remove omp
#define PARALLEL_FLAG

using namespace std;
using namespace cv;
using namespace Eigen;

namespace skelx{
    const vector<vector<vector<int>>> preDefinedNeighbors{
        {{-1,0},{0,-1},{0,1},{1,0}},   // r = 1.1
        {{-1,-1},{-1,1},{1,-1},{1,1}},   // r = 1.5
        {{-2,0},{0,-2},{0,2},{2,0}},   // r = 2
        {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}},   // r = 2.3
        {{-2,-2},{-2,2},{2,-2},{2,2}},   // r = 2.9
        {{-3,0},{0,-3},{0,3},{3,0}},   // r = 3
        {{-3,-1},{-3,1},{-1,-3},{-1,3},{1,-3},{1,3},{3,-1},{3,1}},   // r = 3.2
        {{-3,-2},{-3,2},{-2,-3},{-2,3},{2,-3},{2,3},{3,-2},{3,2}},   // r = 3.7
        {{-4,0},{0,-4},{0,4},{4,0}},   // r = 4
        {{-4,-1},{-4,1},{-1,-4},{-1,4},{1,-4},{1,4},{4,-1},{4,1}},   // r = 4.2
        {{-3,-3},{-3,3},{3,-3},{3,3}},   // r = 4.3
        {{-4,-2},{-4,2},{-2,-4},{-2,4},{2,-4},{2,4},{4,-2},{4,2}},   // r = 4.5
        {{-5,-1},{-5,0},{-5,1},{-4,-3},{-4,3},{-3,-4},{-3,4},{-1,-5},{-1,5},{0,-5},{0,5},{1,-5},{1,5},{3,-4},{3,4},{4,-3},{4,3},{5,-1},{5,0},{5,1}},   // r = 5.1
        {{-5,-2},{-5,2},{-2,-5},{-2,5},{2,-5},{2,5},{5,-2},{5,2}},   // r = 5.4
        {{-4,-4},{-4,4},{4,-4},{4,4}},   // r = 5.7
        {{-5,-3},{-5,3},{-3,-5},{-3,5},{3,-5},{3,5},{5,-3},{5,3}},   // r = 5.9
        {{-6,-1},{-6,0},{-6,1},{-1,-6},{-1,6},{0,-6},{0,6},{1,-6},{1,6},{6,-1},{6,0},{6,1}},   // r = 6.1
        {{-6,-2},{-6,2},{-2,-6},{-2,6},{2,-6},{2,6},{6,-2},{6,2}},   // r = 6.4
        {{-5,-4},{-5,4},{-4,-5},{-4,5},{4,-5},{4,5},{5,-4},{5,4}},   // r = 6.5
        {{-6,-3},{-6,3},{-3,-6},{-3,6},{3,-6},{3,6},{6,-3},{6,3}},   // r = 6.8
        {{-7,-1},{-7,0},{-7,1},{-5,-5},{-5,5},{-1,-7},{-1,7},{0,-7},{0,7},{1,-7},{1,7},{5,-5},{5,5},{7,-1},{7,0},{7,1}},   // r = 7.1
        {{-7,-2},{-7,2},{-6,-4},{-6,4},{-4,-6},{-4,6},{-2,-7},{-2,7},{2,-7},{2,7},{4,-6},{4,6},{6,-4},{6,4},{7,-2},{7,2}},   // r = 7.3
        {{-7,-3},{-7,3},{-3,-7},{-3,7},{3,-7},{3,7},{7,-3},{7,3}},   // r = 7.7
        {{-6,-5},{-6,5},{-5,-6},{-5,6},{5,-6},{5,6},{6,-5},{6,5}},   // r = 7.9
        {{-8,-1},{-8,0},{-8,1},{-7,-4},{-7,4},{-4,-7},{-4,7},{-1,-8},{-1,8},{0,-8},{0,8},{1,-8},{1,8},{4,-7},{4,7},{7,-4},{7,4},{8,-1},{8,0},{8,1}},   // r = 8.1
        {{-8,-2},{-8,2},{-2,-8},{-2,8},{2,-8},{2,8},{8,-2},{8,2}},   // r = 8.3
        {{-6,-6},{-6,6},{6,-6},{6,6}},   // r = 8.5
        {{-8,-3},{-8,3},{-3,-8},{-3,8},{3,-8},{3,8},{8,-3},{8,3}},   // r = 8.6
        {{-7,-5},{-7,5},{-5,-7},{-5,7},{5,-7},{5,7},{7,-5},{7,5}},   // r = 8.7
        {{-8,-4},{-8,4},{-4,-8},{-4,8},{4,-8},{4,8},{8,-4},{8,4}},   // r = 9
        {{-9,-1},{-9,0},{-9,1},{-1,-9},{-1,9},{0,-9},{0,9},{1,-9},{1,9},{9,-1},{9,0},{9,1}},   // r = 9.1
        {{-9,-2},{-9,2},{-7,-6},{-7,6},{-6,-7},{-6,7},{-2,-9},{-2,9},{2,-9},{2,9},{6,-7},{6,7},{7,-6},{7,6},{9,-2},{9,2}},   // r = 9.3
        {{-9,-3},{-9,3},{-8,-5},{-8,5},{-5,-8},{-5,8},{-3,-9},{-3,9},{3,-9},{3,9},{5,-8},{5,8},{8,-5},{8,5},{9,-3},{9,3}},   // r = 9.5
        {{-9,-4},{-9,4},{-7,-7},{-7,7},{-4,-9},{-4,9},{4,-9},{4,9},{7,-7},{7,7},{9,-4},{9,4}},   // r = 9.9
        {{-10,-1},{-10,0},{-10,1},{-8,-6},{-8,6},{-6,-8},{-6,8},{-1,-10},{-1,10},{0,-10},{0,10},{1,-10},{1,10},{6,-8},{6,8},{8,-6},{8,6},{10,-1},{10,0},{10,1}},   // r = 10.1
        {{-10,-2},{-10,2},{-2,-10},{-2,10},{2,-10},{2,10},{10,-2},{10,2}},   // r = 10.2
        {{-9,-5},{-9,5},{-5,-9},{-5,9},{5,-9},{5,9},{9,-5},{9,5}},   // r = 10.3
        {{-10,-3},{-10,3},{-3,-10},{-3,10},{3,-10},{3,10},{10,-3},{10,3}},   // r = 10.5
        {{-8,-7},{-8,7},{-7,-8},{-7,8},{7,-8},{7,8},{8,-7},{8,7}},   // r = 10.7
        {{-10,-4},{-10,4},{-4,-10},{-4,10},{4,-10},{4,10},{10,-4},{10,4}},   // r = 10.8
        {{-9,-6},{-9,6},{-6,-9},{-6,9},{6,-9},{6,9},{9,-6},{9,6}},   // r = 10.9
        {{-11,-1},{-11,0},{-11,1},{-1,-11},{-1,11},{0,-11},{0,11},{1,-11},{1,11},{11,-1},{11,0},{11,1}},   // r = 11.1
        {{-11,-2},{-11,2},{-10,-5},{-10,5},{-5,-10},{-5,10},{-2,-11},{-2,11},{2,-11},{2,11},{5,-10},{5,10},{10,-5},{10,5},{11,-2},{11,2}},   // r = 11.2
        {{-8,-8},{-8,8},{8,-8},{8,8}},   // r = 11.4
        {{-11,-3},{-11,3},{-9,-7},{-9,7},{-7,-9},{-7,9},{-3,-11},{-3,11},{3,-11},{3,11},{7,-9},{7,9},{9,-7},{9,7},{11,-3},{11,3}},   // r = 11.5
        {{-10,-6},{-10,6},{-6,-10},{-6,10},{6,-10},{6,10},{10,-6},{10,6}},   // r = 11.7
        {{-11,-4},{-11,4},{-4,-11},{-4,11},{4,-11},{4,11},{11,-4},{11,4}},   // r = 11.8
        {{-12,-1},{-12,0},{-12,1},{-11,-5},{-11,5},{-9,-8},{-9,8},{-8,-9},{-8,9},{-5,-11},{-5,11},{-1,-12},{-1,12},{0,-12},{0,12},{1,-12},{1,12},{5,-11},{5,11},{8,-9},{8,9},{9,-8},{9,8},{11,-5},{11,5},{12,-1},{12,0},{12,1}},   // r = 12.1
        {{-12,-2},{-12,2},{-2,-12},{-2,12},{2,-12},{2,12},{12,-2},{12,2}},   // r = 12.2
        {{-10,-7},{-10,7},{-7,-10},{-7,10},{7,-10},{7,10},{10,-7},{10,7}},   // r = 12.3
        {{-12,-3},{-12,3},{-3,-12},{-3,12},{3,-12},{3,12},{12,-3},{12,3}},   // r = 12.4
        {{-11,-6},{-11,6},{-6,-11},{-6,11},{6,-11},{6,11},{11,-6},{11,6}},   // r = 12.6
        {{-12,-4},{-12,4},{-4,-12},{-4,12},{4,-12},{4,12},{12,-4},{12,4}},   // r = 12.7
        {{-9,-9},{-9,9},{9,-9},{9,9}},   // r = 12.8
        {{-10,-8},{-10,8},{-8,-10},{-8,10},{8,-10},{8,10},{10,-8},{10,8}},   // r = 12.9
        {{-13,-1},{-13,0},{-13,1},{-12,-5},{-12,5},{-11,-7},{-11,7},{-7,-11},{-7,11},{-5,-12},{-5,12},{-1,-13},{-1,13},{0,-13},{0,13},{1,-13},{1,13},{5,-12},{5,12},{7,-11},{7,11},{11,-7},{11,7},{12,-5},{12,5},{13,-1},{13,0},{13,1}},   // r = 13.1
        {{-13,-2},{-13,2},{-2,-13},{-2,13},{2,-13},{2,13},{13,-2},{13,2}},   // r = 13.2
        {{-13,-3},{-13,3},{-3,-13},{-3,13},{3,-13},{3,13},{13,-3},{13,3}},   // r = 13.4
        {{-12,-6},{-12,6},{-10,-9},{-10,9},{-9,-10},{-9,10},{-6,-12},{-6,12},{6,-12},{6,12},{9,-10},{9,10},{10,-9},{10,9},{12,-6},{12,6}},   // r = 13.5
        {{-13,-4},{-13,4},{-11,-8},{-11,8},{-8,-11},{-8,11},{-4,-13},{-4,13},{4,-13},{4,13},{8,-11},{8,11},{11,-8},{11,8},{13,-4},{13,4}},   // r = 13.7
        {{-12,-7},{-12,7},{-7,-12},{-7,12},{7,-12},{7,12},{12,-7},{12,7}},   // r = 13.9
        {{-13,-5},{-13,5},{-5,-13},{-5,13},{5,-13},{5,13},{13,-5},{13,5}},   // r = 14
        {{-14,-1},{-14,0},{-14,1},{-1,-14},{-1,14},{0,-14},{0,14},{1,-14},{1,14},{14,-1},{14,0},{14,1}},   // r = 14.1
        {{-14,-2},{-14,2},{-10,-10},{-10,10},{-2,-14},{-2,14},{2,-14},{2,14},{10,-10},{10,10},{14,-2},{14,2}},   // r = 14.2
        {{-11,-9},{-11,9},{-9,-11},{-9,11},{9,-11},{9,11},{11,-9},{11,9}},   // r = 14.3
        {{-14,-3},{-14,3},{-13,-6},{-13,6},{-6,-13},{-6,13},{-3,-14},{-3,14},{3,-14},{3,14},{6,-13},{6,13},{13,-6},{13,6},{14,-3},{14,3}},   // r = 14.4
        {{-12,-8},{-12,8},{-8,-12},{-8,12},{8,-12},{8,12},{12,-8},{12,8}},   // r = 14.5
        {{-14,-4},{-14,4},{-4,-14},{-4,14},{4,-14},{4,14},{14,-4},{14,4}},   // r = 14.6
        {{-13,-7},{-13,7},{-7,-13},{-7,13},{7,-13},{7,13},{13,-7},{13,7}},   // r = 14.8
        {{-14,-5},{-14,5},{-11,-10},{-11,10},{-10,-11},{-10,11},{-5,-14},{-5,14},{5,-14},{5,14},{10,-11},{10,11},{11,-10},{11,10},{14,-5},{14,5}},   // r = 14.9
        {{-15,-1},{-15,0},{-15,1},{-12,-9},{-12,9},{-9,-12},{-9,12},{-1,-15},{-1,15},{0,-15},{0,15},{1,-15},{1,15},{9,-12},{9,12},{12,-9},{12,9},{15,-1},{15,0},{15,1}},   // r = 15.1
        {{-15,-2},{-15,2},{-2,-15},{-2,15},{2,-15},{2,15},{15,-2},{15,2}},   // r = 15.2
        {{-15,-3},{-15,3},{-14,-6},{-14,6},{-13,-8},{-13,8},{-8,-13},{-8,13},{-6,-14},{-6,14},{-3,-15},{-3,15},{3,-15},{3,15},{6,-14},{6,14},{8,-13},{8,13},{13,-8},{13,8},{14,-6},{14,6},{15,-3},{15,3}},   // r = 15.3
        {{-15,-4},{-15,4},{-11,-11},{-11,11},{-4,-15},{-4,15},{4,-15},{4,15},{11,-11},{11,11},{15,-4},{15,4}},   // r = 15.6
        {{-14,-7},{-14,7},{-12,-10},{-12,10},{-10,-12},{-10,12},{-7,-14},{-7,14},{7,-14},{7,14},{10,-12},{10,12},{12,-10},{12,10},{14,-7},{14,7}},   // r = 15.7
        {{-15,-5},{-15,5},{-13,-9},{-13,9},{-9,-13},{-9,13},{-5,-15},{-5,15},{5,-15},{5,15},{9,-13},{9,13},{13,-9},{13,9},{15,-5},{15,5}},   // r = 15.9
        {{-16,-1},{-16,0},{-16,1},{-1,-16},{-1,16},{0,-16},{0,16},{1,-16},{1,16},{16,-1},{16,0},{16,1}},   // r = 16.1
        {{-16,-2},{-16,2},{-15,-6},{-15,6},{-14,-8},{-14,8},{-8,-14},{-8,14},{-6,-15},{-6,15},{-2,-16},{-2,16},{2,-16},{2,16},{6,-15},{6,15},{8,-14},{8,14},{14,-8},{14,8},{15,-6},{15,6},{16,-2},{16,2}},   // r = 16.2
        {{-16,-3},{-16,3},{-12,-11},{-12,11},{-11,-12},{-11,12},{-3,-16},{-3,16},{3,-16},{3,16},{11,-12},{11,12},{12,-11},{12,11},{16,-3},{16,3}},   // r = 16.3
        {{-16,-4},{-16,4},{-13,-10},{-13,10},{-10,-13},{-10,13},{-4,-16},{-4,16},{4,-16},{4,16},{10,-13},{10,13},{13,-10},{13,10},{16,-4},{16,4}},   // r = 16.5
        {{-15,-7},{-15,7},{-7,-15},{-7,15},{7,-15},{7,15},{15,-7},{15,7}},   // r = 16.6
        {{-14,-9},{-14,9},{-9,-14},{-9,14},{9,-14},{9,14},{14,-9},{14,9}},   // r = 16.7
        {{-16,-5},{-16,5},{-5,-16},{-5,16},{5,-16},{5,16},{16,-5},{16,5}},   // r = 16.8
        {{-12,-12},{-12,12},{12,-12},{12,12}},   // r = 17
        {{-17,-1},{-17,0},{-17,1},{-16,-6},{-16,6},{-15,-8},{-15,8},{-13,-11},{-13,11},{-11,-13},{-11,13},{-8,-15},{-8,15},{-6,-16},{-6,16},{-1,-17},{-1,17},{0,-17},{0,17},{1,-17},{1,17},{6,-16},{6,16},{8,-15},{8,15},{11,-13},{11,13},{13,-11},{13,11},{15,-8},{15,8},{16,-6},{16,6},{17,-1},{17,0},{17,1}},   // r = 17.1
        {{-17,-2},{-17,2},{-2,-17},{-2,17},{2,-17},{2,17},{17,-2},{17,2}},   // r = 17.2
        {{-17,-3},{-17,3},{-14,-10},{-14,10},{-10,-14},{-10,14},{-3,-17},{-3,17},{3,-17},{3,17},{10,-14},{10,14},{14,-10},{14,10},{17,-3},{17,3}},   // r = 17.3
        {{-17,-4},{-17,4},{-16,-7},{-16,7},{-15,-9},{-15,9},{-9,-15},{-9,15},{-7,-16},{-7,16},{-4,-17},{-4,17},{4,-17},{4,17},{7,-16},{7,16},{9,-15},{9,15},{15,-9},{15,9},{16,-7},{16,7},{17,-4},{17,4}},   // r = 17.5
        {{-13,-12},{-13,12},{-12,-13},{-12,13},{12,-13},{12,13},{13,-12},{13,12}},   // r = 17.7
        {{-17,-5},{-17,5},{-5,-17},{-5,17},{5,-17},{5,17},{17,-5},{17,5}},   // r = 17.8
        {{-16,-8},{-16,8},{-14,-11},{-14,11},{-11,-14},{-11,14},{-8,-16},{-8,16},{8,-16},{8,16},{11,-14},{11,14},{14,-11},{14,11},{16,-8},{16,8}},   // r = 17.9
        {{-18,-1},{-18,0},{-18,1},{-17,-6},{-17,6},{-15,-10},{-15,10},{-10,-15},{-10,15},{-6,-17},{-6,17},{-1,-18},{-1,18},{0,-18},{0,18},{1,-18},{1,18},{6,-17},{6,17},{10,-15},{10,15},{15,-10},{15,10},{17,-6},{17,6},{18,-1},{18,0},{18,1}},   // r = 18.1
        {{-18,-2},{-18,2},{-2,-18},{-2,18},{2,-18},{2,18},{18,-2},{18,2}},   // r = 18.2
        {{-18,-3},{-18,3},{-3,-18},{-3,18},{3,-18},{3,18},{18,-3},{18,3}},   // r = 18.3
        {{-17,-7},{-17,7},{-16,-9},{-16,9},{-13,-13},{-13,13},{-9,-16},{-9,16},{-7,-17},{-7,17},{7,-17},{7,17},{9,-16},{9,16},{13,-13},{13,13},{16,-9},{16,9},{17,-7},{17,7}},   // r = 18.4
        {{-18,-4},{-18,4},{-14,-12},{-14,12},{-12,-14},{-12,14},{-4,-18},{-4,18},{4,-18},{4,18},{12,-14},{12,14},{14,-12},{14,12},{18,-4},{18,4}},   // r = 18.5
        {{-18,-5},{-18,5},{-15,-11},{-15,11},{-11,-15},{-11,15},{-5,-18},{-5,18},{5,-18},{5,18},{11,-15},{11,15},{15,-11},{15,11},{18,-5},{18,5}},   // r = 18.7
        {{-17,-8},{-17,8},{-8,-17},{-8,17},{8,-17},{8,17},{17,-8},{17,8}},   // r = 18.8
        {{-16,-10},{-16,10},{-10,-16},{-10,16},{10,-16},{10,16},{16,-10},{16,10}},   // r = 18.9
        {{-19,0},{-18,-6},{-18,6},{-6,-18},{-6,18},{0,-19},{0,19},{6,-18},{6,18},{18,-6},{18,6},{19,0}},   // r = 19
        {{-19,-1},{-19,1},{-1,-19},{-1,19},{1,-19},{1,19},{19,-1},{19,1}},   // r = 19.1
        {{-19,-2},{-19,2},{-14,-13},{-14,13},{-13,-14},{-13,14},{-2,-19},{-2,19},{2,-19},{2,19},{13,-14},{13,14},{14,-13},{14,13},{19,-2},{19,2}},   // r = 19.2
        {{-19,-3},{-19,3},{-17,-9},{-17,9},{-15,-12},{-15,12},{-12,-15},{-12,15},{-9,-17},{-9,17},{-3,-19},{-3,19},{3,-19},{3,19},{9,-17},{9,17},{12,-15},{12,15},{15,-12},{15,12},{17,-9},{17,9},{19,-3},{19,3}},   // r = 19.3
        {{-18,-7},{-18,7},{-7,-18},{-7,18},{7,-18},{7,18},{18,-7},{18,7}},   // r = 19.4
        {{-19,-4},{-19,4},{-16,-11},{-16,11},{-11,-16},{-11,16},{-4,-19},{-4,19},{4,-19},{4,19},{11,-16},{11,16},{16,-11},{16,11},{19,-4},{19,4}},   // r = 19.5
        {{-19,-5},{-19,5},{-18,-8},{-18,8},{-8,-18},{-8,18},{-5,-19},{-5,19},{5,-19},{5,19},{8,-18},{8,18},{18,-8},{18,8},{19,-5},{19,5}},   // r = 19.7
        {{-17,-10},{-17,10},{-14,-14},{-14,14},{-10,-17},{-10,17},{10,-17},{10,17},{14,-14},{14,14},{17,-10},{17,10}},   // r = 19.8
        {{-15,-13},{-15,13},{-13,-15},{-13,15},{13,-15},{13,15},{15,-13},{15,13}},   // r = 19.9
        {{-20,0},{-19,-6},{-19,6},{-16,-12},{-16,12},{-12,-16},{-12,16},{-6,-19},{-6,19},{0,-20},{0,20},{6,-19},{6,19},{12,-16},{12,16},{16,-12},{16,12},{19,-6},{19,6},{20,0}},   // r = 20
        {{-20,-2},{-20,-1},{-20,1},{-20,2},{-2,-20},{-2,20},{-1,-20},{-1,20},{1,-20},{1,20},{2,-20},{2,20},{20,-2},{20,-1},{20,1},{20,2}},   // r = 20.1
        {{-18,-9},{-18,9},{-9,-18},{-9,18},{9,-18},{9,18},{18,-9},{18,9}},   // r = 20.2
        {{-20,-3},{-20,3},{-19,-7},{-19,7},{-17,-11},{-17,11},{-11,-17},{-11,17},{-7,-19},{-7,19},{-3,-20},{-3,20},{3,-20},{3,20},{7,-19},{7,19},{11,-17},{11,17},{17,-11},{17,11},{19,-7},{19,7},{20,-3},{20,3}},   // r = 20.3
        {{-20,-4},{-20,4},{-4,-20},{-4,20},{4,-20},{4,20},{20,-4},{20,4}},   // r = 20.4
        {{-18,-10},{-18,10},{-15,-14},{-15,14},{-14,-15},{-14,15},{-10,-18},{-10,18},{10,-18},{10,18},{14,-15},{14,15},{15,-14},{15,14},{18,-10},{18,10}},   // r = 20.6
        {{-20,-5},{-20,5},{-19,-8},{-19,8},{-16,-13},{-16,13},{-13,-16},{-13,16},{-8,-19},{-8,19},{-5,-20},{-5,20},{5,-20},{5,20},{8,-19},{8,19},{13,-16},{13,16},{16,-13},{16,13},{19,-8},{19,8},{20,-5},{20,5}},   // r = 20.7
        {{-20,-6},{-20,6},{-17,-12},{-17,12},{-12,-17},{-12,17},{-6,-20},{-6,20},{6,-20},{6,20},{12,-17},{12,17},{17,-12},{17,12},{20,-6},{20,6}},   // r = 20.9
        {{-21,0},{0,-21},{0,21},{21,0}},   // r = 21
        {{-21,-2},{-21,-1},{-21,1},{-21,2},{-19,-9},{-19,9},{-18,-11},{-18,11},{-11,-18},{-11,18},{-9,-19},{-9,19},{-2,-21},{-2,21},{-1,-21},{-1,21},{1,-21},{1,21},{2,-21},{2,21},{9,-19},{9,19},{11,-18},{11,18},{18,-11},{18,11},{19,-9},{19,9},{21,-2},{21,-1},{21,1},{21,2}},   // r = 21.1
        {{-20,-7},{-20,7},{-7,-20},{-7,20},{7,-20},{7,20},{20,-7},{20,7}},   // r = 21.2
        {{-21,-3},{-21,3},{-16,-14},{-16,14},{-15,-15},{-15,15},{-14,-16},{-14,16},{-3,-21},{-3,21},{3,-21},{3,21},{14,-16},{14,16},{15,-15},{15,15},{16,-14},{16,14},{21,-3},{21,3}},   // r = 21.3
        {{-21,-4},{-21,4},{-4,-21},{-4,21},{4,-21},{4,21},{21,-4},{21,4}},   // r = 21.4
        {{-19,-10},{-19,10},{-17,-13},{-17,13},{-13,-17},{-13,17},{-10,-19},{-10,19},{10,-19},{10,19},{13,-17},{13,17},{17,-13},{17,13},{19,-10},{19,10}},   // r = 21.5
        {{-21,-5},{-21,5},{-20,-8},{-20,8},{-8,-20},{-8,20},{-5,-21},{-5,21},{5,-21},{5,21},{8,-20},{8,20},{20,-8},{20,8},{21,-5},{21,5}},   // r = 21.6
        {{-18,-12},{-18,12},{-12,-18},{-12,18},{12,-18},{12,18},{18,-12},{18,12}},   // r = 21.7
        {{-21,-6},{-21,6},{-6,-21},{-6,21},{6,-21},{6,21},{21,-6},{21,6}},   // r = 21.9
        {{-22,0},{-20,-9},{-20,9},{-19,-11},{-19,11},{-16,-15},{-16,15},{-15,-16},{-15,16},{-11,-19},{-11,19},{-9,-20},{-9,20},{0,-22},{0,22},{9,-20},{9,20},{11,-19},{11,19},{15,-16},{15,16},{16,-15},{16,15},{19,-11},{19,11},{20,-9},{20,9},{22,0}},   // r = 22
        {{-22,-2},{-22,-1},{-22,1},{-22,2},{-17,-14},{-17,14},{-14,-17},{-14,17},{-2,-22},{-2,22},{-1,-22},{-1,22},{1,-22},{1,22},{2,-22},{2,22},{14,-17},{14,17},{17,-14},{17,14},{22,-2},{22,-1},{22,1},{22,2}},   // r = 22.1
        {{-21,-7},{-21,7},{-7,-21},{-7,21},{7,-21},{7,21},{21,-7},{21,7}},   // r = 22.2
        {{-22,-3},{-22,3},{-18,-13},{-18,13},{-13,-18},{-13,18},{-3,-22},{-3,22},{3,-22},{3,22},{13,-18},{13,18},{18,-13},{18,13},{22,-3},{22,3}},   // r = 22.3
        {{-22,-4},{-22,4},{-20,-10},{-20,10},{-10,-20},{-10,20},{-4,-22},{-4,22},{4,-22},{4,22},{10,-20},{10,20},{20,-10},{20,10},{22,-4},{22,4}},   // r = 22.4
        {{-21,-8},{-21,8},{-19,-12},{-19,12},{-12,-19},{-12,19},{-8,-21},{-8,21},{8,-21},{8,21},{12,-19},{12,19},{19,-12},{19,12},{21,-8},{21,8}},   // r = 22.5
        {{-22,-5},{-22,5},{-5,-22},{-5,22},{5,-22},{5,22},{22,-5},{22,5}},   // r = 22.6
        {{-17,-15},{-17,15},{-16,-16},{-16,16},{-15,-17},{-15,17},{15,-17},{15,17},{16,-16},{16,16},{17,-15},{17,15}},   // r = 22.7
        {{-22,-6},{-22,6},{-21,-9},{-21,9},{-20,-11},{-20,11},{-18,-14},{-18,14},{-14,-18},{-14,18},{-11,-20},{-11,20},{-9,-21},{-9,21},{-6,-22},{-6,22},{6,-22},{6,22},{9,-21},{9,21},{11,-20},{11,20},{14,-18},{14,18},{18,-14},{18,14},{20,-11},{20,11},{21,-9},{21,9},{22,-6},{22,6}},   // r = 22.9
        {{-23,0},{0,-23},{0,23},{23,0}}   // r = 23
    };
}

namespace skelx{

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
            cout << "Point" << count<< ": deltaX:[ " << p.deltaX[0] << " , " << p.deltaX[1] << " ]" << " cosTheta:[" << p.cosTheta << "] "
                    << "ui:[ " << p.ui[0] << " , " << p.ui[1] << " ] sigma:[" <<p.sigma << "]"<< endl;

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
        int rows = img.rows;
        int cols = img.cols;
        int x = point.pos[0];
        int y = point.pos[1];

        vector<vector<double> > neighbors{};

        // predefined search
        for(const vector<vector<int>> &nei : preDefinedNeighbors){
            if(neighbors.size() < k){
                for(const vector<int> &p : nei){
                    int i = p[0];
                    int j = p[1];
                    if(x + i >= 0 && x + i < img.rows && y + j >= 0 && y + j < img.cols && img.at<uchar>(x + i, y + j) != 0){
                        neighbors.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
                    }
                }
            }else break;
        }

        // overflow control
        if(neighbors.size() < k){
            neighbors = {};
            for(int i = 0; i < 3; ++i){
                for(const vector<int> &p : preDefinedNeighbors[i]){
                    int i = p[0];
                    int j = p[1];
                    if(x + i >= 0 && x + i < img.rows && y + j >= 0 && y + j < img.cols && img.at<uchar>(x + i, y + j) != 0){
                        neighbors.push_back({static_cast<double>(x + i), static_cast<double>(y + j)});
                    }
                }
            }
        }

        // perturbation test
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

        #ifdef PARALLEL_FLAG
            #pragma omp parallel for
        #endif
        for(int i = 0; i < pointset.size(); ++i){
            skelx::Point &p = pointset[i];
            if(p.pos[0] >= 0 && p.pos[0] < src.rows && p.pos[1] >= 0 && p.pos[1] < src.cols) 
                ret.at<uchar>(p.pos[0], p.pos[1]) = 255;
        }
        return ret;
    }

    // move points toward deltaX
    void movePoint(vector<skelx::Point> &pointset){
        for(int i = 0; i < pointset.size(); ++i){
            skelx::Point &p = pointset[i];
            p.pos[0] += static_cast<int>(p.deltaX[0]);
            p.pos[1] += static_cast<int>(p.deltaX[1]);
        }
    }

    // set ui for each xi based on k nearest neighbors.
    // neighbors and ui of xi would be set
    void computeUi(Mat &img, vector<skelx::Point> &pointset, const int k, const bool perturbationFlag){
        // number search
        #ifdef PARALLEL_FLAG
            #pragma omp parallel for
        #endif
        for(int i = 0; i < pointset.size(); ++i){
            skelx::Point &p = pointset[i];
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
        #ifdef PARALLEL_FLAG
            #pragma omp parallel for
        #endif
        for(int i = 0; i < pointset.size(); ++i){
            skelx::Point &xi = pointset[i];
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
            double uiMod = pow(pow(xi.ui[0], 2) + pow(xi.ui[1], 2), 0.5);
            double scale = 10.0;

            xi.deltaX[0] = xi.ui[0] * std::exp(- pow(cosTheta, 2.0) * detailFactor * scale);
            xi.deltaX[1] = xi.ui[1] * std::exp(- pow(cosTheta, 2.0) * detailFactor * scale);
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
        cout<<"area:"<<(right - left) * (down - up)<<endl;
        return min(36,max(21, static_cast<int>(sqrt((right - left) * (down - up) / 400))));
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

    // remove isolate points
    void cleanImage(Mat &img){
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
    }

    // for further thinning
    Mat postProcess(Mat &img, vector<skelx::Point> &pointset){

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

    // return the answer of gauss circle problem
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

    // compute the search radius
    int computeMinimumSearchRadius(const int k){
        int r = 1;
        while(gaussCircleCount(r++) <= k);
        return r - 2;
    }
}

