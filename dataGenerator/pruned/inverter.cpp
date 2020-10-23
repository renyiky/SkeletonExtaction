#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){
    string filename = argv[1];
    Mat img = imread(filename + ".png", IMREAD_COLOR);
    
    MatIterator_<Vec3b> it, end;
            for( it = img.begin<Vec3b>(), end = img.end<Vec3b>(); it != end; ++it)
            {
                if(((*it)[0] == 125 && (*it)[1] == 125 && (*it)[2] == 125)){
                    (*it)[0] = 255;
                    (*it)[1] = 255;
                    (*it)[2] = 255;
                }else if((*it)[0] == 0 && (*it)[1] == 0 && (*it)[2] == 0){
                    (*it)[0] = 255;
                    (*it)[1] = 255;
                    (*it)[2] = 255;
                }else if((*it)[0] == 255 && (*it)[1] == 0 && (*it)[2] == 0){
                    (*it)[0] = 0;
                    (*it)[1] = 0;
                    (*it)[2] = 0;
                }
            }
    imwrite("results/" + filename + ".png", img);
}