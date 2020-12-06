#ifndef POINTSTRUCT_HPP
#define POINTSTRUCT_HPP

#include <vector>

namespace skelx{
    
    struct Point{
    int k;
    std::vector<double> pos{0.0,0.0}, 
                        ui,
                        deltaX,
                        principalVec{0.0, 0.0};

    double  sigma = 0, cosTheta = 0;

    std::vector<std::vector<double> > neighbors, covMat;
    };
}

#endif