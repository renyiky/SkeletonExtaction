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

    double  sigma = 0,
            d3nn,
            cosAlpha = 0;  // to represent the angle of current cluster

    std::vector<std::vector<double> > neighbors, PCAneighbors, covMat;
    };
}

#endif