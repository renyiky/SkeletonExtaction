#ifndef POINTSTRUCT_HPP
#define POINTSTRUCT_HPP

#include <vector>

namespace skelx{
    
    struct Point{
    int k0, k;
    std::vector<double> pos{0.0,0.0}, 
                        ui,
                        deltaX,
                        principalVec{0.0, 0.0};

    double  sigma,
            d3nn, 
            lambda, // confidence
            deltaK;

    std::vector<std::vector<double> > neighbors, PCAneighbors, covMat;
    };

}

#endif