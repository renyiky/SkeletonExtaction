#ifndef POINTSTRUCT_HPP
#define POINTSTRUCT_HPP

#include <vector>

namespace skelx{
    
    struct Point{
    std::vector<double> pos{0.0,0.0}, 
                        ui, 
                        deltaX;

    double  sigma,
            d3nn, 
            lambda; // confidence
    int k, visited=0;
    std::vector<std::vector<int> > neighbors;
    };

}

#endif