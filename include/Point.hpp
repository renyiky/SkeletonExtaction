#ifndef POINTSTRUCT_HPP
#define POINTSTRUCT_HPP

#include <vector>

namespace skelx{
    
    struct Point{
        int k;
        double  sigma, cosTheta;
        std::vector<std::vector<double> > neighbors;
        std::vector<double> pos, 
                            ui,
                            deltaX,
                            principalVec;

        Point():sigma(0), cosTheta(0), pos{0.0, 0.0}, deltaX{0.0, 0.0}, principalVec{0.0, 0.0}{}
        Point(double x, double y):pos{x, y}{}
    };
}

#endif