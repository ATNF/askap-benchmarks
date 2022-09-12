#pragma once

#include "../utilities/Parameters.h"
#include "../utilities/RandomVectorGenerator.h"

#include <vector>
#include <iostream>
#include <complex>
#include <iomanip>

/*
    T0: Real        ->  float, double
    T1: Coord       ->  float, double
    T2: Value       ->  complex<Real>
*/
template <typename T0, typename T1, typename T2> 
class Setup
{
private:
    int& support;
    int& overSample;
    T1& wCellSize;
    std::vector<T1>& u;
    std::vector<T1>& v;
    std::vector<T1>& w;
    std::vector<T1>& freq;
    std::vector<int>& cOffset;
    std::vector<int>& iu;
    std::vector<int>& iv;
    std::vector<T2>& C;

public:
    Setup(int& support,
        int& overSample,
        T1& wCellSize,
        std::vector<T1>& u,
        std::vector<T1>& v,
        std::vector<T1>& w,
        std::vector<T1>& freq,
        std::vector<int>& cOffset,
        std::vector<int>& iu,
        std::vector<int>& iv,
        std::vector<T2>& C) : support{support}, overSample{overSample}, wCellSize{wCellSize},
        u{u}, v{v}, w{w}, freq{freq}, cOffset{cOffset}, iu{iu}, iv{iv}, C{C} {}
    void initCoord();
    void initC();
    void initCOffset();
    void setup();
    
};
