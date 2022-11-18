// Solver interface 
#pragma once

#include "../utilities/Parameters.h"

#include <vector>
#include <iostream>
#include <complex>

class IDegridder
{
protected:
    const std::vector<std::complex<float>>& grid;
    const size_t DSIZE;
    const size_t SSIZE;
    const size_t GSIZE;
    const size_t support;
    const std::vector<std::complex<float>>& C;
    const std::vector<int>& cOffset;
    const std::vector<int>& iu;
    const std::vector<int>& iv;
    std::vector<std::complex<float>>& data;
        
public:
    IDegridder(const std::vector<std::complex<float>>& grid,
        const size_t DSIZE,
        const size_t SSIZE,
        const size_t GSIZE,
        const size_t support,
        const std::vector<std::complex<float>>& C,
        const std::vector<int>& cOffset,
        const std::vector<int>& iu,
        const std::vector<int>& iv,
        std::vector<std::complex<float>>& data) : grid{ grid }, DSIZE{ DSIZE }, SSIZE{ SSIZE }, GSIZE{ GSIZE }, support{ support }, C{ C },
        cOffset{ cOffset }, iu{ iu }, iv{ iv }, data{ data } {}
    virtual ~IDegridder() {}
    virtual void degridder() = 0;
};

