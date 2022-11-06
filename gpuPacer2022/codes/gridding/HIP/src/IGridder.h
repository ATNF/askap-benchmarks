// Solver interface 
#pragma once

#include "../utilities/Parameters.h"

#include <vector>
#include <iostream>
#include <complex>

template <typename T2>
class IGridder
{
protected:
    const size_t support;
    const size_t GSIZE;
    const std::vector<T2>& data;
    const std::vector<T2>& C;
    const std::vector<int>& cOffset;
    const std::vector<int>& iu;
    const std::vector<int>& iv;
    std::vector<T2>& grid;

public:
    IGridder(const size_t support,
        const size_t GSIZE,
        const std::vector<T2>& data,
        const std::vector<T2>& C,
        const std::vector<int>& cOffset,
        const std::vector<int>& iu,
        const std::vector<int>& iv,
        std::vector<T2>& grid) : support{ support }, GSIZE{ GSIZE }, data{ data }, C{ C },
        cOffset{ cOffset }, iu{ iu }, iv{ iv }, grid{ grid } {}
    virtual ~IGridder() {}
    virtual void gridder() = 0;
};

