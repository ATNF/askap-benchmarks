#pragma once

#include <vector>
#include <random>

template <typename T>
class RandomVectorGenerator
{
public:
    void randomVector(std::vector<T>& v) const;
};
