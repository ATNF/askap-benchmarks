#pragma once

#include <vector>
#include <complex>
#include <iostream>

template <typename T>
class MaxError
{
public:
    void maxError(const std::vector<T>& v1, const std::vector<T>& v2) const;
    void reportError(size_t size, double aveErr, double frac_non_zero, double maxErr, int index) const;
    void reportError(size_t size, double aveErr, double frac_non_zero, double maxErr_r , double maxErr_i, int index) const;
};
