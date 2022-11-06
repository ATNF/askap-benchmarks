#include "RandomVectorGenerator.h"

template <typename T>
void RandomVectorGenerator<T>::randomVector(std::vector<T>& v) const
{
    std::random_device rd;
    std::mt19937 randEng(rd());

    std::uniform_real_distribution<T> uniNum{0.0, 1.0};

    for (auto& i : v)
    {
        i = uniNum(randEng);
    }
}

template void RandomVectorGenerator<float>::randomVector(std::vector<float>& v) const;
template void RandomVectorGenerator<double>::randomVector(std::vector<double>& v) const;
