#include "MaxError.h"

using std::cout;
using std::endl;
using std::cerr;
using std::complex;
using std::vector;


template<typename T> void MaxError<T>::reportError(size_t size, double aveErr, double frac_non_zero, double maxErr, int index) const
{
    if (frac_non_zero > 0) aveErr /= frac_non_zero;
    frac_non_zero /= static_cast<double>(size);
    cout <<" Fraction non-zero error = "<<frac_non_zero<<" with average error of "<<aveErr<< endl;
    cout << "Maximum Error = " << maxErr << " at index: " << index<<endl;
}

template<typename T> void MaxError<T>::reportError(size_t size, double aveErr, double frac_non_zero, double maxErr_r, double maxErr_i, int index) const 
{
    if (frac_non_zero > 0) aveErr /= frac_non_zero;
    frac_non_zero /= static_cast<double>(size);
    cout <<" Fraction non-zero error = "<<frac_non_zero<<" with average error of "<<aveErr<< endl;
    cout << "Maximum Error (real, img) = (" << maxErr_r << ", " << maxErr_i << "), at index: " << index<<endl;
}

template void MaxError<float>::reportError(size_t size, double aveErr, double frac_non_zero, double maxErr, int index) const;
template void MaxError<double>::reportError(size_t size, double aveErr, double frac_non_zero, double maxErr, int index) const;
template void MaxError<complex<float>>::reportError(size_t size, double aveErr, double frac_non_zero, double maxErr, int index) const;
template void MaxError<complex<double>>::reportError(size_t size, double aveErr, double frac_non_zero, double maxErr, int index) const;


template<>
void MaxError<complex<float>>::maxError(const std::vector<complex<float>>& v1, const std::vector<complex<float>>& v2) const
{
    float maximumErrorReal = 0.0;
    float maximumErrorImag = 0.0;
    float aveErr = 0 ;
    double frac_non_zero = 0;
    size_t index = 0;
    try
    {
        if (v1.size() != v2.size())
        {
            throw 0;
        }
        for (auto i = 0; i < v1.size(); ++i)
        {
            auto diffi = abs(v1[i].imag() - v2[i].imag());
            auto diffr = abs(v1[i].real() - v2[i].real());
            if (diffi > 0 || diffr > 0) {
                frac_non_zero++;
                aveErr += std::sqrt(diffr*diffr + diffi*diffi);
            }
            if (diffi > maximumErrorImag ||  diffr > maximumErrorReal)
            {
                maximumErrorImag = diffi;
                maximumErrorReal = diffr;
                index = i;
            }
        }
        reportError(v1.size(), aveErr, frac_non_zero, maximumErrorReal, maximumErrorImag, index);
    }

    catch (int& ex)
    {
        cerr << "Sizes of 2 vectors are different." << endl;
    }
}

template<>
void MaxError<complex<double>>::maxError(const std::vector<complex<double>>& v1, const std::vector<complex<double>>& v2) const
{
    double maximumErrorReal = 0.0;
    double maximumErrorImag = 0.0;
    double aveErr = 0 ;
    double frac_non_zero = 0;
    size_t index = 0;
    try
    {
        if (v1.size() != v2.size())
        {
            throw 0;
        }

        for (auto i = 0; i < v1.size(); ++i)
        {
            auto diffi = abs(v1[i].imag() - v2[i].imag());
            auto diffr = abs(v1[i].real() - v2[i].real());
            if (diffi > 0 || diffr > 0) {
                frac_non_zero++;
                aveErr += std::sqrt(diffr*diffr + diffi*diffi);
            }
            if (diffi > maximumErrorImag ||  diffr > maximumErrorReal)
            {
                maximumErrorImag = diffi;
                maximumErrorReal = diffr;
                index = i;
            }
        }
        reportError(v1.size(), aveErr, frac_non_zero, maximumErrorReal, maximumErrorImag, index);
    }
    catch (int& ex)
    {
        cerr << "Sizes of 2 vectors are different." << endl;
    }
}

template <typename T>
void MaxError<T>::maxError(const std::vector<T>& v1, const std::vector<T>& v2) const
{
    T maximumError = 0.0;
    T aveErr = 0 ;
    T frac_non_zero = 0;
    int index;
    try
    {
        if (v1.size() != v2.size())
        {
            throw 0;
        }
        
        for (auto i = 0; i < v1.size(); ++i)
        {
            auto diff = abs(v1[i] - v2[i]);
            if (diff > 0) {
                frac_non_zero++;
                aveErr += diff;
            }
            if (diff > maximumError)
            {
                maximumError = diff;
                index = i;
            }
            reportError(v1.size(), aveErr, frac_non_zero, maximumError, index);
        }
    }
    catch (int& ex)
    {
        cerr << "Sizes of 2 vectors are different." << endl;
    }
}

template void MaxError<float>::maxError(const std::vector<float>& v1, const std::vector<float>& v2) const;
template void MaxError<double>::maxError(const std::vector<double>& v1, const std::vector<double>& v2) const;
template void MaxError<int>::maxError(const std::vector<int>& v1, const std::vector<int>& v2) const;
