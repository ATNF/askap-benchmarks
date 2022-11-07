#include "MaxError.h"

using std::cout;
using std::endl;
using std::cerr;
using std::complex;
using std::vector;

template<>
void MaxError<complex<float>>::maxError(const std::vector<complex<float>>& v1, const std::vector<complex<float>>& v2) const
{
    float maximumErrorReal = 0.0;
    float maximumErrorImag = 0.0;
    size_t index = 0;
    try
    {
        if (v1.size() != v2.size())
        {
            throw 0;
        }

        for (auto i = 0; i < v1.size(); ++i)
        {
            if (abs(v1[i].imag() - v2[i].imag()) > maximumErrorImag || abs(v1[i].real() - v2[i].real()) > maximumErrorReal)
            {
                maximumErrorImag = abs(v1[i].imag() - v2[i].imag());
                maximumErrorReal = abs(v1[i].real() - v2[i].real());
                index = i;
                cout << "Maximum Error: (" << maximumErrorReal << ", " << maximumErrorImag << "), at index: " << index << endl;
            //    exit(-1);
            }
        }
        cout << "Maximum Error: (" << maximumErrorReal << ", " << maximumErrorImag << ")" << endl;
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
    size_t index = 0;
    try
    {
        if (v1.size() != v2.size())
        {
            throw 0;
        }

        for (auto i = 0; i < v1.size(); ++i)
        {
            if (abs(v1[i].imag() - v2[i].imag()) > maximumErrorImag || abs(v1[i].real() - v2[i].real()) > maximumErrorReal)
            {
                maximumErrorImag = abs(v1[i].imag() - v2[i].imag());
                maximumErrorReal = abs(v1[i].real() - v2[i].real());
                index = i;
                cout << "Maximum Error: (" << maximumErrorReal << ", " << maximumErrorImag << "), at index: " << index << endl;
            //    exit(-1);
            }
        }
        cout << "Maximum Error: (" << maximumErrorReal << ", " << maximumErrorImag << ")" << endl;
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
    try
    {
        if (v1.size() != v2.size())
        {
            throw 0;
        }

        for (auto i = 0; i < v1.size(); ++i)
        {
            maximumError = abs(v1[i] - v2[i]) > maximumError ? abs(v1[i] - v2[i]) : maximumError;
        }
    }
    catch (int& ex)
    {
        cerr << "Sizes of 2 vectors are different." << endl;
    }
    cout << "Maximum Error: " << maximumError << endl;
}

template void MaxError<float>::maxError(const std::vector<float>& v1, const std::vector<float>& v2) const;
template void MaxError<double>::maxError(const std::vector<double>& v1, const std::vector<double>& v2) const;
template void MaxError<int>::maxError(const std::vector<int>& v1, const std::vector<int>& v2) const;
