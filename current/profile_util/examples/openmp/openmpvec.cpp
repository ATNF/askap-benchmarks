#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <tuple>
#include <profile_util.h>

#ifdef USEOPENMP
#include <omp.h>
#endif

template<class T> std::vector<T> allocate_and_init_vector(unsigned long long N)
{
    std::vector<T> v(N);
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::normal_distribution<> distr(0,1);
    auto t_init = NewTimer();
    #pragma omp parallel \
    default(none) shared(v,N) firstprivate(gen,distr) \
    if (N>10000)
    {
        LogThreadAffinity();
//        #pragma omp for 
//        for (auto &x:v)
        #pragma omp for 
        for (auto i=0;i<v.size();i++)
        {
            v[i] = distr(gen);
            //x = distr(gen);
        }
    }
    LogTimeTaken(t_init);
    LogMemUsage();
    return v;
}


template<class T> T vector_sq_and_sum(std::vector<T> &v)
{
    T sum = 0;
    auto t1 = NewTimer();
    #pragma omp parallel \
    default(none) shared(v,sum) \
    if (v.size()>1000)
    {
        LogThreadAffinity();
        /*#pragma omp for reduction(+:sum) nowait 
        for (auto &x:v) 
        {
            //x = x*x;
            sum += x*x;
        }*/
        #pragma omp for reduction(+:sum) nowait 
        for (auto i=0;i<v.size();i++)
        {
        	auto x = v[i];
            //x = x*x;
            sum += x*x;
        }
    }
    LogTimeTaken(t1);
    t1 = NewTimer();
    T sum_serial = 0;
    for (auto &x:v) 
    {
        sum_serial += x*x;
    }
    LogTimeTaken(t1);
    std::cout<<__func__<<" "<<__LINE__<<" "<<v.size()<<" omp reduction "<<sum<<" serial sum  "<<sum_serial<<std::endl;
    return sum;
}

template<class T> void recursive_vector(std::vector<T> &v)
{
    T sum = 0;
    if (v.size() < 1000000) {
        sum = vector_sq_and_sum(v);
    }
    else 
    {
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 gen(rd()); // seed the generator
        std::uniform_int_distribution<> distr(0,v.size()); // define the range
        auto split = distr(gen);
        std::vector<T> left, right;
        int level=1;
        #ifdef USEOPENMP
        level = omp_get_level();
        #endif
        std::cout<<"@"<<__func__<<" L"<<__LINE__<<" spliting vector of size "<<v.size()<<" at "<<split<<" at level "<<level<<std::endl;
        #pragma omp parallel \
        default(none) shared(v, split, left, right) 
        {
            LogThreadAffinity();
            #pragma omp single
            {
                // #pragma taskgroup 
                {
                    #pragma task 
                    {
                        LogThreadAffinity();
                        std::string s = "Left split of " + std::to_string(split);
                        printf("%s\n",s.c_str());
                        std::copy(v.begin(), v.begin() + split, std::back_inserter(left));
                        recursive_vector(left);
                    }
                    #pragma task 
                    {
                        std::copy(v.begin() + split, v.end(), std::back_inserter(right));
                        recursive_vector(right);
                    }
                    #pragma taskwait 
                }
            }
            #pragma omp for 
            for (auto i=0;i<split;i++) v[i]=left[i];

            #pragma omp for 
            for (auto i=split;i<v.size();i++) v[i]=right[i-split];
        }
    }
}

// allocate mem and logs time taken and memory usage
void allocate_mem(unsigned long long Nentries, std::vector<int> &x_int, std::vector<int> &y_int, 
    std::vector<float> &x_float, std::vector<float> &y_float, 
    std::vector<double> &x_double, std::vector<double> &y_double) 
{
    auto time_mem = NewTimer();
    std::cout<<"Vectorization test running with "<<Nentries<<" requiring "<<Nentries*2*(sizeof(int)+sizeof(float)+sizeof(double))/1024./1024./1024.<<"GB"<<std::endl;
    x_int.resize(Nentries);
    y_int.resize(Nentries);
    x_float.resize(Nentries);
    y_float.resize(Nentries);
    x_double.resize(Nentries);
    y_double.resize(Nentries);
    LogMemUsage();
    LogTimeTaken(time_mem);
}

void deallocate_mem( std::vector<int> &x_int, std::vector<int> &y_int, 
    std::vector<float> &x_float, std::vector<float> &y_float, 
    std::vector<double> &x_double, std::vector<double> &y_double) 
{
    auto time_mem = NewTimer();
    x_int.clear();
    x_float.clear();
    x_double.clear();
    y_int.clear();
    y_float.clear();
    y_double.clear();
    x_int.shrink_to_fit();
    x_float.shrink_to_fit();
    x_double.shrink_to_fit();
    y_int.shrink_to_fit();
    y_float.shrink_to_fit();
    y_double.shrink_to_fit();
    LogMemUsage();
    LogTimeTaken(time_mem);
}

// tests openmp vectorization, logs time taken
void vector_vectorization_test(unsigned long long Nentries, 
    std::vector<int> &x_int, std::vector<int> &y_int, 
    std::vector<float> &x_float, std::vector<float> &y_float, 
    std::vector<double> &x_double, std::vector<double> &y_double) 
{
    #ifdef USEOPENMP
    int nthreads = omp_get_max_threads();
    #endif
    auto time_sillycalc = NewTimer();

    #ifdef USEOPENMP
    #pragma omp parallel \
    default(none) \
    shared(x_int, y_int, x_float, y_float, x_double, y_double, Nentries) \
    if (nthreads > 1)
    #endif
    {
#ifdef USEOPENMP 
    LogThreadAffinity();
    #pragma omp for schedule(static)
#endif
    for (auto i=0u; i<Nentries; i++) {
      x_int[i] = i;
      auto temp = x_int[i];
      y_int[i] = temp+temp*pow(temp,2) + temp/(temp+1);
    }
    }

    #ifdef USEOPENMP
    #pragma omp parallel for \
    default(none) \
    shared(x_int, y_int, x_float, y_float, x_double, y_double, Nentries) \
    schedule(static) if (nthreads > 1)
    #endif
    for (auto i=0u; i<Nentries; i++) {
      x_int[i] = i;
      auto temp = x_int[i];
      y_int[i] = temp+temp*pow(temp,2) + temp/(temp+1);
    }

    #ifdef USEOPENMP
    #pragma omp parallel for \
    default(none) shared(x_float, y_float, Nentries) \
    schedule(static) if (nthreads > 1)
    #endif
    for (auto i=0u; i<Nentries; i++) {
      x_float[i] = i;
      auto tempf = x_float[i];
      y_float[i] = tempf+tempf*pow(tempf,2) + tempf/(tempf+1);
    }

    #ifdef USEOPENMP
    #pragma omp parallel for \
    default(none) shared(x_double, y_double, Nentries) \
    schedule(static) if (nthreads > 1)
    #endif
    for (auto i=0u; i<Nentries; i++) {
      x_double[i] = i;
      auto tempd = x_double[i];
      y_double[i] = tempd+tempd*pow(tempd,2) + tempd/(tempd+1);
    }
    LogTimeTaken(time_sillycalc);
}

int main(int argc, char **argv) {
    LogParallelAPI();
    LogBinding();

    std::vector<int> x_int, y_int;
    std::vector<float> x_float, y_float;
    std::vector<double> x_double, y_double;
    unsigned long long Nentries = 24.0*1024.0*1024.0*1024.0/8.0/6.0;
    if (argc == 2) Nentries = atoi(argv[2]);

    //allocate, test vectorization and deallocate
    //functions showcase use of logging time taken and mem usage
    allocate_mem(Nentries, x_int, y_int, x_float, y_float, x_double, y_double);
    vector_vectorization_test(Nentries, x_int, y_int, x_float, y_float, x_double, y_double);
    deallocate_mem(x_int, y_int, x_float, y_float, x_double, y_double);

    //allocate mem and init vector using random numbers 
    unsigned long long N=100000000;
    x_double = allocate_and_init_vector<double>(N);
    //recursive call highlights thread affinity reporting
    auto t1 = NewTimer();
    recursive_vector(x_double);
    LogTimeTaken(t1);

}
