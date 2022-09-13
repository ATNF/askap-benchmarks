#pragma once

#include <iostream>
#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>


/// \defgroup HIP_Error_Check
//@{
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __func__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, const char *func, int line, bool abort=true)
{
   if (code != hipSuccess) 
   {
      std::cerr<<"GPUassert: "<<hipGetErrorString(code)<<" @ "<<file<<":func:"<<func<<":L"<<line<<std::endl;
      if (abort) exit(code);
   }
}

// Error checking macro
#define gpuCheckErrors(msg) \
    do { \
        hipError_t __err = hipGetLastError(); \
        if (__err != hipSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, hipGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

//@}
