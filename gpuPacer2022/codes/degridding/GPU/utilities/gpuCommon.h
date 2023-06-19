/*! \file gpuCommon.h
 *  \brief common gpu related items
 */

#ifndef GPUCOMMON_H
#define GPUCOMMON_H

#include <iostream>

#ifdef USEHIP
#define __GPU_API__ "HIP"
#define __GPU_TO_SECONDS__ 1.0/1000.0
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#elif defined(USECUDA)
#define __GPU_API__ "CUDA"
#define __GPU_TO_SECONDS__ 1.0/1000.0
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#endif


#if defined(USEHIP)
#define gpuMalloc hipMalloc
#define gpuHostMalloc hipHostMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetErrorString hipGetErrorString
#define gpuError_t hipError_t
#define gpuErr hipErr
#define gpuGetLastError hipGetLastError
#define gpuSuccess hipSuccess
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuDeviceProp_t hipDeviceProp_t
#define gpuGetDevice hipGetDevice
#define gpuSetDevice hipSetDevice
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceGetPCIBusId hipDeviceGetPCIBusId
#define gpuMemGetInfo hipMemGetInfo
#define gpuDeviceReset hipDeviceReset
#define gpuLaunchKernel(...) hipLaunchKernelGGL(__VA_ARGS__)

// specific instrinsics
#define gpuCmulf hipCmulf
#define gpuCaddf hipCaddf
#define make_gpuComplex make_hipComplex

// types
typedef hipComplex Complex;

// useful parameters 
#define MAX_SSIZE 256

#elif defined(USECUDA)
#define gpuMalloc cudaMalloc
#define gpuHostMalloc cudaMallocHost
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetErrorString cudaGetErrorString
#define gpuError_t cudaError_t
#define gpuErr cudaErr
#define gpuGetLastError cudaGetLastError
#define gpuSuccess cudaSuccess
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuDeviceProp_t cudaDeviceProp
#define gpuGetDevice cudaGetDevice
#define gpuSetDevice cudaSetDevice
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceGetPCIBusId cudaDeviceGetPCIBusId
#define gpuMemGetInfo cudaMemGetInfo
#define gpuDeviceReset cudaDeviceReset
#define gpuLaunchKernel(...) cudaLaunchKernel(__VA_ARGS__)

// specific instrinsics
#define gpuCmulf cuCmulf
#define gpuCaddf cuCaddf
#define make_gpuComplex make_cuComplex

// types
typedef cuComplex Complex;

// useful parameters
#define MAX_SSIZE 256

#endif

// if using a NVIDIA device, regardless of through HIP or CUDA, warpsize is 32
// AMD have 64
#ifdef __NVCC__
#define WARPSIZE 32
#else
#define WARPSIZE 64
#endif


// macro for checking errors in HIP API calls
#define gpuErrorCheck(call)                                                                 \
do{                                                                                         \
    gpuError_t __gpuErr = call;                                                               \
    if(__gpuErr != gpuSuccess){                                                               \
        std::cerr<<__GPU_API__<<" Fatal error : "<<gpuGetErrorString(__gpuErr) \
        <<" - "<<__FILE__<<":"<<__LINE__<<std::endl; \
        std::cerr<<" *** FAILED - ABORTING "<<std::endl; \
        exit(1);                                                                            \
    }                                                                                       \
}while(0)

#define gpuCheckErrors(msg) \
    do { \
        gpuError_t __gpuErr = gpuGetLastError(); \
        if (__gpuErr != gpuSuccess) { \
            std::cerr<<__GPU_API__<<" Fatal error: "<<msg<<" ("<<gpuGetErrorString(__gpuErr) \
            <<" at "<<__FILE__<<":"<<__LINE__<<")"<<std::endl; \
            std::cerr<<" *** FAILED - ABORTING "<<std::endl; \
            exit(1); \
        } \
    } while (0)

#define GPUReportDevice() \
    { \
    int device; \
    gpuDeviceProp_t devprop; \
    gpuGetDevice(&device); \
    gpuGetDeviceProperties(&devprop, device); \
    //std::cout << "[@" << __func__ << " L" << __LINE__ << "] :" << "Using " << __GPU_API__ << " Device " << device << ": " << devprop.name << std::endl; \
    }

#endif
