/*! \file gpuCommon.h
 *  \brief common gpu related items
 */

#ifndef GPUCOMMON_H
#define GPUCOMMON_H

#include <iostream>

#if defined(USEHIP)

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#define __GPU_API__ "HIP"

#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetDevice hipGetDevice
#define gpuSetDevice hipSetDevice
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceProp_t hipDeviceProp_t
#define gpuMemGetInfo hipMemGetInfo
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuError_t hipError_t
#define gpuGetLastError hipGetLastError
#define gpuSuccess hipSuccess
#define gpuGetErrorString(__err) hipGetErrorString(__err)

#elif defined(USECUDA)

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define __GPU_API__ "CUDA"

#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDevice cudaGetDevice
#define gpuSetDevice cudaSetDevice
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceProp_t cudaDeviceProp
#define gpuMemGetInfo cudaMemGetInfo
#define gpuMalloc cudaMalloc
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
#define gpuError_t cudaError_t
#define gpuGetLastError cudaGetLastError
#define gpuSuccess cudaSuccess
#define gpuGetErrorString(__err) cudaGetErrorString(__err)

#endif

// Error checking macro
#define gpuCheckErrors(msg) \
    do { \
        gpuError_t __err = gpuGetLastError(); \
        if (__err != gpuSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, gpuGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
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
