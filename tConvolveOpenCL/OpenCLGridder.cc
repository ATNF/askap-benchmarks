/// @copyright (c) 2007 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia
/// atnf-enquiries@csiro.au
///
/// This file is part of the ASKAP software distribution.
///
/// The ASKAP software distribution is free software: you can redistribute it
/// and/or modify it under the terms of the GNU General Public License as
/// published by the Free Software Foundation; either version 2 of the License,
/// or (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program; if not, write to the Free Software
/// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
///
/// @author Ben Humphreys <ben.humphreys@csiro.au>

// Include own header file first
#include "OpenCLGridder.h"

// System includes
#include <vector>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <string>
#include <cstring>

// OpenCL includes
#include <CL/cl.h>

// Local includes
#include "Stopwatch.h"

typedef float Real;
typedef std::complex<Real> Value;

void checkError(cl_int& error, const std::string& msg)
{
    if (error != CL_SUCCESS) {
        std::cerr << "Error (" << msg << ")" << error << std::endl;
        throw std::runtime_error("OpenCL Error");
    }
}

static std::string readFile(const std::string& filename)
{
    std::string str;
    std::string content;

    std::ifstream in;
    in.open(filename.c_str());
    std::getline(in, str);
    while (in) {
        content += str;
        content += '\n';
        std::getline(in, str);
    }

    return content;
}


void gridKernelOpenCL(const std::vector< std::complex<float> >& data, const int support,
        const std::vector< std::complex<float> >& C, const std::vector<int>& cOffset,
        const std::vector<int>& iu, const std::vector<int>& iv,
        std::vector< std::complex<float> >& grid, const int gSize,
        double &time)
{
    cl_int error;

    // Need to convert all std::vectors to C arrays for CUDA, then call
    // the kernel exec function. NOTE: The std::vector is the only STL
    // container which you can treat as an array like we do here.

    char *sProgramSource;
    {
        const std::string str = readFile("kernel.cl");
        sProgramSource = new char [str.size()+1];
        strcpy(sProgramSource, str.c_str());
    }

    // Get OpenCL platform count
    cl_uint NumPlatforms;
    error = clGetPlatformIDs(0, NULL, &NumPlatforms);
    checkError(error, "clGetPlatformIDs");

    // Get all OpenCL platform IDs
    cl_platform_id* PlatformIDs;
    PlatformIDs = new cl_platform_id[NumPlatforms];
    error = clGetPlatformIDs(NumPlatforms, PlatformIDs, NULL);
    checkError(error, "clGetPlatformIDs");

    // Select NVIDIA platform (this example assumes it IS present)
    char cBuffer[1024];
    cl_uint NvPlatform = 0;
    for(cl_uint i = 0; i < NumPlatforms; ++i)
    {
        clGetPlatformInfo(PlatformIDs[i], CL_PLATFORM_NAME, 1024, cBuffer, NULL);
        if(strstr(cBuffer, "NVIDIA") != NULL)
        {
            NvPlatform = i;
            break;
        }
    }
    // Get a GPU device on Platform (this example assumes one IS present)
    cl_device_id cdDevice;
    error = clGetDeviceIDs(PlatformIDs[NvPlatform], CL_DEVICE_TYPE_GPU, 1,
            &cdDevice, NULL);
    checkError(error, "clGetDeviceIDs");

    // Create a context
    cl_context hContext;
    hContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &error);
    checkError(error, "clCreateContext");

    // Create a command queue for the device in the context
    cl_command_queue hCmdQueue;
    hCmdQueue = clCreateCommandQueue(hContext, cdDevice, 0, &error);
    checkError(error, "clCreateCommandQueue");

    // Create & compile program
    cl_program hProgram;
    hProgram = clCreateProgramWithSource(hContext, 1, (const char**)&sProgramSource, 0, &error);
    checkError(error, "clCreateProgramWithSource");
    error = clBuildProgram(hProgram, 0, 0, "-Werror", 0, 0);
    checkError(error, "clBuildProgram");

    // Create kernel instance
    cl_kernel hKernel;
    hKernel = clCreateKernel(hProgram, "d_gridKernel", &error);
    checkError(error, "clCreateKernel");
    
    // Allocate device memory
    cl_mem d_grid;
    cl_mem d_C;
    cl_mem d_cOffset;
    cl_mem d_iu;
    cl_mem d_iv;
    cl_mem d_data;

    d_grid = clCreateBuffer(hContext,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            grid.size() * sizeof(Value),
            &grid[0],
            &error);
    checkError(error, "clCreateBuffer");

    d_C = clCreateBuffer(hContext,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            C.size() * sizeof(Value),
            (void *)&C[0],
            &error);
    checkError(error, "clCreateBuffer");

    d_cOffset = clCreateBuffer(hContext,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            cOffset.size() * sizeof(unsigned int),
            (void *)&cOffset[0],
            &error);
    checkError(error, "clCreateBuffer");

    d_iu = clCreateBuffer(hContext,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            iu.size() * sizeof(unsigned int),
            (void *)&iu[0],
            &error);
    checkError(error, "clCreateBuffer");

    d_iv = clCreateBuffer(hContext,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            iv.size() * sizeof(unsigned int),
            (void *)&iv[0],
            &error);
    checkError(error, "clCreateBuffer");

    std::cout << "Creatin data sized: " << data.size() * sizeof(Value) << std::endl;
    d_data = clCreateBuffer(hContext,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            data.size() * sizeof(Value),
            (void *)&data[0],
            &error);
    checkError(error, "clCreateBuffer");

    Stopwatch sw;
    sw.start();

    // Execute the kernel on the GPU
    const int sSize=2*support+1;
    int step = 1;

    // Setup parameter values
    error = clSetKernelArg(hKernel, 0, sizeof(cl_mem), &d_data);
    checkError(error, "clSetKernelArg");
    error = clSetKernelArg(hKernel, 1, sizeof(int), &support);
    checkError(error, "clSetKernelArg");
    error = clSetKernelArg(hKernel, 2, sizeof(cl_mem), &d_C);
    checkError(error, "clSetKernelArg");
    error = clSetKernelArg(hKernel, 3, sizeof(cl_mem), &d_cOffset);
    checkError(error, "clSetKernelArg");
    error = clSetKernelArg(hKernel, 4, sizeof(cl_mem), &d_iu);
    checkError(error, "clSetKernelArg");
    error = clSetKernelArg(hKernel, 5, sizeof(cl_mem), &d_iv);
    checkError(error, "clSetKernelArg");
    error = clSetKernelArg(hKernel, 6, sizeof(cl_mem), &d_grid);
    checkError(error, "clSetKernelArg");
    error = clSetKernelArg(hKernel, 7, sizeof(int), &gSize);
    checkError(error, "clSetKernelArg");

    // This loop begs some explanation. It steps through each spectral
    // sample either one at a time or two at a time. It will do two samples
    // if the two regions involved do not overlap. If they do, only a 
    // single point is gridded.
    //
    // Gridding two point is better than one because giving the GPU more
    // work to do allows it to hide memory latency better.
    int dSize = data.size();
    std::cout << "data.size() = " << dSize << std::endl;
    for (int dind = 0; dind < dSize; dind += step) {
        size_t local_work_size = sSize;
        size_t global_work_size = sSize * sSize;

        // Set the last parameter value, the only one that changes within
        // the loop.
        if (dind == 0) {
            error = clSetKernelArg(hKernel, 8, sizeof(int), &dind);
            checkError(error, "clSetKernelArg");
        }

        // Execute kernel
        std::cerr << "Executing kernel..." << dind << std::endl;
        error = clEnqueueNDRangeKernel(hCmdQueue, hKernel, 1, NULL, &global_work_size, &local_work_size, 0, 0, 0);
        checkError(error, "clEnqueueNDRangeKernel");
        std::cerr << "Done" << std::endl;
    }

    error = clFinish(hCmdQueue);
    time = sw.stop();
    checkError(error, "clFinish");

    // Copy results from device back to host
    error = clEnqueueReadBuffer(hCmdQueue, d_grid, CL_TRUE, 0,
            grid.size() * sizeof(Value),
            &grid[0], 0, 0, 0);
    checkError(error, "clEnqueueReadBuffer");

    // Free device memory
    clReleaseMemObject(d_grid);
    clReleaseMemObject(d_C);
    clReleaseMemObject(d_cOffset);
    clReleaseMemObject(d_iu);
    clReleaseMemObject(d_iv);
    clReleaseMemObject(d_data);

    // Other cleanup
    clReleaseKernel(hKernel);
    clReleaseProgram(hProgram);
    clReleaseCommandQueue(hCmdQueue);
    clReleaseContext(hContext);

}

void degridKernelOpenCL(const std::vector< std::complex<float> >& grid,
        const int gSize,
        const int support,
        const std::vector< std::complex<float> >& C,
        const std::vector<int>& cOffset,
        const std::vector<int>& iu,
        const std::vector<int>& iv,
        std::vector< std::complex<float> >& data,
        double &time)
{
    // Need to convert all std::vectors to C arrays for CUDA, then call
    // the kernel exec function. NOTE: The std::vector is the only STL
    // container which you can treat as an array like we do here.
/*
    // Allocate device memory
    Value *d_grid;
    Value *d_C;
    int *d_cOffset;
    int *d_iu;
    int *d_iv;
    Value *d_data;

    cudaError_t err;
    err = cudaMalloc((void **) &d_grid, grid.size() * sizeof(Value));
    checkerror(err);
    err = cudaMalloc((void **) &d_C, C.size() * sizeof(Value));
    checkerror(err);
    err = cudaMalloc((void **) &d_cOffset, cOffset.size() * sizeof(unsigned int));
    checkerror(err);
    err = cudaMalloc((void **) &d_iu, iu.size() * sizeof(unsigned int));
    checkerror(err);
    err = cudaMalloc((void **) &d_iv, iv.size() * sizeof(unsigned int));
    checkerror(err);
    err = cudaMalloc((void **) &d_data, data.size() * sizeof(Value));
    checkerror(err);

    // Copy host vectors to device arrays
    err = cudaMemcpy(d_grid, &grid[0], grid.size() * sizeof(Value), cudaMemcpyHostToDevice);
    checkerror(err);
    err = cudaMemcpy(d_C, &C[0], C.size() * sizeof(Value), cudaMemcpyHostToDevice);
    checkerror(err);
    err = cudaMemcpy(d_cOffset, &cOffset[0], cOffset.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    checkerror(err);
    err = cudaMemcpy(d_iu, &iu[0], iu.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    checkerror(err);
    err = cudaMemcpy(d_iv, &iv[0], iv.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    checkerror(err);
    err = cudaMemcpy(d_data, &data[0], data.size() * sizeof(Value), cudaMemcpyHostToDevice);
    checkerror(err);

    Stopwatch sw;
    sw.start();
    opencl_degridKernel((const Complex *)d_grid, gSize, support,
            (const Complex *)d_C, d_cOffset, d_iu, d_iv,
            (Complex *)d_data, data.size());
    cudaThreadSynchronize();
    time = sw.stop();

    // Copy device arrays back into the host vector
    err = cudaMemcpy(&data[0], d_data, data.size() * sizeof(Value), cudaMemcpyDeviceToHost);
    checkerror(err);

    // Free device memory
    cudaFree(d_grid);
    cudaFree(d_C);
    cudaFree(d_cOffset);
    cudaFree(d_iu);
    cudaFree(d_iv);
    cudaFree(d_data);
*/
}

