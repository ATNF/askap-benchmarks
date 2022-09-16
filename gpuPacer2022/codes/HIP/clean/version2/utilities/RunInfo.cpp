#include "RunInfo.h"

void RunInfo::GetInfo() const
{
    int nDevices=0;
    std::string testname = "HIP";
    std::string runinfo = "";
    hipGetDeviceCount(&nDevices);
    if (nDevices == 0)
    {
        std::cout<<" No HIP devices found "<<std::endl;
        exit(9);
    }
    for (auto i=0;i<nDevices;i++)
    {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        runinfo = "";
        runinfo += "Device name:" + std::string(prop.name);
        runinfo += "\nCompute Units: " + std::to_string(prop.multiProcessorCount);
        runinfo += "\nMax Work Group Size: " + std::to_string(prop.warpSize);
        runinfo += "\nLocal Mem Size: " + std::to_string(prop.sharedMemPerBlock);
        runinfo += "\nGlobal Mem Size: " + std::to_string(prop.totalGlobalMem);
        std::cout<<runinfo<<" "<<std::endl;
    }
}