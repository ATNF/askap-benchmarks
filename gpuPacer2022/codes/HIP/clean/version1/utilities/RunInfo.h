/*! \file RunInfo.h
    \brief Class to report runtime info 
*/
#pragma once

#include <string>
#include <vector>
#include <iostream>
#include "hip/hip_runtime.h"

class RunInfo
{
public:
    void GetInfo() const;
};
