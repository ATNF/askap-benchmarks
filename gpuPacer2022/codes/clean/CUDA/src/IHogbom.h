// Solver interface 
#pragma once

#include <vector>
#include <iostream>

#include "../utilities/Parameters.h"

class IHogbom
{
protected:
    // Private members
    const std::vector<float>& dirty;
    const std::vector<float>& psf;
    const size_t imageWidth;
    std::vector<float>& model;
    std::vector<float>& residual;

    // Private methods
    virtual void subtractPSF(const size_t peakPos,
        const size_t psfPeakPos,
        const float absPeakVal) = 0;

public:
    IHogbom(const std::vector<float>& dirty, 
        const std::vector<float>& psf,
        const size_t imageWidth,
        std::vector<float>& model,
        std::vector<float>& residual) : dirty{dirty}, psf{psf}, imageWidth{imageWidth}, 
        model{model}, residual{residual} {}

    virtual ~IHogbom() { std::cout << "IHogbom interface destructor" << std::endl; }

    // Public methods
    virtual void deconvolve() = 0;
    
};