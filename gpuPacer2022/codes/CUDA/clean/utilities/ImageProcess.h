#pragma once

#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <cmath>
#include <sys/stat.h>

class ImageProcess
{
public:
	std::vector<float> readImage(const std::string& filename);
	void writeImage(const std::string& filename, std::vector<float>& image);
	size_t checkSquare(std::vector<float>& vec);
};