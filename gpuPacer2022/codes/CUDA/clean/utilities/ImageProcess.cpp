#include "ImageProcess.h"

using std::ofstream;
using std::ifstream;
using std::string;
using std::vector;
using std::ios;
using std::cerr;
using std::cout;
using std::endl;

std::vector<float> ImageProcess::readImage(const std::string& filename)
{
    struct stat results;
    if (stat(filename.c_str(), &results) != 0)
    {
        cerr << "Error: Could not stat " << filename << endl;
        exit(1);
    }

    vector<float> image(results.st_size / sizeof(float));
    ifstream file(filename.c_str(), ios::in | ios::binary);
    file.read(reinterpret_cast<char*>(&image[0]), results.st_size);
    file.close();

    return image;
}

void ImageProcess::writeImage(const string& filename, vector<float>& image)
{
    ofstream file(filename.c_str(), ios::out | ios::binary | ios::trunc);
    file.write(reinterpret_cast<char*>(&image[0]), image.size() * sizeof(float));
    file.close();
}

size_t ImageProcess::checkSquare(vector<float>& vec)
{
    const size_t SIZE = vec.size();
    const size_t SINGLE_DIM = sqrt(SIZE);
    if (SINGLE_DIM * SINGLE_DIM != SIZE)
    {
        cerr << "Error: Image is not square" << endl;
        exit(-1);
    }

    return SINGLE_DIM;
}

