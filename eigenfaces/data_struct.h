#ifndef DATA_STRUCT_H
#define DATA_STRUCT_H

#include "data_struct.h"

void printUsage();

struct Data
{
    cv::Mat_<double> images;
    std::vector<unsigned int> labels;
};

#endif //DATA_STRUCT_H
