#include <iostream>
#include <string>

#include "io.h"

/// images should be in row major order
/// componentNumIndicator is either 0              - all of them are used
///                              in (0,1) range - explanatory power based component selection
///                              or larger      - rounded to intiger and that many are used
extern "C" void gpuAssistedClassification( Data* training
                                         , Data* test
                                         , double componentNumIndicator
                                         );

int main(int argc, const char * argv[])
{
    if(4 != argc)
    {
        printUsage();
        exit(0);
    }
    
    std::string trainingDataPath(argv[1]);
    std::string testDataPath(argv[2]);    
    
    Data training = readTrainingData(trainingDataPath);
    Data test     = readTestData(testDataPath);    
    
    double componentNumUserInput = std::atof(argv[3]);
    
    gpuAssistedClassification(&training, &test, componentNumUserInput);
   
    for(std::size_t i = 0; i < test.size; ++i)
    {
        std::cout << i << ".\t-\t" << test.labels[i] << std::endl;
    }
    
    
    return 0;
}
