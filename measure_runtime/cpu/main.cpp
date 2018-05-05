#include <iostream>
#include <string>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "io.h"
#include "calculations.h"

int main(int argc, const char * argv[])
{
    if(4 != argc)
    {
        printUsage();
        exit(0);
    }
    
    auto startLoading = std::chrono::high_resolution_clock::now();
    
    std::string trainingDataPath(argv[1]);
    std::string testDataPath(argv[2]);    
    
    Data training = readTrainingData(trainingDataPath);
    Data test     = readTestData(testDataPath);
    
    cv::Mat A = training.images;
    
    auto finishLoading = std::chrono::high_resolution_clock::now();
    
    auto startTraining = std::chrono::high_resolution_clock::now();
    
    cv::Mat meanImage(1, A.cols, CV_64F);
    cv::reduce(A, meanImage, 0, CV_REDUCE_AVG); // mean for each column

    for (int i = 0; i < A.rows; ++i)
    {
        A.row(i) -= meanImage;
        for (int j = 0; j < A.cols; ++j)
        {
            if (A.at<double>(i,j) < 0)
            {
                A.at<double>(i,j) = 0;
            }
        }
    }

    // Reduced covariance mx (see wikipedia). The transposed one is switched because we use the rows as images.
    cv::Mat S = A*A.t();
    
    cv::Mat eigenvalues;
    cv::Mat eigenvectors;
    
    cv::eigen(S, eigenvalues, eigenvectors);
    
    double componentNumUserInput = std::atof(argv[3]);
    unsigned int componentNum;
    if(componentNumUserInput < 1.0)
    {
        double varianceThreshold = componentNumUserInput;
        componentNum = getSufficientComponentNum(eigenvalues, varianceThreshold);
    }
    else
    {
        componentNum = round(componentNumUserInput);
    }
    
    cv::Mat V = restoreEigenvectors(A, eigenvectors, componentNum);
    
    cv::Mat W = getWeights(A, V);
    
    auto finishTraining = std::chrono::high_resolution_clock::now();
    
    auto startClassifying = std::chrono::high_resolution_clock::now();

    // classification
    
    classifyImages( test, training.labels, meanImage, V, W );
    
    auto finishClassifying = std::chrono::high_resolution_clock::now();
    
    //for(unsigned int i = 0; i < test.labels.size(); ++i)
    //{
    //    std::cout << i << ".\t-\t" << test.labels[i] << std::endl;
    //}
    
    
    std::cout << static_cast<std::chrono::duration<double>>(finishLoading     - startLoading    ).count() << "\t";
    std::cout << static_cast<std::chrono::duration<double>>(finishTraining    - startTraining   ).count() << "\t";
    std::cout << static_cast<std::chrono::duration<double>>(finishClassifying - startClassifying).count() << "\t";
    std::cout << static_cast<std::chrono::duration<double>>(finishClassifying - startLoading    ).count() << "\n";
    
    return 0;
}
