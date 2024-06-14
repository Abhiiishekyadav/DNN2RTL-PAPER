// cnn.h

#ifndef CNN_H
#define CNN_H

#include <cmath>
#include "weights.h"

double relu(double x);
void convolution(const double flattenedImage[], const int kernels[], double output[], int imageWidth, int imageHeight, int imageDepth, int k_h, int k_w, const int biases[], int numKernels,char a_f);
void maxPooling(const double image[], double output[], int imageWidth, int numChannels, int pool_size);
void fullyConnectedLayer(const double input[], double output[], const int weights[], const int bias[], int inputSize, int outputSize,char a_f);
void CNN(const double flattenedImage[], int imageWidth, int imageHeight, double output[]);

#endif // CNN_H
