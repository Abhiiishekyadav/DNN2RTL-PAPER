// #include <cmath>
// #include "weights.h"
// using namespace std;
#include "cnn.h"

double relu(double x)
{
    return (x < 0) ? 0 : x;
}

void softmax(const double input[], int size, double output[]) {
#pragma HLS INLINE
     double max_val = input[0];
    for (int i = 1; i < size; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < size; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        sum_exp += std::exp(input[i] - max_val);
    }

    for (int i = 0; i < size; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] = std::exp(input[i] - max_val) / sum_exp;
        }

}


void convolution(const double flattenedImage[], const int kernels[], double output[], int imageWidth, int imageHeight, int imageDepth, int k_h, int k_w, const int biases[], int numKernels , char a_f)
{
    int output_h = imageHeight - k_h + 1;
    int output_w = imageWidth - k_w + 1;
    // int output_d = k_d;
#pragma HLS UNROLL factor=2
    for (int k = 0; k < numKernels; ++k)
    {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
#pragma HLS UNROLL factor=2
        for (int i = 0; i < output_h; ++i)
        {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
#pragma HLS UNROLL factor=2
            for (int j = 0; j < output_w; ++j)
            {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
#pragma HLS UNROLL factor=2
                output[k * output_h * output_w + i * output_w + j] = biases[k]; // Initialize output with bias for the current kernel

                for (int kd = 0; kd < imageDepth; ++kd)
                {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1

                    for (int ki = 0; ki < k_h; ++ki)
                    {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1

                        for (int kj = 0; kj < k_w; ++kj)
                        {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1

                            output[k * output_h * output_w + i * output_w + j] +=
                                flattenedImage[ (j+kj) + (i+ki)*imageWidth + kd*imageHeight*imageWidth]  * kernels[k * (k_h * k_w*imageDepth) + ki * k_w + kd*k_h*k_w+kj];
                        }
                    }
                }
                if(a_f=='R'){
                    output[k * output_h * output_w + i * output_w + j] =  relu(output[k * output_h * output_w + i * output_w + j]);
                }
            }
        }
    }
}

void maxPooling(const double image[], double output[], int imageWidth, int numChannels, int pool_size) {
    int i_h = (imageWidth);
    int i_w = i_h;
    int output_h = i_h / pool_size;
    int output_w = i_w / pool_size;

    for (int c = 0; c < numChannels; ++c) {
        for (int i = 0; i < output_h; ++i) {
            for (int j = 0; j < output_w; ++j) {
                double max_val = 0.0;
                for (int pi = 0; pi < pool_size;pi+=2) {
                    for (int pj = 0; pj < pool_size; pj+=2) {
                        max_val = fmax(max_val, image[c * (i_h * i_w) + (i * pool_size + pi) * i_w + (j * pool_size + pj)]);
                    }
                }
                output[c * (output_h * output_w) + i * output_w + j] = max_val;
            }
        }
    }
}


void fullyConnectedLayer(const double input[], double output[], const int weights[], const int bias[], int inputSize, int outputSize, char a_f) {
    for (int i = 0; i < outputSize; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] = bias[i];
        for (int j = 0; j < inputSize; ++j) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
            output[i] += input[j] * weights[i * inputSize + j];
        }

        if (a_f=='R') {
            output[i] = (output[i] < 0) ? 0 : output[i];
        }
    }
}







