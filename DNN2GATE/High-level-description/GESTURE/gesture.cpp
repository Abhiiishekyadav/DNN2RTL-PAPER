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


void CNN(const double flattenedImage[], int imageWidth, int imageHeight, double output[]) {
#pragma HLS INTERFACE s_axilite port=imageWidth bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=imageHeight bundle=CTRL
    #pragma HLS INTERFACE m_axi depth=50 port=flattenedImage offset=slave bundle=DATA
    #pragma HLS INTERFACE m_axi depth=50 port=output offset=slave bundle=DATA
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    // Convolutional layer1
    int num_kernels1 = 32;
    int kernelWidth1 = 5;
    int kernelHeight1 = 5;
    int imagedepth1 = 1;
    char a_f = 'R';
    double conv1Output[32*58*158];
    convolution(flattenedImage, convolution1_weights, conv1Output, imageWidth, imageHeight, imagedepth1, kernelHeight1, kernelWidth1 , convolution1_bias,num_kernels1, a_f);
   // Max pooling layer1
    int pool_size = 2;
    double pool1Output[29*79*32];
    maxPooling(conv1Output, pool1Output, imageWidth - kernelWidth1 + 1 ,num_kernels1, pool_size);
       // Convolutional layer2
    int num_kernels2 = 64;
    int kernelWidth2 = 3;
    int kernelHeight2 = 3;
    int imageWidth2=29;
    int imageHeight2 = 79;
    int imagedepth2 = 32;
    // char a_f = 'R';
    double conv2Output[27*77*64];
    convolution(pool1Output, convolution2_weights, conv2Output, imageWidth2, imageHeight2, imagedepth2, kernelHeight2, kernelWidth2 , convolution2_bias, num_kernels2, a_f);
   // Max pooling layer2
    //int pool_size = 2;
    double pool2Output[13*38*64];
    maxPooling(conv2Output, pool2Output, imageWidth2 - kernelWidth2 + 1 ,num_kernels2, pool_size);

       // Convolutional layer3
    int num_kernels3 = 64;
    int kernelWidth3 = 3;
    int kernelHeight3 = 3;
    int imageWidth3=13;
    int imageHeight3 = 38;
    int imagedepth3 = 64;
    // char a_f = 'R';
    double conv3Output[11*36*64];
    convolution(pool2Output, convolution3_weights, conv3Output, imageWidth3, imageHeight3, imagedepth3, kernelHeight3, kernelWidth3 , convolution3_bias, num_kernels3, a_f);
   // Max pooling layer2
   // int pool_size = 2;
    double pool3Output[5*18*64];
    maxPooling(conv3Output, pool3Output, imageWidth3 - kernelWidth3 + 1 ,num_kernels3, pool_size);


    // fully connected layer. 
    char a_f2='R';
    double fully1output[128];
    fullyConnectedLayer(pool3Output,fully1output,dense1_weights,dense1_bias,256,128,a_f2);
 
    // fully connected layer. 
   // char a_f2='R';
    double fully2output[256];
    fullyConnectedLayer(fully1output,fully2output,dense2_weights,dense2_bias,256,256,a_f2);
     // fully connected layer. 
   // char a_f2='R';
    double fully3output[256];
    fullyConnectedLayer(fully2output,fully3output,dense3_weights,dense3_bias,256,256,a_f2);
    
    char a_f3='S';
    double fully4output[10];
    fullyConnectedLayer(fully3output,fully4output,dense4_weights,dense4_bias,256,10,a_f3);
    char a_f4='S';
    if(a_f4=='S'){
        softmax(fully4output,10,output);
    }
}







