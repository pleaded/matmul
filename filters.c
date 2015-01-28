#include <iostream>
#include <cstdlib>
#include "lodepng.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <functional>

void filter(unsigned char* input_image, unsigned char* output_image, int width, int height);

void getError(cudaError_t err);

void getError(cudaError_t err) {
    if(err != cudaSuccess) {
        std::cout << "Error " << cudaGetErrorString(err) << std::endl;
    }
}

__global__
void blur(unsigned char* input_image, unsigned char* output_image, int width, int height) {

    const unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset-x)/width;
    int fsize = 5; // Filter size
    if(offset < width*height) {

        float output_red = 0;
        float output_green = 0;
        float output_blue = 0;
        int hits = 0;
        for(int ox = -fsize; ox < fsize+1; ++ox) {
            for(int oy = -fsize; oy < fsize+1; ++oy) {
                if((x+ox) > -1 && (x+ox) < width && (y+oy) > -1 && (y+oy) < height) {
                    const int currentoffset = (offset+ox+oy*width)*3;
                    output_red += input_image[currentoffset]; 
                    output_green += input_image[currentoffset+1];
                    output_blue += input_image[currentoffset+2];
                    hits++;
                }
            }
        }
        output_image[offset*3] = output_red/hits;
        output_image[offset*3+1] = output_green/hits;
        output_image[offset*3+2] = output_blue/hits;
        }
}

static __constant__ double gausDevice[7][7];


__global__
void kernel2(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    int fsize = 3; // Filter size

    const unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset-x)/width;
    if(offset < width*height) {

        float output_red = 0;
        float output_green = 0;
        float output_blue = 0;
        for(int ox = -fsize; ox < fsize+1; ++ox) {
            for(int oy = -fsize; oy < fsize+1; ++oy) {
                if((x+ox) > -1 && (x+ox) < width && (y+oy) > -1 && (y+oy) < height) {
                    const int currentoffset = (offset+ox+oy*width)*3;
                    output_red += input_image[currentoffset]*gausDevice[ox][oy]; 
                    output_green += input_image[currentoffset+1]*gausDevice[ox][oy];
                    output_blue += input_image[currentoffset+2]*gausDevice[ox][oy];
                }
            }
        }
        output_image[offset*3] = output_red;
        output_image[offset*3+1] = output_green;
        output_image[offset*3+2] = output_blue;
        }
}


const int MAX_STREAM_COUNT = 1;

typedef struct
{
    cudaStream_t stream;
} devicePlan;

void filter (unsigned char* input_image, unsigned char* output_image, int width, int height) {

    unsigned char* dev_input;
    unsigned char* dev_output;
    unsigned char* dev_output2;
    double gaus[7][7] =
    {
    {0.00000067,    0.00002292,     0.00019117,     0.00038771,     0.00019117,     0.00002292,     0.00000067},
    {0.00002292,    0.00078634,     0.00655965,     0.01330373,     0.00655965,     0.00078633,     0.00002292},
    {0.00019117,    0.00655965,     0.05472157,     0.11098164,     0.05472157,     0.00655965,     0.00019117},
    {0.00038771,    0.01330373,     0.11098164,     0.22508352,     0.11098164,     0.01330373,     0.00038771},
    {0.00019117,    0.00655965,     0.05472157,     0.11098164,     0.05472157,     0.00655965,     0.00019117},
    {0.00002292,    0.00078633,     0.00655965,     0.01330373,     0.00655965,     0.00078633,     0.00002292},
    {0.00000067,    0.00002292,     0.00019117,     0.00038771,     0.00019117,     0.00002292,     0.00000067}
    };


    getError(cudaMemcpyToSymbol(gausDevice, gaus, 7 * 7 * sizeof(double)));

    unsigned int heightPerStream = height / MAX_STREAM_COUNT;

    devicePlan plan[MAX_STREAM_COUNT];
    getError(cudaSetDevice(0));

    cudaMalloc((void**) &dev_input, width*height*3*sizeof(unsigned char));
    cudaHostRegister(dev_input, width*height*3*sizeof(unsigned char), 0);
    for (int i = 0; i < MAX_STREAM_COUNT; i++)
    {
        getError(cudaStreamCreate(&plan[i].stream));

        getError(cudaMemcpyAsync(dev_input  + i *  width*heightPerStream*3*sizeof(unsigned char), input_image + i *  width*heightPerStream*3*sizeof(unsigned char), width*heightPerStream*3*sizeof(unsigned char), cudaMemcpyHostToDevice, plan[i].stream));

        getError(cudaMalloc((void**) &dev_output, width*heightPerStream*3*sizeof(unsigned char)));

        getError(cudaMalloc((void**) &dev_output2, width*heightPerStream*3*sizeof(unsigned char)));
    
        dim3 blockDims(512,1,1);
        dim3 gridDims((unsigned int) ceil((double)(width*heightPerStream*3/blockDims.x)), 1, 1 );

        blur<<<gridDims, blockDims, 0, plan[i].stream>>>(dev_input + i *  width*heightPerStream*3*sizeof(unsigned char), dev_output, width, heightPerStream); 
        kernel2<<<gridDims, blockDims, 0, plan[i].stream>>>(dev_output, dev_output2, width, heightPerStream); 

        getError(cudaMemcpyAsync(output_image + i *  width*heightPerStream*3*sizeof(unsigned char), dev_output2 , width*heightPerStream*3*sizeof(unsigned char), cudaMemcpyDeviceToHost, plan[i].stream));
    }



    for (int i = 0; i < MAX_STREAM_COUNT; i++)
    {
        cudaStreamSynchronize(plan[i].stream);

        getError(cudaStreamDestroy(plan[i].stream));
    }

    getError(cudaFree(dev_output));
    getError(cudaFree(dev_output2));
    getError(cudaFree(dev_input));
}

int main(int argc, char ** argv) 
{
    const char * input_file = argv[1];
    const char * output_file = argv[2];

    std::vector<unsigned char> in_image;
    unsigned int width, height;
    
    lodepng::decode(in_image, width, height, input_file);

    unsigned char * input_image = new unsigned char[(in_image.size()*3)/4];
    unsigned char * output_image = new unsigned char[(in_image.size()*3)/4];
    int where = 0;
    for(int i = 0; i < in_image.size(); ++i) {
       if((i + 1) % 4 != 0) {
           input_image[where] = in_image.at(i);
           output_image[where] = 255;
           where++;
       }
    }

    filter(input_image, output_image, width, height); 

    std::vector<unsigned char> out_image;
    for(int i = 0; i < in_image.size(); ++i) {
        out_image.push_back(output_image[i]);
        if((i + 1) % 3 == 0) {
            out_image.push_back(255);
        }
    }
    
    lodepng::encode(output_file, out_image, width, height);

    delete[] input_image;
    delete[] output_image;

    return 0;
}