#include "IL/il.h"
#include <stdlib.h> /* because of malloc() etc. */
#include <IL/devil_cpp_wrapper.hpp>
#include  <stdint.h>
#include <cuda_runtime.h>

#define get_elem(array, Row, Column, RowWidth) \
(((Pixel*)((char*)array + (Row)*(RowWidth)*sizeof(uint32_t)))[(Column)])

__constant__ __device__ int fil_kernel[441];

struct Pixel {
    uint8_t r,g,b,a;

    __host__ __device__ Pixel &operator=(const Pixel &otherPixel) { 
         *((uint32_t *)this) = (*(uint32_t *)&otherPixel);
         return *this;
    }
    __host__ __device__ Pixel(const Pixel &otherPixel) {
       *((uint32_t *)this) = (*(uint32_t *)&otherPixel);
    }

    __host__ __device__ Pixel(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
       this->r=r;
       this->g=g;
       this->b=b;
       this->a=a;
    }
    __host__ __device__ Pixel() {
       this->r=0;
       this->g=0;
       this->b=0;
       this->a=0;
    }

    __host__ __device__ Pixel operator*(int coefficient);

    __host__ __device__ Pixel operator+(const Pixel &otherPixel);

};

__global__ void fil1Kernel(Pixel * source, Pixel * dest, ILuint w, ILuint h, int fils)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i, j;
    Pixel tmp(0,0,0,0), temp;
    if ((row<h)&&(col<w)) {
       for (i=-fils; i<=fils; i++)
          for (j=-fils; j<=fils; j++) {
             tmp=tmp+get_elem(source,row+i, col+j, w+2*fils)*fil_kernel[(i+fils)*(2*fils+1)+j+fils];
           }
      get_elem(dest,row,col,w)=tmp;
    }
}

int main(int argc, const char * argv[])
{
    ILuint handle, w, h;
    int fils;
    cudaEvent_t start,stop;
    float time;
    
        if (argc<4) {
          printf("Usage: filtering <flag: i or m> <path or matrix size> <radius of filters>\n");
          return 0;
        } 

	ilInit();
	fils=atoi(argv[3])-1;

	ilEnable(IL_ORIGIN_SET);
	
	ilGenImages(1, & handle);
	ilBindImage(handle);
	ILboolean loaded = ilLoadImage(argv[2]);
	
	if (loaded == IL_FALSE)
		return -1; /* error encountered during loading */
	
	w = ilGetInteger(IL_IMAGE_WIDTH); // getting image width
	h = ilGetInteger(IL_IMAGE_HEIGHT); // and height
	printf("Our image resolution: %dx%d\n", w, h);
	
	int memory_needed = w * h * 4 * sizeof(unsigned char);
	
	ILubyte * data = (ILubyte *)malloc(memory_needed);
	
	ilCopyPixels(0, 0, 0, w, h, 1, IL_RGBA, IL_UNSIGNED_BYTE, data);
	
	int i, j, k;
        Pixel * pixelstemp=new Pixel[w*h];
        Pixel * pixels=new Pixel[(w+2*fils)*(h+2*fils)];
        Pixel temp;

	for(i = 0; i < memory_needed; i=i+4) {
            temp.r=data[i];
            temp.g=data[i+1];
            temp.b=data[i+2];
            temp.a=0,3*temp.r+0,59*temp.g+0,11*temp.b;
	    pixelstemp[i/4] = temp;
        }
        temp=get_elem(pixelstemp,0,0,w);
        for (i=0; i<fils; i++)
           for (j=0; j<fils; j++)
	     get_elem(pixels, i, j, w+2*fils)=temp;

        temp=get_elem(pixelstemp,0,w-1,w);
        for (i=h+fils; i<h+2*fils; i++)
           for (j=0; j<fils; j++)
	     get_elem(pixels, i, j, w+2*fils)=temp;

        temp=get_elem(pixelstemp,h-1,w-1,w);
        for (i=h+fils; i<h+2*fils; i++)
           for (j=w+fils; j<w+2*fils; j++)
	     get_elem(pixels, i, j, w+2*fils)=temp;

        temp=get_elem(pixelstemp,h-1,0,w);
        for (i=0; i<fils; i++)
           for (j=w+fils; j<w+2*fils; j++)
	     get_elem(pixels, i, j, w+2*fils)=temp;

        for (i=fils; i<h+fils; i++)
           for (j=fils; j<w+fils; j++) {
	     get_elem(pixels, i, j, w+2*fils)=get_elem(pixelstemp, i-fils, j-fils, w);
        }

        for (i=fils; i<h+fils; i++)
           for (j=0; j<fils; j++)
	     get_elem(pixels, i, j, w+2*fils)=get_elem(pixels, i, fils, w+2*fils);

        for (i=fils; i<h+fils; i++)
           for (j=w+fils; j<w+2*fils; j++)
	     get_elem(pixels, i, j, w+2*fils)=get_elem(pixels, i, h+fils-1, w+2*fils);

        for (i=0; i<fils; i++)
           for (j=fils; j<w+fils; j++)
	     get_elem(pixels, i, j, w+2*fils)=get_elem(pixels, fils, j, w+2*fils);

        for (i=h+fils; i<h+2*fils; i++)
           for (j=fils; j<w+fils; j++)
	     get_elem(pixels, i, j, w+2*fils)=get_elem(pixels, w+fils-1, j, w+2*fils);

        int * fil1 = new int[441];
        fil1[(2 * fils + 1) * fils + fils + 1] = 1;
        for (i = 0; i < 2 * fils + 1; i++)
           for (j = 0; j < 2 * fils + 1; j++)
              if ((i != fils + 1) && (j != fils + 1)) {
                 k=min(i, min(j, min(2 * fils - i, 2 * fils - j))) + 1;
                 fil1[i * (2 * fils + 1) + j] =- k;
                 fil1[(2 * fils + 1) * fils + fils + 1] += k;
              }

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaMemcpyToSymbol(fil_kernel, fil1, sizeof(fil1));
        Pixel * source, * dest;
        cudaMalloc(&source, (w+2*fils)*(h+2*fils)*sizeof(Pixel));
        cudaMalloc(&dest, w*h*sizeof(Pixel));

        cudaMemcpy(source, pixels, (w+2*fils)*(h+2*fils)*sizeof(Pixel), cudaMemcpyHostToDevice);

        dim3 dimBlock(32, 32);
        dim3 dimGrid(((w-1) / dimBlock.x) + 1, ((h-1) / dimBlock.y) + 1);

        cudaEventRecord(start, 0);

        fil1Kernel<<<dimGrid, dimBlock>>>(source,dest,w,h,fils);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
    
        cudaEventElapsedTime(&time, start, stop);
        printf("%f ms\n", time);

        cudaMemcpy(pixelstemp, dest, w*h*sizeof(Pixel), cudaMemcpyDeviceToHost);

        k=0;
        for (i=0; i<h; i++)
           for (j=0; j<w; j++) {
              temp=get_elem(pixelstemp,i,j,w);
              data[k]=temp.r;
              data[k+1]=temp.g;
              data[k+2]=temp.b;
              data[k+3]=temp.a;
              k=k+4;
           }

	ilSetPixels(0, 0, 0, w, h, 1, IL_RGBA, IL_UNSIGNED_BYTE, data);
	
	ilSaveImage("our_result.png");
	
	ilDeleteImages(1, & handle);
	free(data); data = NULL; free(pixels); free(pixelstemp);
        cudaFree(source); cudaFree(dest);
	
	return 0;
}
