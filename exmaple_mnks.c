#include <cuda.h>
#include <iostream>
#include <cstdio>
#include <complex>
#include <fstream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
//#define get_elem(array, Row, Column) (((float*)((float*)array.ptr + (Row)*array.pitch/sizeof(float)))[(Column)])
 #define get_elem(array, Row, Column) (((float*)((char*)array.ptr + (Row)*array.pitch))[(Column)])

// Thread block size
//#define BLOCK_SIZE 16
#define BLOCK_WIDTH 32
#define WARP_SIZE 32

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const cudaPitchedPtr, const cudaPitchedPtr, cudaPitchedPtr);
__global__ void MatMulTransKernel(const cudaPitchedPtr, const cudaPitchedPtr, cudaPitchedPtr);
__global__ void MatTransposeKernel(const cudaPitchedPtr, cudaPitchedPtr);

// Matrix multiplication - Host code
int main (int argc, char *argv[])
{
	int m,n,k,pad;
	int blockHeight;
    int trans=0;
	m=16+13;
	n=16+20;
	k=16+31;
	m=16+3;
	n=16;
	k=16+17;
	m=16;
	n=16;
	k=16;
	FILE *f;
	f = fopen("MatMultRes.txt","w");
	FILE *fpr;
	fpr = fopen("MatMultResProve.txt","w");
	
	if (argc<4) {
		printf("Not enough arguments");
		return 0;
	}
	m=atoi(argv[1]);
	n=atoi(argv[2]);
	k=atoi(argv[3]);
	blockHeight=atoi(argv[4]);
	pad=0;
	if (argc==6) pad=atoi(argv[5]);
	printf("m %d, n %d, k %d, pad %d\n",m,n,k,pad);
	//blockHeight=BLOCK_WIDTH;   

	FILE *ftwp;
	if (!pad) ftwp = fopen("WorkTimeWP.txt","w");
	FILE *ftp16;
	if ((pad)&&(blockHeight==16)) ftp16 = fopen("WorkTimeP16.txt","w");
	FILE *ftp32;
	if ((pad)&&(blockHeight==32)) ftp32 = fopen("WorkTimeP32.txt","w");

	cudaPitchedPtr A = make_cudaPitchedPtr(malloc(n*m*sizeof(float)),n*sizeof(float),n,m);
	cudaPitchedPtr B = make_cudaPitchedPtr(malloc(n*k*sizeof(float)),k*sizeof(float),k,n);
	cudaPitchedPtr C = make_cudaPitchedPtr(malloc(m*k*sizeof(float)),k*sizeof(float),k,m);
	cudaPitchedPtr Cpr = make_cudaPitchedPtr(malloc(m*k*sizeof(float)),k*sizeof(float),k,m);

	A.ptr=new float [A.xsize*A.ysize];
	B.ptr=new float [B.xsize*B.ysize];
	C.ptr=new float [C.xsize*C.ysize];
	Cpr.ptr=new float [Cpr.xsize*Cpr.ysize];

	for (int i=0; i<A.xsize*A.ysize;i++){
      ((float*)A.ptr)[i]=(float)rand() / RAND_MAX;
      //((float*)A.ptr)[i]=1;
	}

	for (int i=0; i<B.xsize*B.ysize;i++){
      ((float*)B.ptr)[i]=(float)rand() / RAND_MAX;
      //((float*)B.ptr)[i]=2;
	}

	for(int i = 0; i < A.ysize; i++){
		for(int j = 0; j < B.xsize; j++) {
			float r = 0;
			for(int h = 0; h < A.xsize; h++){
				r += ((float*)A.ptr)[i * A.xsize + h] * ((float*)B.ptr)[h * B.xsize + j];
			}
			((float*)Cpr.ptr)[i * Cpr.xsize + j] = r;
			//fprintf(fpr, "%f ", r);
         
		}
	}
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	size_t size;
	cudaPitchedPtr d_A, d_B, d_C;
    // Load A and B to device memory
	if (!pad){ //ordinary
		d_A = make_cudaPitchedPtr(malloc(n*m*sizeof(float)),n*sizeof(float),A.xsize,A.ysize);
		size = A.pitch * A.ysize;
		cudaMalloc(&d_A.ptr, size);
		cudaMemcpy(d_A.ptr, A.ptr, size,cudaMemcpyHostToDevice);
		d_B = make_cudaPitchedPtr(malloc(n*k*sizeof(float)),k*sizeof(float),B.xsize,B.ysize);
		size = B.pitch * B.ysize;
		cudaMalloc(&d_B.ptr, size);
		cudaMemcpy(d_B.ptr, B.ptr, size, cudaMemcpyHostToDevice);
		d_C = make_cudaPitchedPtr(malloc(k*m*sizeof(float)),k*sizeof(float),C.xsize,C.ysize);
		size = C.pitch * C.ysize;
		cudaMalloc(&d_C.ptr, size);
	} else { //with padding
		d_A = make_cudaPitchedPtr(0,0, A.xsize, A.ysize);
		size_t width_in_bytes = A.xsize * sizeof(float);  
		cudaMallocPitch(&d_A.ptr, &d_A.pitch, width_in_bytes, A.ysize);
		cudaMemcpy2D( d_A.ptr, d_A.pitch, A.ptr, A.pitch, A.pitch, A.ysize, cudaMemcpyHostToDevice);
    
		printf ("Transpose% d \n" ,blockHeight%WARP_SIZE);
    
		if (blockHeight%WARP_SIZE){
			cudaPitchedPtr AT = make_cudaPitchedPtr(malloc(n*m*sizeof(float)),m*sizeof(float),m,n);
			AT.ptr=new float [AT.xsize*AT.ysize];     
			width_in_bytes = A.ysize * sizeof(float);
			cudaPitchedPtr d_AT = make_cudaPitchedPtr(0,0, A.ysize, A.xsize);
			cudaMallocPitch(&d_AT.ptr, &d_AT.pitch, width_in_bytes, AT.ysize);
			dim3 dimBlock1(BLOCK_WIDTH, blockHeight);
			dim3 dimGrid1((A.xsize-1) / dimBlock1.x+1, (A.ysize-1) / dimBlock1.y+1);
			MatTransposeKernel<<<dimGrid1, dimBlock1>>>(d_A, d_AT);
			cudaMemcpy2D( AT.ptr, AT.pitch, d_AT.ptr, d_AT.pitch, AT.pitch, AT.ysize, cudaMemcpyDeviceToHost);
			cudaFree(d_A.ptr);  
			cudaFree(d_AT.ptr);  
			d_A = make_cudaPitchedPtr(0,0, AT.xsize, AT.ysize);
			width_in_bytes = AT.xsize * sizeof(float);  
			cudaMallocPitch(&d_A.ptr, &d_A.pitch, width_in_bytes, AT.ysize);
			cudaMemcpy2D( d_A.ptr, d_A.pitch, AT.ptr, AT.pitch, AT.pitch, AT.ysize, cudaMemcpyHostToDevice);

			trans=1;
		}


    
		d_B = make_cudaPitchedPtr(0,0, B.xsize, B.ysize);
		width_in_bytes = B.xsize * sizeof(float);
		cudaMallocPitch(&d_B.ptr, &d_B.pitch, width_in_bytes, B.ysize);
		cudaMemcpy2D( d_B.ptr, d_B.pitch, B.ptr, B.pitch, B.pitch, B.ysize, cudaMemcpyHostToDevice);

		d_C = make_cudaPitchedPtr(0,0, C.xsize, C.ysize);
		width_in_bytes = C.xsize * sizeof(float);
		cudaMallocPitch(&d_C.ptr, &d_C.pitch, width_in_bytes, C.ysize);



	}


    // Invoke kernel
    dim3 dimBlock(BLOCK_WIDTH, blockHeight);
//printf ("GridSize %d, %d",(B.xsize-1) / dimBlock.x+1, (A.ysize-1) / dimBlock.y+1);
    dim3 dimGrid((B.xsize-1) / dimBlock.x+1, (A.ysize-1) / dimBlock.y+1);
//printf("Bsize %d, %d", d_B.xsize, d_B.ysize);
    

    cudaEventRecord(start, 0);
    printf("Transpose is %d\n", trans);
    if (trans) MatMulTransKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	else MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    //MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop, 0);

    float time = 0;
     //��������������� � �������� ��������� �������
	cudaEventSynchronize(stop);
     //������������ ����� ������ GPU
	cudaEventElapsedTime(&time, start, stop);
 
    //������� ����� ������� � ������
    printf("GPU compute time: %f\n", time);
	if (!pad) fprintf(ftwp, "%f ", time);
	if ((pad)&&(blockHeight==16)) fprintf(ftp16, "%f ", time);
	if ((pad)&&(blockHeight==32)) fprintf(ftp32, "%f ", time);
	// Read C from device memory
	if (!pad) cudaMemcpy(C.ptr, d_C.ptr, size, cudaMemcpyDeviceToHost);
	else cudaMemcpy2D( C.ptr, C.pitch, d_C.ptr, d_C.pitch, C.pitch, C.ysize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.ptr);   
    cudaFree(d_B.ptr);
    cudaFree(d_C.ptr);  
    
	
	for (int i=0; i<C.ysize; i++){
		for (int j=0; j<C.xsize; j++){
			//fprintf(f, "%f ", ((float*)C.ptr)[i*C.xsize+j]);
			if (abs((((float*)((char*)C.ptr + (i)*C.pitch))[(j)])-(((float*)((char*)Cpr.ptr + (i)*Cpr.pitch))[(j)]))>0.001)
			printf( "(%d, %d) ResG %f, resP %f", i,j,((float*)((char*)C.ptr + (i)*C.pitch))[(j)],((float*)((char*)Cpr.ptr + (i)*Cpr.pitch))[(j)]);
		}
	}


	delete [] A.ptr; 
	delete [] B.ptr; 
	delete [] C.ptr; 
	fclose(f); 
	fclose(fpr);
	if (!pad) fclose(ftwp);
	if ((pad)&&(blockHeight==16)) fclose(ftp16);
	if ((pad)&&(blockHeight==32)) fclose(ftp32);
	std::cout << cudaGetErrorString(cudaGetLastError());
	return 0;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    float aelem, belem;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
                
    if ((row<A.ysize)&&(col<B.xsize)){
      //printf("Without Transpose\n");
      for (int e = 0; e < A.xsize; e++){  

         aelem=get_elem(A, row, e);
         belem=get_elem(B, e, col);
         Cvalue = Cvalue+aelem*belem;        
      }
      (((float*)((char*)C.ptr + (row)*C.pitch))[(col)])=Cvalue;
    }
}


// Matrix multiplication kernel called by MatMul()
__global__ void MatMulTransKernel(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    float aelem, belem;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row<A.xsize)&&(col<B.xsize)){
		//printf("Transpose\n");
		for (int e = 0; e < A.ysize; e++){         
			aelem=get_elem(A, e, row);
			belem=get_elem(B, e, col);
			Cvalue = Cvalue+aelem*belem;        
		}
		(((float*)((char*)C.ptr + (row)*C.pitch))[(col)])=Cvalue;
	}
    
}

// Matrix transposition kernel 
__global__ void MatTransposeKernel(cudaPitchedPtr A, cudaPitchedPtr AT)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shmem[32][32];
    shmem[threadIdx.y][threadIdx.x] = get_elem(A, row, col);
    __syncthreads();
     if ((row<A.ysize)&&(col<A.xsize)){
      get_elem(AT, row, col) = shmem[threadIdx.x][threadIdx.y];    
    }
 /*
   if ((row<A.ysize)&&(col<A.xsize)){
      (((float*)((char*)AT.ptr + (col)*AT.pitch))[(row)])=get_elem(A, row, col);    
    }*/
}
