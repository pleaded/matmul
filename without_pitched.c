#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 16

typedef struct 
{
	int width;
	int heigth;
	float * elements;
} Matrix;

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	float Sum = 0.0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row > A.heigth || col > B.width)
		return;

	for(int i = 0; i < A.width; i++)
		Sum += (A.elements[row * A.width + i] * (B.elements[i * B.width + col]));
	C.elements[row * C.width + col] = Sum;
}

void MatMul(const Matrix A, const Matrix B, const Matrix C)
{
	cudaError_t err;
	size_t size;

	// A Matrix
	Matrix d_A;
	d_A.width = A.width;
	d_A.heigth = A.heigth;
	size = d_A.width * d_A.heigth * sizeof(float);

	err = cudaMalloc(&d_A.elements, size);
	printf("[Device]: cudaMalloc A - %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	printf("[Device]: cudaMemcpy A - %s\n", cudaGetErrorString(err));
	
	// B Matrix
	Matrix d_B;
	d_B.width = B.width;
	d_B.heigth = B.heigth;
	size = d_B.width * d_B.heigth * sizeof(float);

	err = cudaMalloc(&d_B.elements, size);
	printf("[Device]: cudaMalloc B - %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
	printf("[Device]: cudaMemcpy B - %s\n", cudaGetErrorString(err));
	
	// C Matrix
	Matrix d_C;
	d_C.width = C.width;
	d_C.heigth = C.heigth;
	size = d_C.width * d_C.heigth * sizeof(float);

	err = cudaMalloc(&d_C.elements, size);
	printf("[Device]: cudaMalloc C - %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);
	printf("[Device]: cudaMemcpy C - %s\n", cudaGetErrorString(err));
	
	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, 
				 (A.heigth + dimBlock.y - 1) / dimBlock.y);
	MatMulKernel <<< dimGrid, dimBlock >>> (d_A, d_B, d_C);

	err = cudaThreadSynchronize();
	printf("[Deivce]: running kernel - %s\n", cudaGetErrorString(err));

	err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	printf("[Deivce]: cudaMemcpy from device to host - %s\n", cudaGetErrorString(err));

	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

int main(int argc, char ** argv)
{
	Matrix A, B, C;
	A.heigth = atoi(argv[1]);
	A.width = atoi(argv[2]);
	B.heigth = atoi(argv[2]);
	B.width = atoi(argv[3]);
	C.heigth = atoi(argv[1]);
	C.width = atoi(argv[3]);
	printf("[Host]: A[%d][%d] x B[%d][%d] = C[%d][%d]\n\n", A.heigth, A.width, B.heigth, B.width, C.heigth, C.width);

	printf("[Host]: Allocating memory ... \n");
	A.elements = (float *) malloc(A.width * A.heigth * sizeof(float));
	B.elements = (float *) malloc(B.width * B.heigth * sizeof(float));
	C.elements = (float *) malloc(C.width * C.heigth * sizeof(float));
	printf("[Host]: Allocating memory ... OK\n\n");
	
	// Init A, B
	printf("[Host]: Generating random data ... \n");
	for(int i = 0; i < A.heigth; i++)
		for(int j = 0; j < A.width; j++)
			A.elements[i * A.width + j] = (float) rand() / (float) RAND_MAX;
	for(int i = 0; i < B.heigth; i++)
		for(int j = 0; j < B.width; j++)
			B.elements[i * B.width + j] = (float) rand() / (float) RAND_MAX;
	printf("[Host]: Generating random data ... OK\n\n");
	
	// Show A
	printf("[Host]: A matrix elements ...\n");
	for(int i = 0; i < A.heigth; i++) {
		for(int j = 0; j < A.width; j++)
			printf("%lf ", A.elements[i * A.width + j]);
		printf("\n");
	}
	printf("\n");

	// Show B
	printf("[Host]: B matrix elements ...\n");
	for(int i = 0; i < B.heigth; i++) {
		for(int j = 0; j < B.width; j++)
			printf("%lf ", B.elements[i * B.width + j]);
		printf("\n");
	}
	printf("\n");

	for(int i = 0; i < A.heigth; i++) {
		for(int j = 0; j < C.width; j++) {
			float sum = 0.0;
			for(int p = 0; p < B.heigth; p++)
				sum += A.elements[i * C.width + p] * B.elements[p * B.heigth + j];
			C.elements[i * C.width + j] = sum;
		}
	}

	// Show C
	printf("[Host]: C matrix elements after multiplication ...\n");
	for(int i = 0; i < C.heigth; i++) {
		for(int j = 0; j < C.width; j++)
			printf("%lf ", C.elements[i * C.width + j]);
		printf("\n");
	}
	printf("\n");

	MatMul(A, B, C);

	// Show C
	printf("[Host]: C matrix elements after multiplication ...\n");
	for(int i = 0; i < C.heigth; i++) {
		for(int j = 0; j < C.width; j++)
			printf("%lf ", C.elements[i * C.width + j]);
		printf("\n");
	}
	printf("\n");
}