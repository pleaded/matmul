#include <stdio.h>
#include <cuda.h>
#define BLOCK_SIZE 16

__device__ float GetElement(float * A, int row, int col, int stride)
{
	return A[row * stride + col];
}
__device__ void SetElement(float * A, int row, int col, float value, int stride)
{
	A[row * stride + col] = value;
}

__device__ float * GetSubMatrix(float * A, int row, int col, int stride)
{
	return &A[stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
}

__global__ void MatMulKernelShared(float * A, float * B, float * C, int m, int n, int k)
{
	/*
	         B  x
	         ###### 
	        y###@## 
             ###|## 
             ###|## 
	            |
	 A x     C  |
	 ######  ###|## 
	 ######  ###|## 
	y##@========@## 
	 ######  ###### 

	@ - submatix BLOCK_SIZE x BLOCK_SIZE
	x - blockCol
	y - blockRow
	*/

	/* blockIdx - The location of a block within the grid. */
	int blockCol = blockIdx.x;
	int blockRow = blockIdx.y;

	/* threadIdx - the location of a thread within its own block. */
	int col = threadIdx.x;
	int row = threadIdx.y;

	// Every thread computes its own Csub according to its own block
	float * Csub = GetSubMatrix(C, blockRow, blockCol, k);
	float Sum = 0.0;

	for(int subCnt = 0; subCnt < (n / BLOCK_SIZE); subCnt++) {
		// Asub - fixed row for each thread
		// Bsub - fixed column for each thread
		float * Asub = GetSubMatrix(A, blockRow, subCnt, n);
		float * Bsub = GetSubMatrix(B, subCnt, blockCol, k);

		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[row][col] = GetElement(Asub, row, col, n);
		Bs[row][col] = GetElement(Bsub, row, col, k);

		__syncthreads();

		for(int e = 0; e < BLOCK_SIZE; e++) {
			Sum += As[row][e] * Bs[e][col];
		}
		__syncthreads();
	}
	SetElement(Csub, row, col, Sum, k);

}

/* 
     nnnn    kkkk    kkkk
    m####   n####   m####
	m#### x n#### = m####
	m####   n####   m####
*/


void MatMul(float * A, float * B, float * C, int m, int n, int k)
{
	cudaEvent_t start, stop;
	float timer = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	

	printf("[MatMul]: m = %d, n = %d, k = %d\n", m, n, k);
	float * dA, * dB, * dC;

	cudaError_t err = cudaMalloc(&dA, m * n * sizeof(float));
	printf("[MatMul]: CUDA malloc A - %s\n", cudaGetErrorString(err));
	cudaEventRecord(start, 0);
	err = cudaMemcpy(dA, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timer, start, stop);
	printf("[MatMul]: CUDA memcpy A - %s (%f ms)\n", cudaGetErrorString(err), timer);

	err = cudaMalloc(&dB, n * k * sizeof(float));
	printf("[MatMul]: CUDA malloc B - %s\n", cudaGetErrorString(err));
	cudaEventRecord(start, 0);
	err = cudaMemcpy(dB, B, n * k * sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timer, start, stop);
	printf("[MatMul]: CUDA memcpy B - %s (%f ms)\n", cudaGetErrorString(err), timer);
	
	err = cudaMalloc(&dC, m * k * sizeof(float));
	printf("[MatMul]: CUDA malloc C - %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(dC, C, m * k * sizeof(float), cudaMemcpyHostToDevice);
	printf("[MatMul]: CUDA memcpy C - %s\n", cudaGetErrorString(err));
	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(k / BLOCK_SIZE, m / BLOCK_SIZE);

	cudaEventRecord(start, 0);
	MatMulKernelShared <<< dimGrid, dimBlock >>> (dA, dB, dC, m, n, k);
	err = cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timer, start, stop);

	printf("[MatMul]: Run kernel - %s (%f ms)\n", cudaGetErrorString(err), timer);

	cudaEventRecord(start, 0);
	err = cudaMemcpy(C, dC, m * k * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timer, start, stop);
	printf("[MatMul]: CUDA memcpy back to host! - %s (%f ms)\n", cudaGetErrorString(err), timer);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

void make_random(float * A, int m, int n)
{
	for(int i = 0; i < m * n; i++)
		A[i] = (float) rand() / (float) RAND_MAX;
}

void show(float * A, int m, int n)
{
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			printf("%f ", A[i * n + j]);
		}
		printf("\n");
	}
	printf("\n");
}



int main(int argc, char ** argv)
{
	int m, n, k;
	sscanf(argv[1], "%d", &m);
	sscanf(argv[2], "%d", &n);
	sscanf(argv[3], "%d", &k);
	float * A = (float *) malloc(m * n * sizeof(float));
	float * B = (float *) malloc(n * k * sizeof(float));
	float * C = (float *) malloc(m * k * sizeof(float));

	make_random(A, m, n);
	make_random(B, n, k);
	make_random(C, m, k);

	MatMul(A, B, C, m, n, k);

	
	float * hC = (float *) malloc(m * k * sizeof(float));
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < k; j++) {
			float sum = 0.0;	
			for(int p = 0; p < n; p++) {
					sum += A[i * k + p] * B[p * n + j];
			}
			hC[i * k + j] = sum;
		}
	}
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < k; j++) {
			printf("%f ", hC[i * n + j] - C[i * n + j]);
		}
		printf("\n");
	}
}
