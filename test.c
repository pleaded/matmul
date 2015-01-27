#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 16

#define get_elem(array, Row, Column) (((float*)((char*)array.ptr + (Row)*array.pitch))[(Column)])

// __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
// {
// 	float Sum = 0.0;
// 	int row = blockIdx.y * blockDim.y + threadIdx.y;
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;

// 	if(row > A.heigth || col > B.width)
// 		return;

// 	for(int i = 0; i < A.width; i++)
// 		Sum += (A.elements[row * A.width + i] * (B.elements[i * B.width + col]));
// 	C.elements[row * C.width + col] = Sum;
// }

// void MatMul(const Matrix A, const Matrix B, const Matrix C)
// {
// 	cudaError_t err;

// 	cudaPitchedPtr 

// 	// A Matrix
// 	Matrix d_A;
// 	d_A.width = A.width;
// 	d_A.heigth = A.heigth;
// 	size = d_A.width * d_A.heigth * sizeof(float);

// 	cudaPitchedPtr d_pA;
// 	d_pA.ptr = d_A.elements;
// 	d_pA.xsize = d_A.width;
// 	d_pA.ysize = d_A.heigth;
// 	err = cudaMallocPitch(&d_pA.ptr, &d_pA.pitch, d_pA.xsize * sizeof(float), d_pA.ysize);
// 	printf("[Device]: cudaMallocPitch A - %s\n", cudaGetErrorString(err));
// 	err = cudaMemcpy2D(d_pA.ptr, d_pA.pitch, A.elements, A.width * sizeof(float), A.width * sizeof(float), A.heigth, cudaMemcpyHostToDevice);
// 	printf("[Device]: cudaMemcpy2D A - %s\n", cudaGetErrorString(err));
	
// 	// B Matrix
// 	Matrix d_B;
// 	d_B.width = B.width;
// 	d_B.heigth = B.heigth;
// 	size = d_B.width * d_B.heigth * sizeof(float);

// 	cudaPitchedPtr d_pB;
// 	d_pB.ptr = d_B.elements;
// 	d_pB.xsize = d_B.width;
// 	d_pB.ysize = d_B.heigth;
	
// 	err = cudaMallocPitch(&d_pB.ptr, &d_pB.pitch, d_pB.xsize * sizeof(float), d_pB.ysize);
// 	printf("[Device]: cudaMallocPitch B - %s\n", cudaGetErrorString(err));
// 	err = cudaMemcpy2D(d_pB.ptr, d_pB.pitch, B.elements, B.width * sizeof(float), B.width * sizeof(float), B.heigth, cudaMemcpyHostToDevice);
// 	printf("[Device]: cudaMemcpy2D B - %s\n", cudaGetErrorString(err));
	
// 	// C Matrix
// 	Matrix d_C;
// 	d_C.width = C.width;
// 	d_C.heigth = C.heigth;
// 	size = d_C.width * d_C.heigth * sizeof(float);

// 	cudaPitchedPtr d_pC;
// 	d_pC.ptr = d_C.elements;
// 	d_pC.xsize = d_C.width;
// 	d_pC.ysize = d_C.heigth;

// 	err = cudaMallocPitch(&d_pC.ptr, &d_pC.pitch, d_pC.xsize * sizeof(float), d_pC.ysize);
// 	printf("[Device]: cudaMallocPitch C - %s\n", cudaGetErrorString(err));
// 	//err = cudaMemcpy(d_C.elements, C.elements, size, cudaMemcpyHostToDevice);
// 	//printf("[Device]: cudaMemcpy C - %s\n", cudaGetErrorString(err));
	
// 	// Invoke kernel
// 	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
// 	dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, 
// 				 (A.heigth + dimBlock.y - 1) / dimBlock.y);
// 	MatMulKernel <<< dimGrid, dimBlock >>> (d_A, d_B, d_C);

// 	err = cudaThreadSynchronize();
// 	printf("[Deivce]: running kernel - %s\n", cudaGetErrorString(err));

// 	err = cudaMemcpy2D(C.elements, C.width * sizeof(float), d_pC.ptr, d_pC.pitch, d_pC.pitch, d_pC.ysize, cudaMemcpyDeviceToHost);
// 	printf("[Deivce]: cudaMemcpy2D from device to host - %s\n", cudaGetErrorString(err));

// 	cudaFree(d_A.elements);
// 	cudaFree(d_B.elements);
// 	cudaFree(d_C.elements);
// }


void make_random(cudaPitchedPtr A)
{
	for(int i = 0; i < A.xsize; i++)
		for(int j = 0; j < A.ysize; j++)
			get_elem(A, i, j) = (float) rand() / (float) RAND_MAX;
}

void show(cudaPitchedPtr A)
{
	for(int i = 0; i < A.ysize; i++)
	{
		for(int j = 0; j < A.xsize; j++)
		{
			printf("%lf ", get_elem(A, i, j));
		}
		printf("\n");
	}
}

int main(int argc, char ** argv)
{
	cudaError_t err;

	int m, n, k;
	sscanf(argv[1], "%d", &m); 
	sscanf(argv[2], "%d", &n);
	sscanf(argv[3], "%d", &k);

	cudaPitchedPtr A, B;
	A = make_cudaPitchedPtr(malloc(m * n * sizeof(float)), n * sizeof(float), n, m);
	printf("A at %d pitch = %d xsize = %d ysize = %d\n", A.ptr, A.pitch, A.xsize, A.ysize);
	
	B = make_cudaPitchedPtr(malloc(m * n * sizeof(float)), n * sizeof(float), n, m);
	printf("B at %d pitch = %d xsize = %d ysize = %d\n", B.ptr, B.pitch, B.xsize, B.ysize);
	
	make_random(A);
	show(A);

	cudaPitchedPtr dA;
	dA = make_cudaPitchedPtr(0, 0, n, m);
	err = cudaMallocPitch(&dA.ptr, &dA.pitch, n * sizeof(float), m);
	printf("dA at %d pitch = %d xsize = %d ysize = %d\n", dA.ptr, dA.pitch, dA.xsize, dA.ysize);
	printf("[Deivce]: cudaMallocPitch - %s\n", cudaGetErrorString(err));

	err = cudaMemcpy2D(dA.ptr, dA.pitch, A.ptr, A.pitch, A.xsize * sizeof(float), A.ysize, cudaMemcpyHostToDevice);
	printf("[Deivce]: cudaMemcpy2D - %s\n", cudaGetErrorString(err));

	printf("%d <> %d * %d = %d\n", dA.pitch, dA.xsize, sizeof(float), dA.xsize * sizeof(float));
	err =cudaMemcpy2D(B.ptr, B.pitch, dA.ptr, dA.pitch, dA.xsize * sizeof(float), dA.ysize, cudaMemcpyDeviceToHost);
	printf("[Deivce]: cudaMemcpy2D - %s\n", cudaGetErrorString(err));

	show(B);


	// printf("[Host][my_make_cudaPitchedPtr]: A at %d pitch = %d xsize = %d ysize = %d\n", A.ptr, A.pitch, A.xsize, A.ysize);
	// cudaMallocPitch(&A.ptr, &A.pitch, A.xsize * sizeof(float), A.ysize);
	// printf("[Host][my_make_cudaPitchedPtr]: A at %d pitch = %d xsize = %d ysize = %d\n", A.ptr, A.pitch, A.xsize, A.ysize);
	
	// // Show A
	// printf("[Host]: A matrix elements ...\n");
	// for(int i = 0; i < A.heigth; i++) {
	// 	for(int j = 0; j < A.width; j++)
	// 		printf("%lf ", A.elements[i * A.width + j]);
	// 	printf("\n");
	// }
	// printf("\n");

	// // Show B
	// printf("[Host]: B matrix elements ...\n");
	// for(int i = 0; i < B.heigth; i++) {
	// 	for(int j = 0; j < B.width; j++)
	// 		printf("%lf ", B.elements[i * B.width + j]);
	// 	printf("\n");
	// }
	// printf("\n");


	// MatMul(A, B, C);

	// // Show C
	// printf("[Host]: C matrix elements after multiplication on DEVICE...\n");
	// for(int i = 0; i < C.heigth; i++) {
	// 	for(int j = 0; j < C.width; j++)
	// 		printf("%lf ", C.elements[i * C.width + j]);
	// 	printf("\n");
	// }
	// printf("\n");


	// // Show C
	// for(int i = 0; i < A.heigth; i++) {
	// 	for(int j = 0; j < C2.width; j++) {
	// 		float sum = 0.0;
	// 		for(int p = 0; p < B.heigth; p++)
	// 			sum += A.elements[i * C2.width + p] * B.elements[p * B.heigth + j];
	// 		C2.elements[i * C2.width + j] = sum;
	// 	}
	// }

	// printf("[Host]: C matrix elements after multiplication on HOST...\n");
	// for(int i = 0; i < C2.heigth; i++) {
	// 	for(int j = 0; j < C2.width; j++)
	// 		printf("%lf ", C2.elements[i * C2.width + j] - C.elements[i * C.width + j]);
	// 	printf("\n");
	// }
	// printf("\n");

}
