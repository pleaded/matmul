#include <stdio.h>
#include <time.h>
#include <cuda.h>
#define BLOCK_SIZE 16
#define BLOCK_SIZE_H 16
#define BLOCK_SIZE_W 16
#define SHARED_W 16
#define SHARED_H 16

#define get_elem(array, Row, Column) (((float*)((char*)array.ptr + (Row)*array.pitch))[(Column)])
#define set_elem(array, Row, Column, elem) \
(((float*)((char*)array.ptr + (Row)*array.pitch))[(Column)] = elem)

#define type_get_elem(array, Row, Column) \
(((float*)((char*)array.ptr + (Row)*array.pitch))[(Column)])

#define type_set_elem(array, Row, Column, elem) \
(((float*)((char*)array.ptr + (Row)*array.pitch))[(Column)] = elem)


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

__global__ void MatMulKernel(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C)
{
	float Sum = 0.0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row > A.ysize || col > B.xsize)
		return;

	for(int i = 0; i < A.ysize; i++)
		Sum += get_elem(A, row, i) * get_elem(B, i, col);
	get_elem(C, row, col) = Sum;
}

__global__
void kernel_shared(cudaPitchedPtr l_in, cudaPitchedPtr r_in, cudaPitchedPtr out, int n, int m, int k, int block_h) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = gridDim.y;
    if (row >= n || col >= k ){
        return;
    }
    
    float sum1, sum2 = 0.0;
    extern __shared__  float ptr[];
    float *AS = ptr;
    float *BS = ptr + sizeof(float) * (block_h * SHARED_W);
    
    int aBegin = m * BLOCK_SIZE_H * blockIdx.y;
    int aEnd = aBegin + m - 1;
    int bBegin = m * BLOCK_SIZE_W * blockIdx.x;
    int aStep = BLOCK_SIZE_H * m;
    int bStep = BLOCK_SIZE_W * k;
    
    
    sum1 = 0.0;
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        AS[threadIdx.y * BLOCK_SIZE_W + threadIdx.x] = type_get_elem(l_in, threadIdx.y, a + threadIdx.x);
        BS[threadIdx.y * BLOCK_SIZE_W + threadIdx.x] = type_get_elem(r_in, threadIdx.y, b + threadIdx.x);
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE_H; k++) {
            sum1 += AS[threadIdx.y * BLOCK_SIZE_H + k] * BS[k * BLOCK_SIZE_H + threadIdx.x];
        }
        __syncthreads();
    }


    type_set_elem(out, row, col, sum1);
}

__global__ void MatMulKernelShared(float * A, float * B, float * C, int n, int m, int k)
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

	for(int subCnt = 0; subCnt < (m / BLOCK_SIZE); subCnt++) {
		// Asub - fixed row for each thread
		// Bsub - fixed column for each thread
		float * Asub = GetSubMatrix(A, blockRow, subCnt, m);
		float * Bsub = GetSubMatrix(B, subCnt, blockCol, k);

		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[row][col] = GetElement(Asub, row, col, m);
		Bs[row][col] = GetElement(Bsub, row, col, k);

		__syncthreads();

		for(int e = 0; e < BLOCK_SIZE; e++) {
			Sum += As[row][e] * Bs[e][col];
		}
		__syncthreads();
	}
	SetElement(Csub, row, col, Sum, k);

}


#define type_get_elem(array, Row, Column) \
(((float*)((char*)array.ptr + (Row)*array.pitch))[(Column)])

#define type_set_elem(array, Row, Column, elem) \
(((float*)((char*)array.ptr + (Row)*array.pitch))[(Column)] = elem)


__global__
void kernel_pitch(cudaPitchedPtr l_in, cudaPitchedPtr r_in, cudaPitchedPtr out, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= k ){
        return;
    }
    
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        
        sum += type_get_elem(l_in, row, i) * type_get_elem(r_in, i, col);
    }
    
    type_set_elem(out, row, col, sum);
}


void info(cudaPitchedPtr A)
{
	printf("A at %d pitch = %d xsize = %d ysize = %d\n", A.ptr, A.pitch, A.xsize, A.ysize);
}

/* 
     nnnn    kkkk    kkkk
    n####   m####   n####
	n#### x m#### = n####
	n####   m####   n####
*/

cudaPitchedPtr cuda_zmalloc(int m, int n) {
    cudaPitchedPtr res;
    res.ysize = ((m - 1) / 32 + 1) * 32;
    res.pitch = ((n - 1) / 32 + 1) * 32 * sizeof(float);
    res.xsize = n;
    cudaMalloc(&(res.ptr), res.pitch * res.ysize);
    cudaMemset(res.ptr, 0 , res.ysize * res.pitch);
    
    return res;
}


void launch_kernel_shared(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C, int m, int n, int k, int block_h)
{
    cudaPitchedPtr dA, dB, dC;
    dA = cuda_zmalloc(m , n);
    dB = cuda_zmalloc(n, k);
    dC = cuda_zmalloc(m, k);
    
    cudaMemcpy2D(dA.ptr, dA.pitch, A.ptr, n * sizeof(float), n * sizeof(float), m, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dB.ptr, dB.pitch, B.ptr, k * sizeof(float), k * sizeof(float), n, cudaMemcpyHostToDevice);
    
    dim3 threadBlock(BLOCK_SIZE_W, BLOCK_SIZE_H);
    int bw = (m - 1) / BLOCK_SIZE_W + 1;
    int bh = (k - 1) / BLOCK_SIZE_H + 1;
    dim3 grid(bw, bh);
    
    kernel_shared<<<grid, threadBlock, 2 * block_h * BLOCK_SIZE_W * sizeof(float) >>>(dA, dB, dC, m, n, k, block_h);
    
    
    cudaFree(dA.ptr);
    cudaFree(dB.ptr);
    
    cudaMemcpy2D(C.ptr, k * sizeof(float), dC.ptr, dC.pitch, k * sizeof(float), m, cudaMemcpyDeviceToHost);
    cudaFree(dC.ptr);
}

void MatMul(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C, int n, int m, int k)
{
	cudaError_t err;
	cudaEvent_t start, stop;
	float timer = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);	
	
	cudaPitchedPtr dA, dB, dC;
	dA = make_cudaPitchedPtr(0, 0, m, n);
	dB = make_cudaPitchedPtr(0, 0, k, m);
	dC = make_cudaPitchedPtr(0, 0, k, n);

	/* dA, dB, dC mem allocation */
	err = cudaMallocPitch(&dA.ptr, &dA.pitch, m * sizeof(float), n);
	printf("[MatMul]: cudaMallocPitch A - %s\n", cudaGetErrorString(err));
	err = cudaMallocPitch(&dB.ptr, &dB.pitch, k * sizeof(float), m);
	printf("[MatMul]: cudaMallocPitch B - %s\n", cudaGetErrorString(err));
	err = cudaMallocPitch(&dC.ptr, &dC.pitch, k * sizeof(float), n);
	printf("[MatMul]: cudaMallocPitch C - %s\n", cudaGetErrorString(err));
	/* ------------------------- */

	/* A --cudaMemcpy2D--> dA */
	cudaEventRecord(start, 0);
		err = cudaMemcpy2D(dA.ptr, dA.pitch, A.ptr, A.pitch, A.xsize * sizeof(float), A.ysize, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&timer, start, stop);
	printf("[MatMul]: cudaMemcpy2D A - %s (%f ms)\n", cudaGetErrorString(err), timer);
	/* ---------------------- */

	/* B --cudaMemcpy2D--> dB */
	cudaEventRecord(start, 0);
		err = cudaMemcpy2D(dB.ptr, dB.pitch, B.ptr, B.pitch, B.xsize * sizeof(float), B.ysize, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&timer, start, stop);
	printf("[MatMul]: cudaMemcpy2D B - %s (%f ms)\n", cudaGetErrorString(err), timer);
	/* ---------------------- */
	
	// info(dA);
	// info(dB);
	// info(dC);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	int bw = (n - 1) / BLOCK_SIZE + 1;
	int bh = (n - 1) / BLOCK_SIZE + 1;
	//dim3 dimGrid(k / BLOCK_SIZE, n / BLOCK_SIZE);
	dim3 dimGrid(bw, bh);

	/* Launch the kernel      */
	cudaEventRecord(start, 0);
		MatMulKernelShared <<< dimGrid, dimBlock >>> ((float *) dA.ptr,(float *) dB.ptr,(float *) dC.ptr, n, m, k);
		//MatMulKernel <<< dimGrid, dimBlock >>> (dA, dB, dC);
		//kernel_shared <<< dimGrid, dimBlock >>> (dA, dB, dC, n, m, k, BLOCK_SIZE);

	err = cudaThreadSynchronize(); cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&timer, start, stop);
	printf("[MatMul]: Run kernel - %s (%f ms)\n", cudaGetErrorString(err), timer);
	/* ---------------------- */
	
	/* dC --cudaMemcpy2D--> C */
	cudaEventRecord(start, 0);
		err = cudaMemcpy2D(C.ptr, C.pitch, dC.ptr, dC.pitch, dC.xsize * sizeof(float), dC.ysize, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&timer, start, stop);
	printf("[MatMul]: CUDA memcpy back to host! - %s (%f ms)\n", cudaGetErrorString(err), timer);
	/* ---------------------- */
	
	cudaFree(dA.ptr);
	cudaFree(dB.ptr);
	cudaFree(dC.ptr);
}

void make_random(cudaPitchedPtr A)
{
	for(int i = 0; i < A.ysize; i++) {
		for(int j = 0; j < A.xsize; j++) {
			get_elem(A, i, j) = (float) rand() / (float) RAND_MAX;
		}
	}
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
	printf("\n");
}

/* Simple host matrix multiplication */
void hostmul(cudaPitchedPtr A, cudaPitchedPtr B, cudaPitchedPtr C)
{
	for(int i = 0; i < A.ysize; i++) {
		for(int j = 0; j < C.xsize; j++) {
			float sum = 0.0;
			for(int p = 0; p < B.ysize; p++)
				sum += get_elem(A, i, p) * get_elem(B, p, j);
			get_elem(C, i, j) = sum;
		}
	}
}

/* Compares two matrixes and gives the result of the comparity */
int check(cudaPitchedPtr C, cudaPitchedPtr hC, float eps)
{
	if(C.xsize != hC.xsize || C.ysize != hC.ysize) {
		printf("[check]: memory error, matrixes are not equal!\n");
		return 0;
	}
	for(int i = 0; i < C.ysize; i++) {
		for(int j = 0; j < C.xsize; j++) {
			if(get_elem(C, i, j) - get_elem(hC, i, j) >= eps) {
				printf("[check]: not equal (eps = %f)\n", eps);
				return 0;
			}
		}
	}
	printf("[check]: everything is OK!\n");
	return 1;
}

int main(int argc, char ** argv)
{
	srand(time(NULL));

	int n, m, k;
	sscanf(argv[1], "%d", &n);
	sscanf(argv[2], "%d", &m);
	sscanf(argv[3], "%d", &k);

	cudaPitchedPtr A, B, C, hC;

	A = make_cudaPitchedPtr(malloc(n * m * sizeof(float)), m * sizeof(float), m, n);
	B = make_cudaPitchedPtr(malloc(m * k * sizeof(float)), k * sizeof(float), k, m);
	C = make_cudaPitchedPtr(malloc(n * k * sizeof(float)), k * sizeof(float), k, n);
	hC = make_cudaPitchedPtr(malloc(n * k * sizeof(float)), k * sizeof(float), k, n);
	
	// info(A);
	// info(B);
	// info(C);

	make_random(A);
	make_random(B);
	make_random(C);

	hostmul(A, B, hC);
	//show(hC);
	launch_kernel_shared(A, B, C, n, m, k, BLOCK_SIZE_H);

	check(C, hC, 0.0001);

	// show(A);
	// show(B);
	//show(C);
}
