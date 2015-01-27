#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
using namespace std;

enum
{
#ifdef DOUBLE_ELEMS
    BLOCK_SIZE_H = 32,
#else
    BLOCK_SIZE_H = 16,
#endif
    BLOCK_SIZE_W = 32
};
typedef float type;

#define type_get_elem(array, Row, Column) \
(((type*)((char*)array.ptr + (Row)*array.pitch))[(Column)])

#define type_set_elem(array, Row, Column, elem) \
(((type*)((char*)array.ptr + (Row)*array.pitch))[(Column)] = elem)


//C  == A * B
// A n * m, B m * k C n * k

__global__
void kernel_naive(cudaPitchedPtr l_in, cudaPitchedPtr r_in, cudaPitchedPtr out, int n, int m, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= k ){
        return;
    }
    
    type sum = 0.0;
    for (int i = 0; i < m; i++) {
        sum += type_get_elem(l_in, row, i) * type_get_elem(r_in, i, col);
    }
    
    type_set_elem(out, row, col, sum);
}


__global__
void kernel_pitch(cudaPitchedPtr l_in, cudaPitchedPtr r_in, cudaPitchedPtr out, int n, int m, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= k ){
        return;
    }
    
    type sum = 0.0;
    for (int i = 0; i < m; i++) {
        
        sum += type_get_elem(l_in, row, i) * type_get_elem(r_in, i, col);
    }
    
    type_set_elem(out, row, col, sum);
}


enum
{
#ifndef DOUBLE_ELEMS 
    SHARED_W = 32,
#else
    SHARED_W = 16,
#endif
    SHARED_H = 32
};

__global__
void kernel_shared(cudaPitchedPtr l_in, cudaPitchedPtr r_in, cudaPitchedPtr out, int n, int m, int k, int block_h) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = gridDim.y;
    if (row >= n || col >= k ){
        return;
    }
    
    type sum1, sum2 = 0.0;
    //__shared__ type *AS = new type[SHARED_H * SHARED_W]
    //__shared__ type *BS = new type[SHARED_H * SHARED_W]
    extern __shared__  type ptr[];
    type *AS = ptr;
    type *BS = ptr + sizeof(type) * (block_h * SHARED_W);
//     for (unsigned i = 0; i < gy; i++) {
//         printf("Acessing a[%d][%d]\n", row, i * SHARED_W + threadIdx.y);
//         AS[threadIdx.x * SHARED_W + threadIdx.y] = type_get_elem(l_in, row, i * SHARED_W + threadIdx.y);
// #ifdef DOUBLE_ELEMS 
//         AS[threadIdx.x * SHARED_W + threadIdx.y + 16] = type_get_elem(l_in, row + 16, i * SHARED_W + threadIdx.y);
// #endif
//         
//         printf("Acessing b[%d][%d]\n", i * SHARED_H + threadIdx.y, col);
//         BS[threadIdx.y * SHARED_W + threadIdx.y] = type_get_elem(r_in, i * SHARED_H + threadIdx.x, col);
//         __syncthreads(); 
//         
//         for (int j = 0; j < SHARED_H; j++) {
// #ifdef DOUBLE_ELEMS
//             sum2 += AS[j * SHARED_W + threadIdx.x + 16] * BS[j * SHARED_W + threadIdx.y];
// #endif
//             sum1 += AS[j * SHARED_W + threadIdx.x] * BS[j * SHARED_W + threadIdx.y];
// 
//         }
//         __syncthreads();
//     }
    
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
    //type_set_elem(out, row + 16, col, sum2);
}




enum
{
    RND = 100
};

void init_matr(type *a, int w, int h) {

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            a[i * w + j] = rand() % (RND);
        }
    }
}

void host_mul(type *a, type *b, type *c, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            type tsum = 0.0f;
            for (int l = 0; l < m; l++) {
                tsum += a[i * m + l] * b[l * k + j];
            }
            c[i * k + j] = tsum;
        }
    }
}



void log(const char *s)
{
    fprintf(stderr, "%s\n", s);
}

void print_matrix(type *a, int w, int h) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            fprintf(stdout, "%f ", a[i * w + j]);
        }
        fprintf(stdout, "\n");
    }
}

inline  cudaPitchedPtr toPitch(type *a, int n, int m) {
    cudaPitchedPtr res;
    res.ptr = a;
    res.xsize = m;
    res.ysize = n;
    res.pitch = m * sizeof(type);
    return res;
}

//return res
type * launch_kernel(type *a, type *b, type *c, int n, int m, int k)
{
    type *gpuA, *gpuB, *gpuC;
    cudaMalloc(&gpuA, n * m * sizeof(type));
    cudaMalloc(&gpuB, m * k * sizeof(type));
    cudaMalloc(&gpuC, n * k * sizeof(type));
    
    cudaPitchedPtr gA = toPitch(gpuA, n, m);
    cudaPitchedPtr gB = toPitch(gpuB, m, k);
    cudaPitchedPtr gC = toPitch(gpuC, n, k);
    
    cudaMemcpy(gpuA, a, n * m * sizeof(type), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, b, m * k * sizeof(type), cudaMemcpyHostToDevice);
    
    
    
    dim3 threadBlock(BLOCK_SIZE_W, BLOCK_SIZE_H);
    int bw = (n - 1) / BLOCK_SIZE_W + 1;
    int bh = (k - 1) / BLOCK_SIZE_H + 1;
    dim3 grid(bw, bh);
    
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    kernel_naive<<<grid, threadBlock>>>(gA, gB, gC, n, m, k);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    fprintf(stderr, "KERNEL NAIVE ELAPSED TIME == %.06f\n", time);
    fprintf(stdout, "%d\t%.06f\n", n, time);
    
    cudaFree(gpuA);
    cudaFree(gpuB);
    type *gpures = new type[n * k];
    cudaMemcpy(gpures, gpuC, n * k * sizeof(type) , cudaMemcpyDeviceToHost);
    cudaFree(gpuC);
    return gpures;
    
}

type * launch_kernel_pitch(type *a, type *b, type *c, int n, int m, int k)
{
    //type *gpuA, *gpuB, *gpuC;
    //cudaMalloc(&gpuA, n * m * sizeof(type));
    //cudaMalloc(&gpuB, m * k * sizeof(type));
    //cudaMalloc(&gpuC, n * k * sizeof(type));
    cudaPitchedPtr gpuA, gpuB, gpuC;
    gpuA.xsize = m;
    gpuA.ysize = n;
    
    gpuB.xsize = k;
    gpuB.ysize = m;
    
    gpuC.xsize = k;
    gpuC.ysize = n;
    
    cudaMallocPitch(&gpuA.ptr, &gpuA.pitch, m * sizeof(type), n);
    cudaMallocPitch(&gpuB.ptr, &gpuB.pitch, k * sizeof(type), m);
    cudaMallocPitch(&gpuC.ptr, &gpuC.pitch, k * sizeof(type), n);
    
    
    
    
    cudaMemcpy2D(gpuA.ptr, gpuA.pitch, a, m * sizeof(type), m * sizeof(type), n, cudaMemcpyHostToDevice);
    cudaMemcpy2D(gpuB.ptr, gpuB.pitch, b, k * sizeof(type), k * sizeof(type), m, cudaMemcpyHostToDevice);
    
    dim3 threadBlock(BLOCK_SIZE_W, BLOCK_SIZE_H);
    int bw = (n - 1) / BLOCK_SIZE_W + 1;
    int bh = (k - 1) / BLOCK_SIZE_H + 1;
    dim3 grid(bw, bh);
    
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    kernel_pitch<<<grid, threadBlock>>>(gpuA, gpuB, gpuC, n, m, k);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    fprintf(stderr, "KERNEL PITCHED ELAPSED TIME == %.06f\n", time);
    fprintf(stdout, "%d\t%.06f\n", n, time);
    
    
    cudaFree(gpuA.ptr);
    cudaFree(gpuB.ptr);
    type *gpures = new type[n * k];
    cudaMemcpy2D(gpures, k * sizeof(type), gpuC.ptr, gpuC.pitch, k *sizeof(type), n, cudaMemcpyDeviceToHost);
    
    //cudaFree(gpuC.ptr);
    return gpures;
}

cudaPitchedPtr cuda_zmalloc(int n, int m) {
    cudaPitchedPtr res;
    res.ysize = ((n - 1) / 32 + 1) * 32;
    res.pitch = ((m - 1) / 32 + 1) * 32 * sizeof(type);
    res.xsize = m;
    cudaMalloc(&(res.ptr), res.pitch * res.ysize);
    cudaMemset(res.ptr, 0 , res.ysize * res.pitch);
    
    return res;
}


type * launch_kernel_shared(type *a, type *b, type *c, int n, int m, int k, int block_h)
{
    //type *gpuA, *gpuB, *gpuC;
    //cudaMalloc(&gpuA, n * m * sizeof(type));
    //cudaMalloc(&gpuB, m * k * sizeof(type));
    //cudaMalloc(&gpuC, n * k * sizeof(type));
    cudaPitchedPtr gpuA, gpuB, gpuC;
    gpuA = cuda_zmalloc(n , m);
    gpuB = cuda_zmalloc(m, k);
    gpuC = cuda_zmalloc(n, k);
    
    
    
    
    cudaMemcpy2D(gpuA.ptr, gpuA.pitch, a, m * sizeof(type), m * sizeof(type), n, cudaMemcpyHostToDevice);
    cudaMemcpy2D(gpuB.ptr, gpuB.pitch, b, k * sizeof(type), k * sizeof(type), m, cudaMemcpyHostToDevice);
    
    dim3 threadBlock(BLOCK_SIZE_W, BLOCK_SIZE_H);
    int bw = (n - 1) / BLOCK_SIZE_W + 1;
    int bh = (k - 1) / BLOCK_SIZE_H + 1;
    dim3 grid(bw, bh);
    
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    fprintf(stderr, "SIZE == %d\n", 2 * block_h * BLOCK_SIZE_W);
    kernel_shared<<<grid, threadBlock, 2 * block_h * BLOCK_SIZE_W * sizeof(type) >>>(gpuA, gpuB, gpuC, n, m, k, block_h);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    fprintf(stderr, "KERNEL SHARED ELAPSED TIME == %.06f\n", time);
    fprintf(stdout, "%d\t%.06f\n", n, time);
    
    
    cudaFree(gpuA.ptr);
    cudaFree(gpuB.ptr);
    type *gpures = new type[n * k];
    
    cudaMemcpy2D(gpures, k * sizeof(type), gpuC.ptr, gpuC.pitch, k *sizeof(type), n, cudaMemcpyDeviceToHost);
    
    cudaFree(gpuC.ptr);
    return gpures;
}

int main(int argc, char **argv) {
    //cudaSetDevice(1);
    if (argc < 4) {
        fprintf(stderr, "USAGE %s N M K\n", argv[0]);
        return -1;
    }
    int n, m, k;
    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%d", &m);
    sscanf(argv[3], "%d", &k);
    fprintf(stderr, "ARGC == %d, N== %d M==%d K==%d\n", argc, n, m, k);
    type *ha = new type[n * m];
    type *hb = new type[m * k];
    type *hc = new type[n * k];
    init_matr(ha, n, m);
    init_matr(hb, m, k);
    //print_matrix(ha, n, m);
    //print_matrix(hb,m, k);
    
    log("MEM HAVE BEEN COPIED");
    log("STARTING KERNEL");
    ///fprintf(stderr, "ARGC==%d\n", argc);

    type * gpures;
    if (argc == 4) {
        gpures = launch_kernel(ha, hb, hc, n, m, k);
    } else if (argc > 4) {
        fprintf(stderr, "argv4 == %s\n", argv[4]);
        if (strcmp(argv[4], "pitch") == 0) {
            gpures = launch_kernel_pitch(ha, hb, hc, n, m, k);
        }
        if (strcmp(argv[4], "shared") == 0) {
            int hsize = BLOCK_SIZE_H;
            if (argc > 5) {
                sscanf(argv[5], "%d", &hsize);
            }
            gpures = launch_kernel_shared(ha, hb, hc, n, m, k, hsize);
        }
    }
    log("STARTING CPU");
    host_mul(ha, hb, hc, n, m, k);
    log("CHECKING");
    std::cerr << cudaGetErrorString(cudaGetLastError()) << endl;
    const type eps = 1e-5;
    
    std::cerr << cudaGetErrorString(cudaGetLastError()) << endl;
    if (n <= 512) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                if (fabs(gpures[i * k + j] - hc[i * k + j]) > eps) {
                    /*print_matrix(ha, n, m);
                    log("");
                    print_matrix(hb, m, k);
                    fprintf(stdout, "CPURES\n");
                    print_matrix(hc, n, k);
                    fprintf(stdout, "GPURES\n");
                    print_matrix(gpures, n, k);*/
                    fprintf(stderr, "ERROR %d %d, %f\n", i, j, fabs(gpures[i * k + j] - hc[i * k + j]));
                    abort();
                }
            }
        }
        log("OK!");
    }

    return 0;
}