#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void naiveGemm(float *A, float *B, float *C, 
                            const int M, const int N, const int K) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty < M && tx < N) {
        float c = 0;
        for (int i = 0; i < K; i++) {
            c += A[ty * K + i] * B[i * N + tx];
        }
        C[ty * N + tx] = c;
    }
}

__global__ void tiledGemm(float *A, float *B, float *C, 
                            const int M, const int N, const int K) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty < M && tx < N) {
        float c = 0;
        for (int i = 0; i < K; i++) {
            c += A[ty * K + i] * B[i * N + tx];
        }
        C[ty * N + tx] = c;
    }
}

float testNaiveGemmPerformance(int M, int N, int K, int repeat, int block_num, int thread_num) {
    float *A, *B, *C;
    float *h_A, *h_B, *h_C;
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    //allocate memory on CPU
    h_A = (float *)malloc(size_a);
    h_B = (float *)malloc(size_b);
    h_C = (float *)malloc(size_c);

    // allocate memory on GPU
    cudaMalloc(&A, size_a);
    cudaMalloc(&B, size_b);
    cudaMalloc(&C, size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_A[i] = (rand() / float(RAND_MAX));
    for (int i = 0; i < K * N; i++)
        h_B[i] = (rand() / float(RAND_MAX));
    
    cudaMemcpy(A, h_A, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B, size_b, cudaMemcpyHostToDevice);
    // cudaMemcpy(C, h_C, size_c, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    for (int i = 0; i < repeat; i++) {
        naiveGemm<<<block_num, thread_num>>>(A, B, C, M, N, K);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    free(h_A); free(h_B); free(h_C);
    cudaFree(A); cudaFree(B); cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return sec;
}

int main(int argc, char **argv) {
    int M = 2048;
    int N = 2048;
    int K = 2048;

    float sec = testNaiveGemmPerformance(M, N, K, 5, 1024, 1024);
    printf("%lf s\n", sec);
    return 0;
}