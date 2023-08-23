#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>


int main() {

    const int m = 100; // Number of rows of matrix A
    const int n = 150; // Number of columns of matrix B and output matrix C
    const int k = 200; // Number of columns of matrix A and rows of matrix B

    std::vector<float> h_A(m * k, 2.0f); // Input matrix A with all elements initialized to 2.0
    std::vector<float> h_B(k * n, 3.0f); // Input matrix B with all elements initialized to 3.0
    std::vector<float> h_C(m * n, 0.0f); // Output matrix C with all elements initialized to 0.0

    float* d_A, * d_B, * d_C; // Device pointers for matrices A, B, and C

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_A, sizeof(float) * m * k);
    cudaMalloc((void**)&d_B, sizeof(float) * k * n);
    cudaMalloc((void**)&d_C, sizeof(float) * m * n);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A.data(), sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeof(float) * k * n, cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    checkCublasStatus(cublasCreate(&handle), "Failed to create cuBLAS handle.");

    const float alpha = 1.0f; // Scalar alpha
    const float beta = 0.0f; // Scalar beta

    // Perform matrix multiplication using cuBLAS
    checkCublasStatus(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
        d_B, n, d_A, k, &beta, d_C, n), "cuBLAS matrix multiplication failed.");

    // Copy the result matrix C from device to host
    cudaMemcpy(h_C.data(), d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    // Print the result matrix C
    // std::cout << "Result matrix C:" << std::endl;
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         std::cout << h_C[i * n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Destroy the cuBLAS handle
    checkCublasStatus(cublasDestroy(handle), "Failed to destroy cuBLAS handle.");

    // Free the allocated memory on the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}