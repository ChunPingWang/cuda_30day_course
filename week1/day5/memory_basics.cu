#include <stdio.h>
#include <stdlib.h>

/**
 * Error checking macro
 */
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (line %d)\n", \
               cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

/**
 * Simple kernel function: multiply each element by 2
 */
__global__ void doubleArray(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = arr[idx] * 2.0f;
    }
}

int main() {
    printf("========================================\n");
    printf("    CUDA Memory Management Basics\n");
    printf("========================================\n\n");

    const int n = 10;
    size_t bytes = n * sizeof(float);

    // ========== Host Memory ==========
    printf("1. Allocate host memory\n");
    float *h_data = (float*)malloc(bytes);
    if (h_data == NULL) {
        printf("Error: Cannot allocate host memory\n");
        return 1;
    }
    printf("   Allocated %zu bytes on CPU\n\n", bytes);

    // Initialize data
    printf("2. Initialize data\n");
    printf("   Original data: [ ");
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i;
        printf("%.1f ", h_data[i]);
    }
    printf("]\n\n");

    // ========== Device Memory ==========
    printf("3. Allocate device memory\n");
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    printf("   Allocated %zu bytes on GPU\n\n", bytes);

    // ========== Data Transfer: CPU -> GPU ==========
    printf("4. Copy data to GPU (cudaMemcpyHostToDevice)\n");
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    printf("   Transfer complete!\n\n");

    // ========== Execute Kernel ==========
    printf("5. Execute kernel (multiply each element by 2)\n");
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    doubleArray<<<blocks, threadsPerBlock>>>(d_data, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("   Kernel execution complete!\n\n");

    // ========== Data Transfer: GPU -> CPU ==========
    printf("6. Copy results back to CPU (cudaMemcpyDeviceToHost)\n");
    CHECK_CUDA(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    printf("   Transfer complete!\n\n");

    // Display results
    printf("7. Results\n");
    printf("   After processing: [ ");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("]\n\n");

    // ========== Using cudaMemset ==========
    printf("8. Using cudaMemset to clear GPU memory\n");
    CHECK_CUDA(cudaMemset(d_data, 0, bytes));
    CHECK_CUDA(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    printf("   After clearing: [ ");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("]\n\n");

    // ========== Free Memory ==========
    printf("9. Free memory\n");
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
    printf("   Memory freed\n");

    printf("\n========================================\n");
    printf(" Memory management demo complete!\n");
    printf("========================================\n");

    return 0;
}
