#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/**
 * CUDA kernel function: Vector Addition
 */
__global__ void vectorAddGPU(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * CPU version: Vector Addition
 */
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/**
 * Initialize array
 */
void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % 100) / 10.0f;
    }
}

/**
 * Verify results
 */
bool verifyResult(float *gpu, float *cpu, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(gpu[i] - cpu[i]) > 0.001f) {
            return false;
        }
    }
    return true;
}

int main() {
    printf("========================================\n");
    printf("    CPU vs GPU Performance Comparison\n");
    printf("========================================\n\n");

    // Test different array sizes
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    int numSizes = 5;

    srand((unsigned int)time(NULL));

    printf("%-15s %-15s %-15s %-10s\n",
           "Array Size", "CPU Time(ms)", "GPU Time(ms)", "Speedup");
    printf("--------------------------------------------------------\n");

    for (int s = 0; s < numSizes; s++) {
        int n = sizes[s];
        size_t bytes = n * sizeof(float);

        // Allocate host memory
        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C_cpu = (float*)malloc(bytes);
        float *h_C_gpu = (float*)malloc(bytes);

        // Initialize data
        initArray(h_A, n);
        initArray(h_B, n);

        // ========== CPU Timing ==========
        clock_t cpuStart = clock();
        vectorAddCPU(h_A, h_B, h_C_cpu, n);
        clock_t cpuEnd = clock();
        double cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC * 1000;

        // ========== GPU Timing ==========
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Copy data to GPU
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

        // Configure kernel function
        int threadsPerBlock = 256;
        int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

        // Start timing
        cudaEventRecord(start);

        // Execute kernel function
        vectorAddGPU<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

        // Stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpuTime;
        cudaEventElapsedTime(&gpuTime, start, stop);

        // Copy results back to CPU
        cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

        // Verify results
        bool correct = verifyResult(h_C_gpu, h_C_cpu, n);

        // Calculate speedup
        double speedup = cpuTime / gpuTime;

        // Output results
        printf("%-15d %-15.3f %-15.3f %-10.2fx %s\n",
               n, cpuTime, gpuTime, speedup,
               correct ? "[OK]" : "[FAIL]");

        // Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
    }

    printf("\n========================================\n");
    printf("Observations:\n");
    printf("1. GPU may not be faster for small arrays\n");
    printf("   (due to data transfer overhead)\n");
    printf("2. GPU advantage is significant for large arrays\n");
    printf("3. Speedup increases with array size\n");
    printf("========================================\n");

    return 0;
}
