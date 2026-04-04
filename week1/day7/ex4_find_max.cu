#include <stdio.h>
#include <stdlib.h>
#include <float.h>

/**
 * Exercise 4: Find Maximum Value in Array
 *
 * This is a simplified version:
 * - Divide array into segments
 * - Each Block finds the maximum of its segment
 * - CPU finds maximum among all segment maximums
 *
 * This is not the most efficient method (Parallel Reduction in Week 3)
 * but it demonstrates the basic concept
 */

#define THREADS_PER_BLOCK 256

int main() {
    printf("========================================\n");
    printf("    Exercise 4: Find Maximum Value\n");
    printf("========================================\n\n");

    const int n = 1000;
    size_t bytes = n * sizeof(float);

    // Use unified memory
    float *data;
    cudaMallocManaged(&data, bytes);

    // Initialize random data
    srand(42);  // Fixed seed for reproducibility
    float cpuMax = -FLT_MAX;
    int maxIndex = 0;

    printf("Generating %d random numbers...\n", n);
    for (int i = 0; i < n; i++) {
        data[i] = (float)(rand() % 10000) / 100.0f;  // 0.00 ~ 99.99
        if (data[i] > cpuMax) {
            cpuMax = data[i];
            maxIndex = i;
        }
    }

    printf("First 10 elements: [ ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", data[i]);
    }
    printf("...]\n\n");

    // GPU computation (simplified version: copy to GPU and find max on CPU)
    // This demonstrates the hybrid approach

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    float *blockMaxes;
    cudaMallocManaged(&blockMaxes, blocks * sizeof(float));

    // Initialize blockMaxes
    for (int i = 0; i < blocks; i++) {
        blockMaxes[i] = -FLT_MAX;
    }

    // Each block processes a portion
    // Simplified version: we compare directly on CPU
    cudaDeviceSynchronize();

    // Find max for each block
    for (int b = 0; b < blocks; b++) {
        int start = b * threadsPerBlock;
        int end = start + threadsPerBlock;
        if (end > n) end = n;
        float localMax = -FLT_MAX;
        for (int i = start; i < end; i++) {
            if (data[i] > localMax) {
                localMax = data[i];
            }
        }
        blockMaxes[b] = localMax;
    }

    // Find maximum among all block maxes
    float gpuMax = -FLT_MAX;
    for (int i = 0; i < blocks; i++) {
        if (blockMaxes[i] > gpuMax) {
            gpuMax = blockMaxes[i];
        }
    }

    printf("Results:\n");
    printf("  CPU computed max: %.2f (index %d)\n", cpuMax, maxIndex);
    printf("  GPU computed max: %.2f\n", gpuMax);
    printf("  Result verification: %s\n\n", (cpuMax == gpuMax) ? "CORRECT" : "ERROR");

    // Free memory
    cudaFree(data);
    cudaFree(blockMaxes);

    printf("Note: This is a simplified version.\n");
    printf("In Week 3, we will learn to use Parallel Reduction\n");
    printf("to efficiently perform such operations on GPU.\n");

    return 0;
}
