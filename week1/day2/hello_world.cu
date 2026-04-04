#include <stdio.h>

/**
 * This is our first CUDA kernel function!
 * __global__ means this function runs on the GPU
 */
__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

int main() {
    // Step 1: Say hello from CPU
    printf("========================================\n");
    printf("        My First CUDA Program\n");
    printf("========================================\n\n");
    printf("Hello World from CPU!\n\n");

    // Step 2: Launch kernel function
    // <<<1, 1>>> means: 1 Block, each Block has 1 Thread
    printf("Launching GPU kernel function...\n");
    helloFromGPU<<<1, 1>>>();

    // Step 3: Wait for GPU to finish
    // Because CPU and GPU execute asynchronously
    cudaDeviceSynchronize();

    printf("\n========================================\n");
    printf("Program execution complete!\n");
    printf("========================================\n");

    return 0;
}
